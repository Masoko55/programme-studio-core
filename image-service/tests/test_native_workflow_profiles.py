import unittest
import json
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from unittest.mock import AsyncMock

from app.config.settings import settings
from app.services.comfyui_client import (
    _attempt_budget_exhausted,
    _engine_positive_prompt,
    _transport_prompts,
    ComfyUIClient,
    ComfyUIError,
    _valid_background_dimensions,
    build_workflow,
)
from app.services.candidate_spec import build_candidate_spec
from app.services.prompt_compiler import compile_candidate_prompt
from app.services.image_execution import _resume_retry_context, _prepare_engine
from app.services.image_execution import execute_image_job
from app.services.image_execution import _pre_attempt_slot_is_clean
from app.schemas.image_job_state import ImageJobState, ImageOutputRecord
from app.services.job_persistence import load_image_job_state, persist_image_job_state


class NativeWorkflowProfileTests(unittest.TestCase):
    def test_release_accepts_low_torch_memory_with_high_physical_reservation(self):
        class Response:
            def __init__(self, data):
                self.data = data

            def json(self):
                return self.data

        async def request(_self, method, path, **_kwargs):
            if path == "/queue":
                return Response({"queue_running": [], "queue_pending": []})
            if path == "/free":
                return Response({})
            if path == "/system_stats":
                return Response({"devices": [{"torch_vram_total": 32 * 1024 * 1024,
                                               "vram_total": 24 * 1024**3,
                                               "vram_free": 18 * 1024**3}]})
            raise AssertionError(path)

        clock = iter(range(1000000))
        with patch.object(ComfyUIClient, "request", new=request), \
             patch("app.services.comfyui_client.time.monotonic", side_effect=lambda: next(clock)), \
             patch("app.services.comfyui_client.asyncio.sleep", new_callable=AsyncMock) as sleep:
            asyncio.run(ComfyUIClient().release_models())
        sleep.assert_not_awaited()

    def test_release_times_out_when_torch_allocation_remains_high(self):
        class Response:
            def __init__(self, data):
                self.data = data

            def json(self):
                return self.data

        async def request(_self, _method, path, **_kwargs):
            if path == "/queue":
                return Response({"queue_running": [], "queue_pending": []})
            if path == "/free":
                return Response({})
            if path == "/system_stats":
                return Response({"devices": [{"torch_vram_total": 128 * 1024**2,
                                               "vram_total": 24 * 1024**3,
                                               "vram_free": 24 * 1024**3}]})
            raise AssertionError(path)

        clock = iter(range(1000000))
        with patch.object(ComfyUIClient, "request", new=request), \
             patch("app.services.comfyui_client.time.monotonic", side_effect=lambda: next(clock)), \
             patch("app.services.comfyui_client.asyncio.sleep", new_callable=AsyncMock):
            with self.assertRaisesRegex(ComfyUIError, "not confirmed"):
                asyncio.run(ComfyUIClient().release_models())

    def test_release_keeps_queue_and_api_failures_hard(self):
        class Response:
            def __init__(self, data):
                self.data = data

            def json(self):
                return self.data

        async def request(_self, _method, path, **_kwargs):
            if path == "/queue":
                return Response({"queue_running": [[1]], "queue_pending": []})
            raise AssertionError(path)

        with patch.object(ComfyUIClient, "request", new=request):
            with self.assertRaisesRegex(ComfyUIError, "active jobs"):
                asyncio.run(ComfyUIClient().release_models())
        for failed_path in ("/free", "/system_stats"):
            async def failing(_self, _method, path, **_kwargs):
                if path == "/queue":
                    return Response({"queue_running": [], "queue_pending": []})
                if path == failed_path:
                    raise ComfyUIError(f"{failed_path} unavailable")
                return Response({})

            with patch.object(ComfyUIClient, "request", new=failing):
                with self.assertRaisesRegex(ComfyUIError, "unavailable"):
                    asyncio.run(ComfyUIClient().release_models())

    def test_pre_attempt_evidence_rejects_histories_and_corrupt_record(self):
        output = ImageOutputRecord(engine_id="sdxl-1-0", engine_label="SDXL",
                                   direction_id="B", status="pending")
        with TemporaryDirectory() as directory, \
             patch.object(settings, "programme_data_path", Path(directory)):
            self.assertTrue(_pre_attempt_slot_is_clean("ABCDEF-123456", output, {}))
            for record in ({"attempt_count": 1},
                           {"status": "submitted", "attempt_count": 0},
                           {"failed_attempts": [{"attempt_count": 1}]},
                           {"rejected_attempts": [{"attempt_count": 1}]}):
                self.assertFalse(_pre_attempt_slot_is_clean("ABCDEF-123456", output, record))
            output.status = "failed"
            self.assertFalse(_pre_attempt_slot_is_clean("ABCDEF-123456", output, {}))
            output.status = "pending"
            path = Path(directory) / "ABCDEF-123456" / "backgrounds" / "sdxl-1-0" / "image-b.json"
            path.parent.mkdir(parents=True)
            path.write_text("{broken", encoding="utf-8")
            self.assertFalse(_pre_attempt_slot_is_clean("ABCDEF-123456", output, {}))

    def test_same_engine_reuse_does_not_bypass_foreign_queue_guard(self):
        class Response:
            def json(self):
                return {"queue_running": [[1, "foreign", {}, {"programme_request_id": "other"}]],
                        "queue_pending": []}

        class Client:
            async def request(self, *_args):
                return Response()

            async def release_models(self):
                raise AssertionError("foreign work must not be released")

        output = ImageOutputRecord(engine_id="sdxl-1-0", engine_label="SDXL",
                                   direction_id="B", status="pending")
        with TemporaryDirectory() as directory, \
             patch.object(settings, "programme_data_path", Path(directory)):
            with self.assertRaisesRegex(RuntimeError, "another active job"):
                asyncio.run(_prepare_engine(Client(), "ABCDEF-123456", output, "sdxl-1-0"))

    def test_preparation_failure_is_resumable_without_consuming_attempt(self):
        from app.services import image_execution as execution

        reference = "ABCDEF-123456"
        source = {"brief": {}, "directions": []}
        plan = {"execution_order": [{"engine_id": "flux-2", "direction": "A"},
                                    {"engine_id": "sdxl-1-0", "direction": "B"}]}
        outputs = [ImageOutputRecord(engine_id="flux-2", engine_label="FLUX", direction_id="A",
                                     status="complete", output_path="accepted.png"),
                   ImageOutputRecord(engine_id="sdxl-1-0", engine_label="SDXL", direction_id="B",
                                     status="pending")]
        state = ImageJobState(reference_number=reference, status="ready", current_stage="planned",
                              total_outputs=2, completed_outputs=1, outputs=outputs)

        class Lease:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

        class Response:
            def json(self):
                return {"models": [], "queue_running": [], "queue_pending": []}

            def raise_for_status(self):
                return None

        class Http:
            def __init__(self, **_kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def get(self, _url):
                return Response()

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def request(self, *_args):
                return Response()

            async def release_models(self):
                return None

        async def healthy(_client, _reference, output, _current):
            return output.engine_id

        async def wave(**kwargs):
            self.assertEqual(kwargs["output"].status, "pending")
            self.assertEqual(kwargs["output"].seed, None)
            self.assertEqual(execution._resume_retry_context(
                execution._candidate_record(reference, "sdxl-1-0", "B"))[0], 0)
            from app.services.job_persistence import update_output
            update_output(kwargs["state"], "sdxl-1-0", "B", "complete", "new.png", "sha")
            return True, 1

        with TemporaryDirectory() as directory, \
             patch.object(settings, "programme_data_path", Path(directory)), \
             patch.object(execution, "GPULease", Lease), \
             patch.object(execution.httpx, "AsyncClient", Http), \
             patch.object(execution, "ComfyUIClient", Client), \
             patch.object(execution, "load_prompts_document", return_value=source), \
             patch.object(execution, "build_execution_plan", return_value=plan), \
             patch.object(execution, "_attempt_output_wave", side_effect=wave) as attempted:
            persist_image_job_state(state)
            with patch.object(execution, "_prepare_engine", side_effect=ComfyUIError("temporary release timeout")):
                first = asyncio.run(execute_image_job(reference))
            self.assertEqual(first.outputs[0].status, "complete")
            self.assertEqual(first.outputs[1].status, "pending")
            self.assertEqual(first.status, "processing")
            self.assertEqual(first.outputs[1].seed, None)
            self.assertFalse((Path(directory) / reference / "backgrounds" / "sdxl-1-0" / "image-b.json").exists())
            attempted.assert_not_called()
            with patch.object(execution, "_prepare_engine", side_effect=healthy):
                second = asyncio.run(execute_image_job(reference))
            self.assertEqual(second.outputs[0].output_path, "accepted.png")
            self.assertEqual(second.outputs[1].output_path, "new.png")
            self.assertEqual(load_image_job_state(reference).completed_outputs, 2)
            attempted.assert_called_once()

    def test_candidate_record_stores_compiled_spec_before_remote_generation(self):
        document = {
            "reference_number": "ABCDEF-123456",
            "brief": {
                "event_type": "birthday", "theme": "comic city",
                "creative_description": "energetic celebration",
                "background_inspiration": "city skyline and web arcs",
                "theme_reference_treatment": "comic geometry without a character",
                "primary_colour": "royal blue", "secondary_colour": "red",
            },
            "directions": [{"direction": {
                "direction_id": "B", "role": "expressive",
                "positive_prompt": "dynamic asymmetric skyline",
                "composition_strategy": "asymmetric layered skyline",
                "focal_strategy": "off-centre tower focus",
                "depth_strategy": "foreground to background depth",
                "border_or_perimeter_strategy": "open irregular perimeter",
                "title_safe_region_strategy": "quiet upper reading area",
                "programme_safe_region_strategy": "quiet lower reading area",
                "negative_prompt": "people", "layout_guidance": {},
            }}],
        }
        spec = build_candidate_spec(document, settings.engine_2_id, "B")
        compiled = compile_candidate_prompt(spec, 1)

        class Response:
            def __init__(self, data):
                self.data = data

            def json(self):
                return self.data

        async def fake_request(self, method, path, **kwargs):
            if path == "/upload/image":
                return Response({"name": kwargs["files"]["image"][0], "subfolder": ""})
            if path == "/object_info":
                return Response({})
            if path == "/queue":
                return Response({"queue_running": [], "queue_pending": []})
            if path == "/prompt":
                return Response({"node_errors": {"diagnostic": "stop before generation"}})
            raise AssertionError(path)

        with TemporaryDirectory() as directory:
            job_dir = Path(directory) / "ABCDEF-123456"
            job_dir.mkdir()
            (job_dir / "prompts.json").write_text(json.dumps(document), encoding="utf-8")
            with patch.object(settings, "programme_data_path", Path(directory)), \
                 patch.object(settings, "template_conditioning_enabled", True), \
                 patch("app.services.comfyui_client.validate_workflow"), \
                 patch.object(ComfyUIClient, "request", new=fake_request):
                async def check():
                    async with ComfyUIClient() as client:
                        with self.assertRaises(ComfyUIError):
                            await client.generate_image(
                                spec.reference_number, spec.engine_id, spec.direction_id,
                                compiled.positive, compiled.negative,
                                compiled_prompt=compiled, spec_sha256=spec.spec_sha256,
                                direction_role=spec.direction_role,
                            )
                import asyncio
                asyncio.run(check())
            saved = json.loads((job_dir / "backgrounds" / spec.engine_id / "image-b.json")
                                  .read_text(encoding="utf-8"))
            self.assertEqual(saved["spec_sha256"], spec.spec_sha256)
            self.assertEqual(saved["direction_role"], "expressive")
            self.assertEqual(saved["compiled_prompt"]["positive"], compiled.positive)
            self.assertEqual(saved["retry_stage"], "normal")
            self.assertEqual(saved["sampling_profile"]["name"], "baseline")
            self.assertEqual(saved["workflow"]["3"]["inputs"]["sampler_name"], "dpmpp_2m")
            self.assertEqual(saved["generation_mode"], "native_text_to_image")
            self.assertEqual(saved["GENERATION_MODE"], "native_text_to_image")
            self.assertFalse(saved["TEMPLATE_CONDITIONING_ENABLED"])
            self.assertFalse(saved["MASKED_DENOISING_ENABLED"])
            self.assertIn("city skyline", saved["visual_anchors"][0])
            self.assertFalse(saved["template_conditioning"]["enabled"])
            self.assertFalse(saved["template_conditioning"]["masked_denoising_enabled"])
            self.assertEqual(saved["workflow"]["3"]["inputs"]["latent_image"], ["53", 0])
            self.assertFalse(any(node["class_type"] in {"LoadImage", "VAEEncode", "SetLatentNoiseMask"}
                                 for node in saved["workflow"].values()))

    def test_resume_keeps_attempt_budget_and_correction_context(self):
        reason = "Generated background became a smooth gradient"
        self.assertEqual(
            _resume_retry_context({"status": "rejected", "attempt_count": 8,
                                   "rejection_reason": reason}),
            (8, reason),
        )
        self.assertEqual(
            _resume_retry_context({"status": "submitted", "attempt_count": 9,
                                   "rejected_attempts": [{"attempt_count": 8,
                                                          "rejection_reason": reason}]}),
            (8, reason),
        )
        self.assertEqual(_resume_retry_context({"status": "rejected", "attempt_count": 9,
                                                "rejection_reason": reason})[0], 9)
        with self.assertRaises(ValueError):
            _resume_retry_context({"status": "submitted", "attempt_count": 5})

    def test_existing_candidate_rejects_changed_spec_before_submission(self):
        with TemporaryDirectory() as directory:
            record = Path(directory) / "ABCDEF-123456" / "backgrounds" / settings.engine_2_id / "image-a.json"
            record.parent.mkdir(parents=True)
            record.write_text(json.dumps({"status": "rejected", "attempt_count": 1,
                                          "spec_sha256": "old"}), encoding="utf-8")
            with patch.object(settings, "programme_data_path", Path(directory)):
                async def check():
                    client = ComfyUIClient()
                    with self.assertRaisesRegex(ComfyUIError, "specification changed"):
                        await client.generate_image("ABCDEF-123456", settings.engine_2_id,
                                                    "A", "prompt", "", spec_sha256="new")
                import asyncio
                asyncio.run(check())

    def test_compiled_native_prompts_cross_transport_unchanged(self):
        document = {
            "reference_number": "ABCDEF-123456",
            "brief": {
                "event_type": "birthday", "theme": "comic city",
                "creative_description": "dynamic skyline",
                "background_inspiration": "angular city skyline and spiderweb geometry",
                "theme_reference_treatment": "radial webs, no character",
                "primary_colour": "royal blue", "secondary_colour": "red",
            },
            "directions": [{"direction": {
                "direction_id": "B", "role": "expressive",
                "positive_prompt": "dynamic asymmetric city architecture",
                "composition_strategy": "asymmetric layered skyline",
                "focal_strategy": "off-centre tower focus",
                "depth_strategy": "foreground to background depth",
                "border_or_perimeter_strategy": "open irregular perimeter",
                "title_safe_region_strategy": "quiet upper reading area",
                "programme_safe_region_strategy": "quiet lower reading area",
                "negative_prompt": "people", "layout_guidance": {},
            }}],
        }
        for engine in (settings.engine_2_id, settings.engine_3_id):
            compiled = compile_candidate_prompt(
                build_candidate_spec(document, engine, "B"), 5,
                "Generated background contains colours that were not requested",
            )
            positive, negative = _transport_prompts(
                engine, "ignored", "ignored", "royal blue", "red", compiled,
            )
            self.assertEqual((positive, negative), (compiled.positive, compiled.negative))
            workflow = build_workflow(engine, positive, negative, 123, "test/native", compiled)
            if engine == settings.engine_2_id:
                self.assertEqual(workflow["16"]["class_type"], "CLIPTextEncodeSDXL")
                self.assertEqual(workflow["16"]["inputs"]["text_g"], compiled.clip_g)
                self.assertEqual(workflow["16"]["inputs"]["text_l"], compiled.clip_l)
                self.assertIn("spiderweb", workflow["16"]["inputs"]["text_g"])
                self.assertIn("spiderweb", workflow["16"]["inputs"]["text_l"])
                self.assertIn("Correction:", workflow["16"]["inputs"]["text_g"])
                self.assertIn("Correction:", workflow["16"]["inputs"]["text_l"])
                self.assertEqual(workflow["40"]["inputs"]["text"], compiled.negative)
            else:
                self.assertEqual(workflow["16"]["inputs"]["clip_l"], compiled.clip_l)
                self.assertEqual(workflow["16"]["inputs"]["clip_g"], compiled.clip_g)
                self.assertEqual(workflow["16"]["inputs"]["t5xxl"], compiled.t5)
                self.assertEqual(workflow["40"]["inputs"]["t5xxl"], compiled.negative)

    def test_sd35_keeps_its_subject_without_negative_concepts_in_positive(self):
        prompt = _engine_positive_prompt(
            settings.engine_3_id,
            "Comic city skyline with red web geometry on royal blue",
            "royal blue",
            "red",
        )

        self.assertTrue(prompt.startswith("Comic city skyline"))
        self.assertNotIn("no people", prompt)
        self.assertNotIn("STRICT COLOUR PALETTE CONTRACT", prompt)

    def test_persisted_candidate_cannot_exceed_nine_attempts(self):
        self.assertFalse(_attempt_budget_exhausted({"attempt_count": 8}))
        self.assertTrue(_attempt_budget_exhausted({"attempt_count": 9}))
        self.assertFalse(_attempt_budget_exhausted({"attempt_count": "unknown"}))

    def test_sdxl_uses_its_native_portrait_bucket(self):
        workflow = build_workflow(
            settings.engine_2_id,
            "large floral forms around the page edge",
            "text, people",
            123,
            "test/sdxl",
        )

        self.assertEqual(
            workflow["53"]["inputs"]["width"],
            settings.sdxl_generation_width,
        )
        self.assertEqual(
            workflow["53"]["inputs"]["height"],
            settings.sdxl_generation_height,
        )

    def test_sd35_uses_its_native_bucket_and_empty_negative_conditioning(self):
        workflow = build_workflow(
            settings.engine_3_id,
            "large floral forms around the page edge",
            "text, people, scanlines",
            123,
            "test/sd35",
        )

        self.assertEqual(
            workflow["53"]["inputs"]["width"],
            settings.sd35_generation_width,
        )
        self.assertEqual(
            workflow["53"]["inputs"]["height"],
            settings.sd35_generation_height,
        )
        self.assertEqual(workflow["40"]["inputs"]["clip_l"], "")
        self.assertEqual(workflow["40"]["inputs"]["clip_g"], "")
        self.assertEqual(workflow["40"]["inputs"]["t5xxl"], "")
        self.assertEqual(workflow["16"]["inputs"]["clip"], ["54", 0])
        self.assertEqual(workflow["40"]["inputs"]["clip"], ["54", 0])
        self.assertEqual(workflow["54"]["class_type"], "TripleCLIPLoader")
        self.assertEqual(workflow["8"]["inputs"]["vae"], ["4", 2])
        self.assertNotIn("56", workflow)

    def test_native_dimensions_are_accepted(self):
        self.assertIn(
            (
                settings.sdxl_generation_width,
                settings.sdxl_generation_height,
            ),
            _valid_background_dimensions(),
        )
        self.assertIn(
            (
                settings.sd35_generation_width,
                settings.sd35_generation_height,
            ),
            _valid_background_dimensions(),
        )

if __name__ == "__main__":
    unittest.main()
