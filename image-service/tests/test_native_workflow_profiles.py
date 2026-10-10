import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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
from app.services.image_execution import _resume_retry_context


class NativeWorkflowProfileTests(unittest.TestCase):
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

    def test_sd35_keeps_its_subject_before_safety_suffix(self):
        prompt = _engine_positive_prompt(
            settings.engine_3_id,
            "Comic city skyline with red web geometry on royal blue",
            "royal blue",
            "red",
        )

        self.assertTrue(prompt.startswith("Comic city skyline"))
        self.assertIn("no people", prompt)
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
