import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.config.settings import settings
from app.services.comfyui_client import ComfyUIClient, ComfyUIError, _candidate_attempt_number
from app.services.image_execution import _attempt_output_wave, _retry_candidate_after_error
from app.services.execution_plan import build_execution_plan
from app.services.sampling_profiles import select_sampling_profile


REASON = "Generated background failed visual quality validation: smooth gradient colour wash"


def document():
    return {
        "reference_number": "ABCDEF-123456",
        "brief": {
            "event_type": "birthday", "theme": "comic city",
            "creative_description": "energetic party",
            "background_inspiration": "angular comic city skyline at the edges",
            "theme_reference_treatment": "spiderweb-inspired geometry without a character",
            "primary_colour": "royal blue", "secondary_colour": "red",
        },
        "directions": [{"direction": {
            "direction_id": "B", "role": "expressive",
            "positive_prompt": "dynamic city architecture and bold web arcs",
            "composition_strategy": "asymmetric layered skyline",
            "focal_strategy": "off-centre tower focus",
            "depth_strategy": "foreground to background depth",
            "border_or_perimeter_strategy": "open irregular perimeter",
            "title_safe_region_strategy": "quiet upper reading area",
            "programme_safe_region_strategy": "quiet lower reading area",
            "negative_prompt": "people, text", "layout_guidance": {},
        }}],
    }


class FailingEngine:
    def __init__(self):
        self.calls = []

    async def generate(self, positive_prompt, negative_prompt, **kwargs):
        self.calls.append((positive_prompt, negative_prompt, kwargs))
        raise ComfyUIError(REASON)


class CandidateRetryIntegrationTests(unittest.TestCase):
    def test_nonretryable_error_is_preserved(self):
        error = ComfyUIError("Remote workflow is invalid")
        output = SimpleNamespace(engine_id=settings.engine_2_id, direction_id="B")

        async def check():
            with patch("app.services.image_execution._rejection_reason", return_value=None), \
                 patch("app.services.image_execution._candidate_was_rejected", return_value=False):
                with self.assertRaises(ComfyUIError) as raised:
                    await _retry_candidate_after_error(
                        SimpleNamespace(), output, "ABCDEF-123456", None, 0, error,
                    )
                self.assertIs(raised.exception, error)

        asyncio.run(check())

    def test_execution_plan_keeps_three_engines_and_three_directions(self):
        source = document()
        prototype = source["directions"][0]["direction"]
        source["directions"] = [
            {"direction": {**prototype, "direction_id": direction}}
            for direction in "ABC"
        ]
        plan = build_execution_plan(source)
        self.assertEqual(plan["total_outputs"], 9)
        self.assertEqual(
            {(item["engine_id"], item["direction"]) for item in plan["execution_order"]},
            {(engine, direction) for engine in (settings.engine_1_id,
                                                 settings.engine_2_id,
                                                 settings.engine_3_id)
             for direction in "ABC"},
        )

    def run_wave(self, existing, reason=REASON):
        engine = FailingEngine()
        output = SimpleNamespace(engine_id=settings.engine_2_id, direction_id="B")

        async def no_sleep(_):
            return None

        async def execute():
            return await _attempt_output_wave(
                state=SimpleNamespace(), document=document(), output=output,
                reference_number="ABCDEF-123456", primary_colour="royal blue",
                secondary_colour="red",
            )

        with patch("app.services.image_execution._candidate_record", return_value=existing), \
             patch("app.services.image_execution._candidate_was_rejected", return_value=True), \
             patch("app.services.image_execution._rejection_reason", return_value=reason), \
             patch("app.services.image_execution.get_engine", return_value=engine), \
             patch("app.services.image_execution.update_output"), \
             patch("app.services.image_execution.asyncio.sleep", new=no_sleep):
            result = asyncio.run(execute())
        return engine.calls, result

    def test_one_wave_has_nine_spec_locked_attempts(self):
        calls, result = self.run_wave({})
        self.assertEqual(result, (False, 0))
        self.assertEqual(len(calls), 9)
        self.assertEqual(len({call[2]["spec_sha256"] for call in calls}), 1)
        for positive, _, kwargs in calls:
            for anchor in ("city skyline", "spiderweb", "royal blue", "red"):
                self.assertIn(anchor, positive.lower())
            self.assertNotIn("expressive", positive.lower())
            self.assertEqual(kwargs["compiled_prompt"].positive, positive)

    def test_other_quality_rejections_consume_the_same_nine_attempt_budget(self):
        for reason in ("HUMAN: human silhouette", "MOTIF_DRIFT: missing lattice",
                       "AUDIENCE_DRIFT: unsuitable atmosphere"):
            with self.subTest(reason=reason):
                calls, result = self.run_wave({}, reason)
                self.assertEqual(result, (False, 0))
                self.assertEqual(len(calls), 9)

    def test_each_attempt_selects_profile_from_persisted_prior_failures(self):
        engine = FailingEngine()
        output = SimpleNamespace(engine_id=settings.engine_2_id, direction_id="B")
        palette_reason = "Generated background contains colours that were not requested (off-palette ratio 0.20)"

        def record(*_):
            completed = len(engine.calls)
            if not completed:
                return {}
            attempts = [{"attempt_count": number, "rejection_reason": palette_reason}
                        for number in range(1, completed + 1)]
            return {**attempts[-1], "status": "rejected", "rejected_attempts": attempts[:-1]}

        async def no_sleep(_):
            return None

        with patch("app.services.image_execution._candidate_record", side_effect=record), \
             patch("app.services.image_execution._candidate_was_rejected", return_value=True), \
             patch("app.services.image_execution._rejection_reason", return_value=palette_reason), \
             patch("app.services.image_execution.get_engine", return_value=engine), \
             patch("app.services.image_execution.update_output"), \
             patch("app.services.image_execution.asyncio.sleep", new=no_sleep), \
             patch("app.services.image_execution.select_sampling_profile", wraps=select_sampling_profile) as selector:
            asyncio.run(_attempt_output_wave(
                state=SimpleNamespace(), document=document(), output=output,
                reference_number="ABCDEF-123456", primary_colour="royal blue",
                secondary_colour="red"))
        self.assertEqual([call.args[1] for call in selector.call_args_list], list(range(1, 10)))
        self.assertEqual([call.args[2].count("PALETTE_OFF") for call in selector.call_args_list],
                         list(range(9)))
        self.assertEqual(engine.calls[3][2]["sampling_profile"].name, "palette_alternate")
        self.assertEqual(engine.calls[2][2]["sampling_profile"].name, "palette_guidance")
        self.assertEqual(engine.calls[6][2]["sampling_profile"].name, "palette_strict")

    def test_sd35_resume_uses_attempt_six_and_prior_history(self):
        engine = FailingEngine()
        output = SimpleNamespace(engine_id=settings.engine_3_id, direction_id="B")
        reason = "Generated background contains colours that were not requested (off-palette ratio 0.20)"

        def record(*_):
            completed = 5 + len(engine.calls)
            attempts = [{"attempt_count": number, "rejection_reason": reason}
                        for number in range(1, completed + 1)]
            return {**attempts[-1], "status": "rejected", "rejected_attempts": attempts[:-1]}

        async def no_sleep(_):
            return None

        with patch("app.services.image_execution._candidate_record", side_effect=record), \
             patch("app.services.image_execution._candidate_was_rejected", return_value=True), \
             patch("app.services.image_execution._rejection_reason", return_value=reason), \
             patch("app.services.image_execution.get_engine", return_value=engine), \
             patch("app.services.image_execution.update_output"), \
             patch("app.services.image_execution.asyncio.sleep", new=no_sleep), \
             patch("app.services.image_execution.select_sampling_profile", wraps=select_sampling_profile) as selector:
            asyncio.run(_attempt_output_wave(
                state=SimpleNamespace(), document=document(), output=output,
                reference_number="ABCDEF-123456", primary_colour="royal blue",
                secondary_colour="red"))
        self.assertEqual([call.args[1] for call in selector.call_args_list], [6, 7, 8, 9])
        self.assertEqual([call.args[2].count("PALETTE_OFF") for call in selector.call_args_list],
                         [5, 6, 7, 8])
        self.assertEqual([call[2]["sampling_profile"].name for call in engine.calls],
                         ["alternate_sampler", "raster_rescue_trial", "raster_rescue_trial", "raster_rescue_trial"])

    def test_ocr_infrastructure_failure_retries_fresh_image_within_budget(self):
        engine = FailingEngine()
        output = SimpleNamespace(engine_id=settings.engine_2_id, direction_id="B")
        reason = "OCR_TIMEOUT: text validation did not complete"
        async def fail_once(*args, **kwargs):
            engine.calls.append((args, kwargs))
            if len(engine.calls) == 1:
                raise ComfyUIError(reason)
            return {"seed": 2, "prompt_id": "second", "workflow_sha256": "workflow",
                    "output_path": "candidate.png", "sha256": "validated", "reused": False}
        engine.generate = fail_once
        with patch("app.services.image_execution._candidate_record", return_value={}), \
             patch("app.services.image_execution._rejection_reason", return_value=reason), \
             patch("app.services.image_execution.update_output") as update, \
             patch("app.services.image_execution.get_engine", return_value=engine), \
             patch("app.services.image_execution._candidate_was_rejected", return_value=True), \
             patch("app.services.image_execution.asyncio.sleep", new=AsyncMock()):
            self.assertEqual(asyncio.run(_attempt_output_wave(
                state=SimpleNamespace(), document=document(), output=output,
                reference_number="ABCDEF-123456", primary_colour="royal blue",
                secondary_colour="red")), (True, 1))
        self.assertEqual(len(engine.calls), 2)
        self.assertEqual(update.call_args.args[3], "complete")
        self.assertEqual(engine.calls[0][1]["spec_sha256"], engine.calls[1][1]["spec_sha256"])

    def test_persisted_ocr_rejection_selects_fresh_seed_and_preserves_history(self):
        previous = {"status": "rejected", "primary_failure_category": "OCR_TIMEOUT",
                    "rejection_reason": "OCR_TIMEOUT: text validation did not complete",
                    "attempt_count": 1, "seed": 17, "spec_sha256": "frozen"}
        rejected, failed = ComfyUIClient._retry_states(previous)
        self.assertTrue(rejected)
        self.assertFalse(failed)
        with patch("app.services.comfyui_client.secrets.randbits", return_value=23):
            self.assertEqual(ComfyUIClient._candidate_seed(previous, rejected, None), 23)
        self.assertEqual(_candidate_attempt_number(previous, rejected), 2)
        self.assertEqual(previous["spec_sha256"], "frozen")
        history, _ = ComfyUIClient()._prior_attempts(previous, rejected, failed)
        self.assertEqual(history[0]["primary_failure_category"], "OCR_TIMEOUT")
        self.assertEqual(history[0]["attempt_count"], 1)
        async def check():
            with tempfile.TemporaryDirectory() as directory, \
                 patch("app.services.comfyui_client.get_job_directory", return_value=Path(directory)), \
                 patch.object(ComfyUIClient, "_load_existing_candidate", return_value=(previous, None)), \
                 patch.object(ComfyUIClient, "_candidate_seed", side_effect=ValueError("fresh seed selected")):
                async with ComfyUIClient() as client:
                    with self.assertRaisesRegex(ValueError, "fresh seed selected"):
                        await client.generate_image("ABCDEF-123456", settings.engine_2_id, "B", "scene", "")
        asyncio.run(check())

    def test_repeated_ocr_infrastructure_failures_exhaust_at_nine(self):
        for reason in ("OCR_TIMEOUT: timed out", "OCR_UNAVAILABLE: worker absent",
                       "OCR_PARTIAL_INSUFFICIENT: incomplete checks",
                       "TEXT: HAPPY BIRTHDAY visible"):
            with self.subTest(reason=reason):
                calls, result = self.run_wave({}, reason=reason)
                self.assertEqual(result, (False, 0))
                self.assertEqual(len(calls), 9)
                self.assertEqual(len({call[2]["spec_sha256"] for call in calls}), 1)

    def test_resume_after_eight_attempts_has_one_remaining(self):
        calls, result = self.run_wave({
            "status": "rejected", "attempt_count": 8,
            "rejection_reason": REASON,
        })
        self.assertEqual(result, (False, 0))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2]["compiled_prompt"].retry_stage, "rescue")

    def test_exhausted_candidate_makes_no_submission(self):
        calls, result = self.run_wave({
            "status": "rejected", "attempt_count": 9,
            "rejection_reason": REASON,
        })
        self.assertEqual(result, (False, 0))
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
