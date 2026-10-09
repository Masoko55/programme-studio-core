import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.config.settings import settings
from app.services.comfyui_client import ComfyUIError
from app.services.image_execution import _attempt_output_wave
from app.services.execution_plan import build_execution_plan


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

    def run_wave(self, existing):
        engine = FailingEngine()
        output = SimpleNamespace(engine_id=settings.engine_2_id, direction_id="B")

        async def no_sleep(_):
            return None

        async def execute():
            return await _attempt_output_wave(
                client=None, state=SimpleNamespace(), document=document(), output=output,
                reference_number="ABCDEF-123456", primary_colour="royal blue",
                secondary_colour="red",
            )

        with patch("app.services.image_execution._candidate_record", return_value=existing), \
             patch("app.services.image_execution._candidate_was_rejected", return_value=True), \
             patch("app.services.image_execution._rejection_reason", return_value=REASON), \
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
            for anchor in ("city skyline", "spiderweb", "expressive", "royal blue", "red"):
                self.assertIn(anchor, positive.lower())
            self.assertEqual(kwargs["compiled_prompt"].positive, positive)

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
