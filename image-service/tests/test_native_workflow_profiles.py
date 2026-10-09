import unittest

from app.config.settings import settings
from app.services.comfyui_client import (
    _attempt_budget_exhausted,
    _engine_positive_prompt,
    _valid_background_dimensions,
    build_workflow,
)


class NativeWorkflowProfileTests(unittest.TestCase):
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
        self.assertEqual(workflow["56"]["inputs"]["vae_name"], settings.sd35_vae)
        self.assertEqual(workflow["8"]["inputs"]["vae"], ["56", 0])

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
