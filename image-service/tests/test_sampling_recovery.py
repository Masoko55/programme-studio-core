import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from app.services.comfyui_client import _palette_only_recovery, _recoverable_palette_error, build_workflow
from app.services.sampling_profiles import select_sampling_profile, summarize_failures


def rejected(category: str, attempt: int) -> dict:
    reasons = {
        "PALETTE_OFF": "off-palette ratio 0.77",
        "RASTER": "fine repetitive vertical raster-line structure",
        "HUMAN": "human figure or face",
    }
    return {"attempt_count": attempt, "rejection_reason": reasons[category],
            "failure_category": "INITIAL"}


class SamplingRecoveryTests(unittest.TestCase):
    def test_large_palette_drift_is_not_recoloured(self):
        self.assertTrue(_recoverable_palette_error("off-palette ratio 0.20"))
        self.assertFalse(_recoverable_palette_error("off-palette ratio 0.77"))
        self.assertFalse(_recoverable_palette_error("raster-line structure"))

    def test_history_uses_outcome_not_prompt_correction(self):
        record = {"status": "rejected", "attempt_count": 4,
                  "rejected_attempts": [rejected("RASTER", 1), rejected("RASTER", 2),
                                        rejected("HUMAN", 3)],
                  **rejected("RASTER", 4)}
        summary = summarize_failures(record)
        self.assertEqual(summary.counts["RASTER"], 3)
        self.assertEqual(summary.consecutive_count, 1)

    def test_sdxl_palette_and_sd35_raster_escalate_separately(self):
        for count, expected in ((0, "baseline"), (1, "palette_guidance"),
                                (2, "palette_alternate"), (4, "palette_strict")):
            history = summarize_failures({"status": "rejected", "attempt_count": count,
                                          "rejected_attempts": [rejected("PALETTE_OFF", i)
                                                                for i in range(1, count)],
                                          **(rejected("PALETTE_OFF", count) if count else {})})
            self.assertEqual(select_sampling_profile("sdxl-1-0", max(1, count + 1), history).name,
                             expected)
        for count, expected in ((0, "official_baseline"), (1, "official_baseline"),
                                (2, "alternate_sampler"), (3, "revised_schedule"),
                                (4, "raster_rescue_trial")):
            history = summarize_failures({"status": "rejected", "attempt_count": count,
                                          "rejected_attempts": [rejected("RASTER", i)
                                                                for i in range(1, count)],
                                          **(rejected("RASTER", count) if count else {})})
            self.assertEqual(select_sampling_profile("sd-3-5-medium", max(1, count + 1), history).name,
                             expected)
        self.assertIsNone(select_sampling_profile("flux-2", 4, summarize_failures(None)))
        self.assertEqual(select_sampling_profile("sd-3-5-medium", 9, summarize_failures(None)).name,
                         "raster_rescue_trial")

    def test_workflows_receive_dynamic_sampling_values_and_preserve_encoders(self):
        for engine in ("sdxl-1-0", "sd-3-5-medium"):
            profile = select_sampling_profile(engine, 9, summarize_failures(None))
            workflow = build_workflow(engine, "subject", "", 123, "test", sampling_profile=profile)
            sampler = workflow["3"]["inputs"]
            self.assertEqual(sampler["steps"], profile.steps)
            self.assertEqual(sampler["cfg"], profile.cfg)
            self.assertEqual(sampler["sampler_name"], profile.sampler_name)
            self.assertEqual(sampler["scheduler"], profile.scheduler)
            if engine == "sd-3-5-medium":
                self.assertEqual(workflow["55"]["inputs"]["shift"], profile.shift)
                self.assertEqual(workflow["16"]["class_type"], "CLIPTextEncodeSD3")
                self.assertEqual(workflow["54"]["class_type"], "TripleCLIPLoader")
                baseline = build_workflow(engine, "subject", "", 123, "test")
                self.assertNotIn("55", baseline)
                self.assertEqual(baseline["3"]["inputs"]["model"], ["4", 0])

    def test_recolour_requires_raw_structure_and_revalidates_result(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "raw.png"
            target = Path(directory) / "normalized.png"
            Image.new("RGB", (32, 32), (100, 120, 130)).save(source)
            calls = []

            def validate(path, **kwargs):
                calls.append((path, kwargs.get("enforce_palette", True)))
                return {"structural_edge_density": 0.2, "sha256": "test"}

            with patch("app.services.comfyui_client.validate_background", side_effect=validate), \
                 patch("app.services.comfyui_client._edge_geometry_similarity", return_value=0.9):
                _palette_only_recovery(source, target, "ABCDEF-123456", "royal blue", "red")
            self.assertEqual(calls, [(source, False), (target, True)])
            self.assertTrue(target.exists())

            target.unlink()
            with patch("app.services.comfyui_client.validate_background",
                       side_effect=ValueError("raster structure")), \
                 patch("app.services.comfyui_client.constrain_to_requested_palette") as colour:
                with self.assertRaisesRegex(ValueError, "raster"):
                    _palette_only_recovery(source, target, "ABCDEF-123456", "royal blue", "red")
                colour.assert_not_called()
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
