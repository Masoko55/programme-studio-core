"""Offline checks for the simplified production path."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from PIL import Image

from app.services.candidate_spec import build_candidate_spec
from app.services.execution_plan import build_execution_plan
from app.services.image_execution import _finalize_generation_status
from app.services.prompt_compiler import compile_candidate_prompt, runtime_retry_category
from app.services.comfyui_client import validate_background
from app.schemas.image_job_state import ImageJobState, ImageOutputRecord
from tests.test_semantic_foundation import document


class SimplifiedPipelineTests(unittest.TestCase):
    def test_bad_semantic_metadata_does_not_block_valid_brief(self):
        source = document()
        source["brief"]["semantic_contract"] = "stale metadata"
        source["brief"]["semantic_contract_sha256"] = "bad hash"
        spec = build_candidate_spec(source, "sdxl-1-0", "A")
        self.assertEqual(spec.visual_design["primary_visual_subject"],
                         source["brief"]["background_inspiration"])

    def test_nine_slots_are_grouped_by_direction(self):
        source = document()
        template = source["directions"][0]
        source["directions"] = [{**template, "direction": {**template["direction"], "direction_id": direction}}
                                for direction in "ABC"]
        plan = build_execution_plan(source)
        slots = [(item["direction"], item["engine_id"]) for item in plan["execution_order"]]
        self.assertEqual(len(slots), 9)
        self.assertEqual([direction for direction, _ in slots], list("AAABBBCCC"))
        for direction in "ABC":
            self.assertEqual(len({engine for slot, engine in slots if slot == direction}), 3)

    def test_concise_prompt_keeps_subject_mood_palette_and_spatial_difference(self):
        source = document()
        for engine in ("sdxl-1-0", "sd-3-5-medium", "flux-2"):
            spec = build_candidate_spec(source, engine, "A")
            compiled = compile_candidate_prompt(spec, 1, concise=True)
            self.assertIn(spec.visual_design["primary_visual_subject"], compiled.positive)
            self.assertIn(spec.primary_colour, compiled.positive)
            self.assertLess(len(compiled.clip_l.split()), 35)
            self.assertLess(len(compiled.clip_g.split()), 100)
            for phrase in ("masterpiece", "trending on artstation", "programme", "safe region"):
                self.assertNotIn(phrase, " ".join((compiled.positive, compiled.clip_l,
                                                   compiled.clip_g, compiled.t5 or "")).casefold())
            if engine == "sd-3-5-medium":
                self.assertLess(len(compiled.t5.split()), 120)
            if engine == "flux-2":
                self.assertFalse(compiled.negative_prompt_channel_active)
                self.assertIn("Environment-only", compiled.positive)

    def test_unrequested_object_in_direction_strategy_does_not_reach_renderer(self):
        source = document()
        source["directions"][0]["direction"]["focal_strategy"] = (
            "Glowing portal at the left with a diagonal visual flow")
        compiled = compile_candidate_prompt(
            build_candidate_spec(source, "flux-2", "A"), 1, concise=True)
        self.assertNotIn("portal", compiled.positive.casefold())
        self.assertIn("focal placement", compiled.positive)

    def test_retry_taxonomy_is_small(self):
        reasons = {"TEXT", "HUMAN", "PALETTE_OFF", "SUBJECT_DRIFT",
                   "RUNTIME", "DENSITY_DRIFT", "MOTIF_DRIFT"}
        self.assertEqual({runtime_retry_category(reason) for reason in reasons},
                         {"TEXT_OR_LOGO", "HUMAN", "PALETTE", "SUBJECT",
                          "GENERATION_FAILURE", "UNKNOWN"})

    def test_one_failed_slot_does_not_hide_valid_backgrounds(self):
        outputs = [ImageOutputRecord(engine_id=f"engine-{i}", engine_label=str(i),
                                     direction_id="A", status="complete" if i != 1 else "failed")
                   for i in range(3)]
        state = ImageJobState(reference_number="ABCDEF-123456", status="processing",
                              current_stage="generating", total_outputs=3,
                              completed_outputs=2, outputs=outputs)
        with patch("app.services.image_execution.persist_image_job_state"):
            _finalize_generation_status(state, 2, False)
        self.assertEqual(state.status, "awaiting_selection")
        self.assertIn("2/3", state.error)

    def test_density_and_visual_quality_are_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "image.png"
            Image.new("RGB", (896, 1280), "navy").save(path)
            ocr = {"confirmed_tokens": [], "ocr_diagnostics": {"decision": "OCR_COMPLETE"},
                   "text_detected": False, "tokens": [], "text_lines": [], "text_area_ratio": 0}
            human = {"detected": False, "ambiguous_face_signal": False,
                     "evidence": [], "strong_signals": [], "weak_signals": [],
                     "people": [], "faces": [], "profiles": [], "upper_bodies": []}
            density = {"decision": "DENSITY_DRIFT", "macro_form_balance": 0}
            with patch("app.services.comfyui_client._cached_text_evidence", return_value=ocr), \
                 patch("app.services.comfyui_client.detect_human_signals", side_effect=lambda _: dict(human)), \
                 patch("app.services.comfyui_client.boilerplate_metrics", return_value={"boilerplate_score": .6}), \
                 patch("app.services.comfyui_client.validate_visual_quality", side_effect=ValueError("FLAT: low detail")), \
                 patch("app.services.comfyui_client.assess_scene_density", return_value=density):
                result = validate_background(path, enforce_palette=False,
                                             visual_design={"density_preference": "moderate"})
            self.assertEqual(result["validation"]["decision"], "accepted")
            self.assertIn("DENSITY_DRIFT", result["validation"]["diagnostic_findings"])


if __name__ == "__main__":
    unittest.main()
