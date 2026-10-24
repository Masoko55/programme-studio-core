"""Offline checks for the simplified production path."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from PIL import Image, ImageDraw

from app.services.candidate_spec import build_candidate_spec
from app.services.execution_plan import build_execution_plan
from app.services.image_execution import _finalize_generation_status
from app.services.prompt_compiler import (CompiledPrompt, _RETRY_CORRECTIONS,
                                          _concise_runtime_prompt, compile_candidate_prompt,
                                          failure_category, runtime_retry_category)
from app.services.comfyui_client import _record_safe_region_retry, validate_background
from app.services.scene_density import assess_explicit_safe_regions
from app.schemas.image_job_state import ImageJobState, ImageOutputRecord
from tests.test_semantic_foundation import document


class SimplifiedPipelineTests(unittest.TestCase):
    def test_hog_only_vision_disagreement_does_not_override_independent_human_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "image.png"
            Image.new("RGB", (896, 1280), "white").save(path)
            ocr = {"confirmed_tokens": [], "ocr_diagnostics": {"decision": "complete"},
                   "text_detected": False, "tokens": [], "text_lines": [], "text_area_ratio": 0}
            human = {"detected": True, "ambiguous_face_signal": False,
                     "strong_signals": [{"source": "hog", "weight": 1.6}], "weak_signals": [],
                     "people": [(300, 450, 70, 140)], "faces": [], "profiles": [], "upper_bodies": [],
                     "human_detection_reason": "strong HOG weight"}
            with patch("app.services.comfyui_client._cached_text_evidence", return_value=ocr), \
                 patch("app.services.comfyui_client.detect_human_signals", side_effect=lambda _: dict(human)), \
                 patch("app.services.comfyui_client.boilerplate_metrics", return_value={"boilerplate_score": 0}), \
                 patch("app.services.comfyui_client.validate_visual_quality", return_value={}):
                result = validate_background(path, enforce_palette=False, vision_human_signal=False)
                self.assertEqual(result["validation"]["decision"], "accepted")
                with self.assertRaisesRegex(ValueError, "human"):
                    validate_background(path, enforce_palette=False, vision_human_signal=None)
                human["strong_signals"].append({"source": "frontal_face", "confirmed": True})
                with self.assertRaisesRegex(ValueError, "human"):
                    validate_background(path, enforce_palette=False, vision_human_signal=False)

    def test_safe_region_retry_records_comparable_scores(self):
        record = {"rejected_attempts": [{"validation": {"quality": {"explicit_safe_regions": {
            "regions": {"title": {"usability_score": .125}}}}}}]}
        current = {"quality": {"explicit_safe_regions": {"regions": {
            "title": {"usability_score": .75}}}}}
        _record_safe_region_retry(record, current)
        self.assertEqual(record["safe_region_retry"], {
            "previous_score": .125, "current_score": .75, "improved": True})

    def test_global_pass_with_borderline_region_is_diagnostic(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "image.png"
            image = Image.new("RGB", (896, 1280), "white")
            ImageDraw.Draw(image).rectangle((260, 50, 635, 500), fill="black")
            image.save(path)
            ocr = {"confirmed_tokens": [], "ocr_diagnostics": {"decision": "complete"},
                   "text_detected": False, "tokens": [], "text_lines": [], "text_area_ratio": 0}
            human = {"detected": False, "ambiguous_face_signal": False,
                     "people": [], "faces": [], "profiles": [], "upper_bodies": [],
                     "evidence": [], "strong_signals": [], "weak_signals": []}
            with patch("app.services.comfyui_client._cached_text_evidence", return_value=ocr), \
                 patch("app.services.comfyui_client.detect_human_signals", return_value=human), \
                 patch("app.services.comfyui_client.boilerplate_metrics", return_value={"boilerplate_score": 0}), \
                 patch("app.services.comfyui_client.validate_visual_quality", return_value={}):
                result = validate_background(path, enforce_palette=False,
                                             visual_design={"density_preference": "moderate",
                                                            "title_safe_region_strategy": "quiet upper centre"})
        validation = result["validation"]
        self.assertEqual(validation["decision"], "accepted")
        self.assertEqual(validation["quality"]["global_density_decision"], "PASS")
        self.assertEqual(validation["quality"]["explicit_safe_region_decision"], "BORDERLINE")
        self.assertIn("SAFE_REGION_BORDERLINE", validation["diagnostic_findings"])

    def test_safe_region_coordinates_and_global_decision_are_independent(self):
        tiles = [.18] * 16
        zones = {"title": "quiet top-left", "programme": "calm middle-right"}
        for index in (0, 1, 4, 5):
            tiles[index] = .01
        density = {"decision": "PASS", "tile_edge_fractions": tiles}
        result = assess_explicit_safe_regions(density, zones)
        self.assertEqual(result["regions"]["title"]["tile_indexes"], [0, 1, 4, 5])
        self.assertEqual(result["regions"]["programme"]["tile_indexes"], [6, 7, 10, 11])
        self.assertEqual(result["regions"]["title"]["decision"], "PASS")
        self.assertEqual(result["regions"]["programme"]["decision"], "FAIL")
        self.assertEqual(result["decision"], "FAIL")
        bottom = assess_explicit_safe_regions(density, {"programme": "quiet bottom-center"})
        self.assertEqual(bottom["regions"]["programme"]["tile_indexes"], [9, 10, 13, 14])
        self.assertEqual(bottom["decision"], "FAIL")
        for index in (9, 10, 13, 14):
            tiles[index] = .01
        self.assertEqual(assess_explicit_safe_regions(density, {"programme": "quiet bottom-center"})["decision"], "PASS")

    def test_known_rejection_categories_remain_stable(self):
        for category in ("TEXT", "TEXT_OR_LOGO", "HUMAN", "PALETTE_ROLE_DRIFT", "DENSE_SAFE_REGION",
                         "SUBJECT_DRIFT", "TREATMENT_DRIFT", "UNREQUESTED_LITERAL_OBJECT",
                         "VISION_UNAVAILABLE"):
            self.assertEqual(failure_category(category + ": example"), category)
        self.assertEqual(failure_category("Generated background contains grouped glyph-like rows"), "TEXT_OR_LOGO")

    def test_concise_runtime_prompt_keeps_compiled_prompt_contract(self):
        for engine in ("sdxl-1-0", "sd-3-5-medium", "flux-2"):
            with self.subTest(engine=engine):
                spec = build_candidate_spec(document(), engine, "A")
                original = compile_candidate_prompt(spec, 4, "AUDIENCE_DRIFT", concise=False)
                result = _concise_runtime_prompt(original, spec, 4, "AUDIENCE_DRIFT")
                self.assertIsInstance(result, CompiledPrompt)
                self.assertTrue(result.positive)
                self.assertTrue(result.clip_l)
                self.assertTrue(result.clip_g)
                self.assertEqual(bool(result.t5), engine == "sd-3-5-medium")
                self.assertEqual(result.failure_category, "AUDIENCE_DRIFT")
                self.assertEqual(result.retry_stage, "targeted_recovery")
                self.assertEqual(result.retry_strength_level, "TARGETED_RECOVERY")
                self.assertEqual(result.spec_sha256, original.spec_sha256)

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

    def test_retry_taxonomy_preserves_failed_dimension(self):
        reasons = {"TEXT", "HUMAN", "PALETTE_OFF", "SUBJECT_DRIFT",
                   "RUNTIME", "DENSITY_DRIFT", "MOTIF_DRIFT"}
        self.assertEqual({runtime_retry_category(reason) for reason in reasons},
                         {"TEXT_OR_LOGO", "HUMAN", "PALETTE_OFF", "SUBJECT_DRIFT",
                          "MOTIF_DRIFT", "GENERATION_FAILURE", "DENSITY_DRIFT"})

    def test_production_concise_corrections_are_specific_and_stage_aware(self):
        source = document()
        expected = {
            "AUDIENCE_DRIFT": "audience-appropriate",
            "ATMOSPHERE_DRIFT": "supplied atmosphere",
            "PRIMARY_MISSING": "primary colour across substantial surfaces",
            "SECONDARY_MISSING": "secondary colour in several substantial regions",
            "MOTIF_DRIFT": "motif topology",
            "DENSITY_DRIFT": "density and visual hierarchy",
            "RASTER": "scanline or grid",
            "FLAT": "medium and large subject forms",
            "GRADIENT": "discrete subject-specific forms",
            "NOISE": "controlled detail",
            "PALETTE_OFF": "requested colour families",
            "PALETTE_ROLE_DRIFT": "primary-to-secondary visual-mass",
            "SUBJECT_DRIFT": "subject forms and visual anchors",
            "HUMAN": "non-human environment",
            "TEXT": "without glyph- or signage-like marks",
            "WEAK_OUTER_STRUCTURE": "structure where this direction permits",
            "DENSE_SAFE_REGION": "away from integrated calm regions",
            "DIRECTION_DUPLICATE": "focal placement, viewpoint",
            "RUNTIME": "simpler technical execution",
        }
        for engine in ("sdxl-1-0", "sd-3-5-medium", "flux-2"):
            spec = build_candidate_spec(source, engine, "A")
            for failure, phrase in expected.items():
                with self.subTest(engine=engine, failure=failure):
                    prompts = [compile_candidate_prompt(spec, attempt, failure, concise=True)
                               for attempt in (2, 4, 7, 9)]
                    self.assertEqual([item.retry_stage for item in prompts],
                                     ["normal", "targeted_recovery", "strict_recovery", "rescue"])
                    self.assertEqual(len({item.positive for item in prompts}), 4)
                    for compiled in prompts:
                        positive = compiled.positive
                        if engine == "sdxl-1-0" and failure in {
                                "PALETTE_ROLE_DRIFT", "PRIMARY_MISSING", "SECONDARY_MISSING"}:
                            self.assertIn({
                                "PALETTE_ROLE_DRIFT": "substantial chromatic masses",
                                "PRIMARY_MISSING": "across principal surfaces",
                                "SECONDARY_MISSING": "across several substantial regions",
                            }[failure], positive)
                        else:
                            self.assertIn(phrase, positive)
                        self.assertIn("Preserve all original requested semantics", positive)
                        for semantic in (spec.visual_design["primary_visual_subject"],
                                         spec.primary_colour,
                                         spec.visual_design["visual_semantics"]["motif_structure_description"],
                                         spec.visual_design["visual_semantics"]["atmosphere_descriptors"][0],
                                         spec.visual_design["visual_semantics"]["audience_descriptors"][0]):
                            self.assertIn(semantic, positive)
                        self.assertEqual(compiled.spec_sha256, spec.spec_sha256)
                        self.assertNotIn("spider", positive.casefold())
                        self.assertNotIn("parchment", positive.casefold())


    def test_progressive_retries_preserve_immutable_spec_without_history(self):
        source = document()
        levels = ("BASE", "CORRECTION", "CORRECTION", "TARGETED_RECOVERY")
        for engine in ("sdxl-1-0", "sd-3-5-medium", "flux-2"):
            spec = build_candidate_spec(source, engine, "A")
            design = spec.visual_design
            for category, phrases in _RETRY_CORRECTIONS.items():
                with self.subTest(engine=engine, category=category):
                    attempts = [compile_candidate_prompt(spec, attempt,
                                category if attempt > 1 else "", concise=True)
                                for attempt in range(1, 5)]
                    self.assertEqual(tuple(item.retry_strength_level for item in attempts), levels)
                    self.assertEqual({item.spec_sha256 for item in attempts}, {spec.spec_sha256})
                    self.assertEqual({item.clip_l for item in attempts}, {attempts[0].clip_l})
                    for index, compiled in enumerate(attempts):
                        effective = " ".join((compiled.positive, compiled.clip_g, compiled.t5 or ""))
                        for expected in (design["primary_visual_subject"], spec.primary_colour,
                                         design["visual_semantics"]["atmosphere_descriptors"][0],
                                         design["visual_semantics"]["audience_descriptors"][0],
                                         design["visual_semantics"]["motif_structure_description"],
                                         design["theme_reference_treatment"]):
                            self.assertIn(expected, effective)
                        self.assertIn("Composition:", effective)
                        if index:
                            if engine == "sdxl-1-0" and category in {
                                    "PALETTE_ROLE_DRIFT", "PRIMARY_MISSING", "SECONDARY_MISSING"}:
                                self.assertIn({
                                    "PALETTE_ROLE_DRIFT": "substantial chromatic masses",
                                    "PRIMARY_MISSING": "across principal surfaces",
                                    "SECONDARY_MISSING": "across several substantial regions",
                                }[category], effective)
                            else:
                                self.assertIn(phrases, effective)
                            self.assertIn("Preserve all original requested semantics", effective)
                    if engine == "sdxl-1-0":
                        self.assertIn("targeted_recovery:", attempts[3].clip_g)
                    elif engine == "sd-3-5-medium":
                        self.assertIn("targeted_recovery:", attempts[3].t5)
                    else:
                        self.assertFalse(attempts[3].negative_prompt_channel_active)
                        self.assertIn(phrases, attempts[3].positive)

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

    def test_explicit_quiet_region_with_no_quiet_tiles_is_diagnostic(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "image.png"
            artwork = Image.new("RGB", (896, 1280), "white")
            draw = ImageDraw.Draw(artwork)
            for coordinate in range(0, 896, 8):
                draw.line((coordinate, 0, coordinate, 640), fill="black", width=2)
            for coordinate in range(0, 640, 8):
                draw.line((0, coordinate, 895, coordinate), fill="black", width=2)
            artwork.save(path)
            ocr = {"confirmed_tokens": [], "ocr_diagnostics": {"decision": "OCR_COMPLETE"},
                   "text_detected": False, "tokens": [], "text_lines": [], "text_area_ratio": 0}
            human = {"detected": False, "ambiguous_face_signal": False,
                     "people": [], "faces": [], "profiles": [], "upper_bodies": [],
                     "evidence": [], "strong_signals": [], "weak_signals": []}
            density = {"decision": "PASS", "macro_form_balance": 0,
                       "occupied_tile_fraction": 1, "quiet_tile_fraction": 0,
                       "tile_edge_fractions": [.2] * 16}
            with patch("app.services.comfyui_client._cached_text_evidence", return_value=ocr), \
                 patch("app.services.comfyui_client.detect_human_signals", return_value=human), \
                 patch("app.services.comfyui_client.boilerplate_metrics", return_value={"boilerplate_score": 0}), \
                 patch("app.services.comfyui_client.validate_visual_quality", return_value={}), \
                 patch("app.services.comfyui_client.assess_scene_density", return_value=density):
                result = validate_background(path, enforce_palette=False,
                                             visual_design={"density_preference": "moderate",
                                                            "title_safe_region_strategy": "integrated quiet area"})
            validation = result["validation"]
            self.assertEqual(validation["decision"], "accepted")
            self.assertIn("SAFE_REGION_BORDERLINE", validation["diagnostic_findings"])
            self.assertEqual(validation["quality"]["global_density_decision"], "PASS")
            self.assertEqual(validation["quality"]["explicit_safe_region_decision"], "BORDERLINE")

    def test_pseudo_text_without_ocr_tokens_rejects(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "image.png"
            Image.new("RGB", (896, 1280), "navy").save(path)
            evidence = {"confirmed_tokens": [], "ocr_diagnostics": {"decision": "OCR_COMPLETE"},
                        "text_detected": True, "tokens": [], "text_lines": [], "text_area_ratio": 0,
                        "pseudo_text": {"detected": True, "aligned_rows": 2}}
            human = {"detected": False, "ambiguous_face_signal": False}
            with patch("app.services.comfyui_client._cached_text_evidence", return_value=evidence), \
                 patch("app.services.comfyui_client.detect_human_signals", return_value=human), \
                 patch("app.services.comfyui_client.boilerplate_metrics", return_value={"boilerplate_score": 0}), \
                 patch("app.services.comfyui_client.validate_visual_quality", return_value={}):
                with self.assertRaisesRegex(ValueError, "grouped glyph-like rows"):
                    validate_background(path, enforce_palette=False)

    def test_explicit_quiet_region_passes_when_its_tiles_are_calm(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "image.png"
            Image.new("RGB", (896, 1280), "navy").save(path)
            ocr = {"confirmed_tokens": [], "ocr_diagnostics": {"decision": "OCR_COMPLETE"},
                   "text_detected": False, "tokens": [], "text_lines": [], "text_area_ratio": 0}
            human = {"detected": False, "ambiguous_face_signal": False,
                     "people": [], "faces": [], "profiles": [], "upper_bodies": [],
                     "evidence": [], "strong_signals": [], "weak_signals": []}
            tiles = [.2] * 16
            for index in (0, 1, 4, 5):
                tiles[index] = .01
            density = {"decision": "PASS", "macro_form_balance": 0,
                       "tile_edge_fractions": tiles}
            with patch("app.services.comfyui_client._cached_text_evidence", return_value=ocr), \
                 patch("app.services.comfyui_client.detect_human_signals", return_value=human), \
                 patch("app.services.comfyui_client.boilerplate_metrics", return_value={"boilerplate_score": 0}), \
                 patch("app.services.comfyui_client.validate_visual_quality", return_value={}), \
                 patch("app.services.comfyui_client.assess_scene_density", return_value=density):
                result = validate_background(path, enforce_palette=False,
                                             visual_design={"density_preference": "moderate",
                                                            "title_safe_region_strategy": "quiet top-left area"})
            self.assertEqual(result["validation"]["decision"], "accepted")
            self.assertEqual(result["validation"]["quality"]["global_density_decision"], "PASS")
            self.assertEqual(result["validation"]["quality"]["explicit_safe_region_decision"], "PASS")


if __name__ == "__main__":
    unittest.main()
