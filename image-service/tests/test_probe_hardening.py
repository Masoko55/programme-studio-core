"""Static regression checks for the six-probe production findings."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.services.background_policy import _palette_role_metrics, validate_palette
from app.services.candidate_spec import build_candidate_spec
from app.services.comfyui_client import ComfyUIClient, ComfyUIError, detect_text_evidence, validate_background
from app.services.prompt_compiler import compile_candidate_prompt
from app.services.prompt_freshness import source_prompt_manifest
from app.services.scene_density import assess_scene_density
from app.services.subject_adherence import _extract_numeric_score, assess_subject_adherence
from tests.test_semantic_foundation import document


class ProbeHardeningTests(unittest.TestCase):
    def test_default_palette_relationship_is_diagnostic(self):
        for primary, secondary, counts in (
            ("green", "cream", {"green": 5000, "cream": 17000}),
            ("navy", "white", {"navy": 5000, "white": 17000}),
        ):
            with self.subTest(primary=primary):
                metrics = _palette_role_metrics(primary, secondary, counts, 22528,
                    "requested colours dominate with natural tonal variation", "dominant", None, "default")
                self.assertEqual(metrics["palette_role_warning"], "PALETTE_ROLE_WARNING")
                with self.assertRaisesRegex(ValueError, "PALETTE_ROLE_DRIFT"):
                    _palette_role_metrics(primary, secondary, counts, 22528,
                        f"{primary} must dominate, {secondary} only as accents", "dominant", None, "user")

    def test_palette_retry_diagnostics_recognize_role_and_presence(self):
        from app.services.image_execution import _is_palette_failure
        from app.services.prompt_compiler import runtime_retry_category
        for reason in (
            "PALETTE_ROLE_DRIFT: requested colour dominance relationship is reversed",
            "PALETTE_ROLE_DRIFT: requested primary colour is not meaningfully present",
            "PALETTE_ROLE_DRIFT: required secondary colour is absent or too small",
            "PRIMARY_MISSING", "SECONDARY_MISSING",
        ):
            with self.subTest(reason=reason):
                self.assertEqual(runtime_retry_category(reason), "PALETTE")
                self.assertTrue(_is_palette_failure(reason))

    def test_structured_scores_and_bounded_diagnostics(self):
        cases = [(0.8, .8), ("0.8", .8), ({"score": .8}, .8),
                 ({"value": "0.8"}, .8), ({"rating": .8}, .8),
                 ({"subject_score": {"value": .8}}, .8)]
        for value, expected in cases:
            diagnostics = []
            self.assertEqual(_extract_numeric_score(value, "subject_score", diagnostics), expected)
            self.assertEqual(diagnostics[0]["parsed_value"], expected)
            self.assertNotIn(str(value)[:30], str(diagnostics) if isinstance(value, dict) else "")
        for value in ({}, {"score": .7, "rating": .8}, {"score": "bad"},
                      float("nan"), float("inf"), [.8], None):
            diagnostics = []
            with self.subTest(value=value), self.assertRaises(ValueError):
                _extract_numeric_score(value, "subject_score", diagnostics)
            self.assertTrue(diagnostics[0]["parse_failure_reason"])

    def test_structured_minicpm_response_is_available(self):
        class Response:
            status_code = 200
            headers = {"content-type": "application/json"}
            def raise_for_status(self):
                pass
            def json(self):
                return {"message": {"content": json.dumps({
                    "subject_score": {"score": .9}, "atmosphere_score": {"value": .8},
                    "motif_score": {"rating": .75}, "audience_fit_score": "0.8",
                    "background_usability_score": {"score": .7},
                    "human_visible": False, "observed_evidence": ["layered foliage"]})}}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "candidate.png"
            Image.new("RGB", (64, 64), "green").save(path)
            with patch("app.services.subject_adherence.httpx.post", return_value=Response()):
                result = assess_subject_adherence(path, {"primary_visual_subject": "layered foliage"},
                                                  model="minicpm-v4.5:latest")
        self.assertEqual(result["decision"], "pass")
        self.assertEqual(result["vision_human_signal"], False)
        self.assertEqual(result["vision_debug"]["score_parsing"][0]["extracted_field"], "score")

    def test_palette_roles_are_independent_of_membership(self):
        counts = {"royal blue": 15000, "red": 0}
        with self.assertRaisesRegex(ValueError, "PALETTE_ROLE_DRIFT"):
            _palette_role_metrics("royal blue", "red", counts, 22528,
                                  "royal blue dominant with strong red secondary accents", "dominant", None)
        with self.assertRaisesRegex(ValueError, "PALETTE_ROLE_DRIFT"):
            _palette_role_metrics("royal blue", "red", {"royal blue": 2700, "red": 19828}, 22528,
                                  "royal blue should dominate", "dominant", None)
        balanced = _palette_role_metrics("black", "white", {"black": 11000, "white": 11528}, 22528,
                                         "black and white balanced", "strict", None)
        self.assertEqual(balanced["dominance_relationship"], "BALANCED")
        self.assertTrue(balanced["dominance_relationship_passed"])
        optional = _palette_role_metrics("green", "cream", {"green": 22000, "cream": 0}, 22528,
                                         "cream optional neutral support only", "dominant", None)
        self.assertTrue(optional["secondary_presence"])
        mono = _palette_role_metrics("green", None, {"green": 22528}, 22528, "monochromatic", "monochrome", None)
        self.assertEqual(mono["dominance_relationship"], "MONOCHROMATIC")
        three = _palette_role_metrics("green", "cream", {"green": 14000, "cream": 5000, "pink": 3000}, 22528,
                                      "green dominant with cream support", "dominant",
                                      [{"role": "primary"}, {"role": "supporting"}, {"role": "accent"}])
        self.assertTrue(three["secondary_presence"])

    def test_palette_membership_can_pass_while_role_fails(self):
        pixels = np.zeros((176, 128, 3), dtype=np.uint8)
        pixels[:] = (25, 35, 130)
        with self.assertRaisesRegex(ValueError, "PALETTE_ROLE_DRIFT") as caught:
            validate_palette(Image.fromarray(pixels), "royal blue", "red",
                             relationship="royal blue dominant with red secondary", quality_result={})
        self.assertLess(caught.exception.palette_metrics["off_palette_ratio"], .01)
        self.assertFalse(caught.exception.palette_metrics["secondary_presence"])

    def test_semantic_diagnostics_survive_other_hard_rejections(self):
        source = document()
        with tempfile.TemporaryDirectory() as root:
            candidate = Path(root) / "image-a.part.png"
            for category, subject_score in (("OCR_TIMEOUT", .2), ("PALETTE_OFF", .9), ("HUMAN", .9)):
                Image.new("RGB", (64, 64), "blue").save(candidate)
                record = {"reference_number": "ABCDEF-123456", "engine_id": "sdxl-1-0",
                          "direction_id": "A", "attempt_count": 1}
                error = ValueError(category + ": hard rejection")
                error.diagnostics = {"validation": {"decision": "rejected", "failure_category": category},
                                     "primary_failure_category": category, "all_failure_categories": [category]}
                fidelity = {"decision": "SUBJECT_DRIFT" if subject_score < .5 else "BRIEF_FIDELITY_DRIFT",
                            "failure_category": "SUBJECT_DRIFT" if subject_score < .5 else "MOTIF_DRIFT",
                            "subject_score": subject_score, "atmosphere_score": .8,
                            "motif_score": .2, "audience_fit_score": .8,
                            "failed_dimensions": ["subject_score", "motif_score"] if subject_score < .5 else ["motif_score"],
                            "vision_human_signal": False}
                with patch("app.services.comfyui_client.load_prompts_document", return_value=source), \
                     patch("app.services.comfyui_client.compare_directions_metrics", return_value={"similarity": 0}), \
                     patch("app.services.comfyui_client.validate_background", side_effect=error), \
                     patch("app.services.comfyui_client.assess_subject_adherence", return_value=fidelity), \
                     patch("app.services.comfyui_client.write_json"):
                    with self.assertRaises(ComfyUIError):
                        ComfyUIClient()._validate_downloaded_candidate(record, False,
                            Path(root) / "image-a.png", candidate, {}, ("navy", "gold"))
                self.assertEqual(record["primary_failure_category"], category)
                self.assertEqual(record["semantic_dimension_scores"]["subject_score"], subject_score)
                self.assertIn("MOTIF_DRIFT", record["all_failure_categories"])

    def test_ocr_partial_evidence_quorum(self):
        def run(completed):
            def fake(_image, _config, variant, diagnostics):
                diagnostics["attempted"].append(variant)
                if variant in completed:
                    diagnostics["completed_variants"].append(variant)
                    return {"text": [], "conf": []}
                diagnostics["timed_out_variants"].append(variant)
                return None
            with patch("app.services.comfyui_client._ocr_data", side_effect=fake):
                return detect_text_evidence(Image.new("RGB", (64, 64), "white"))
        sufficient = run({"original", "contrast"})
        self.assertEqual(sufficient["ocr_diagnostics"]["decision"], "OCR_PARTIAL_SUFFICIENT")
        self.assertFalse(sufficient["text_detected"])
        self.assertEqual(run({"original"})["ocr_diagnostics"]["decision"], "OCR_PARTIAL_INSUFFICIENT")

    def test_vision_disambiguates_weak_face_without_overriding_clear_human_or_text(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "image.png"
            Image.new("RGB", (896, 1280), "green").save(path)
            ocr = {"confirmed_tokens": [], "ocr_diagnostics": {"decision": "complete"},
                   "text_detected": False, "tokens": [], "text_lines": [], "text_area_ratio": 0}
            weak = {"detected": False, "ambiguous_face_signal": True,
                    "evidence": [], "strong_signals": [], "weak_signals": [{"source": "frontal_face"}],
                    "people": [], "faces": [], "profiles": [], "upper_bodies": []}
            with patch("app.services.comfyui_client._cached_text_evidence", return_value=ocr), \
                 patch("app.services.comfyui_client.detect_human_signals", side_effect=lambda _: dict(weak)), \
                 patch("app.services.comfyui_client.boilerplate_metrics", return_value={"boilerplate_score": 0}), \
                 patch("app.services.comfyui_client.validate_visual_quality", return_value={}):
                self.assertEqual(validate_background(path, enforce_palette=False,
                                                      vision_human_signal=False)["validation"]["decision"], "accepted")
                with self.assertRaisesRegex(ValueError, "human"):
                    validate_background(path, enforce_palette=False, vision_human_signal=True)
                ocr["confirmed_tokens"] = ["Surofatal"]
                with self.assertRaisesRegex(ValueError, "OCR text"):
                    validate_background(path, enforce_palette=False, vision_human_signal=False)
                ocr["ocr_diagnostics"]["decision"] = "OCR_UNAVAILABLE"
                with self.assertRaises(ValueError) as caught:
                    validate_background(path, enforce_palette=False, vision_human_signal=False)
                self.assertEqual(caught.exception.diagnostics["primary_failure_category"], "TEXT")
                ocr["ocr_diagnostics"]["decision"] = "complete"
                ocr["confirmed_tokens"] = []
                weak["ambiguous_face_signal"] = False
                with self.assertRaisesRegex(ValueError, "human"):
                    validate_background(path, enforce_palette=False, vision_human_signal=True)
                weak["ambiguous_face_signal"] = True
            strong = {**weak, "detected": True, "ambiguous_face_signal": False}
            strong["people"] = [(80, 100, 90, 250), (300, 110, 85, 240)]
            with patch("app.services.comfyui_client._cached_text_evidence", return_value=ocr), \
                 patch("app.services.comfyui_client.detect_human_signals", return_value=strong), \
                 patch("app.services.comfyui_client.boilerplate_metrics", return_value={"boilerplate_score": 0}), \
                 patch("app.services.comfyui_client.validate_visual_quality", return_value={}):
                with self.assertRaisesRegex(ValueError, "human"):
                    validate_background(path, enforce_palette=False, vision_human_signal=None)

    def test_flux_and_all_channels_are_sanitized(self):
        data = document()
        data["directions"][0]["direction"]["positive_prompt"] = (
            "Glowing invented portal with title panel, typography, logos and schedule information")
        data["brief"]["visual_semantics"]["incompatible_treatments"] = ["no people"]
        for engine in ("flux-2", "sdxl-1-0", "sd-3-5-medium"):
            prompt = compile_candidate_prompt(build_candidate_spec(data, engine, "A"), 1)
            channels = (prompt.positive, prompt.clip_l, prompt.clip_g, prompt.t5 or "")
            for channel in channels:
                self.assertNotIn("invented portal", channel.casefold())
                self.assertNotIn("brief-compatible or characters", channel.casefold())
                for word in ("title panel", "typography", "logos", "schedule information"):
                    self.assertNotIn(word, channel.casefold())
            if engine == "flux-2":
                self.assertFalse(prompt.negative_prompt_channel_active)
                self.assertIn("uninterrupted natural image surfaces", prompt.positive)

    def test_source_prompt_freshness(self):
        data = document()
        data["brief"]["semantic_contract"] = {"semantic_contract_version": "1.0"}
        data["creative_direction_version"] = "2.0"
        data["prompt_schema_version"] = "1.0"
        data["prompt_generation_sha256"] = hashlib.sha256(json.dumps(
            data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(source_prompt_manifest(data)["freshness_status"], "CURRENT")
        data["directions"][0]["direction"]["positive_prompt"] = "changed"
        self.assertEqual(source_prompt_manifest(data)["freshness_status"], "STALE_FOR_EXPERIMENT")

    def test_density_is_continuous_across_tile_boundary(self):
        rng = np.random.default_rng(9)
        pixels = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        full = assess_scene_density(Image.fromarray(pixels), "moderate")
        pixels[:64, :64] = np.tile(np.arange(64, dtype=np.uint8), (64, 1))[:, :, None]
        almost = assess_scene_density(Image.fromarray(pixels), "moderate")
        self.assertEqual(full["decision"], almost["decision"])
        self.assertIn("density_pressure", almost["normalized_measurements"])
        self.assertIn("density_pressure", almost["thresholds_used"])


if __name__ == "__main__":
    unittest.main()
