"""Offline regressions for event fit and bounded OCR."""

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from app.config.settings import settings
from app.services.candidate_spec import build_candidate_spec
from app.services.comfyui_client import (
    _cached_text_evidence,
    _ocr_data,
    _OCR_CACHE,
    build_workflow,
    detect_text_evidence,
    effective_workflow_summary,
    validate_background,
)
from app.services.prompt_compiler import compile_candidate_prompt
from app.services.prompt_compiler import failure_category, runtime_retry_category
from app.services.sampling_profiles import profile_by_name
from app.services.subject_adherence import (
    assess_subject_adherence,
    hard_fidelity_failure,
)
from scripts.probe_native_candidate import classify_probe_failure
from tests.test_renderer_brief_matrix import fixture


class BriefFidelityRecoveryTests(unittest.TestCase):
    def test_explicit_character_constraint_reaches_frozen_candidate_and_retry(self):
        document = fixture(1, "celebration", "abstract geometry", "blue", "red", "dominant",
                           "style inspired", "asymmetric")
        document["brief"]["grill_me_context"] = {"generation_constraints": ["No people or characters"]}
        spec = build_candidate_spec(document, "sdxl-1-0", "A")
        self.assertTrue(spec.visual_design["characters_prohibited"])
        self.assertEqual(failure_category("CHARACTER: depicted costumed figure"), "CHARACTER")
        self.assertEqual(failure_category("CHARACTER: depicted humanoid figure"), "CHARACTER")
        self.assertEqual(runtime_retry_category("CHARACTER: depicted costumed figure"), "CHARACTER")

    def test_prohibited_depicted_character_is_a_hard_semantic_failure(self):
        design = {"primary_visual_subject": "abstract celebratory background",
                  "characters_prohibited": True}
        for evidence in ("masked costumed figure with raised arm at center",
                         "stylized comic humanoid with body and legs at left",
                         "fictional character with cape and flying pose in foreground",
                         "large superhero-like silhouette with body and outstretched arm"):
            with self.subTest(evidence=evidence):
                result, _ = self._assess(design, {
                    "subject_score": .95, "human_visible": False,
                    "character_visible": True, "character_description": evidence,
                    "unrequested_literal_object_visible": False})
                self.assertEqual(result["decision"], "CHARACTER")
                self.assertTrue(hard_fidelity_failure(result))

    def test_character_motifs_and_abstract_shapes_are_not_characters(self):
        design = {"primary_visual_subject": "abstract background", "characters_prohibited": True}
        for evidence in ("spider-web geometry", "bat-shaped abstract motif", "black and yellow shapes",
                         "foliage", "floral pattern", "architecture", "face-like decoration",
                         "human-like geometry", "sculpture-like form with a face pattern",
                         "abstract silhouette without body evidence"):
            with self.subTest(evidence=evidence):
                result, _ = self._assess(design, {"subject_score": .95,
                    "character_visible": False, "character_description": evidence})
                self.assertEqual(result["decision"], "pass")
        allowed, _ = self._assess({"primary_visual_subject": "illustrated character",
                                   "authorized_literal_subject": "illustrated character"},
                                  {"subject_score": .95, "character_visible": True,
                                   "character_description": "costumed figure at center"})
        self.assertEqual(allowed["decision"], "pass")

    def test_structured_character_evidence_overrides_inconsistent_boolean(self):
        design = {"primary_visual_subject": "abstract celebratory background",
                  "characters_prohibited": True}
        result, _ = self._assess(design, {"subject_score": 1.0, "human_visible": False,
            "character_visible": False, "observed_evidence": ["Heroic superhero figure in dynamic pose"],
            "unrequested_literal_object_visible": False})
        self.assertEqual(result["decision"], "CHARACTER")
        self.assertIn("figure", result["character_description"])
        missing, _ = self._assess(design, {"subject_score": .9,
            "observed_evidence": ["abstract web pattern"]})
        self.assertEqual(missing["decision"], "unavailable")
        self.assertTrue(hard_fidelity_failure(missing))
        malformed, _ = self._assess(design, {"subject_score": .9,
            "character_visible": "false", "observed_evidence": ["abstract web pattern"]})
        self.assertEqual(malformed["decision"], "unavailable")

    def test_inconsistent_literal_claim_retries_then_only_literal_dimension_is_unavailable(self):
        brief = {"primary_visual_subject": "layered abstract marks",
                 "motif_literalization_policy": "motif_only",
                 "visual_semantics": {"motif_descriptors": ["curved line pattern"]}}
        scores = {"subject_score": .95, "motif_score": .92,
                  "unrequested_literal_object_visible": True}
        result, _ = self._assess(brief, scores)
        self.assertEqual(result["subject_score"], .95)
        self.assertEqual(result["motif_score"], .92)
        self.assertEqual(result["literal_object_dimension_availability"], "UNAVAILABLE")
        self.assertEqual(result["vision_debug"]["assessment_attempts"], 2)
        self.assertNotEqual(result["decision"], "pass")
        self.assertNotEqual(result["decision"], "UNREQUESTED_LITERAL_OBJECT")
        self.assertFalse(hard_fidelity_failure(result))
        valid, _ = self._assess(brief, {**scores, "unrequested_literal_object_visible": False})
        self.assertEqual(valid["decision"], "pass")
        self.assertEqual(valid["literal_object_dimension_availability"], "PASS")

    def test_requested_literal_subject_is_not_unauthorized_motif_object(self):
        for subject, visible in (("flowers in a garden", "flowers"),
                                 ("foliage in a landscape", "foliage")):
            with self.subTest(subject=subject):
                brief = {"primary_visual_subject": subject,
                         "authorized_literal_subject": subject,
                         "motif_literalization_policy": "motif_only",
                         "visual_semantics": {"motif_descriptors": ["repeating organic pattern"]}}
                result, _ = self._assess(brief, {
                    "subject_score": .95, "motif_score": .91,
                    "unrequested_literal_object_visible": True,
                    "unrequested_literal_object_evidence": {
                        "object": visible, "location": "upper left",
                        "why_unrequested": "claimed to originate only from the motif"}})
                self.assertNotEqual(result["decision"], "UNREQUESTED_LITERAL_OBJECT")
                self.assertEqual(result["subject_score"], .95)

    def test_inspiration_word_does_not_authorize_its_literal_source_object(self):
        brief = {"primary_visual_subject": "bird-inspired abstract geometry",
                 "authorized_literal_subject": "",
                 "motif_literalization_policy": "motif_only"}
        result, _ = self._assess(brief, {
            "subject_score": .95, "unrequested_literal_object_visible": True,
            "unrequested_literal_object_evidence": {
                "object": "bird", "location": "upper left",
                "why_unrequested": "literal source of the pattern, absent from requested subject"}})
        self.assertEqual(result["decision"], "UNREQUESTED_LITERAL_OBJECT")

    def test_inconsistent_literal_claim_recovers_on_second_answer(self):
        brief = {"primary_visual_subject": "abstract forms",
                 "motif_literalization_policy": "motif_only"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.png"
            Image.new("RGB", (32, 32), "white").save(path)
            def response(value):
                result = Mock(status_code=200, headers={"content-type": "application/json"})
                result.json.return_value = {"message": {"content": json.dumps(value)}}
                return result
            scores = {"subject_score": .95, "observed_evidence": ["abstract forms"]}
            with patch("app.services.subject_adherence.httpx.post", side_effect=[
                response({**scores, "unrequested_literal_object_visible": True}),
                response({**scores, "unrequested_literal_object_visible": False})]) as send:
                result = assess_subject_adherence(path, brief, model="minicpm-v4.5:latest")
        self.assertEqual(send.call_count, 2)
        self.assertIn("Recheck unrequested_literal_object_visible",
                      send.call_args.kwargs["json"]["messages"][0]["content"])
        self.assertEqual(result["decision"], "pass")

    def test_inferred_audience_mood_is_diagnostic_but_explicit_hard_audience_rejects(self):
        brief = {"primary_visual_subject": "layered abstract landscape",
                 "audience_context": "families", "age_group": "children",
                 "visual_semantics": {"audience_descriptors": ["welcoming"]},
                 "semantic_provenance": {"audience_descriptors": "SEMANTIC_INFERENCE"}}
        scores = {"subject_score": .9, "audience_fit_score": .3}
        result, _ = self._assess(brief, scores)
        self.assertEqual(result["audience_fit_score"], .3)
        self.assertIn("audience_fit_score", result["failed_dimensions"])
        self.assertNotIn("audience_fit_score", result["hard_failed_dimensions"])
        self.assertFalse(hard_fidelity_failure(result))
        brief["hard_dimensions"] = ["audience_fit_score"]
        hard, _ = self._assess(brief, scores)
        self.assertIn("audience_fit_score", hard["hard_failed_dimensions"])
        self.assertTrue(hard_fidelity_failure(hard))

    def test_vision_human_requires_anatomy_or_posture_and_location(self):
        brief = {"primary_visual_subject": "architectural shapes"}
        base = {"subject_score": .9, "human_visible": True}
        for description in ("figure-like floral shape", "possible silhouette", "human-like geometry at left"):
            result, _ = self._assess(brief, {**base, "human_description": description})
            self.assertIsNone(result["vision_human_signal"])
            self.assertTrue(result["vision_debug"]["unconfirmed_human_report"])
        for description in ("standing person on the left", "small upright person beside doorway"):
            result, _ = self._assess(brief, {**base, "human_description": description})
            self.assertTrue(result["vision_human_signal"])

    def test_explicit_distant_people_report_is_confirmed_without_anatomy_detail(self):
        brief = {"primary_visual_subject": "architectural shapes"}
        descriptions = (
            "Two small, distant human figures: visible only as silhouettes, posture upright, "
            "location at far end of the hallway, facing away from the viewer.",
            "three small people at the end of the corridor",
            "several people in the background",
            "a distant person walking beside the doorway",
        )
        for description in descriptions:
            with self.subTest(description=description):
                result, _ = self._assess(brief, {
                    "subject_score": .9, "human_visible": True,
                    "human_description": description})
                self.assertTrue(result["vision_human_signal"])
                self.assertFalse(result["vision_debug"]["unconfirmed_human_report"])

        for description in (
            "face-like floral structure in the background",
            "human-like decorative pattern in the distance",
            "figure-like architecture at the far end",
            "weak isolated cascade hit beside the doorway",
        ):
            with self.subTest(description=description):
                result, _ = self._assess(brief, {
                    "subject_score": .9, "human_visible": True,
                    "human_description": description})
                self.assertIsNone(result["vision_human_signal"])

    def test_human_evidence_confirms_empty_description_without_ambiguous_shape_false_positive(self):
        brief = {"primary_visual_subject": "architectural shapes"}
        person, _ = self._assess(brief, {
            "subject_score": .9, "human_visible": True, "human_description": "",
            "observed_evidence": ["One person in a suit (formal, posture upright, walking in foreground)"],
        })
        self.assertTrue(person["vision_human_signal"])
        self.assertFalse(person["vision_debug"]["unconfirmed_human_report"])

        for evidence in (
            "face-like floral structure in the background",
            "figure-like architecture at the far end",
            "human-like geometry beside the doorway",
        ):
            with self.subTest(evidence=evidence):
                ambiguous, _ = self._assess(brief, {
                    "subject_score": .9, "human_visible": True, "human_description": "",
                    "observed_evidence": [evidence],
                })
                self.assertIsNone(ambiguous["vision_human_signal"])

    def test_good_subject_with_weak_treatment_keeps_its_category(self):
        brief = {"primary_visual_subject": "layered city forms",
                 "theme_reference_treatment": "first-person movement through the city",
                 "semantic_provenance": {"theme_reference_treatment": "USER_EXPLICIT"}}
        result, _ = self._assess(brief, {"subject_score": .95, "treatment_score": .1,
                                          "motif_score": .9, "atmosphere_score": .9})
        self.assertEqual(result["subject_score"], .95)
        self.assertEqual(result["decision"], "TREATMENT_DRIFT")
        self.assertEqual(result["failure_category"], "TREATMENT_DRIFT")
        self.assertNotIn("subject_score", result["hard_failed_dimensions"])

    def test_malformed_json_once_then_valid_uses_same_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.png"
            Image.new("RGB", (32, 32), "white").save(path)
            def response(content):
                result = Mock(status_code=200, headers={"content-type": "application/json"})
                result.json.return_value = {"message": {"content": content}}
                return result
            valid = json.dumps({"subject_score": .9, "observed_evidence": ["correct forms"]})
            with patch("app.services.subject_adherence.httpx.post",
                       side_effect=[response("not json"), response(valid)]) as send:
                result = assess_subject_adherence(path, {"primary_visual_subject": "landscape"},
                                                  model="minicpm-v4.5:latest")
                self.assertEqual(send.call_count, 2)
                self.assertTrue(result["checked"])
                self.assertEqual(result["vision_debug"]["assessment_attempts"], 2)
            with patch("app.services.subject_adherence.httpx.post",
                       side_effect=[response("not json"), response("not json")]) as send:
                result = assess_subject_adherence(path, {"primary_visual_subject": "landscape"},
                                                  model="minicpm-v4.5:latest")
                self.assertEqual(send.call_count, 2)
                self.assertEqual(result["decision"], "unavailable")
                self.assertNotEqual(result.get("failure_category"), "SUBJECT_DRIFT")

    def test_borderline_artistic_scores_are_diagnostic_but_wrong_subject_is_hard(self):
        brief = {"primary_visual_subject": "layered river landscape",
                 "theme_reference_treatment": "first-person viewpoint through curved banks",
                 "semantic_provenance": {"theme_reference_treatment": "USER_EXPLICIT"},
                 "visual_semantics": {"motif_descriptors": ["fine light ripples"],
                                      "audience_descriptors": ["family guests"]}}
        for field in ("subject_score", "treatment_score", "audience_fit_score"):
            scores = {"subject_score": .9, "motif_score": .9, "audience_fit_score": .9,
                      "treatment_score": .9, field: .5}
            result, _ = self._assess(brief, scores)
            self.assertEqual(result["decision"], "BRIEF_FIDELITY_DRIFT")
            self.assertIn(field, result["failed_dimensions"])
            self.assertFalse(hard_fidelity_failure(result))
        wrong, _ = self._assess(brief, {"subject_score": .2, "motif_score": .9,
                                         "audience_fit_score": .9, "treatment_score": .9})
        self.assertEqual(wrong["decision"], "SUBJECT_DRIFT")
        self.assertTrue(hard_fidelity_failure(wrong))

    def test_vision_text_requires_concrete_evidence(self):
        brief = {"primary_visual_subject": "abstract layered marks"}
        scores = {"subject_score": .9, "unrequested_text_visible": True,
                  "text_evidence": "two rows of letter-like glyphs in the upper left"}
        result, prompt = self._assess(brief, scores)
        self.assertEqual(result["decision"], "TEXT_OR_LOGO")
        self.assertTrue(hard_fidelity_failure(result))
        self.assertIn("unrequested_text_visible", prompt)
        for evidence in ("", "abstract decorative lines"):
            result, _ = self._assess(brief, {**scores, "text_evidence": evidence})
            self.assertFalse(hard_fidelity_failure(result))

    def test_motif_only_retries_one_missing_boolean_then_fails_closed(self):
        brief = {"primary_visual_subject": "abstract layered marks",
                 "motif_literalization_policy": "motif_only",
                 "visual_semantics": {"motif_descriptors": ["curved strokes"]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.png"
            Image.new("RGB", (32, 32), "white").save(path)
            def response(body):
                result = Mock(status_code=200, headers={"content-type": "application/json"})
                result.json.return_value = {"message": {"content": json.dumps(body)}}
                return result
            scores = {"subject_score": .95, "motif_score": .95,
                      "observed_evidence": ["matching forms"], "reason": "brief fit"}
            with patch("app.services.subject_adherence.httpx.post",
                       side_effect=[response(scores), response({**scores, "unrequested_literal_object_visible": False})]) as send:
                result = assess_subject_adherence(path, brief, model="minicpm-v4.5:latest")
                self.assertEqual(send.call_count, 2)
                self.assertTrue(result["checked"])
            with patch("app.services.subject_adherence.httpx.post",
                       side_effect=[response(scores), response(scores)]) as send:
                result = assess_subject_adherence(path, brief, model="minicpm-v4.5:latest")
                self.assertEqual(send.call_count, 2)
                self.assertEqual(result["decision"], "unavailable")
                self.assertTrue(hard_fidelity_failure(result))

    def test_motif_only_requires_explicit_boolean_literal_object_signal(self):
        brief = {
            "primary_visual_subject": "abstract layered marks",
            "visual_semantics": {"motif_descriptors": ["feather-like strokes"]},
            "motif_literalization_policy": "motif_only",
            "authorized_literal_subject": "",
        }
        scores = {"subject_score": .95, "motif_score": .95}

        result, _ = self._assess(
            brief, {**scores, "unrequested_literal_object_visible": False}
        )
        self.assertTrue(result["checked"])
        self.assertNotEqual(result["decision"], "SUBJECT_DRIFT")

        for response in (
            scores,
            {**scores, "unrequested_literal_object_visible": None},
            {**scores, "unrequested_literal_object_visible": "false"},
            {**scores, "unrequested_literal_object_visible": 0},
        ):
            result, _ = self._assess(brief, response)
            self.assertFalse(result["checked"])
            self.assertEqual(result["decision"], "unavailable")
            self.assertEqual(result["dimension_availability"]["subject_score"], "UNAVAILABLE")

        result, _ = self._assess(
            {**brief, "motif_literalization_policy": "literal_allowed"}, scores
        )
        self.assertTrue(result["checked"])
        self.assertNotEqual(result["decision"], "unavailable")
        self.assertEqual(result["literal_object_dimension_availability"], "NOT_REQUIRED")

    def test_unrequested_literal_objects_fail_policy_without_changing_subject_score(self):
        scores = {"subject_score": .95, "motif_score": .95,
                  "unrequested_literal_object_visible": True,
                  "unrequested_literal_object_evidence": {
                      "object": "bird", "location": "upper left",
                      "why_unrequested": "literal object introduced from motif, absent from requested subject"}}
        for motif in ("feather-like strokes", "repeating scale texture"):
            brief = {"primary_visual_subject": "abstract layered marks",
                     "visual_semantics": {"motif_descriptors": [motif]},
                     "motif_literalization_policy": "motif_only",
                     "authorized_literal_subject": ""}
            result, prompt = self._assess(brief, scores)
            self.assertEqual(result["decision"], "UNREQUESTED_LITERAL_OBJECT")
            self.assertEqual(result["subject_score"], .95)
            self.assertEqual(result["motif_score"], .95)
            self.assertTrue(hard_fidelity_failure(result))
            self.assertIn("unrequested_literal_object_visible", prompt)
        scores["unrequested_literal_object_visible"] = False
        explicit = {"primary_visual_subject": "a bird perched on a branch",
                    "visual_semantics": {"motif_descriptors": ["feather pattern"]},
                    "motif_literalization_policy": "motif_only",
                    "authorized_literal_subject": "a bird perched on a branch"}
        result, prompt = self._assess(explicit, scores)
        self.assertFalse(hard_fidelity_failure(result))
        self.assertIn("a bird perched on a branch", prompt)

    def test_explicit_viewpoint_is_scored_independently_of_subject_and_motif(self):
        brief = {
            "primary_visual_subject": "layered city geometry",
            "theme_reference_treatment": "Solar Sentinel is moving through the city in first person view",
            "visual_design": {"theme_reference_treatment": "first-person viewpoint; movement through the city"},
            "semantic_provenance": {"theme_reference_treatment": "USER_EXPLICIT"},
        }
        scores = {"subject_score": .9, "atmosphere_score": .9, "motif_score": .9,
                  "audience_fit_score": .9, "treatment_score": .1}
        result, prompt = self._assess(brief, scores)
        self.assertEqual(result["decision"], "TREATMENT_DRIFT")
        self.assertTrue(hard_fidelity_failure(result))
        self.assertIn("treatment_score", prompt)
        scores["treatment_score"] = .9
        result, _ = self._assess(brief, scores)
        self.assertFalse(hard_fidelity_failure(result))

    def _assess(
        self,
        brief,
        scores,
    ):
        with tempfile.TemporaryDirectory() as directory:
            image = (
                Path(directory)
                / "candidate.png"
            )

            Image.new(
                "RGB",
                (
                    32,
                    32,
                ),
                "white",
            ).save(
                image
            )

            response = Mock(
                status_code=200,
                headers={
                    "content-type": (
                        "application/json"
                    )
                },
            )

            response.raise_for_status.return_value = (
                None
            )

            response.json.return_value = {
                "message": {
                    "content": json.dumps(
                        {
                            "observed_evidence": [
                                "visible forms"
                            ],
                            "reason": (
                                "brief fit"
                            ),
                            **scores,
                        }
                    )
                }
            }

            with patch(
                "app.services.subject_adherence.httpx.post",
                return_value=response,
            ) as send:
                result = (
                    assess_subject_adherence(
                        image,
                        brief,
                        model=(
                            "minicpm-v4.5:latest"
                        ),
                    )
                )

        prompt = (
            send.call_args
            .kwargs[
                "json"
            ][
                "messages"
            ][
                0
            ][
                "content"
            ]
        )

        self.assertNotIn(
            "correct colours",
            prompt,
        )

        return (
            result,
            prompt,
        )

    def test_visible_city_with_wrong_event_fit_fails(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            image = (
                Path(directory)
                / "candidate.png"
            )

            Image.new(
                "RGB",
                (
                    32,
                    32,
                ),
                "white",
            ).save(
                image
            )

            response = Mock(
                status_code=200,
                headers={
                    "content-type": (
                        "application/json"
                    )
                },
            )

            response.raise_for_status.return_value = (
                None
            )

            response.json.return_value = {
                "message": {
                    "content": json.dumps(
                        {
                            "subject_score": .85,
                            "atmosphere_score": .2,
                            "motif_score": .1,
                            "audience_fit_score": .15,
                            "observed_evidence": [
                                "dark city"
                            ],
                            "reason": (
                                "No playful mood or connected web motif"
                            ),
                        }
                    )
                }
            }

            brief = {
                "primary_visual_subject": (
                    "comic city"
                ),
                "event_context": (
                    "children's birthday"
                ),
                "audience_context": (
                    "children"
                ),
                "hard_dimensions": ["audience_fit_score"],
                "creative_description": (
                    "joyful playful celebration"
                ),
                "symbolic_motifs": [
                    "connected web geometry"
                ],
            }

            with patch(
                "app.services.subject_adherence.httpx.post",
                return_value=response,
            ):
                result = (
                    assess_subject_adherence(
                        image,
                        brief,
                        model=(
                            "minicpm-v4.5:latest"
                        ),
                    )
                )

        self.assertEqual(
            result[
                "decision"
            ],
            "BRIEF_FIDELITY_DRIFT",
        )

        self.assertTrue(
            hard_fidelity_failure(
                result
            )
        )

        self.assertEqual(
            result[
                "failure_category"
            ],
            "MOTIF_DRIFT",
        )

        self.assertIn(
            "motif_score",
            result[
                "hard_failed_dimensions"
            ],
        )

        self.assertIn(
            "audience_fit_score",
            result[
                "hard_failed_dimensions"
            ],
        )

        self.assertEqual(
            result[
                "subject_score"
            ],
            .85,
        )

        self.assertEqual(
            result[
                "motif_score"
            ],
            .1,
        )

    def test_missing_web_topology_fails_even_with_good_subject_and_mood(
        self,
    ):
        brief = {
            "primary_visual_subject": (
                "comic city"
            ),
            "event_context": (
                "children birthday"
            ),
            "creative_description": (
                "joyful playful celebration"
            ),
            "audience_context": (
                "children"
            ),
            "symbolic_motifs": [
                "connected web geometry"
            ],
            "visual_semantics": {
                "motif_structure_description": (
                    "connected radial strands with curved cross-links; "
                    "isolated rays are insufficient"
                )
            },
        }

        result, prompt = self._assess(
            brief,
            {
                "subject_score": .9,
                "atmosphere_score": .9,
                "motif_score": .1,
                "audience_fit_score": .9,
            },
        )

        self.assertEqual(
            result[
                "decision"
            ],
            "BRIEF_FIDELITY_DRIFT",
        )

        self.assertEqual(
            result[
                "failure_category"
            ],
            "MOTIF_DRIFT",
        )

        self.assertIn(
            "motif_score",
            result[
                "hard_failed_dimensions"
            ],
        )

        self.assertTrue(
            hard_fidelity_failure(
                result
            )
        )

        self.assertIn(
            "curved cross-links",
            prompt,
        )

        self.assertIn(
            "isolated rays",
            prompt,
        )

    def test_inferred_motif_is_diagnostic_for_simple_floral_subject(
        self,
    ):
        brief = {
            "primary_visual_subject": (
                "flowers"
            ),
            "visual_semantics": {
                "motif_descriptors": [
                    "soft petals"
                ]
            },
            "semantic_provenance": {
                "motif_descriptors": (
                    "SEMANTIC_INFERENCE"
                )
            },
        }

        result, _ = self._assess(
            brief,
            {
                "subject_score": .9,
                "atmosphere_score": .8,
                "motif_score": .2,
                "audience_fit_score": .8,
            },
        )

        self.assertEqual(
            result[
                "decision"
            ],
            "BRIEF_FIDELITY_DRIFT",
        )

        self.assertFalse(
            result[
                "compound_fidelity_failure"
            ]
        )

    def test_explicit_anchor_does_not_promote_inferred_motif(
        self,
    ):
        brief = {
            "background_inspiration": (
                "city with connected forms"
            ),
            "primary_visual_subject": (
                "city with connected forms"
            ),
            "visual_anchors": [
                "city skyline"
            ],
            "visual_semantics": {
                "motif_descriptors": [
                    "connected geometry"
                ]
            },
            "semantic_provenance": {
                "primary_visual_subject": (
                    "USER_EXPLICIT"
                ),
                "visual_anchors": (
                    "USER_EXPLICIT"
                ),
                "motif_descriptors": (
                    "SEMANTIC_INFERENCE"
                ),
            },
        }

        result, _ = self._assess(
            brief,
            {
                "subject_score": .9,
                "atmosphere_score": .8,
                "motif_score": .2,
                "audience_fit_score": .8,
            },
        )

        self.assertEqual(
            result[
                "decision"
            ],
            "BRIEF_FIDELITY_DRIFT",
        )

        self.assertEqual(
            result[
                "failure_category"
            ],
            "MOTIF_DRIFT",
        )

        self.assertFalse(
            result[
                "compound_fidelity_failure"
            ]
        )

    def test_explicit_treatment_promotes_missing_motif(
        self,
    ):
        brief = {
            "background_inspiration": (
                "city with connected forms"
            ),
            "primary_visual_subject": (
                "city with connected forms"
            ),
            "visual_anchors": [
                "city skyline"
            ],
            "theme_reference_treatment": (
                "connected web geometry"
            ),
            "visual_semantics": {
                "motif_descriptors": [
                    "connected geometry"
                ]
            },
            "semantic_provenance": {
                "primary_visual_subject": (
                    "USER_EXPLICIT"
                ),
                "visual_anchors": (
                    "USER_EXPLICIT"
                ),
                "motif_descriptors": (
                    "SEMANTIC_INFERENCE"
                ),
                "theme_reference_treatment": (
                    "USER_EXPLICIT"
                ),
            },
        }

        result, _ = self._assess(
            brief,
            {
                "subject_score": .9,
                "atmosphere_score": .8,
                "motif_score": .2,
                "audience_fit_score": .8,
            },
        )

        self.assertEqual(
            result[
                "decision"
            ],
            "BRIEF_FIDELITY_DRIFT",
        )

        self.assertEqual(
            result[
                "failure_category"
            ],
            "MOTIF_DRIFT",
        )

        self.assertTrue(
            result[
                "compound_fidelity_failure"
            ]
        )

        self.assertIn(
            "motif_score",
            result[
                "hard_failed_dimensions"
            ],
        )

        self.assertTrue(
            hard_fidelity_failure(
                result
            )
        )

    def test_ocr_timeout_is_bounded_and_explicit(
        self,
    ):
        with patch(
            "app.services.comfyui_client.pytesseract.image_to_data",
            side_effect=RuntimeError(
                "Tesseract process timeout"
            ),
        ) as ocr:
            result = (
                detect_text_evidence(
                    Image.new(
                        "RGB",
                        (
                            64,
                            64,
                        ),
                        "white",
                    )
                )
            )

        self.assertGreater(
            ocr.call_args.kwargs[
                "timeout"
            ],
            0,
        )

        self.assertEqual(
            result[
                "ocr_diagnostics"
            ][
                "timed_out_variants"
            ],
            [
                "original",
                "contrast",
                "threshold",
                "inverted",
            ],
        )

        self.assertEqual(
            result[
                "ocr_diagnostics"
            ][
                "decision"
            ],
            "OCR_UNAVAILABLE",
        )

    def test_exhausted_ocr_budget_starts_no_more_processes(
        self,
    ):
        diagnostics = {
            "attempted": [],
            "completed_variants": [],
            "timed_out_variants": [],
            "failed_variants": [],
            "failure_reasons": {},
            "_deadline": 0.0,
        }

        with patch(
            "app.services.comfyui_client.pytesseract.image_to_data"
        ) as ocr:
            self.assertIsNone(
                _ocr_data(
                    Image.new(
                        "L",
                        (
                            16,
                            16,
                        ),
                    ),
                    "--psm 11",
                    "original",
                    diagnostics,
                )
            )

        ocr.assert_not_called()

        self.assertEqual(
            diagnostics[
                "timed_out_variants"
            ],
            [
                "original"
            ],
        )

    def test_five_event_families_require_atmosphere_and_audience_when_given(
        self,
    ):
        cases = (
            (
                "black-tie gala",
                "refined formal ballroom",
                "adults",
                "formal",
            ),
            (
                "garden celebration",
                "warm organic garden",
                "families",
                "organic",
            ),
            (
                "modern awards",
                "prestigious contemporary abstraction",
                "adults",
                "prestigious",
            ),
            (
                "corporate conference",
                "professional polished stage",
                "professionals",
                "professional",
            ),
            (
                "memorial service",
                "dignified restrained memorial",
                "family",
                "dignified",
            ),
        )

        for (
            event,
            atmosphere,
            audience,
            expected,
        ) in cases:
            with self.subTest(
                event=event
            ):
                brief = {
                    "primary_visual_subject": (
                        "abstract forms"
                        if event
                        == "modern awards"
                        else event
                    ),
                    "event_context": (
                        event
                    ),
                    "creative_description": (
                        atmosphere
                    ),
                    "audience_context": (
                        audience
                    ),
                }

                result, prompt = (
                    self._assess(
                        brief,
                        {
                            "subject_score": .9,
                            "atmosphere_score": .9,
                            "audience_fit_score": .9,
                        },
                    )
                )

                self.assertEqual(
                    result[
                        "decision"
                    ],
                    "pass",
                )

                self.assertIn(
                    expected,
                    prompt,
                )

                wrong, _ = (
                    self._assess(
                        brief,
                        {
                            "subject_score": .9,
                            "atmosphere_score": .2,
                            "audience_fit_score": .9,
                        },
                    )
                )

                self.assertEqual(
                    wrong[
                        "decision"
                    ],
                    "BRIEF_FIDELITY_DRIFT",
                )

    def test_sdxl_child_mood_and_web_topology_are_explicit(
        self,
    ):
        document = fixture(
            401,
            "children birthday",
            "comic city web geometry",
            "royal blue",
            "red",
            "dominant",
            "none",
            "asymmetric",
        )

        document[
            "brief"
        ].update(
            creative_description=(
                "joyful playful celebration"
            ),
            audience_context=(
                "children"
            ),
            symbolic_motifs=[
                "connected web network"
            ],
        )

        document[
            "brief"
        ][
            "visual_semantics"
        ] = {
            "atmosphere_descriptors": [
                "joyful",
                "playful",
            ],
            "audience_descriptors": [
                "child-friendly",
                "children",
            ],
            "motif_structure_description": (
                "connected radial strands "
                "with curved cross-links "
                "and cross-link arcs"
            ),
            "incompatible_treatments": [
                "dystopian",
                "fractured perimeter",
            ],
            "scene_family": (
                "general"
            ),
        }

        direction = (
            document[
                "directions"
            ][
                0
            ][
                "direction"
            ]
        )

        direction[
            "composition_strategy"
        ] = (
            "dramatic atmospheric perspective "
            "with fractured perimeter and spikes"
        )

        prompt = (
            compile_candidate_prompt(
                build_candidate_spec(
                    document,
                    "sdxl-1-0",
                    "A",
                ),
                1,
            )
        )

        for field in (
            prompt.clip_l,
            prompt.clip_g,
        ):
            self.assertIn(
                "joyful",
                field,
            )

            self.assertIn(
                "child-friendly",
                field,
            )

            self.assertNotIn(
                "fractured perimeter",
                field,
            )

        self.assertIn(
            "cross-link arcs",
            prompt.clip_g,
        )

        self.assertIn(
            "dystopian",
            prompt.negative,
        )

    def test_sdxl_formal_and_botanical_mood_do_not_inherit_playful_city(
        self,
    ):
        for (
            event,
            scene,
            description,
            expected,
        ) in (
            (
                "black-tie gala",
                "ballroom columns",
                "refined formal luxury",
                "refined",
            ),
            (
                "garden celebration",
                "botanical garden",
                "warm organic garden",
                "organic",
            ),
            (
                "modern awards",
                "abstract sculptural geometry",
                "prestigious contemporary awards",
                "prestigious",
            ),
        ):
            with self.subTest(
                event=event
            ):
                document = fixture(
                    403,
                    event,
                    scene,
                    "purple",
                    "silver",
                    "dominant",
                    "none",
                    "asymmetric",
                )

                document[
                    "brief"
                ][
                    "creative_description"
                ] = description

                document[
                    "brief"
                ][
                    "visual_semantics"
                ] = {
                    "atmosphere_descriptors": [
                        description
                    ]
                }

                prompt = (
                    compile_candidate_prompt(
                        build_candidate_spec(
                            document,
                            "sdxl-1-0",
                            "A",
                        ),
                        1,
                    )
                )

                self.assertIn(
                    expected,
                    prompt.clip_g,
                )

                self.assertNotIn(
                    "child-friendly",
                    prompt.clip_g,
                )

                self.assertNotIn(
                    "web-like",
                    prompt.clip_g,
                )

    def test_ocr_same_sha_reuses_one_result(
        self,
    ):
        _OCR_CACHE.clear()

        image = Image.new(
            "RGB",
            (
                32,
                32,
            ),
            "white",
        )

        with patch(
            "app.services.comfyui_client.detect_text_evidence",
            return_value={
                "confirmed_tokens": []
            },
        ) as detect:
            first = (
                _cached_text_evidence(
                    image,
                    "same-sha",
                )
            )

            second = (
                _cached_text_evidence(
                    image,
                    "same-sha",
                )
            )

        self.assertIs(
            first,
            second,
        )

        self.assertEqual(
            detect.call_count,
            1,
        )

    def test_validator_rejects_ocr_timeout_and_reuses_source_sha(
        self,
    ):
        _OCR_CACHE.clear()

        evidence = {
            "confirmed_tokens": [],
            "text_detected": False,
            "tokens": [],
            "text_lines": [],
            "text_area_ratio": 0,
            "ocr_diagnostics": {
                "decision": (
                    "OCR_UNAVAILABLE"
                ),
                "timed_out_variants": [
                    "original"
                ],
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            image = (
                Path(directory)
                / "candidate.png"
            )

            Image.new(
                "RGB",
                (
                    64,
                    64,
                ),
                "white",
            ).save(
                image
            )

            with (
                patch(
                    "app.services.comfyui_client.detect_text_evidence",
                    return_value=evidence,
                ) as ocr,
                patch(
                    "app.services.comfyui_client.detect_human_signals",
                    return_value={
                        "detected": False,
                        "people": [],
                        "faces": [],
                        "profiles": [],
                        "upper_bodies": [],
                    },
                ),
                patch(
                    "app.services.comfyui_client.boilerplate_metrics",
                    return_value={
                        "boilerplate_score": 0
                    },
                ),
                patch(
                    "app.services.comfyui_client.validate_visual_quality",
                    return_value={},
                ),
                patch(
                    "app.services.comfyui_client._valid_background_dimensions",
                    return_value={
                        (
                            64,
                            64,
                        )
                    },
                ),
            ):
                for _ in range(
                    2
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "OCR_TIMEOUT",
                    ) as caught:
                        validate_background(
                            image
                        )

                    self.assertEqual(
                        caught.exception.diagnostics[
                            "primary_failure_category"
                        ],
                        "OCR_TIMEOUT",
                    )

                    self.assertEqual(
                        caught.exception.diagnostics[
                            "ocr_diagnostics"
                        ],
                        evidence[
                            "ocr_diagnostics"
                        ],
                    )

            self.assertEqual(
                ocr.call_count,
                4,
            )
            self.assertEqual([call.kwargs["timeout_seconds"] for call in ocr.call_args_list],
                             [settings.ocr_timeout_seconds,
                              min(settings.ocr_timeout_seconds * 2, 60.0)] * 2)

    def test_transient_ocr_timeout_revalidates_identical_png(self):
        _OCR_CACHE.clear()
        unavailable = {"confirmed_tokens": [], "text_detected": False,
                       "tokens": [], "text_lines": [], "text_area_ratio": 0,
                       "ocr_diagnostics": {"decision": "OCR_UNAVAILABLE",
                                           "timed_out_variants": ["original"]}}
        sufficient = {**unavailable, "ocr_diagnostics": {
            "decision": "OCR_PARTIAL_SUFFICIENT", "timed_out_variants": ["inverted"]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.png"
            Image.new("RGB", (64, 64), "white").save(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch("app.services.comfyui_client.detect_text_evidence",
                       side_effect=[unavailable, sufficient]) as ocr, \
                 patch("app.services.comfyui_client.detect_human_signals",
                       return_value={"detected": False, "people": [], "faces": [],
                                     "profiles": [], "upper_bodies": []}), \
                 patch("app.services.comfyui_client.boilerplate_metrics",
                       return_value={"boilerplate_score": 0}), \
                 patch("app.services.comfyui_client.validate_visual_quality", return_value={}), \
                 patch("app.services.comfyui_client._valid_background_dimensions",
                       return_value={(64, 64)}), \
                 patch("app.services.comfyui_client.logger.warning") as warning:
                result = validate_background(path)
                self.assertEqual(result["sha256"], digest)
                self.assertEqual(result["ocr_validation_sha256"], digest)
                self.assertEqual(result["ocr_validation_attempts"], 2)
                self.assertEqual(result["ocr_diagnostics"]["decision"], "OCR_PARTIAL_SUFFICIENT")
                self.assertEqual(ocr.call_count, 2)
                self.assertEqual([call.kwargs["timeout_seconds"] for call in ocr.call_args_list],
                                 [settings.ocr_timeout_seconds,
                                  min(settings.ocr_timeout_seconds * 2, 60.0)])
                self.assertTrue(any("event=ocr_infrastructure_retry" in call.args[0]
                                    for call in warning.call_args_list))
                self.assertEqual(validate_background(path)["sha256"], digest)
                self.assertEqual(ocr.call_count, 2)

    def test_readable_text_found_on_same_png_recheck_remains_hard_failure(self):
        unavailable = {"confirmed_tokens": [], "text_detected": False, "tokens": [],
                       "text_lines": [], "text_area_ratio": 0,
                       "ocr_diagnostics": {"decision": "OCR_UNAVAILABLE",
                                           "timed_out_variants": ["original"]}}
        readable = {**unavailable, "confirmed_tokens": ["WELCOME"],
                    "text_detected": True, "tokens": ["WELCOME"],
                    "ocr_diagnostics": {"decision": "complete", "timed_out_variants": []}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.png"
            Image.new("RGB", (64, 64), "white").save(path)
            with patch("app.services.comfyui_client._cached_text_evidence",
                       side_effect=[unavailable, readable]) as ocr, \
                 patch("app.services.comfyui_client.detect_human_signals",
                       return_value={"detected": False, "people": [], "faces": [],
                                     "profiles": [], "upper_bodies": []}), \
                 patch("app.services.comfyui_client.boilerplate_metrics",
                       return_value={"boilerplate_score": 0}), \
                 patch("app.services.comfyui_client.validate_visual_quality", return_value={}), \
                 patch("app.services.comfyui_client._valid_background_dimensions",
                       return_value={(64, 64)}):
                with self.assertRaises(ValueError) as caught:
                    validate_background(path)
            self.assertEqual(caught.exception.diagnostics["primary_failure_category"], "TEXT")
            self.assertEqual(ocr.call_count, 2)
            self.assertEqual(ocr.call_args_list[0].args[1], ocr.call_args_list[1].args[1])

    def test_sd35_effective_workflow_wiring_and_profile(
        self,
    ):
        document = fixture(
            402,
            "garden celebration",
            "botanical garden",
            "green",
            "cream",
            "dominant",
            "none",
            "asymmetric",
        )

        compiled = (
            compile_candidate_prompt(
                build_candidate_spec(
                    document,
                    settings.engine_3_id,
                    "A",
                ),
                1,
            )
        )

        profile = profile_by_name(
            settings.engine_3_id,
            "alternate_sampler",
        )

        seed = 20261009

        workflow = build_workflow(
            settings.engine_3_id,
            compiled.positive,
            compiled.negative,
            seed,
            "probe/static",
            compiled_prompt=compiled,
            sampling_profile=profile,
        )

        audit = (
            effective_workflow_summary(
                workflow,
                settings.engine_3_id,
            )
        )

        self.assertEqual(
            audit[
                "checkpoint"
            ],
            settings.sd35_checkpoint,
        )

        self.assertEqual(
            (
                audit[
                    "clip_l"
                ],
                audit[
                    "clip_g"
                ],
                audit[
                    "t5"
                ],
            ),
            (
                settings.sd35_clip_l,
                settings.sd35_clip_g,
                settings.sd35_t5,
            ),
        )

        self.assertEqual(
            (
                audit[
                    "width"
                ],
                audit[
                    "height"
                ],
            ),
            (
                settings.sd35_generation_width,
                settings.sd35_generation_height,
            ),
        )

        self.assertEqual(
            audit[
                "seed"
            ],
            seed,
        )

        self.assertEqual(
            (
                audit[
                    "sampler"
                ],
                audit[
                    "scheduler"
                ],
                audit[
                    "cfg"
                ],
                audit[
                    "steps"
                ],
                audit[
                    "shift"
                ],
            ),
            (
                profile.sampler_name,
                profile.scheduler,
                profile.cfg,
                profile.steps,
                profile.shift,
            ),
        )

        self.assertEqual(
            audit[
                "positive_fields"
            ],
            {
                "clip_l": (
                    compiled.clip_l
                ),
                "clip_g": (
                    compiled.clip_g
                ),
                "t5xxl": (
                    compiled.t5
                ),
            },
        )

        self.assertEqual(
            audit[
                "positive_route"
            ],
            [
                "16",
                0,
            ],
        )

        self.assertEqual(
            audit[
                "negative_route"
            ],
            [
                "40",
                0,
            ],
        )

        self.assertEqual(
            audit[
                "vae"
            ],
            [
                "4",
                2,
            ],
        )

        baseline = profile_by_name(
            settings.engine_3_id,
            "official_baseline",
        )

        native = build_workflow(
            settings.engine_3_id,
            compiled.positive,
            compiled.negative,
            seed,
            "probe/native",
            compiled_prompt=compiled,
            sampling_profile=baseline,
        )

        native_audit = (
            effective_workflow_summary(
                native,
                settings.engine_3_id,
            )
        )

        self.assertIsNone(
            native_audit[
                "shift"
            ]
        )

        self.assertEqual(
            native_audit[
                "model_route"
            ],
            [
                "4",
                0,
            ],
        )

        self.assertEqual(
            audit[
                "model_route"
            ],
            [
                "4",
                0,
            ],
        )

        self.assertEqual(
            native_audit[
                "seed"
            ],
            seed,
        )

        self.assertEqual(
            audit[
                "positive_fields"
            ],
            native_audit[
                "positive_fields"
            ],
        )

        shifted_profile = (
            profile_by_name(
                settings.engine_3_id,
                "raster_rescue_trial",
            )
        )

        shifted = build_workflow(
            settings.engine_3_id,
            compiled.positive,
            compiled.negative,
            seed,
            "probe/shifted",
            compiled_prompt=compiled,
            sampling_profile=shifted_profile,
        )

        shifted_audit = (
            effective_workflow_summary(
                shifted,
                settings.engine_3_id,
            )
        )

        self.assertEqual(
            shifted_audit[
                "model_route"
            ],
            [
                "55",
                0,
            ],
        )

        self.assertEqual(
            shifted_audit[
                "shift"
            ],
            shifted_profile.shift,
        )

    def test_probe_failure_categories_stay_distinct(
        self,
    ):
        for category in (
            "OCR_TIMEOUT",
            "OCR_UNAVAILABLE",
            "VISION_UNAVAILABLE",
        ):
            self.assertEqual(
                classify_probe_failure(
                    {
                        "primary_failure_category": (
                            category
                        )
                    },
                    RuntimeError(
                        "failed"
                    ),
                ),
                category,
            )

        self.assertEqual(
            classify_probe_failure(
                {},
                TimeoutError(
                    "ComfyUI timed out"
                ),
            ),
            "COMFYUI_TIMEOUT",
        )


if __name__ == "__main__":
    unittest.main()
