"""Prompt-only direction retry and diversity regression checks."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.graph.workflow import (
    _log_validation_failure,
    generate_direction_c,
    route_after_validation_a,
    route_after_validation_b,
    route_after_validation_c,
    validate_direction_c,
)
from app.services.creative_direction import build_creative_direction_prompt
from app.services.direction_diversity import build_diversity_constraints, locked_structural_fields
from app.services.validation import direction_signature, validate_direction_diversity
from scripts.generate_prompts_for_reference import main as prompt_only_main, parse_args
from tests.test_direction_structure_contract import direction


class DirectionRetryBoundaryTests(unittest.TestCase):
    def test_general_composition_choices_are_structural_and_keep_alternatives(self):
        brief = {"visual_semantics": {"scene_family": "general"}}
        for letter in "ABC":
            plan = build_diversity_constraints(letter, [], brief)
            choices = plan["available_alternatives"]["composition_family"]
            self.assertIn("central-focal", choices)
            self.assertNotIn("central-monument", choices)
            self.assertGreaterEqual(len(choices), 2)

    def test_central_focal_and_legacy_monument_normalize_together(self):
        current = direction("A", "central-focal composition")
        legacy = direction("B", "central monument composition")
        self.assertEqual(direction_signature(current)["composition_family"], "central-focal")
        self.assertEqual(direction_signature(legacy)["composition_family"], "central-focal")

    def _spatial_brief(self, *affinities, source="USER_EXPLICIT"):
        return {"visual_semantics": {"scene_family": "architectural-interior",
                                     "composition_affinities": list(affinities)},
                "semantic_provenance": {"composition_affinities": source}}

    def _spatial_direction(self, letter, mass="left-heavy", quiet="upper/lower"):
        result = direction(letter, f"first-person curved traversal with {mass} mass")
        result.focal_strategy = "first-person focal building at left"
        result.depth_strategy = "strong receding perspective through layers"
        result.title_safe_region_strategy = f"quiet {quiet.split('/')[0]} region"
        result.programme_safe_region_strategy = f"quiet {quiet.split('/')[1]} region"
        return result

    def test_explicit_spatial_locks_allow_two_other_changes(self):
        brief = self._spatial_brief("immersive first-person viewpoint",
                                   "forward traversal and receding spatial depth",
                                   "arcing spatial motion")
        locked = locked_structural_fields(brief)
        self.assertEqual(locked, {"viewpoint": "first-person", "depth_plan": "receding",
                                  "major_flow": "curved"})
        first = self._spatial_direction("A")
        plan = build_diversity_constraints("B", [first], brief)
        self.assertEqual(plan["suggested_combination"]["viewpoint"], "first-person")
        self.assertEqual(plan["suggested_combination"]["depth_plan"], "receding")
        self.assertNotIn("viewpoint", plan["available_alternatives"])
        second = self._spatial_direction("B", "right-heavy", "left/right")
        validate_direction_diversity(second, [first], locked)
        third = self._spatial_direction("C", "distributed", "right/lower")
        third.focal_strategy = "first-person focal building at right"
        validate_direction_diversity(third, [first, second], locked)
        with self.assertRaisesRegex(ValueError, "repeats") as caught:
            validate_direction_diversity(self._spatial_direction("C"), [first], locked)
        diagnostic = caught.exception.diversity_diagnostics
        self.assertIn("viewpoint", diagnostic["locked_fields"])
        self.assertEqual(diagnostic["changed_fields"], [])
        retry = build_diversity_constraints("B", [first], brief, retry_count=1,
                                            failed_diagnostic=diagnostic)
        self.assertNotEqual(plan["suggested_combination"], retry["suggested_combination"])

    def test_locked_prompt_keeps_viewpoint_and_varies_unlocked_fields(self):
        brief = self._spatial_brief("immersive first-person viewpoint", "receding spatial depth")
        first = self._spatial_direction("A")
        plan = build_diversity_constraints("B", [first], brief)
        prompt = json.loads(build_creative_direction_prompt(
            brief, "B", "visual designer", diversity_constraints=plan))
        requirements = " ".join(prompt["requirements"])
        self.assertIn("locked structural", requirements)
        self.assertNotIn("least three independent", requirements)
        self.assertEqual(prompt["structural_diversity_plan"]["locked_fields"]["viewpoint"], "first-person")

    def test_aerial_lock_and_inferred_viewpoint_policy(self):
        brief = self._spatial_brief("aerial overhead viewpoint")
        self.assertEqual(locked_structural_fields(brief)["viewpoint"], "aerial")
        for letter in "ABC":
            self.assertEqual(build_diversity_constraints(letter, [], brief)["suggested_combination"]["viewpoint"], "aerial")
        brief["semantic_provenance"]["composition_affinities"] = "SEMANTIC_INFERENCE"
        self.assertEqual(locked_structural_fields(brief), {})
        self.assertGreater(len(build_diversity_constraints("B", [], brief)["available_alternatives"]["viewpoint"]), 1)

    def test_all_directions_fail_closed_after_retry_exhaustion(self):
        cases = (
            (
                "A",
                route_after_validation_a,
                {
                    "direction_a": None,
                    "retry_count_a": 3,
                    "validation_error_a": "invalid model output",
                },
            ),
            (
                "B",
                route_after_validation_b,
                {
                    "direction_b": None,
                    "retry_count_b": 3,
                    "validation_error_b": "same spatial plan",
                },
            ),
            (
                "C",
                route_after_validation_c,
                {
                    "direction_c": None,
                    "retry_count_c": 3,
                    "validation_error_c": "invalid model output",
                },
            ),
        )

        for direction_id, router, state in cases:
            with self.subTest(direction_id=direction_id):
                with self.assertRaises(ValueError) as caught:
                    router(state)

                self.assertIn(
                    f"PROMPT_DIRECTION_FAILED:{direction_id}:",
                    str(caught.exception),
                )

    def test_direction_c_allows_two_retries_then_fails_closed(self):
        state = {
            "direction_c": None,
            "retry_count_c": 2,
            "validation_error_c": "Direction C repeats A's spatial plan",
        }

        self.assertEqual(
            route_after_validation_c(state),
            "direction_c",
        )

        state["retry_count_c"] = 3

        with self.assertRaises(ValueError) as caught:
            route_after_validation_c(state)

        self.assertIn(
            "PROMPT_DIRECTION_FAILED:C:",
            str(caught.exception),
        )

    def test_final_failure_log_does_not_claim_retry_three_of_two(self):
        with patch("app.graph.workflow.logger.warning") as warning:
            _log_validation_failure("C", 3, ValueError("duplicate"))
        rendered = warning.call_args.args[0] % warning.call_args.args[1:]
        self.assertNotIn("retry=3/2", rendered)
        self.assertIn("retry=2/2", rendered)

    def test_direction_c_receives_structural_occupancy_before_first_attempt(self):
        first = direction("A", "central architectural arrangement")
        second = direction("B", "diagonal sculptural planes")
        generator = AsyncMock(return_value=direction("C"))
        with patch("app.graph.workflow.generate_creative_direction", generator):
            asyncio.run(generate_direction_c({"brief": {"event_type": "awards ceremony",
                                                       "background_inspiration": "abstract sculptural geometry"},
                                              "direction_a": first, "direction_b": second}))
        contract = generator.call_args.kwargs["diversity_constraints"]
        self.assertEqual(len(contract["previous_signatures"]), 2)
        self.assertGreaterEqual(len(contract["available_alternatives"]["composition_family"]), 1)

    def test_fixed_briefs_have_viable_direction_c_alternatives(self):
        briefs = (
            ({"event_type": "adult thirtieth birthday", "theme": "elegant black-tie ballroom",
              "background_inspiration": "ballroom architecture", "primary_colour": "black",
              "secondary_colour": "white", "palette_mode": "achromatic"}, "architectural-interior"),
            ({"event_type": "garden celebration", "theme": "botanical garden",
              "background_inspiration": "foliage and paths", "primary_colour": "green",
              "secondary_colour": "cream", "palette_mode": "dominant"}, "organic"),
            ({"event_type": "modern awards ceremony", "theme": "abstract sculptural geometry",
              "background_inspiration": "polished geometric forms", "primary_colour": "purple",
              "secondary_colour": "silver", "palette_mode": "dominant"}, "abstract"),
        )
        first = direction("A", "central monument with framed geometry")
        first.focal_strategy = "upper left focal sculpture"
        first.depth_strategy = "foreground layered surfaces"
        first.border_or_perimeter_strategy = "framed perimeter"
        second = direction("B", "diagonal planes and right-heavy mass")
        second.focal_strategy = "lower right focal sculpture"
        second.depth_strategy = "receding perspective planes"
        second.border_or_perimeter_strategy = "open perimeter"
        for brief, expected_kind in briefs:
            with self.subTest(kind=expected_kind):
                brief["visual_semantics"] = {"scene_family": expected_kind}
                first_plan = build_diversity_constraints("A", [], brief)
                second_plan = build_diversity_constraints("B", [first], brief)
                plan = build_diversity_constraints("C", [first, second], brief)
                self.assertEqual(first_plan["scene_kind"], expected_kind)
                self.assertEqual(second_plan["scene_kind"], expected_kind)
                self.assertEqual(plan["scene_kind"], expected_kind)
                self.assertEqual(plan["required_material_changes_per_previous"], 2)
                self.assertTrue(second_plan["available_alternatives"]["composition_family"])
                first_signature = direction_signature(first)
                second_changes = sum(
                    value != first_signature[field]
                    for field, value in second_plan["suggested_combination"].items()
                    if first_signature[field] != "unspecified"
                )
                self.assertGreaterEqual(second_changes, 2)
                for earlier in (first, second):
                    signature = direction_signature(earlier)
                    material = sum(value != signature[field] for field, value in plan["suggested_combination"].items()
                                   if signature[field] != "unspecified")
                    self.assertGreaterEqual(material, 2)
                text = json.dumps(plan).casefold()
                for irrelevant in ("city", "web", "child", "street-canyon", "aerial"):
                    self.assertNotIn(irrelevant, text)

    def test_unknown_scene_family_uses_general_choices(self):
        brief = {"event_type": "unfamiliar event", "visual_semantics": {"scene_family": "subterranean"}}
        plan = build_diversity_constraints("A", [], brief)
        self.assertEqual(plan["scene_kind"], "general")
        self.assertTrue(plan["available_alternatives"]["composition_family"])

    def test_ungrounded_scene_inference_is_not_sent_to_direction_model(self):
        brief = {"background_inspiration": "abstract nested mineral loops",
                 "visual_semantics": {"scene_family": "architectural-interior",
                                      "motif_descriptors": ["nested loops"]},
                 "semantic_provenance": {"scene_family": "SEMANTIC_INFERENCE"},
                 "semantic_confidence": {"scene_family": .97}}
        plan = build_diversity_constraints("A", [], brief)
        payload = json.loads(build_creative_direction_prompt(
            brief, "A", "visual designer", diversity_constraints=plan))
        self.assertEqual(plan["scene_kind"], "general")
        self.assertEqual(payload["event_brief"]["visual_semantics"]["scene_family"], "general")
        self.assertEqual(brief["visual_semantics"]["scene_family"], "architectural-interior")

    def test_diversity_rejection_has_bounded_signature_diagnostic(self):
        first = direction("A")
        repeated = direction("C", "layered asymmetric arrangement")
        with self.assertRaises(ValueError) as caught:
            validate_direction_diversity(repeated, [first])
        detail = caught.exception.diversity_diagnostics
        self.assertEqual(detail["compared_to"], "A")
        self.assertEqual(detail["required_material_changes"], 2)
        self.assertLess(len(detail["changed_fields"]), 2)

    def test_workflow_logs_and_persists_compact_diversity_diagnostic(self):
        first = direction("A")
        repeated = direction("C", "layered asymmetric arrangement")
        with patch("app.graph.workflow.validate_creative_direction"), \
             patch("app.graph.workflow.logger.warning") as warning:
            result = validate_direction_c({"brief": {}, "direction_a": first,
                                           "direction_b": None, "direction_c": repeated})
        self.assertEqual(result["diversity_diagnostics_c"]["compared_to"], "A")
        messages = [call.args[0] for call in warning.call_args_list]
        self.assertIn("direction_diversity=%s", messages)
        self.assertNotIn("Sculptural mineral arches", str(warning.call_args_list))

    def test_retry_prompts_escalate_diversity_correction(self):
        brief = {"event_type": "modern awards ceremony", "theme": "abstract sculpture"}
        first, second = direction("A"), direction("B", "diagonal arrangement")
        diagnostic = {"compared_to": "A", "repeated_fields": ["depth_plan"],
                      "changed_fields": ["major_flow"], "required_material_changes": 3}
        prompts = []
        for retry_count in (1, 2):
            plan = build_diversity_constraints("C", [first, second], brief,
                                               retry_count=retry_count, failed_diagnostic=diagnostic)
            prompts.append(json.loads(build_creative_direction_prompt(
                brief, "C", "visual designer", "Direction C repeats A's spatial plan",
                diversity_constraints=plan,
            )))
        self.assertIn("at least two", prompts[0]["correction_required"]["instruction"])
        self.assertIn("clearly unused", prompts[1]["correction_required"]["instruction"])
        for prompt in prompts:
            instruction = prompt["correction_required"]["instruction"]
            self.assertIn("Preserve exactly the subject", instruction)
            self.assertNotIn("at least three", instruction)
        self.assertNotEqual(prompts[0]["correction_required"]["instruction"],
                            prompts[1]["correction_required"]["instruction"])
        self.assertEqual(prompts[1]["correction_required"]["diversity_overlap"], diagnostic)

    def test_prompt_only_resume_flag_reuses_checkpoints(self):
        self.assertFalse(parse_args(["ABCDEF-123456"]).resume)
        self.assertTrue(parse_args(["ABCDEF-123456", "--resume"]).resume)
        with tempfile.TemporaryDirectory() as directory:
            brief_path = Path(directory) / "ABCDEF-123456" / "brief.json"
            brief_path.parent.mkdir()
            brief_path.write_text('{"event_type":"gala"}', encoding="utf-8")
            workflow = AsyncMock(return_value={"brief_path": str(brief_path), "prompts_path": "prompts.json"})
            with patch("scripts.generate_prompts_for_reference.settings.programme_data_path", Path(directory)), \
                 patch("scripts.generate_prompts_for_reference.run_prompt_workflow", workflow):
                asyncio.run(prompt_only_main(["ABCDEF-123456", "--resume"]))
            self.assertTrue(workflow.call_args.kwargs["resume"])
