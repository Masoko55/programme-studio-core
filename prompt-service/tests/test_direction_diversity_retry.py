"""Prompt-only direction retry and diversity regression checks."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.graph.workflow import (_log_validation_failure, generate_direction_c,
                                route_after_validation_a, route_after_validation_b,
                                route_after_validation_c, validate_direction_c,
                                fallback_direction_a, fallback_direction_b,
                                fallback_direction_c)
from app.services.creative_direction import build_creative_direction_prompt
from app.services.direction_diversity import build_diversity_constraints
from app.services.validation import direction_signature, validate_direction_diversity
from scripts.generate_prompts_for_reference import main as prompt_only_main, parse_args
from tests.test_direction_structure_contract import direction


class DirectionRetryBoundaryTests(unittest.TestCase):
    def test_all_directions_have_bounded_fallback(self):
        brief = {"background_inspiration": "layered dunes", "primary_colour": "sand",
                 "secondary_colour": "navy", "mood": "calm"}
        a_state = {"brief": brief, "direction_a": None, "retry_count_a": 3,
                   "validation_error_a": "invalid model output"}
        self.assertEqual(route_after_validation_a(a_state), "fallback_direction_a")
        a = fallback_direction_a(a_state)["direction_a"]
        b_state = {"direction_a": a, "direction_b": None, "retry_count_b": 3,
                   "validation_error_b": "same spatial plan"}
        self.assertEqual(route_after_validation_b(b_state), "fallback_direction_b")
        b = fallback_direction_b(b_state)["direction_b"]
        c = fallback_direction_c({"direction_a": a})["direction_c"]
        self.assertEqual([item.direction_id for item in (a, b, c)], list("ABC"))
        self.assertEqual({item.positive_prompt for item in (a, b, c)}, {a.positive_prompt})
        validate_direction_diversity(b, [a])
        validate_direction_diversity(c, [a, b])

    def test_direction_c_allows_two_retries_then_stops(self):
        state = {"direction_c": None, "retry_count_c": 2,
                 "validation_error_c": "Direction C repeats A's spatial plan"}
        self.assertEqual(route_after_validation_c(state), "direction_c")
        state["retry_count_c"] = 3
        state["direction_a"] = direction("A")
        self.assertEqual(route_after_validation_c(state), "fallback_direction_c")
        self.assertEqual(fallback_direction_c(state)["direction_c"].direction_id, "C")

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
        self.assertIn("at least three", prompts[0]["correction_required"]["instruction"])
        self.assertIn("unused focal location", prompts[1]["correction_required"]["instruction"])
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

