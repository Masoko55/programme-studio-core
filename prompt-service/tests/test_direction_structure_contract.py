import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from app.graph.workflow import generate_direction_a, generate_direction_b, generate_direction_c
from app.schemas.creative_direction import CreativeDirectionOutput
from app.services.creative_direction import direction_structural_summary, resolve_palette_mode
from app.services.validation import validate_direction_diversity


def direction(letter: str, composition: str = "asymmetric layered arrangement") -> CreativeDirectionOutput:
    return CreativeDirectionOutput.model_validate({
        "direction_id": letter,
        "role": "brief led",
        "positive_prompt": "Sculptural mineral arches in atmospheric depth",
        "negative_prompt": "people, text, logos, watermarks",
        "title_suggestions": ["Event"],
        "short_design_rationale": "A distinct spatial arrangement",
        "composition_strategy": composition,
        "focal_strategy": "off centre sculptural focus",
        "depth_strategy": "deep receding mineral layers",
        "border_or_perimeter_strategy": "open asymmetrical perimeter",
        "title_safe_region_strategy": "quiet space above the focal form",
        "programme_safe_region_strategy": "quiet space beside the lower forms",
        "layout_guidance": {
            "title_zone": {"x": .1, "y": .08, "width": .8, "height": .12},
            "programme_zone": {"x": .1, "y": .47, "width": .8, "height": .4},
        },
    })


class DirectionStructureContractTests(unittest.TestCase):
    def test_missing_or_blank_strategy_is_invalid(self):
        for field in (
            "composition_strategy", "focal_strategy", "depth_strategy",
            "border_or_perimeter_strategy", "title_safe_region_strategy",
            "programme_safe_region_strategy",
        ):
            for value in (None, "", "     "):
                with self.subTest(field=field, value=value):
                    data = direction("A").model_dump()
                    if value is None:
                        data.pop(field)
                    else:
                        data[field] = value
                    with self.assertRaises(ValidationError):
                        CreativeDirectionOutput.model_validate(data)

    def test_repeated_spatial_plan_is_rejected_before_image_generation(self):
        first = direction("A")
        repeated = direction("B", "layered asymmetric arrangement")
        with self.assertRaisesRegex(ValueError, "repeats"):
            validate_direction_diversity(repeated, [first])
        different = direction("C", "wide horizontal band beneath a tall focal form")
        different.focal_strategy = "high distant focal cluster"
        different.depth_strategy = "shallow overlapping planes"
        different.border_or_perimeter_strategy = "broken diagonal perimeter"
        validate_direction_diversity(different, [first])

    def test_missing_strategy_triggers_prompt_workflow_retry(self):
        incomplete = direction("A").model_dump()
        incomplete.pop("focal_strategy")
        with patch("app.services.creative_direction.generate_text", new_callable=AsyncMock,
                   return_value={"response": json.dumps(incomplete)}):
            result = asyncio.run(generate_direction_a({"brief": {"event_type": "gala"}}))
        self.assertEqual(result["retry_count_a"], 1)
        self.assertIn("focal_strategy", result["validation_error_a"])

    def test_b_and_c_receive_structural_summaries(self):
        brief = {"background_inspiration": "mineral arches", "visual_anchors": ["arches"]}
        first, second = direction("A"), direction("B", "radial ascending arrangement")
        generator = AsyncMock(return_value=direction("C"))
        with patch("app.graph.workflow.generate_creative_direction", generator):
            asyncio.run(generate_direction_b({"brief": brief, "direction_a": first}))
            b_previous = generator.call_args.kwargs["previous_directions"]
            asyncio.run(generate_direction_c({"brief": brief, "direction_a": first, "direction_b": second}))
            c_previous = generator.call_args.kwargs["previous_directions"]
        self.assertEqual(b_previous, [direction_structural_summary(first, brief)])
        self.assertEqual(c_previous, [direction_structural_summary(first, brief), direction_structural_summary(second, brief)])
        self.assertIn("focal_strategy", b_previous[0])

    def test_palette_exclusivity_requires_explicit_brief(self):
        self.assertEqual(resolve_palette_mode({"primary_colour": "royal blue", "secondary_colour": "red"}), "dominant")
        self.assertEqual(resolve_palette_mode({"primary_colour": "royal blue", "secondary_colour": "red",
                                               "creative_description": "Only these colours, no third colour"}), "strict")
        self.assertEqual(resolve_palette_mode({"primary_colour": "royal blue", "secondary_colour": "red",
                                               "creative_description": "Use only royal blue and red"}), "strict")


if __name__ == "__main__":
    unittest.main()
