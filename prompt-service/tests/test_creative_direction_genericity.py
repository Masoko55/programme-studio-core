import unittest

from app.services.creative_direction import (
    background_design_summary,
    background_positive_requirements,
    build_creative_direction_prompt,
    positive_palette_description,
    reference_visual_language,
)


class CreativeDirectionGenericityTests(unittest.TestCase):
    def test_reference_visual_language_uses_clarified_fields_without_named_branches(self):
        brief = {
            "event_type": "invented celebration",
            "theme": "Comet Guild",
            "background_inspiration": "floating crystalline arches over a misty landscape",
            "theme_reference_treatment": "kinetic diagonal geometry and luminous arc patterns",
            "visual_style": "editorial surrealism",
            "material_language": "frosted glass and brushed metal",
            "texture_language": "fine etched texture",
            "lighting_language": "cool rim lighting",
            "visual_anchors": ["crystalline arches", "mist layers"],
        }
        language = reference_visual_language(brief)
        self.assertIn("kinetic diagonal geometry", language)
        self.assertIn("editorial surrealism", language)
        self.assertIn("frosted glass", language)
        self.assertIn("crystalline arches", language)

    def test_palette_description_accepts_arbitrary_values(self):
        brief = {
            "palette": [
                {"colour": "#13A8B3", "role": "primary"},
                {"colour": "rebeccapurple", "role": "secondary"},
                {"colour": "rgb(242,170,76)", "role": "accent"},
            ],
            "palette_relationship": "balanced with a sparse warm accent",
        }
        description = positive_palette_description(brief)
        self.assertIn("#13a8b3", description.lower())
        self.assertIn("rebeccapurple", description.lower())
        self.assertIn("rgb(242,170,76)", description.lower())

    def test_background_requirements_do_not_force_stock_edge_frame(self):
        brief = {
            "event_type": "unknown",
            "background_inspiration": "suspended fabric waves crossing a deep atmospheric field",
        }
        text = background_positive_requirements(brief).lower()
        self.assertIn("original composition", text)
        self.assertIn("negative space", text)
        self.assertNotIn("stock poster", text)
        self.assertNotIn("side slabs", text)
        self.assertNotIn("keep decoration at the outer edges and lower corners", text)

    def test_direction_prompt_requires_structural_difference_without_slot_style(self):
        brief = {
            "event_type": "custom event",
            "theme": "floating mineral garden",
            "background_inspiration": "floating mineral formations in atmospheric depth",
            "primary_colour": "#214B5B",
            "secondary_colour": "#E2C97A",
        }
        prompt = build_creative_direction_prompt(
            brief,
            "B",
            "brief-led",
            previous_directions=[
                "symmetrical arch arrangement with shallow depth",
            ],
        )
        self.assertIn("no slot has a preset style", prompt.lower())
        self.assertIn("materially different arrangement", prompt.lower())
        self.assertIn("stock programme template", prompt.lower())
        self.assertIn("floating mineral formations", prompt.lower())

    def test_background_summary_preserves_brief_specific_subject(self):
        brief = {
            "event_type": "custom event",
            "background_inspiration": "layered wind-carved sandstone ribbons",
            "theme_reference_treatment": "organic flowing geometry",
        }
        summary = background_design_summary(brief)
        self.assertIn("layered wind-carved sandstone ribbons", summary)
        self.assertIn("organic flowing geometry", summary)
        self.assertIn("rather than a boilerplate frame", summary)


if __name__ == "__main__":
    unittest.main()
