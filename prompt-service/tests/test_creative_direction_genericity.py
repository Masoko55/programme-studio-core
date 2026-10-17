import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from app.services.creative_direction import (
    background_design_summary,
    background_positive_requirements,
    build_creative_direction_prompt,
    generate_creative_direction,
    positive_palette_description,
    reference_visual_language,
    sanitize_positive_prompt,
)
from app.services.validation import (
    validate_positive_prompt_quality,
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
        self.assertNotIn("do not", description.lower())

    def test_background_requirements_do_not_force_stock_edge_frame(self):
        brief = {
            "event_type": "unknown",
            "background_inspiration": "suspended fabric waves crossing a deep atmospheric field",
        }
        text = background_positive_requirements(brief).lower()
        self.assertIn("original", text)
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

    def test_named_reference_is_translated_to_semantics_in_positive_prompt(self):
        brief = {
            "event_type": "children celebration",
            "background_inspiration": "A Solar Sentinel inspired celebration",
            "primary_colour": "blue",
            "secondary_colour": "red",
            "palette_relationship": (
                "requested colours dominate with natural tonal variation"
            ),
            "theme_reference_treatment": (
                "Use Solar Sentinel-inspired visual traits through "
                "interconnected radial geometry and energetic movement. "
                "Do not depict Solar Sentinel, masks, people or readable text."
            ),
            "grill_me_context": {
                "reference_detected": True,
                "reference_name": "Solar Sentinel",
                "reference_type": "character",
                "visual_semantics": {
                    "motif_descriptors": [
                        "interconnected radial geometry",
                        "layered urban forms",
                    ],
                    "motif_structure_description": (
                        "Radial strands connect through curved cross-links "
                        "across layered urban forms."
                    ),
                    "atmosphere_descriptors": [
                        "energetic and playful",
                    ],
                    "scene_family": "abstract",
                },
            },
            "visual_semantics": {
                "motif_descriptors": [
                    "interconnected radial geometry",
                    "layered urban forms",
                ],
                "motif_structure_description": (
                    "Radial strands connect through curved cross-links "
                    "across layered urban forms."
                ),
                "atmosphere_descriptors": [
                    "energetic and playful",
                ],
                "scene_family": "abstract",
            },
        }

        prompt = sanitize_positive_prompt(
            (
                "Solar Sentinel standing above a city. "
                "Do not show lettering. "
                "Blue and red radial geometry with layered urban depth."
            ),
            brief,
            "Visual Designer",
        )

        lowered = prompt.lower()

        self.assertNotIn("solar sentinel", lowered)
        self.assertNotIn("do not", lowered)
        self.assertNotIn("people", lowered)
        self.assertNotIn("character", lowered)
        self.assertNotIn("readable text", lowered)
        self.assertIn("radial", lowered)
        self.assertIn("urban", lowered)

        class Direction:
            direction_id = "A"
            positive_prompt = prompt

        validate_positive_prompt_quality(
            Direction()
        )

    def test_negative_clarification_sentences_do_not_leak_into_visual_language(self):
        brief = {
            "grill_me_context": {
                "reference_detected": True,
                "reference_name": "Clockwork Falcon",
                "visual_semantics": {
                    "motif_descriptors": [
                        "sweeping mechanical arcs",
                    ],
                    "atmosphere_descriptors": [
                        "kinetic and luminous",
                    ],
                },
            },
            "visual_semantics": {
                "motif_descriptors": [
                    "sweeping mechanical arcs",
                ],
                "atmosphere_descriptors": [
                    "kinetic and luminous",
                ],
            },
            "theme_reference_treatment": (
                "Use Clockwork Falcon-inspired sweeping mechanical arcs. "
                "Never depict the reference itself. "
                "Avoid faces, logos and writing."
            ),
        }

        language = reference_visual_language(
            brief
        ).lower()

        self.assertIn(
            "sweeping mechanical arcs",
            language,
        )
        self.assertIn(
            "kinetic and luminous",
            language,
        )
        self.assertNotIn(
            "clockwork falcon",
            language,
        )
        self.assertNotIn(
            "never",
            language,
        )
        self.assertNotIn(
            "avoid",
            language,
        )

    def test_empty_treatments_do_not_authorize_invented_materials(self):
        brief = {"event_type": "invented gathering", "background_inspiration": "interlocking arc lattice",
                 "primary_colour": "blue", "secondary_colour": "red",
                 "visual_semantics": {"motif_descriptors": ["interlocking arc lattice"],
                                      "material_descriptors": [], "texture_descriptors": [],
                                      "lighting_descriptors": []}}
        prompt = sanitize_positive_prompt(
            "Interlocking arc lattice with open depth. Aged parchment texture and metallic highlights. "
            "Coherent foreground and background separation.", brief, "visual designer")
        self.assertIn("interlocking arc lattice", prompt.lower())
        self.assertNotIn("parchment", prompt.lower())
        self.assertNotIn("metallic", prompt.lower())

    def test_explicit_material_texture_lighting_remain_available(self):
        brief = {"event_type": "invented gathering", "background_inspiration": "frosted glass arcs",
                 "primary_colour": "blue", "secondary_colour": "red",
                 "visual_semantics": {"material_descriptors": ["frosted glass"],
                                      "texture_descriptors": ["etched texture"],
                                      "lighting_descriptors": ["cool rim lighting"]},
                 "grill_me_context": {"semantic_provenance": {
                     "material_descriptors": "USER_EXPLICIT", "texture_descriptors": "USER_EXPLICIT",
                     "lighting_descriptors": "USER_EXPLICIT"}}}
        prompt = sanitize_positive_prompt(
            "Frosted glass arcs with etched texture and cool rim lighting. "
            "Open depth and quiet upper space.", brief, "visual designer")
        for phrase in ("frosted glass", "etched texture", "cool rim lighting"):
            self.assertIn(phrase, prompt.lower())

    def test_unsupported_colour_fields_and_styles_are_removed_but_layout_survives(self):
        brief = {"background_inspiration": "interlocking arc lattice",
                 "primary_colour": "blue", "secondary_colour": "red",
                 "visual_semantics": {"material_descriptors": [], "texture_descriptors": [],
                                      "lighting_descriptors": []}}
        prompt = sanitize_positive_prompt(
            "Interlocking arcs with an off-centre focal point. Dark atmospheric background. "
            "Soft atmospheric perspective. Forest green and brown colour field. "
            "Layered depth and diagonal rhythm.", brief, "visual designer")
        self.assertIn("off-centre focal point", prompt)
        self.assertIn("layered depth and diagonal rhythm", prompt.lower())
        for term in ("dark atmospheric", "atmospheric perspective", "forest green", "brown"):
            self.assertNotIn(term, prompt.lower())

    def test_explicit_dark_background_is_preserved(self):
        brief = {"background_inspiration": "interlocking arcs on a dark background",
                 "primary_colour": "blue", "secondary_colour": "red"}
        prompt = sanitize_positive_prompt("Dark atmospheric background with interlocking arcs.",
                                          brief, "visual designer")
        self.assertIn("dark atmospheric background", prompt.lower())

    def test_inferred_material_is_not_turned_into_new_treatment(self):
        brief = {"background_inspiration": "interlocking arcs", "primary_colour": "blue",
                 "visual_semantics": {"material_descriptors": ["metallic finish"]},
                 "grill_me_context": {"semantic_provenance": {
                     "material_descriptors": "SEMANTIC_INFERENCE"}}}
        prompt = sanitize_positive_prompt("Metallic finish around interlocking arcs. Open focal placement with layered arcs and a quiet upper region.",
                                          brief, "visual designer")
        self.assertNotIn("metallic", prompt.lower())
        self.assertIn("open focal placement", prompt.lower())

    def test_model_invention_is_removed_before_direction_is_returned(self):
        brief = {"event_type": "invented gathering", "background_inspiration": "interlocking arc lattice",
                 "primary_colour": "blue", "secondary_colour": "red",
                 "visual_semantics": {"material_descriptors": [], "texture_descriptors": [],
                                      "lighting_descriptors": []}}
        output = {
            "direction_id": "A", "role": "visual designer",
            "positive_prompt": "Interlocking arc lattice with open depth. Aged parchment texture and metallic highlights.",
            "negative_prompt": "people, text, logos", "title_suggestions": [],
            "short_design_rationale": "Open arc composition",
            "composition_strategy": "Asymmetric arcs around an open central field",
            "focal_strategy": "Arc intersection at the lower left",
            "depth_strategy": "Layered near and distant arcs",
            "border_or_perimeter_strategy": "Open perimeter with separated arcs",
            "title_safe_region_strategy": "Quiet upper central area",
            "programme_safe_region_strategy": "Quiet lower right area",
            "layout_guidance": {"title_zone": {"x": .1, "y": .08, "width": .8, "height": .12},
                                "programme_zone": {"x": .1, "y": .6, "width": .8, "height": .25}},
        }
        with patch("app.services.creative_direction.generate_text", new_callable=AsyncMock,
                   return_value={"response": json.dumps(output)}):
            direction = asyncio.run(generate_creative_direction("A", brief))
        self.assertNotIn("parchment", direction.positive_prompt.lower())
        self.assertNotIn("metallic", direction.positive_prompt.lower())


if __name__ == "__main__":
    unittest.main()
