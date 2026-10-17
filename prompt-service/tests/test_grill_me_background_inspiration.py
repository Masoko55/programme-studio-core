import unittest
from unittest.mock import patch

from app.schemas.grill_me import GrillMeAnswers, GrillMeSession, ProgrammeItem
from app.services.creative_direction import (
    background_design_summary,
    background_positive_requirements,
    reference_visual_language,
)
from app.services.grill_me_semantic import _safe_context
from app.services.grill_me_service import evaluate_session, freeze_for_generation
from app.services.semantic_contract import build_semantic_contract, resolve_palette_relationship


class GrillMeBackgroundInspirationTests(unittest.TestCase):
    def test_named_reference_keeps_explicit_viewpoint_and_movement_without_assets(self):
        answers = {"theme_reference_treatment": "Solar Sentinel is moving through the city in first person view"}
        review = {"reference_detected": True, "reference_name": "Solar Sentinel",
                  "visual_semantics": {"composition_affinities": ["Solar Sentinel character", "metallic armour"]}}
        context = _safe_context(review, answers)
        affinities = context["visual_semantics"]["composition_affinities"]
        self.assertTrue(any("first-person" in item for item in affinities))
        self.assertTrue(any("traversal" in item for item in affinities))
        self.assertFalse(any("Solar Sentinel" in item or "armour" in item for item in affinities))

    def test_vague_reference_keeps_only_safe_supported_inference(self):
        answers = {"theme_reference_treatment": "Solar Sentinel inspired visual traits",
                   "background_inspiration": "Solar Sentinel inspired background"}
        review = {
            "reference_detected": True, "reference_name": "Solar Sentinel",
            "reference_type": "character", "reference_needs_clarification": True,
            "visual_semantics": {
                "motif_descriptors": ["interconnected radial patterns", "Solar Sentinel logo",
                                      "Solar Sentinel emblem", "Solar Sentinel character",
                                      "Solar Sentinel mask", "exact costume"],
                "material_descriptors": ["cardboard and paper"],
                "texture_descriptors": ["smooth glossy finish"],
                "lighting_descriptors": ["dramatic rim light"],
                "composition_affinities": ["symmetrical and balanced"],
                "scene_family": "abstract",
            },
            "semantic_confidence": {"motif_descriptors": .8, "material_descriptors": .7},
        }
        context = _safe_context(review, answers)
        self.assertTrue(context["reference_detected"])
        self.assertEqual((context["reference_name"], context["reference_type"]),
                         ("Solar Sentinel", "character"))
        self.assertFalse(context["reference_needs_clarification"])
        semantics = context["visual_semantics"]
        self.assertEqual(semantics["motif_descriptors"], ["interconnected radial patterns"])
        for key in ("material_descriptors", "texture_descriptors", "lighting_descriptors",
                    "composition_affinities"):
            self.assertEqual(semantics[key], [])
            self.assertNotIn(key, context["semantic_confidence"])
        self.assertEqual(semantics["scene_family"], "general")
        self.assertEqual(context["semantic_provenance"]["motif_descriptors"], "SEMANTIC_INFERENCE")
        self.assertEqual(context["semantic_confidence"]["motif_descriptors"], .8)

    def test_explicit_reference_visual_detail_survives(self):
        answers = {
            "theme_reference_treatment": "Use frosted glass textures, sweeping arc geometry and cool rim lighting",
            "visual_semantics": {"material_descriptors": ["frosted glass"],
                                 "texture_descriptors": ["frosted glass textures"],
                                 "motif_descriptors": ["sweeping arc geometry"],
                                 "lighting_descriptors": ["cool rim lighting"]},
        }
        context = _safe_context({"reference_detected": True, "reference_name": "Clockwork Falcon",
                                 "reference_type": "character",
                                 "visual_semantics": {"composition_affinities": ["symmetrical composition"]}}, answers)
        for key, expected in answers["visual_semantics"].items():
            self.assertEqual(context["visual_semantics"][key], expected)
            self.assertEqual(context["semantic_provenance"][key], "USER_EXPLICIT")
        self.assertEqual(context["visual_semantics"]["composition_affinities"], [])

    def test_background_constraints_remove_final_artwork_instructions(self):
        constraints = ["Clear readable programme text", "include title text",
                       "programme wording should be readable", "show the event title",
                       "add a logo", "background only", "no readable text", "no people",
                       "soft atmospheric depth", "gentle directional lighting"]
        context = _safe_context({"generation_constraints": constraints}, {})
        for item in constraints[:5]:
            self.assertNotIn(item, context["generation_constraints"])
        for item in ("background only", "no people", "no characters", "no logos",
                     "no readable text", "leave quiet areas for later overlays",
                     "soft atmospheric depth", "gentle directional lighting"):
            self.assertIn(item, context["generation_constraints"])

    def test_user_colour_relationship_is_preserved_from_visual_description(self):
        for primary, secondary, relationship in (
            ("green", "cream", "green dominant with cream supporting"),
            ("black", "white", "mostly black with white accents"),
            ("red", "blue", "red should be the main colour"),
            ("navy", "gold", "navy background with small gold details"),
        ):
            with self.subTest(relationship=relationship):
                brief = {
                    "primary_colour": primary,
                    "secondary_colour": secondary,
                    "creative_description": relationship,
                    "palette_mode": "dominant",
                }
                phrase, source = resolve_palette_relationship(brief)
                self.assertEqual((phrase, source), (relationship, "user_explicit"))
                brief.update(
                    palette_relationship=phrase,
                    palette_relationship_source=source,
                )
                self.assertEqual(
                    build_semantic_contract(brief)["semantic_provenance"][
                        "palette_relationship"
                    ],
                    "USER_EXPLICIT",
                )

        _, source = resolve_palette_relationship(
            {
                "primary_colour": "navy",
                "secondary_colour": "white",
                "creative_description": "A luminous architectural hall",
            }
        )
        self.assertEqual(source, "default")

        _, source = resolve_palette_relationship(
            {
                "primary_colour": "navy",
                "secondary_colour": "white",
                "creative_description": "The main hall has navy and white walls",
            }
        )
        self.assertEqual(source, "default")

    def test_default_palette_relationship_provenance(self):
        brief = {
            "primary_colour": "navy",
            "secondary_colour": "white",
            "palette_relationship": (
                "requested colours dominate with natural tonal variation"
            ),
            "palette_relationship_source": "default",
        }
        contract = build_semantic_contract(brief)
        self.assertEqual(
            contract["semantic_provenance"]["palette_relationship"],
            "SAFE_FALLBACK",
        )

        brief["palette_relationship_source"] = "user"
        contract = build_semantic_contract(brief)
        self.assertEqual(
            contract["semantic_provenance"]["palette_relationship"],
            "USER_EXPLICIT",
        )

    def test_accessibility_stays_in_final_composition(self):
        session = GrillMeSession(
            session_id="test",
            status="ready",
            answers=GrillMeAnswers(
                event_type="garden gathering",
                theme="flowers",
                age_group="family",
                primary_colour="green",
                secondary_colour="cream",
                creative_description="soft garden",
                title_preference="Welcome",
                event_date="2026-10-15",
                start_time="10:00",
                accessibility_preferences=["clear readable programme text"],
                programme=[ProgrammeItem(time="10:00", title="Arrival")],
            ),
        )

        with (
            patch(
                "app.services.grill_me_service.load_session",
                return_value=session,
            ),
            patch(
                "app.services.grill_me_service.evaluate_session",
                return_value=[],
            ),
            patch("app.services.grill_me_service._save"),
        ):
            _, background, final = freeze_for_generation(
                "test",
                "ABCDEF-123456",
            )

        self.assertNotIn("accessibility_preferences", background)
        self.assertNotIn("clear readable programme text", str(background))
        self.assertIn(
            "clear readable programme text",
            final["accessibility_preferences"],
        )

    def test_semantic_review_preserves_unfamiliar_descriptors(self):
        context = _safe_context(
            {
                "visual_semantics": {
                    "atmosphere_descriptors": [
                        "quietly luminous",
                        "meditative",
                        "weightless",
                    ],
                    "motif_structure_description": (
                        "irregular nested mineral clusters"
                    ),
                    "incompatible_treatments": [
                        "harsh metallic glare"
                    ],
                    "scene_family": "abstract",
                }
            },
            {
                "event_type": "unfamiliar gathering",
                "theme": "mineral forms",
                "creative_description": (
                    "quietly luminous, meditative and weightless"
                ),
            },
        )

        semantics = context["visual_semantics"]
        self.assertEqual(
            semantics["atmosphere_descriptors"],
            ["quietly luminous", "meditative", "weightless"],
        )
        self.assertEqual(
            semantics["motif_structure_description"],
            "irregular nested mineral clusters",
        )
        self.assertEqual(
            semantics["incompatible_treatments"],
            ["harsh metallic glare"],
        )
        self.assertEqual(semantics["scene_family"], "abstract")
        self.assertEqual(
            context["semantic_provenance"]["atmosphere_descriptors"],
            "SEMANTIC_INFERENCE",
        )
        self.assertNotIn(
            "atmosphere_descriptors",
            context["semantic_confidence"],
        )

    def test_generic_brief_gets_one_plain_language_background_question(self):
        context = _safe_context(
            {},
            {
                "event_type": "birthday party",
                "theme": "celebration",
                "creative_description": "fun and joyful",
            },
        )

        questions = context["clarification_questions"]
        self.assertEqual(
            questions[-1]["field"],
            "background_inspiration",
        )
        self.assertIn(
            "What would you like people to see",
            questions[-1]["question"],
        )

    def test_background_question_replaces_redundant_theme_question(self):
        context = _safe_context(
            {
                "clarification_questions": [
                    {
                        "field": "theme",
                        "question": "What should the theme look like?",
                    },
                ],
            },
            {
                "event_type": "birthday party",
                "theme": "City adventure",
                "creative_description": "fun and joyful",
            },
        )

        self.assertEqual(
            [
                item["field"]
                for item in context["clarification_questions"]
            ],
            ["background_inspiration"],
        )

    def test_original_superhero_brief_detects_reference_across_fields(self):
        answers = {
            "event_type": "Children birthday party",
            "theme": "Superheroes",
            "age_group": "Children aged 7 to 10",
            "primary_colour": "blue",
            "secondary_colour": "red",
            "creative_description": (
                "A kids party for a kid who is into superheroes, "
                "most especially spiderman."
            ),
            "background_inspiration": (
                "A spiderman inspired theme for a kid's birthday"
            ),
            "theme_reference_treatment": None,
        }

        context = _safe_context({}, answers)

        self.assertTrue(context["reference_detected"])
        self.assertEqual(context["reference_name"], "Spiderman")
        self.assertEqual(context["reference_type"], "character")
        self.assertTrue(context["reference_needs_clarification"])
        self.assertEqual(
            context["clarification_questions"][0]["field"],
            "theme_reference_treatment",
        )
        self.assertIn(
            "Spiderman",
            context["clarification_questions"][0]["question"],
        )

    def test_generic_inspired_language_does_not_become_character_reference(self):
        context = _safe_context(
            {},
            {
                "event_type": "garden gathering",
                "theme": "botanical",
                "creative_description": (
                    "A calm nature inspired garden celebration"
                ),
                "background_inspiration": (
                    "A nature inspired botanical pattern"
                ),
            },
        )

        self.assertFalse(context["reference_detected"])
        self.assertEqual(context["reference_type"], "none")

    def test_existing_reference_detection_is_not_overwritten(self):
        context = _safe_context(
            {
                "reference_detected": True,
                "reference_name": "Existing Reference",
                "reference_type": "franchise",
            },
            {
                "event_type": "birthday party",
                "theme": "Superheroes",
                "creative_description": (
                    "A superhero celebration"
                ),
                "background_inspiration": (
                    "A spiderman inspired theme"
                ),
                "theme_reference_treatment": (
                    "web geometry and energetic city movement"
                ),
            },
        )

        self.assertEqual(
            context["reference_name"],
            "Existing Reference",
        )
        self.assertEqual(
            context["reference_type"],
            "franchise",
        )
        self.assertFalse(
            context["reference_needs_clarification"],
        )

    def test_inspiration_is_used_by_direction_prompt_helpers(self):
        brief = {
            "background_inspiration": (
                "A moonlit botanical garden with oversized paper flowers"
            ),
            "asset_type": "none",
        }

        self.assertIn(
            "moonlit botanical garden",
            background_design_summary(brief),
        )
        self.assertIn(
            "moonlit botanical garden",
            background_positive_requirements(brief),
        )

    def test_spider_man_theme_uses_safe_web_visual_language(self):
        traits = reference_visual_language(
            {
                "theme": "Spider-Man birthday",
                "theme_reference_treatment": (
                    "web geometry and city energy"
                ),
            }
        )

        self.assertIn("web geometry", traits)
        self.assertNotIn("Spider-Man", traits)

    def test_asset_sessions_ask_for_placement_before_other_asset_choices(self):
        answers = GrillMeAnswers(
            event_type="birthday party",
            theme="garden celebration",
            age_group="children",
            primary_colour="blue",
            secondary_colour="red",
            creative_description="joyful and bright",
            background_inspiration="a storybook flower garden",
            event_date="2030-01-01",
            start_time="14:00",
            title_preference="Mia's party",
            venue="The garden hall",
            programme=[
                ProgrammeItem(
                    time="14:00",
                    title="Welcome",
                )
            ],
            asset_type="headshot",
        )

        session = GrillMeSession(
            session_id="test",
            status="reviewing",
            answers=answers,
        )

        fields = [
            question.field
            for question in evaluate_session(session)
        ]

        self.assertEqual(
            fields[:3],
            [
                "asset_placement",
                "headshot_shape",
                "rights_and_consent_confirmed",
            ],
        )


if __name__ == "__main__":
    unittest.main()
