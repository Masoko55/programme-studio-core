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
    def test_user_colour_relationship_is_preserved_from_visual_description(self):
        for primary, secondary, relationship in (
            ("green", "cream", "green dominant with cream supporting"),
            ("black", "white", "mostly black with white accents"),
            ("red", "blue", "red should be the main colour"),
            ("navy", "gold", "navy background with small gold details"),
        ):
            with self.subTest(relationship=relationship):
                brief = {"primary_colour": primary, "secondary_colour": secondary,
                         "creative_description": relationship, "palette_mode": "dominant"}
                phrase, source = resolve_palette_relationship(brief)
                self.assertEqual((phrase, source), (relationship, "user_explicit"))
                brief.update(palette_relationship=phrase, palette_relationship_source=source)
                self.assertEqual(build_semantic_contract(brief)["semantic_provenance"]["palette_relationship"], "USER_EXPLICIT")
        _, source = resolve_palette_relationship({"primary_colour": "navy", "secondary_colour": "white",
                                                   "creative_description": "A luminous architectural hall"})
        self.assertEqual(source, "default")
        _, source = resolve_palette_relationship({"primary_colour": "navy", "secondary_colour": "white",
                                                   "creative_description": "The main hall has navy and white walls"})
        self.assertEqual(source, "default")

    def test_default_palette_relationship_provenance(self):
        brief = {"primary_colour": "navy", "secondary_colour": "white",
                 "palette_relationship": "requested colours dominate with natural tonal variation",
                 "palette_relationship_source": "default"}
        contract = build_semantic_contract(brief)
        self.assertEqual(contract["semantic_provenance"]["palette_relationship"], "SAFE_FALLBACK")
        brief["palette_relationship_source"] = "user"
        contract = build_semantic_contract(brief)
        self.assertEqual(contract["semantic_provenance"]["palette_relationship"], "USER_EXPLICIT")

    def test_accessibility_stays_in_final_composition(self):
        session = GrillMeSession(session_id="test", status="ready", answers=GrillMeAnswers(
            event_type="garden gathering", theme="flowers", age_group="family",
            primary_colour="green", secondary_colour="cream", creative_description="soft garden",
            title_preference="Welcome", event_date="2026-10-15", start_time="10:00",
            accessibility_preferences=["clear readable programme text"],
            programme=[ProgrammeItem(time="10:00", title="Arrival")],
        ))
        with patch("app.services.grill_me_service.load_session", return_value=session), \
             patch("app.services.grill_me_service.evaluate_session", return_value=[]), \
             patch("app.services.grill_me_service._save"):
            _, background, final = freeze_for_generation("test", "ABCDEF-123456")
        self.assertNotIn("accessibility_preferences", background)
        self.assertNotIn("clear readable programme text", str(background))
        self.assertIn("clear readable programme text", final["accessibility_preferences"])

    def test_semantic_review_preserves_unfamiliar_descriptors(self):
        context = _safe_context({"visual_semantics": {
            "atmosphere_descriptors": ["quietly luminous", "meditative", "weightless"],
            "motif_structure_description": "irregular nested mineral clusters",
            "incompatible_treatments": ["harsh metallic glare"],
            "scene_family": "abstract",
        }}, {"event_type": "unfamiliar gathering", "theme": "mineral forms",
             "creative_description": "quietly luminous, meditative and weightless"})
        semantics = context["visual_semantics"]
        self.assertEqual(semantics["atmosphere_descriptors"],
                         ["quietly luminous", "meditative", "weightless"])
        self.assertEqual(semantics["motif_structure_description"], "irregular nested mineral clusters")
        self.assertEqual(semantics["incompatible_treatments"], ["harsh metallic glare"])
        self.assertEqual(semantics["scene_family"], "abstract")
        self.assertEqual(context["semantic_provenance"]["atmosphere_descriptors"], "SEMANTIC_INFERENCE")
        self.assertNotIn("atmosphere_descriptors", context["semantic_confidence"])

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
        self.assertEqual(questions[-1]["field"], "background_inspiration")
        self.assertIn("What would you like people to see", questions[-1]["question"])

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
            [item["field"] for item in context["clarification_questions"]],
            ["background_inspiration"],
        )

    def test_inspiration_is_used_by_direction_prompt_helpers(self):
        brief = {
            "background_inspiration": (
                "A moonlit botanical garden with oversized paper flowers"
            ),
            "asset_type": "none",
        }

        self.assertIn("moonlit botanical garden", background_design_summary(brief))
        self.assertIn("moonlit botanical garden", background_positive_requirements(brief))

    def test_spider_man_theme_uses_safe_web_visual_language(self):
        traits = reference_visual_language({
            "theme": "Spider-Man birthday",
            "theme_reference_treatment": "web geometry and city energy",
        })

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
            programme=[ProgrammeItem(time="14:00", title="Welcome")],
            asset_type="headshot",
        )
        session = GrillMeSession(
            session_id="test",
            status="reviewing",
            answers=answers,
        )

        fields = [question.field for question in evaluate_session(session)]
        self.assertEqual(fields[:3], [
            "asset_placement",
            "headshot_shape",
            "rights_and_consent_confirmed",
        ])


if __name__ == "__main__":
    unittest.main()
