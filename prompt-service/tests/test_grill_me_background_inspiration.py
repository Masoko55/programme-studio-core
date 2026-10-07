import unittest

from app.schemas.grill_me import GrillMeAnswers, GrillMeSession, ProgrammeItem
from app.services.creative_direction import (
    background_design_summary,
    background_positive_requirements,
)
from app.services.grill_me_semantic import _safe_context
from app.services.grill_me_service import evaluate_session


class GrillMeBackgroundInspirationTests(unittest.TestCase):
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

    def test_inspiration_is_used_by_direction_prompt_helpers(self):
        brief = {
            "background_inspiration": (
                "A moonlit botanical garden with oversized paper flowers"
            ),
            "asset_type": "none",
        }

        self.assertIn("moonlit botanical garden", background_design_summary(brief))
        self.assertIn("moonlit botanical garden", background_positive_requirements(brief))

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
