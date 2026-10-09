import unittest

from app.config.settings import settings
from app.services.image_execution import (
    _initial_engine_prompts,
    _strengthen_prompts,
)


class Sd35PromptBoundsTests(unittest.TestCase):
    def test_native_retries_keep_the_requested_visual_subject(self):
        base = "Comic city skyline with bold spiderweb arcs around the edges."
        reasons = (
            "Generated background failed visual quality validation: fine repetitive vertical raster-line structure covers too much of the page",
            "Generated background contains colours that were not requested (off-palette ratio 0.50)",
        )
        for engine in (settings.engine_2_id, settings.engine_3_id):
            for reason in reasons:
                with self.subTest(engine=engine, reason=reason):
                    positive, _ = _strengthen_prompts(
                        base, "people, text", reason, "royal blue", "red", 8, engine
                    )
                    self.assertIn("city skyline", positive.lower())
                    self.assertIn("spiderweb arcs", positive.lower())

    def test_native_profiles_share_subject_with_engine_specific_conditioning(self):
        base = (
            "Art deco ballroom pillars and geometric fan motifs, soft gold "
            "ornament on deep navy. A4 portrait decorative event background, "
            "controlled navy and gold palette. Keep decoration at the outer "
            "edges and lower corners, with a calm open central field."
        )

        sdxl_positive, sdxl_negative = _initial_engine_prompts(
            settings.engine_2_id, base, "unused", "navy", "gold"
        )
        sd35_positive, sd35_negative = _initial_engine_prompts(
            settings.engine_3_id, base, "unused", "navy", "gold"
        )

        for positive in (sdxl_positive, sd35_positive):
            self.assertIn("Art deco ballroom pillars", positive)
            self.assertIn("navy", positive)
            self.assertIn("gold", positive)
            self.assertIn("perimeter composition", positive)
            self.assertNotIn("A4 portrait", positive)

        self.assertIn("scanlines", sdxl_negative)
        self.assertEqual(sd35_negative, "")

    def test_sd35_quality_recovery_stays_within_encoder_budget(self):
        positive, negative = _strengthen_prompts(
            base_positive_prompt=(
                "A comic city skyline with architecture on every edge and "
                "a calm central title area. " * 40
            ),
            base_negative_prompt=(
                "text, people, scanlines, colour drift, noise, gradients, " * 40
            ),
            reason=(
                "Generated background failed visual quality validation: the "
                "outer background area contains too little useful decorative "
                "structure for a programme background."
            ),
            primary_colour="royal blue",
            secondary_colour="red",
            retry_number=8,
            engine_id=settings.engine_3_id,
        )

        self.assertLessEqual(len(positive), 900)
        self.assertLessEqual(len(negative), 650)
        self.assertIn("OUTER STRUCTURE RECOVERY", positive)


    def test_flux_quality_recovery_uses_a_compact_replacement_prompt(self):
        positive, negative = _strengthen_prompts(
            base_positive_prompt="A detailed floral background. " * 120,
            base_negative_prompt="text, people, noisy texture, " * 120,
            reason=(
                "Generated background failed visual quality validation: the "
                "outer background area contains too little useful decorative "
                "structure for a programme background."
            ),
            primary_colour="pink",
            secondary_colour="white",
            retry_number=3,
            engine_id=settings.engine_1_id,
        )

        self.assertLessEqual(len(positive), 1400)
        self.assertLessEqual(len(negative), 1000)
        self.assertIn("OUTER STRUCTURE RECOVERY", positive)


if __name__ == "__main__":
    unittest.main()
