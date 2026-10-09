import unittest

from app.services.candidate_spec import build_candidate_spec
from app.services.prompt_compiler import compile_candidate_prompt, failure_category


CASES = (
    (
        "children's birthday", "Spider-Man city adventure",
        "A stylised comic-book city skyline with bold angular architecture",
        "Spiderweb-inspired geometry and dynamic motion shapes; no character",
        "expressive", "royal blue", "red", "city skyline", "spiderweb",
    ),
    (
        "30th birthday", "black-tie celebration",
        "A formal ballroom with chandeliers and black-tie ornament",
        "Architectural ballroom details in crisp monochrome",
        "elegant", "black", "white", "ballroom", "black-tie",
    ),
    (
        "bridal shower", "paper flower garden",
        "Large layered paper flowers around the border",
        "Floral paper craft with visible petals and leaves",
        "contemporary", "pink", "white", "paper flowers", "floral",
    ),
)

REASONS = (
    "",
    "Generated background contains colours that were not requested (off-palette ratio 0.5)",
    "Generated background contains a human silhouette",
    "Generated background contains readable text",
    "Generated background became a smooth gradient colour wash",
    "Outer background area contains too little useful decorative structure",
    "Generated background has repetitive raster lines",
    "Generated background failed due to runtime timeout",
    "Generated background has missing secondary colour",
)


def document_for(case, direction_id="B"):
    event, theme, inspiration, treatment, role, primary, secondary, *_ = case
    return {
        "reference_number": "ABCDEF-123456",
        "brief": {
            "event_type": event,
            "theme": theme,
            "creative_description": f"A distinctive {event} background",
            "background_inspiration": inspiration,
            "theme_reference_treatment": treatment,
            "primary_colour": primary,
            "secondary_colour": secondary,
        },
        "directions": [{
            "direction": {
                "direction_id": direction_id,
                "role": role,
                "positive_prompt": f"Render {inspiration} with {treatment}",
                "negative_prompt": "people, readable text",
                "layout_guidance": {
                    "title_zone": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.12},
                    "programme_zone": {"x": 0.1, "y": 0.45, "width": 0.8, "height": 0.4},
                },
            }
        }],
    }


class CandidateSpecCompilerTests(unittest.TestCase):
    def test_spec_is_structured_and_deeply_immutable(self):
        document = document_for(CASES[0])
        spec = build_candidate_spec(document, "sdxl-1-0", "B")
        digest = spec.spec_sha256
        self.assertEqual(spec.background_inspiration, CASES[0][2])
        self.assertEqual(spec.theme_reference_treatment, CASES[0][3])
        self.assertEqual(spec.direction_role, "expressive")
        self.assertEqual((spec.primary_colour, spec.secondary_colour), ("royal blue", "red"))
        self.assertIn("programme_zone", spec.layout_guidance)
        spec.layout_guidance["title_zone"]["x"] = 0.9
        self.assertEqual(spec.layout_guidance["title_zone"]["x"], 0.1)
        self.assertEqual(spec.spec_sha256, digest)
        document["brief"]["background_inspiration"] = "a different scene"
        self.assertEqual(spec.spec_sha256, digest)

    def test_all_native_attempts_retain_each_brief_and_direction(self):
        for case in CASES:
            for engine in ("sdxl-1-0", "sd-3-5-medium"):
                spec = build_candidate_spec(document_for(case), engine, "B")
                for attempt in range(1, 10):
                    reason = REASONS[(attempt - 1) % len(REASONS)] if attempt > 1 else ""
                    compiled = compile_candidate_prompt(spec, attempt, reason)
                    with self.subTest(case=case[1], engine=engine, attempt=attempt):
                        text = compiled.positive.lower()
                        for anchor in (case[7], case[8], case[4], case[5], case[6]):
                            self.assertIn(anchor.lower(), text)
                        self.assertIn("title zone", text)
                        self.assertIn("programme zone", text)
                        self.assertEqual(compiled.spec_sha256, spec.spec_sha256)
                        self.assertNotIn("previous attempt", text)

    def test_recovery_categories_keep_subject_and_one_targeted_correction(self):
        spec = build_candidate_spec(document_for(CASES[0]), "sdxl-1-0", "B")
        expected = {
            "PALETTE_OFF": "requested colour families",
            "HUMAN": "Remove people",
            "TEXT": "Remove readable text",
            "GRADIENT": "discrete matte forms",
            "WEAK_OUTER_STRUCTURE": "outer edges",
            "RASTER": "scanlines",
            "RUNTIME": "fewer, larger forms",
            "SECONDARY_MISSING": "secondary colour",
        }
        for reason in REASONS[1:]:
            compiled = compile_candidate_prompt(spec, 9, reason)
            self.assertIn(CASES[0][2], compiled.positive)
            self.assertIn(CASES[0][3], compiled.positive)
            self.assertIn(expected[compiled.failure_category], compiled.positive)
            self.assertEqual(compiled.retry_stage, "rescue")
            self.assertIn("four to six large intentional forms", compiled.positive)

    def test_sd35_conditioning_preserves_subject_direction_and_layout(self):
        spec = build_candidate_spec(document_for(CASES[0]), "sd-3-5-medium", "B")
        compiled = compile_candidate_prompt(spec, 7, REASONS[4])
        self.assertIn("city skyline", compiled.clip_l.lower())
        self.assertIn("expressive", compiled.clip_g.lower())
        self.assertIn("city skyline", compiled.clip_g.lower())
        self.assertIn("spiderweb", compiled.clip_g.lower())
        self.assertIn("royal blue", compiled.clip_g.lower())
        self.assertIn("title zone", compiled.clip_g.lower())
        for anchor in ("birthday", "Spider-Man", "city skyline", "spiderweb", "expressive", "royal blue", "red"):
            self.assertIn(anchor.lower(), compiled.t5.lower())
        self.assertLess(len(compiled.negative), 140)

    def test_retry_limit_and_failure_classification(self):
        self.assertEqual(failure_category(REASONS[1]), "PALETTE_OFF")
        with self.assertRaises(ValueError):
            compile_candidate_prompt(build_candidate_spec(document_for(CASES[0]), "sdxl-1-0", "B"), 10, "bad")


if __name__ == "__main__":
    unittest.main()
