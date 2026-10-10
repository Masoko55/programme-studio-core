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

DENSE_SAFE_REGION_REASON = (
    "Generated background failed visual quality validation: "
    "programme safe region is too visually dense "
    "(center structural edge density 0.123, "
    "outer structural edge density 0.119, "
    "center-to-outer ratio 1.03)."
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
                        self.assertIn("readable space", text)
                        self.assertIn("keep key motifs outside", text)
                        self.assertEqual(compiled.spec_sha256, spec.spec_sha256)
                        self.assertNotIn("previous attempt", text)

    def test_recovery_categories_keep_subject_and_one_targeted_correction(self):
        spec = build_candidate_spec(document_for(CASES[0]), "sdxl-1-0", "B")
        expected = {
            "PALETTE_OFF": "requested colour relationship",
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
            self.assertIn("coherent, varied visual forms", compiled.positive)

    def test_dense_safe_region_is_not_misclassified_as_weak_outer_structure(self):
        self.assertEqual(
            failure_category(DENSE_SAFE_REGION_REASON),
            "DENSE_SAFE_REGION",
        )

        spec = build_candidate_spec(
            document_for(CASES[0]),
            "sd-3-5-medium",
            "B",
        )

        compiled = compile_candidate_prompt(
            spec,
            2,
            DENSE_SAFE_REGION_REASON,
        )

        self.assertEqual(
            compiled.failure_category,
            "DENSE_SAFE_REGION",
        )
        self.assertIn(
            "Move detail from the calm overlay zones",
            compiled.positive,
        )
        self.assertNotIn(
            "Increase the requested motifs at the outer edges",
            compiled.positive,
        )

    def test_sd35_conditioning_preserves_subject_direction_and_layout(self):
        spec = build_candidate_spec(document_for(CASES[0]), "sd-3-5-medium", "B")
        compiled = compile_candidate_prompt(spec, 7, REASONS[4])
        self.assertIn("city skyline", compiled.clip_l.lower())
        self.assertIn("expressive", compiled.clip_g.lower())
        self.assertIn("city skyline", compiled.clip_g.lower())
        self.assertIn("spiderweb", compiled.clip_g.lower())
        self.assertIn("royal blue", compiled.clip_g.lower())
        self.assertIn("readable space", compiled.t5.lower())
        for anchor in ("city skyline", "spiderweb", "expressive", "royal blue", "red"):
            self.assertIn(anchor.lower(), compiled.t5.lower())
        self.assertNotIn("spider-man", compiled.t5.lower())
        self.assertLess(len(compiled.negative), 200)

    def test_missing_inspiration_uses_specific_creative_scene_first(self):
        document = document_for(CASES[0], "A")
        document["brief"]["background_inspiration"] = None
        document["brief"]["creative_description"] = (
            "Kid's party for a child into superheroes. "
            "Create a comic-book city skyline background with blue and red "
            "colour blocking, web geometry around the corners and side edges, "
            "and a calm central area for programme text."
        )
        document["directions"][0]["direction"]["positive_prompt"] = (
            "Use a balanced border-led composition. " * 20
            + "Make the visual subject an elegant abstract event background."
        )
        for engine in ("sdxl-1-0", "sd-3-5-medium"):
            spec = build_candidate_spec(document, engine, "A")
            compiled = compile_candidate_prompt(spec, 1)
            self.assertEqual(compiled.spec_sha256, spec.spec_sha256)
            if engine == "sdxl-1-0":
                self.assertTrue(compiled.positive.startswith("Background inspiration:"))
            else:
                self.assertTrue(compiled.positive.startswith("Direction artwork:"))
            self.assertIn("Background inspiration: Create a comic-book city skyline", compiled.positive[:450])
            if engine == "sd-3-5-medium":
                self.assertIn("Visual subject: Create a comic-book city skyline", compiled.clip_l)
                self.assertIn("Direction:", compiled.clip_g)
                self.assertIn("city skyline", compiled.t5)

    def test_missing_inspiration_fallback_is_theme_and_palette_independent(self):
        examples = (
            ("paper flower garden", "pink", "cream", "Illustrate layered paper flowers along the border"),
            ("black-tie celebration", "navy", "gold", "Depict ballroom pillars and geometric fan motifs"),
            ("coastal wedding", "teal", "sand", "Render dune grasses and shoreline curves"),
        )
        for theme, primary, secondary, scene in examples:
            document = document_for(CASES[0], "A")
            document["brief"].update({
                "theme": theme,
                "primary_colour": primary,
                "secondary_colour": secondary,
                "background_inspiration": None,
                "creative_description": f"An invitation background. {scene} with a calm central text area.",
            })
            document["directions"][0]["direction"]["positive_prompt"] = (
                "An elegant abstract event background with edge decoration."
            )
            for engine in ("sdxl-1-0", "sd-3-5-medium"):
                spec = build_candidate_spec(document, engine, "A")
                compiled = compile_candidate_prompt(spec, 1)
                self.assertIn(f"Background inspiration: {scene}", compiled.positive[:450])
                self.assertIn(primary, compiled.positive)
                self.assertIn(secondary, compiled.positive)
                if engine == "sd-3-5-medium":
                    self.assertIn(f"Visual subject: {scene}", compiled.clip_l)

    def test_sdxl_colour_contract_covers_rendered_materials_for_any_palette(self):
        for primary, secondary in (("royal blue", "red"), ("black", "white"), ("pink", "white")):
            document = document_for(CASES[0], "B")
            document["brief"]["primary_colour"] = primary
            document["brief"]["secondary_colour"] = secondary
            compiled = compile_candidate_prompt(
                build_candidate_spec(document, "sdxl-1-0", "B"), 1
            )
            for field in (compiled.clip_g, compiled.clip_l):
                self.assertTrue(field.startswith("Front-facing"))
                self.assertIn(primary, field)
                self.assertIn(secondary, field)
                self.assertIn("requested palette", field.lower())
                self.assertIn("spiderweb", field.lower())
                self.assertLess(field.lower().index("spiderweb"), field.lower().index("requested palette"))

    def test_sdxl_retries_keep_source_inspiration_in_both_encoders(self):
        document = document_for(CASES[0], "B")
        spec = build_candidate_spec(document, "sdxl-1-0", "B")
        for attempt in range(1, 10):
            reason = REASONS[(attempt - 1) % len(REASONS)] if attempt > 1 else ""
            compiled = compile_candidate_prompt(spec, attempt, reason)
            for field in (compiled.clip_g, compiled.clip_l):
                self.assertIn(spec.background_inspiration, field)
                self.assertIn(spec.theme_reference_treatment, compiled.positive)
                self.assertIn("spiderweb", field.lower())
                if attempt > 1:
                    self.assertIn("Correction:", field)
            self.assertEqual(compiled.spec_sha256, spec.spec_sha256)

    def test_architectural_frames_remain_background_artwork(self):
        document = document_for(CASES[1], "A")
        document["brief"]["background_inspiration"] = (
            "Black architectural frames and white geometric panels around the edges"
        )
        for engine in ("sdxl-1-0", "sd-3-5-medium"):
            spec = build_candidate_spec(document, engine, "A")
            compiled = compile_candidate_prompt(spec, 9, REASONS[6])
            for field in (compiled.clip_l, compiled.clip_g):
                self.assertIn("background", field.lower())
                self.assertIn("architectural frames", field.lower())
            self.assertNotIn("four to six", compiled.positive)
            if engine == "sdxl-1-0":
                self.assertIn("graphic background filling a flat page", compiled.clip_l)
                self.assertIn("No lettering or logos", compiled.clip_g)
                self.assertIn("poster mockup", compiled.negative)

    def test_named_character_in_backstory_does_not_become_image_subject(self):
        for name in ("Spider-Man", "Captain Comet"):
            document = document_for(CASES[0], "A")
            document["brief"].update({
                "theme": f"{name} city adventure",
                "background_inspiration": None,
                "theme_reference_treatment": None,
                "creative_description": (
                    f"A child loves {name}. "
                    "Create a comic-book city skyline with radial web geometry "
                    "at the border and an open centre for programme text."
                ),
            })
            document["directions"][0]["direction"]["positive_prompt"] = (
                "Radial web geometry around the edges, angular skyline silhouettes, "
                "bright geometric colour blocks."
            )
            for engine in ("sdxl-1-0", "sd-3-5-medium"):
                spec = build_candidate_spec(document, engine, "A")
                compiled = compile_candidate_prompt(spec, 1)
                self.assertIn(name, spec.creative_description)
                self.assertNotIn(name.lower(), compiled.positive.lower())
                self.assertIn("city skyline", compiled.positive.lower())
                self.assertIn("radial web geometry", compiled.positive.lower())
                if engine == "sd-3-5-medium":
                    for field in (compiled.clip_l, compiled.clip_g, compiled.t5):
                        self.assertNotIn(name.lower(), field.lower())

    def test_backstory_without_visual_instruction_uses_direction_design(self):
        document = document_for(CASES[0], "A")
        document["brief"].update({
            "theme": "Captain Comet celebration",
            "background_inspiration": None,
            "creative_description": "The guest of honour loves Captain Comet.",
        })
        document["directions"][0]["direction"]["positive_prompt"] = (
            "Angular skyline silhouettes, radial linework at the outer border, "
            "clear central text field."
        )
        for engine in ("sdxl-1-0", "sd-3-5-medium"):
            compiled = compile_candidate_prompt(build_candidate_spec(document, engine, "A"), 1)
            self.assertNotIn("Captain Comet", compiled.positive)
            self.assertIn("Angular skyline silhouettes", compiled.positive)

    def test_retry_limit_and_failure_classification(self):
        self.assertEqual(failure_category(REASONS[1]), "PALETTE_OFF")
        self.assertEqual(
            failure_category(DENSE_SAFE_REGION_REASON),
            "DENSE_SAFE_REGION",
        )
        with self.assertRaises(ValueError):
            compile_candidate_prompt(build_candidate_spec(document_for(CASES[0]), "sdxl-1-0", "B"), 10, "bad")


if __name__ == "__main__":
    unittest.main()
