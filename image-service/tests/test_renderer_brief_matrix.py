"""Fixed and seeded generic briefs for static native-conditioning contracts."""
import random
import unittest

from app.services.candidate_spec import build_candidate_spec
from app.services.prompt_compiler import compile_candidate_prompt


RANDOM_SEED = 20261013
EVENTS = (
    "kids birthday", "adult birthday", "bridal shower", "baby shower",
    "wedding", "graduation", "black-tie gala", "awards ceremony",
    "corporate conference", "school event", "memorial service", "community celebration",
)
ENVIRONMENTS = (
    "city skyline", "garden paths", "ballroom columns", "abstract sculptural geometry",
    "performance stage", "stars and night sky", "botanical flower beds", "coastal dunes",
    "futuristic technology hall", "sports stadium", "layered fabric and light", "cultural pattern environment",
)
PALETTES = (("navy", "gold"), ("pink", "cream"), ("black", "white"),
            ("emerald", "white"), ("purple", "silver"), ("blue", "orange"),
            ("burgundy", "cream"), ("teal", "coral"))
REFERENCES = ("none", "fictional-character inspired", "era/style inspired",
              "sport inspired", "cultural/event-style inspired")
COMPOSITIONS = ("asymmetric", "central monument", "panoramic", "diagonal",
                "open perimeter", "edge-framed", "layered perspective", "radial",
                "horizontal", "vertical")


def fixture(index, event, environment, primary, secondary, mode, reference, composition):
    return {"reference_number": f"{index:06X}-123456", "brief": {
        "event_type": event, "theme": environment,
        "background_inspiration": f"Illustrated {environment} with coherent environmental depth",
        "theme_reference_treatment": f"Translate {reference} into abstract design traits without depicting people",
        "primary_colour": primary, "secondary_colour": secondary,
        "palette_mode": mode, "visual_anchors": [environment, "recognizable environmental forms"],
        "palette_relationship": f"{primary} is primary and {secondary} is secondary",
    }, "directions": [{"direction": {
        "direction_id": "A", "role": "Creative Direction for A4 Portrait Event Background",
        "positive_prompt": f"Illustrate {environment} as layered background scenery",
        "negative_prompt": "people, readable text, characters",
        "composition_strategy": f"{composition} arrangement of the requested environment",
        "focal_strategy": "off-centre environmental focal forms",
        "depth_strategy": "foreground middle and distant environmental layers",
        "border_or_perimeter_strategy": "open irregular perimeter following the subject",
        "title_safe_region_strategy": "quiet upper region integrated with scenery",
        "programme_safe_region_strategy": "quiet lower region integrated with scenery",
        "layout_guidance": {},
    }}]}


FIXED = (
    ("children birthday", "comic-book city architecture and web geometry", "royal blue", "red", "dominant"),
    ("adult thirtieth birthday", "elegant black-tie ballroom", "black", "white", "achromatic"),
    ("bridal celebration", "soft flowers and flowing fabric", "pink", "cream", "dominant"),
    ("corporate conference", "clean professional technology stage", "navy", "white", "dominant"),
    ("garden celebration", "botanical garden with organic foliage", "green", "cream", "dominant"),
    ("modern awards", "abstract sculptural geometry", "purple", "silver", "dominant"),
    ("single-colour design", "minimal graphic forms", "royal blue", "white", "dominant"),
    ("formal event", "black and white architectural forms", "black", "white", "strict"),
)


class RendererBriefMatrixTests(unittest.TestCase):
    def check_document(self, document):
        subject = document["brief"]["background_inspiration"]
        primary = document["brief"]["primary_colour"]
        secondary = document["brief"]["secondary_colour"]
        for engine in ("sdxl-1-0", "sd-3-5-medium", "flux-2"):
            spec = build_candidate_spec(document, engine, "A")
            compiled = compile_candidate_prompt(spec, 1)
            self.assertIn(subject, compiled.positive)
            self.assertIn(primary, compiled.positive)
            self.assertIn(secondary, compiled.positive)
            self.assertNotIn(spec.direction_role, compiled.positive)
            self.assertNotIn("title panel", compiled.positive.lower())
            if engine == "sd-3-5-medium":
                self.assertIn(subject, compiled.clip_l)
                self.assertIn(subject, compiled.clip_g)
                self.assertIn(subject, compiled.t5)
                self.assertIn("Event atmosphere:", compiled.clip_g)
                self.assertIn("no people or characters", compiled.t5)
                self.assertLess(compiled.t5.index("Visual subject"), compiled.t5.index("Event atmosphere"))
                self.assertLess(compiled.t5.index("Event atmosphere"), compiled.t5.index("Composition"))
                self.assertLess(compiled.t5.index("Composition"), compiled.t5.index("Palette:"))
            if engine == "sdxl-1-0":
                if {primary, secondary} == {"black", "white"}:
                    self.assertNotIn("avoiding", compiled.clip_l)
                else:
                    self.assertIn("avoiding", compiled.clip_l)
                    self.assertNotIn("avoiding black", compiled.clip_l)
                    self.assertNotIn("avoiding charcoal", compiled.clip_l)
                    self.assertNotIn("avoiding grey", compiled.clip_l)
                self.assertIn(primary, compiled.clip_l)
                self.assertIn(primary, compiled.clip_g)
            if {primary, secondary} == {"black", "white"}:
                self.assertIn("achromatic", compiled.positive)

    def test_eight_fixed_families(self):
        for index, (event, environment, primary, secondary, mode) in enumerate(FIXED):
            with self.subTest(event=event):
                self.check_document(fixture(index, event, environment, primary, secondary,
                                            mode, "none", "asymmetric"))

    def test_event_atmosphere_and_web_topology_do_not_leak(self):
        city = fixture(200, "children birthday", "comic-book city web geometry", "royal blue", "red",
                       "dominant", "none", "asymmetric")
        city["brief"]["creative_description"] = "A joyful high-energy celebration with playful comic-book city atmosphere"
        city["brief"]["symbolic_motifs"] = ["web-inspired geometry"]
        city["brief"]["audience_context"] = "Children aged 7 to 10"
        compiled = compile_candidate_prompt(build_candidate_spec(city, "sd-3-5-medium", "A"), 1)
        self.assertIn("joyful", compiled.clip_g)
        self.assertIn("child-friendly", compiled.t5)
        self.assertIn("curved cross-links", compiled.t5)
        garden = fixture(201, "garden celebration", "botanical garden", "green", "cream",
                         "dominant", "none", "asymmetric")
        other = compile_candidate_prompt(build_candidate_spec(garden, "sd-3-5-medium", "A"), 1)
        self.assertNotIn("web-like", other.t5)
        self.assertNotIn("city", other.t5)
        self.assertNotIn("child-friendly", other.t5)

    def test_twelve_seeded_randomized_briefs(self):
        rng = random.Random(RANDOM_SEED)
        environments = list(ENVIRONMENTS)
        rng.shuffle(environments)
        generated = []
        for index, event in enumerate(EVENTS):
            primary, secondary = rng.choice(PALETTES)
            mode = rng.choice(("dominant", "strict", "achromatic"))
            if mode == "achromatic":
                primary, secondary = "black", "white"
            document = fixture(index + 100, event, environments[index], primary, secondary,
                               mode, rng.choice(REFERENCES), rng.choice(COMPOSITIONS))
            generated.append(document)
            with self.subTest(index=index, event=event):
                self.check_document(document)
        self.assertEqual(len(generated), 12)
        self.assertEqual(len({item["brief"]["background_inspiration"] for item in generated}), 12)
