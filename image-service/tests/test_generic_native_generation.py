import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from app.services.candidate_spec import build_candidate_spec
from app.services.background_policy import parse_colour, validate_palette
from app.services.composition_diversity import (
    boilerplate_metrics,
    boilerplate_score,
    direction_similarity,
)
from app.services.comfyui_client import build_workflow
from app.services.prompt_compiler import compile_candidate_prompt


def synthetic_document(event, theme, palette):
    return {
        "reference_number": "ABCDEF-123456",
        "brief": {
            "event_type": event,
            "theme": theme,
            "background_inspiration": theme,
            "creative_description": f"Layered visual interpretation of {theme}",
            "palette": palette,
            "palette_relationship": "balanced colours",
            "visual_anchors": [theme, "floating sculptural forms"],
            "material_language": "layered paper and translucent resin",
            "texture_language": "fine tactile detail",
            "lighting_language": "soft directional light",
        },
        "directions": [{
            "direction": {
                "direction_id": letter,
                "role": strategy,
                "positive_prompt": f"{strategy} arrangement of {theme}",
                "composition_strategy": strategy,
                "focal_strategy": focal,
                "depth_strategy": depth,
                "border_or_perimeter_strategy": perimeter,
                "title_safe_region_strategy": safe,
                "programme_safe_region_strategy": "preserve a separate quiet reading area",
                "negative_prompt": "text, people",
                "layout_guidance": {},
            }
        } for letter, strategy, focal, depth, perimeter, safe in (
            (
                "A",
                "sweeping diagonal",
                "off-centre visual emphasis",
                "layered near-to-far depth",
                "open asymmetrical perimeter",
                "upper left breathing room",
            ),
            (
                "B",
                "layered circular focal point",
                "distributed ring emphasis",
                "shallow overlapping planes",
                "broken radial perimeter",
                "upper right breathing room",
            ),
            (
                "C",
                "receding perspective",
                "low horizon emphasis",
                "deep atmospheric recession",
                "minimal framing",
                "top band breathing room",
            ),
        )],
    }


class GenericNativeGenerationTests(unittest.TestCase):
    def test_rgb_hex_css_and_multicolour_palette(self):
        self.assertEqual(parse_colour("rgb(19, 168, 179)"), (19, 168, 179))
        self.assertEqual(parse_colour("#13A8B3"), (19, 168, 179))
        self.assertEqual(parse_colour("rebeccapurple"), (102, 51, 153))
        palette = [
            {"colour": "#13A8B3", "role": "primary"},
            {"colour": "#8178B7", "role": "secondary"},
            {"colour": "#F2AA4C", "role": "accent"},
        ]
        image = Image.new("RGB", (300, 300), "#13A8B3")
        image.paste("#8178B7", (100, 0, 200, 300))
        image.paste("#F2AA4C", (200, 0, 300, 300))
        with patch("app.services.background_policy.validate_visual_quality", return_value={}):
            result = validate_palette(
                image,
                None,
                None,
                palette=palette,
                relationship="balanced",
            )
        self.assertEqual(result["off_palette_ratio"], 0)

        monochrome = Image.new("RGB", (300, 300), "#13A8B3")
        with patch("app.services.background_policy.validate_visual_quality", return_value={}):
            mono_result = validate_palette(
                monochrome,
                None,
                None,
                palette=[{"colour": "#13A8B3", "role": "primary"}],
            )
        self.assertEqual(mono_result["off_palette_ratio"], 0)

    def test_unknown_briefs_palette_and_direction_strategies(self):
        briefs = (
            (
                "orbital harvest ritual",
                "glowing ceramic satellites",
                [{"colour": "#13A8B3", "role": "primary"}],
            ),
            (
                "insect orchestra",
                "folded paper beetle instruments",
                [
                    {"colour": "#101820", "role": "neutral"},
                    {"colour": "#F2AA4C", "role": "accent"},
                    {"colour": "#8178B7", "role": "secondary"},
                ],
            ),
        )
        for event, theme, palette in briefs:
            document = synthetic_document(event, theme, palette)
            specs = [build_candidate_spec(document, "flux-2", letter) for letter in "ABC"]
            self.assertEqual(
                len({spec.visual_design["composition_strategy"] for spec in specs}),
                3,
            )
            self.assertEqual(
                len({spec.visual_design["title_safe_region_strategy"] for spec in specs}),
                3,
            )
            for spec in specs:
                self.assertEqual(spec.visual_design["event_context"], event)
                self.assertEqual(spec.visual_design["requested_palette"], palette)
                self.assertIn(theme, spec.visual_design["visual_anchors"])
                prompt = compile_candidate_prompt(spec, 1).positive
                self.assertIn(theme, prompt)
                self.assertIn("layered paper", prompt)
                self.assertIn("soft directional light", prompt)
                for colour in palette:
                    self.assertIn(colour["colour"], prompt)

    def test_all_engines_use_native_latent_without_image_input(self):
        document = synthetic_document(
            "unknown event",
            "silver rain sculpture",
            [{"colour": "#819CAD"}],
        )
        for engine in ("flux-2", "sdxl-1-0", "sd-3-5-medium"):
            spec = build_candidate_spec(document, engine, "A")
            for attempt in (1, 9):
                compiled = compile_candidate_prompt(
                    spec,
                    attempt,
                    "BOILERPLATE_COMPOSITION" if attempt > 1 else "",
                )
                graph = build_workflow(
                    engine,
                    compiled.positive,
                    compiled.negative,
                    attempt,
                    "test/native",
                    compiled,
                )
                classes = {node["class_type"] for node in graph.values()}
                self.assertFalse(classes & {"LoadImage", "VAEEncode", "SetLatentNoiseMask"})
                self.assertTrue(classes & {
                    "EmptyLatentImage",
                    "EmptyFlux2LatentImage",
                    "EmptySD3LatentImage",
                })

    def test_boilerplate_detection_is_colour_independent_and_handles_multiple_slabs(self):
        def slabs(background, first, second):
            image = Image.new("RGB", (400, 600), background)
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 80, 600), fill=first)
            draw.rectangle((80, 40, 320, 560), fill=second)
            draw.rectangle((320, 0, 400, 600), fill=first)
            return image

        first_palette = slabs("#003366", "#C00030", "#1640A0")
        second_palette = slabs("#F3E8C9", "#355E3B", "#D6B85A")
        first_metrics = boilerplate_metrics(first_palette)
        second_metrics = boilerplate_metrics(second_palette)
        self.assertGreater(boilerplate_score(first_palette), 0.40)
        self.assertGreater(boilerplate_score(second_palette), 0.40)
        self.assertGreater(first_metrics["large_flat_rectangle_count"], 1)
        self.assertGreater(second_metrics["large_flat_rectangle_count"], 1)

    def test_geometry_checks_ignore_palette_but_distinguish_layout(self):
        def shape(background, foreground, kind):
            image = Image.new("RGB", (400, 600), background)
            draw = ImageDraw.Draw(image)
            if kind == "rectangle":
                draw.rectangle((70, 60, 330, 540), fill=foreground)
            elif kind == "circle":
                draw.ellipse((70, 60, 330, 540), fill=foreground)
            else:
                draw.polygon([(30, 500), (200, 80), (370, 500)], fill=foreground)
            return image

        rectangle = shape("#ff1155", "#00aaee", "rectangle")
        recoloured = shape("#225500", "#ffaa22", "rectangle")
        circle = shape("#ff1155", "#00aaee", "circle")
        triangle = shape("#ff1155", "#00aaee", "triangle")
        self.assertGreater(direction_similarity(rectangle, recoloured), 0.88)
        self.assertLess(direction_similarity(rectangle, circle), 0.88)
        self.assertLess(direction_similarity(rectangle, triangle), 0.88)

    def test_compiled_prompt_does_not_force_flat_graphic_medium(self):
        document = synthetic_document(
            "custom ceremony",
            "misty glass garden with suspended lantern forms",
            [
                {"colour": "#244A5A", "role": "primary"},
                {"colour": "#F4D8A8", "role": "accent"},
            ],
        )
        for engine in ("flux-2", "sdxl-1-0", "sd-3-5-medium"):
            compiled = compile_candidate_prompt(
                build_candidate_spec(document, engine, "B"),
                1,
            )
            text = " ".join(filter(None, (
                compiled.positive,
                compiled.clip_l,
                compiled.clip_g,
                compiled.t5,
            ))).lower()
            self.assertNotIn("matte forms", text)
            self.assertNotIn("template boxes", text.split("no people")[0] if "no people" in text else "")
            self.assertIn("misty glass garden", text)


if __name__ == "__main__":
    unittest.main()