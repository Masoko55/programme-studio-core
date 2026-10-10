import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from app.services.candidate_spec import build_candidate_spec
from app.services.background_policy import parse_colour, validate_palette
from app.services.composition_diversity import boilerplate_score, direction_similarity
from app.services.comfyui_client import build_workflow
from app.services.prompt_compiler import compile_candidate_prompt


def synthetic_document(event, theme, palette):
    return {
        "reference_number": "ABCDEF-123456",
        "brief": {
            "event_type": event, "theme": theme, "background_inspiration": theme,
            "creative_description": f"Layered visual interpretation of {theme}",
            "palette": palette, "palette_relationship": "balanced colours",
            "visual_anchors": [theme, "floating sculptural forms"],
        },
        "directions": [{"direction": {
            "direction_id": letter, "role": strategy,
            "positive_prompt": f"{strategy} arrangement of {theme}",
            "composition_strategy": strategy,
            "title_safe_region_strategy": safe,
            "negative_prompt": "text, people", "layout_guidance": {},
        }} for letter, strategy, safe in (
            ("A", "sweeping diagonal", "upper left breathing room"),
            ("B", "layered circular focal point", "upper right breathing room"),
            ("C", "receding perspective", "top band breathing room"),
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
            result = validate_palette(image, None, None, palette=palette, relationship="balanced")
        self.assertEqual(result["off_palette_ratio"], 0)
        monochrome = Image.new("RGB", (300, 300), "#13A8B3")
        with patch("app.services.background_policy.validate_visual_quality", return_value={}):
            mono_result = validate_palette(monochrome, None, None,
                                           palette=[{"colour": "#13A8B3", "role": "primary"}])
        self.assertEqual(mono_result["off_palette_ratio"], 0)

    def test_unknown_briefs_palette_and_direction_strategies(self):
        briefs = (
            ("orbital harvest ritual", "glowing ceramic satellites", [{"colour": "#13A8B3", "role": "primary"}]),
            ("insect orchestra", "folded paper beetle instruments", [
                {"colour": "#101820", "role": "neutral"},
                {"colour": "#F2AA4C", "role": "accent"},
                {"colour": "#8178B7", "role": "secondary"},
            ]),
        )
        for event, theme, palette in briefs:
            document = synthetic_document(event, theme, palette)
            specs = [build_candidate_spec(document, "flux-2", letter) for letter in "ABC"]
            self.assertEqual(len({spec.visual_design["composition_strategy"] for spec in specs}), 3)
            self.assertEqual(len({spec.visual_design["title_safe_region_strategy"] for spec in specs}), 3)
            for spec in specs:
                self.assertEqual(spec.visual_design["event_context"], event)
                self.assertEqual(spec.visual_design["requested_palette"], palette)
                self.assertIn(theme, spec.visual_design["visual_anchors"])
                prompt = compile_candidate_prompt(spec, 1).positive
                self.assertIn(theme, prompt)
                for colour in palette:
                    self.assertIn(colour["colour"], prompt)

    def test_all_engines_use_native_latent_without_image_input(self):
        document = synthetic_document("unknown event", "silver rain sculpture", [{"colour": "#819CAD"}])
        for engine in ("flux-2", "sdxl-1-0", "sd-3-5-medium"):
            spec = build_candidate_spec(document, engine, "A")
            for attempt in (1, 9):
                compiled = compile_candidate_prompt(spec, attempt, "BOILERPLATE_COMPOSITION" if attempt > 1 else "")
                graph = build_workflow(engine, compiled.positive, compiled.negative, attempt, "test/native", compiled)
                classes = {node["class_type"] for node in graph.values()}
                self.assertFalse(classes & {"LoadImage", "VAEEncode", "SetLatentNoiseMask"})
                self.assertTrue(classes & {"EmptyLatentImage", "EmptyFlux2LatentImage", "EmptySD3LatentImage"})

    def test_geometry_checks_ignore_palette(self):
        def shape(background, foreground, kind):
            image = Image.new("RGB", (400, 600), background)
            draw = ImageDraw.Draw(image)
            if kind == "rectangle":
                draw.rectangle((70, 60, 330, 540), fill=foreground)
            else:
                draw.ellipse((70, 60, 330, 540), fill=foreground)
            return image

        rectangle = shape("#ff1155", "#00aaee", "rectangle")
        recoloured = shape("#225500", "#ffaa22", "rectangle")
        circle = shape("#ff1155", "#00aaee", "circle")
        self.assertGreater(boilerplate_score(rectangle), 0.42)
        self.assertGreater(direction_similarity(rectangle, recoloured), 0.88)
        self.assertLess(direction_similarity(rectangle, circle), 0.88)


if __name__ == "__main__":
    unittest.main()
