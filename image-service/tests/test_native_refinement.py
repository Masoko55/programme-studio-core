"""Focused regression checks for generic native-background refinement."""

import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from app.services.background_policy import detect_human_signals, validate_palette
from app.services.composition_diversity import boilerplate_metrics, direction_similarity
from app.services.comfyui_client import _workflow_encoder_prompts


class NativeRefinementTests(unittest.TestCase):
    def test_tonal_derivatives_are_in_family(self):
        cases = (
            (("black", "white"), ((0, 0, 0), (130, 130, 130), (255, 255, 255))),
            (("pink", "white"), ((245, 183, 210), (252, 226, 236), (255, 255, 255))),
            (("royal blue", "red"), ((42, 72, 185), (21, 31, 87), (170, 35, 45))),
            (("#1856c2", "#d52536"), ((24, 86, 194), (17, 44, 105), (175, 29, 43))),
            (("rgb(24, 86, 194)", "rgb(213, 37, 54)"), ((24, 86, 194), (17, 44, 105), (175, 29, 43))),
        )
        for palette, tones in cases:
            with self.subTest(palette=palette):
                image = Image.new("RGB", (300, 300))
                draw = ImageDraw.Draw(image)
                for index, tone in enumerate(tones):
                    draw.rectangle((index * 100, 0, index * 100 + 99, 299), fill=tone)
                result = validate_palette(image, *palette, quality_result={})
                self.assertLessEqual(result["off_palette_ratio"], 0.05)

    def test_unrelated_large_colour_still_fails(self):
        for colour, palette in (("green", ("pink", "white")), ("yellow", ("royal blue", "red"))):
            with self.subTest(colour=colour):
                with self.assertRaisesRegex(ValueError, "not requested"):
                    validate_palette(Image.new("RGB", (80, 80), colour), *palette, quality_result={})

    def test_dominant_palette_allows_small_neutral_rendering_but_exclusive_does_not(self):
        image = Image.new("RGB", (100, 100), "pink")
        draw = ImageDraw.Draw(image)
        for x in range(0, 100, 10):
            for y in range(0, 100, 10):
                draw.rectangle((x, y, x + 3, y + 2), fill=(125, 125, 125))
        ordinary = validate_palette(image, "pink", "white", quality_result={})
        self.assertEqual(ordinary["palette_mode"], "dominant-requested-colour-families")
        with self.assertRaisesRegex(ValueError, "not requested"):
            validate_palette(image, "pink", "white", relationship="only these colours", quality_result={})

    def test_explicit_strict_black_white_rejects_chromatic_region(self):
        image = Image.new("RGB", (100, 100), "white")
        ImageDraw.Draw(image).rectangle((0, 50, 99, 99), fill="black")
        ImageDraw.Draw(image).rectangle((0, 0, 99, 7), fill="red")
        self.assertEqual(validate_palette(image, "black", "white", quality_result={})["palette_mode"], "black-white")
        with self.assertRaisesRegex(ValueError, "black-and-white"):
            validate_palette(image, "black", "white", mode="strict", quality_result={})

    def test_irregular_frame_is_detected_but_focal_structure_survives(self):
        frame = Image.new("RGB", (512, 704), "white")
        draw = ImageDraw.Draw(frame)
        for x in range(0, 512, 28):
            draw.ellipse((x, -20, x + 60, 95), fill="pink")
            draw.ellipse((x, 609, x + 60, 724), fill="pink")
        for y in range(0, 704, 30):
            draw.ellipse((-20, y, 75, y + 65), fill="pink")
            draw.ellipse((437, y, 532, y + 65), fill="pink")
        focal = Image.new("RGB", (512, 704), "white")
        draw = ImageDraw.Draw(focal)
        draw.ellipse((50, 50, 300, 330), fill="pink")
        draw.line((200, 330, 500, 690), fill="black", width=20)
        self.assertGreaterEqual(boilerplate_metrics(frame)["boilerplate_score"], 0.40)
        self.assertLess(boilerplate_metrics(focal)["boilerplate_score"], 0.40)

    def test_city_canyon_and_asymmetric_florals_are_not_boilerplate(self):
        city = Image.new("RGB", (512, 704), "white")
        draw = ImageDraw.Draw(city)
        for index, x in enumerate((0, 40, 85, 390, 440, 480)):
            draw.rectangle((x, 80 + index % 3 * 40, x + 32, 620), fill="#152c57")
            for y in range(180, 590, 40):
                draw.rectangle((x + 8, y, x + 20, y + 12), fill="white")
        draw.polygon(((150, 704), (256, 370), (360, 704)), fill="#445577")
        draw.line((256, 370, 80, 704), fill="black", width=5)
        draw.line((256, 370, 430, 704), fill="black", width=5)
        floral = Image.new("RGB", (512, 704), "white")
        draw = ImageDraw.Draw(floral)
        for index in range(8):
            draw.ellipse((index * 30 - 20, index * 55 - 30,
                          index * 30 + 130, index * 55 + 100), fill="pink")
        draw.line((140, 400, 380, 650), fill="green", width=15)
        for image in (city, floral):
            self.assertLess(boilerplate_metrics(image)["boilerplate_score"], 0.40)

    def test_duplicate_comparison_ignores_colour_but_respects_geometry(self):
        first = Image.new("RGB", (512, 704), "white")
        second = Image.new("RGB", (512, 704), "white")
        different = Image.new("RGB", (512, 704), "white")
        ImageDraw.Draw(first).ellipse((80, 90, 360, 440), fill="red")
        ImageDraw.Draw(second).ellipse((80, 90, 360, 440), fill="blue")
        ImageDraw.Draw(different).ellipse((260, 300, 490, 620), fill="red")
        self.assertGreaterEqual(direction_similarity(first, second), 0.88)
        self.assertLess(direction_similarity(first, different), 0.88)

    def test_uncompiled_fallback_uses_subject_not_fixed_layout(self):
        subject = "Sculptural flowers beneath warm chandeliers"
        clip_l, clip_g, t5, _ = _workflow_encoder_prompts("sd-3-5-medium", subject, None)
        for field in (clip_l, clip_g, t5):
            self.assertIn("flowers", field)
            self.assertNotIn("edge-weighted", field)
            self.assertNotIn("matte graphic forms", field)

    def test_flower_and_curtain_shapes_do_not_confirm_human(self):
        flower = Image.new("RGB", (320, 448), "white")
        draw = ImageDraw.Draw(flower)
        for offset in range(0, 360, 45):
            draw.pieslice((55, 95, 265, 305), offset, offset + 30, fill="pink")
        curtain = Image.new("RGB", (320, 448), "white")
        draw = ImageDraw.Draw(curtain)
        for x in range(0, 320, 32):
            draw.polygon(((x, 0), (x + 20, 0), (x + 48, 448), (x + 12, 448)), fill="black")
        for image in (flower, curtain):
            with self.subTest(shape=image.getpixel((160, 160))):
                self.assertFalse(detect_human_signals(image)["detected"])

    def test_isolated_face_pattern_is_diagnostic_not_confirmed(self):
        class EmptyHog:
            def setSVMDetector(self, value):
                pass

            def detectMultiScale(self, image, **kwargs):
                return [], []

        def cascade(detector, gray, size, neighbors):
            return [(100, 100, 55, 55)] if neighbors == 6 else []

        with patch("app.services.background_policy.cv2.HOGDescriptor", return_value=EmptyHog()), \
             patch("app.services.background_policy.cv2.HOGDescriptor_getDefaultPeopleDetector", return_value=None), \
             patch("app.services.background_policy._load_cascade", return_value=object()), \
             patch("app.services.background_policy._cascade_detect", side_effect=cascade):
            result = detect_human_signals(Image.new("RGB", (320, 448), "white"))
        self.assertFalse(result["detected"])
        self.assertTrue(any(item["source"] == "frontal_face" and not item["confirmed"]
                            for item in result["evidence"]))

    def test_credible_hog_person_remains_confirmed(self):
        class PersonHog:
            def setSVMDetector(self, value):
                pass

            def detectMultiScale(self, image, **kwargs):
                return [(80, 100, 64, 180)], [1.7]

        with patch("app.services.background_policy.cv2.HOGDescriptor", return_value=PersonHog()), \
             patch("app.services.background_policy.cv2.HOGDescriptor_getDefaultPeopleDetector", return_value=None), \
             patch("app.services.background_policy._load_cascade", return_value=None):
            result = detect_human_signals(Image.new("RGB", (320, 448), "white"))
        self.assertTrue(result["detected"])
        self.assertTrue(any(item["source"] == "hog" and item["confirmed"]
                            for item in result["evidence"]))

    def test_moderate_hog_pattern_needs_independent_corroboration(self):
        class PatternHog:
            def setSVMDetector(self, value):
                pass

            def detectMultiScale(self, image, **kwargs):
                return [(80, 100, 64, 180)], [1.0]

        with patch("app.services.background_policy.cv2.HOGDescriptor", return_value=PatternHog()), \
             patch("app.services.background_policy.cv2.HOGDescriptor_getDefaultPeopleDetector", return_value=None), \
             patch("app.services.background_policy._load_cascade", return_value=None):
            result = detect_human_signals(Image.new("RGB", (320, 448), "white"))
        self.assertFalse(result["detected"])
        self.assertTrue(any(item["source"] == "hog" and not item["confirmed"]
                            for item in result["evidence"]))


if __name__ == "__main__":
    unittest.main()
