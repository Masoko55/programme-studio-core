"""Offline examples of text-overlay usability, independent of event theme."""

import unittest

import numpy as np
from PIL import Image, ImageDraw

from app.services.scene_density import assess_explicit_safe_regions, assess_scene_density


class SafeRegionUsabilityTests(unittest.TestCase):
    def assess(self, image, strategy="quiet top-left area", global_decision=None):
        density = assess_scene_density(image, "moderate")
        if global_decision:
            density["decision"] = global_decision
        return assess_explicit_safe_regions(density, {"title": strategy}, image)

    def test_sparse_low_contrast_lines_are_usable(self):
        image = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(image)
        for x in range(16, 256, 48):
            draw.line((x, 0, x + 50, 255), fill=(205, 205, 205), width=2)
        result = self.assess(image)
        self.assertEqual(result["decision"], "PASS")
        self.assertIsNone(result["regions"]["title"]["reason"])

    def test_dense_high_contrast_lines_interfere(self):
        image = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(image)
        for coordinate in range(0, 256, 8):
            draw.line((coordinate, 0, coordinate, 255), fill="black", width=2)
            draw.line((0, coordinate, 255, coordinate), fill="black", width=2)
        result = self.assess(image)
        self.assertEqual(result["decision"], "FAIL")
        self.assertEqual(result["regions"]["title"]["reason"], "dense_high_contrast_detail")

    def test_low_contrast_texture_is_usable(self):
        pixels = np.full((512, 512, 3), 210, dtype=np.uint8)
        for y in range(0, 256, 4):
            for x in range(0, 256, 4):
                pixels[y:y+4, x:x+4] = 200 if (x + y) % 8 else 220
        self.assertEqual(self.assess(Image.fromarray(pixels))["decision"], "PASS")

    def test_high_frequency_clutter_interferes(self):
        pixels = np.full((512, 512, 3), 255, dtype=np.uint8)
        for y in range(0, 256, 8):
            for x in range(0, 256, 8):
                pixels[y:y+8, x:x+8] = 0 if (x + y) % 16 else 255
        self.assertEqual(self.assess(Image.fromarray(pixels))["decision"], "FAIL")

    def test_single_large_focal_form_is_diagnostic(self):
        image = Image.new("RGB", (512, 512), "white")
        ImageDraw.Draw(image).rectangle((30, 30, 225, 225), fill="black")
        result = self.assess(image)
        self.assertEqual(result["decision"], "BORDERLINE")
        self.assertEqual(result["regions"]["title"]["reason"], "large_contrasting_form")
        self.assertGreaterEqual(result["regions"]["title"]["calm_tile_fraction"], .25)

    def test_decorative_border_with_open_title_area_is_usable(self):
        image = Image.new("RGB", (512, 512), "#eee8df")
        draw = ImageDraw.Draw(image)
        for y in range(0, 512, 36):
            draw.ellipse((0, y, 70, y + 58), fill="#345735")
            draw.ellipse((442, y, 511, y + 58), fill="#345735")
        self.assertNotEqual(self.assess(image, "quiet top-center title")["decision"], "FAIL")

    def test_broad_architectural_planes_leave_readable_area(self):
        image = Image.new("RGB", (512, 512), "#d9d5cb")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 165, 511), fill="#282a31")
        draw.rectangle((345, 0, 511, 511), fill="#373941")
        draw.rectangle((185, 0, 325, 511), fill="#bbb9b2")
        self.assertNotEqual(self.assess(image, "calm top-center")["decision"], "FAIL")

    def test_competing_detail_across_programme_area_fails(self):
        image = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(image)
        for x in range(128, 384, 10):
            for y in range(256, 512, 10):
                draw.rectangle((x, y, x + 6, y + 6), fill="black")
        self.assertEqual(self.assess(image, "quiet bottom-center programme")["decision"], "FAIL")

    def test_global_and_local_decisions_are_independent(self):
        calm = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(calm)
        for x in range(256, 512, 8):
            draw.line((x, 0, x, 511), fill="black", width=2)
        self.assertEqual(self.assess(calm, global_decision="DENSITY_DRIFT")["decision"], "PASS")

        busy = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(busy)
        for coordinate in range(0, 256, 8):
            draw.line((coordinate, 0, coordinate, 255), fill="black", width=2)
            draw.line((0, coordinate, 255, coordinate), fill="black", width=2)
        self.assertEqual(self.assess(busy, global_decision="PASS")["decision"], "FAIL")

    def test_geometry_outside_requested_area_does_not_interfere(self):
        image = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(image)
        for coordinate in range(264, 512, 8):
            draw.line((coordinate, 0, coordinate, 511), fill="black", width=2)
            draw.line((256, coordinate, 511, coordinate), fill="black", width=2)
        self.assertEqual(self.assess(image)["decision"], "PASS")

    def test_location_mapping_keeps_upper_right_and_top_center(self):
        image = Image.new("RGB", (512, 512), "white")
        density = assess_scene_density(image, "moderate")
        result = assess_explicit_safe_regions(density, {
            "title": "quiet upper-right", "programme": "calm top-center"}, image)
        self.assertEqual(result["regions"]["title"]["tile_indexes"], [2, 3, 6, 7])
        self.assertEqual(result["regions"]["programme"]["tile_indexes"], [1, 2, 5, 6])


if __name__ == "__main__":
    unittest.main()
