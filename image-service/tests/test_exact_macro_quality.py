import unittest

import numpy as np
from PIL import Image, ImageDraw

from app.services.visual_quality import validate_visual_quality


SIZE = (512, 704)


def skyline(forms=1, calm=False):
    image = Image.new("RGB", SIZE, (205, 220, 239))
    draw = ImageDraw.Draw(image)
    if forms == 1:
        draw.polygon([(210, 640), (210, 235), (255, 165), (300, 235), (300, 640)], fill=(25, 42, 82))
        draw.rectangle((0, 640, 511, 703), fill=(35, 50, 90))
    else:
        for x, top, width in ((20, 290, 80), (145, 180, 95), (285, 250, 80), (410, 355, 85)):
            draw.rectangle((x, top, x + width, 703), fill=(25, 42, 82))
    if calm:
        draw.rectangle((0, 0, 511, 155), fill=(205, 220, 239))
    return image


class ExactMacroQualityCases(unittest.TestCase):
    def assert_metrics(self, result):
        for key in ("macro_structure_score", "coarse_edge_density", "structural_edge_density", "highpass_mean"):
            self.assertIn(key, result)

    def test_single_skyscraper_silhouette(self):
        result = validate_visual_quality(skyline())
        self.assertTrue(result["visual_quality_passed"])
        self.assert_metrics(result)

    def test_four_large_forms(self):
        result = validate_visual_quality(skyline(4))
        self.assertTrue(result["visual_quality_passed"])
        self.assert_metrics(result)

    def test_silhouette_with_calm_region(self):
        result = validate_visual_quality(skyline(calm=True))
        self.assertTrue(result["visual_quality_passed"])
        self.assert_metrics(result)

    def test_near_blank_field(self):
        with self.assertRaises(ValueError):
            validate_visual_quality(Image.new("RGB", SIZE, (220, 222, 225)))

    def test_smooth_gradient(self):
        a = np.linspace(85, 185, SIZE[1], dtype=np.uint8)
        image = Image.fromarray(np.repeat(a[:, None], SIZE[0], axis=1), "L").convert("RGB")
        with self.assertRaises(ValueError):
            validate_visual_quality(image)

    def test_scanline_pattern(self):
        a = np.full((SIZE[1], SIZE[0], 3), 180, dtype=np.uint8)
        a[::2] = 45
        with self.assertRaisesRegex(ValueError, "raster|scanline|line"):
            validate_visual_quality(Image.fromarray(a))

    def test_fine_noise_without_macro_form(self):
        rng = np.random.default_rng(7)
        a = rng.integers(90, 170, size=(SIZE[1], SIZE[0], 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            validate_visual_quality(Image.fromarray(a))
