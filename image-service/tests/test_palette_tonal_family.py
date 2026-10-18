"""Synthetic colour blocks distinguish requested shading from third hues."""

import unittest

import numpy as np
from PIL import Image

from app.services.background_policy import NAMED_COLOURS, validate_palette


BLUE_SHADE = (32, 24, 48)
RED_SHADE = (56, 16, 40)


def blocks(*colours):
    pixels = np.zeros((176, 128, 3), dtype=np.uint8)
    for index, colour in enumerate(colours):
        start = index * 128 // len(colours)
        stop = (index + 1) * 128 // len(colours)
        pixels[:, start:stop] = NAMED_COLOURS.get(colour, colour) if isinstance(colour, str) else colour
    return Image.fromarray(pixels)


def assess(image, mode="dominant"):
    return validate_palette(image, "blue", "red", mode=mode,
                            relationship="requested colours dominate with natural tonal variation",
                            quality_result={})


class PaletteTonalFamilyTests(unittest.TestCase):
    def test_default_chromatic_palette_rejects_neutral_visual_majority(self):
        image = blocks(*(["blue"] * 39 + ["red"] * 6 + ["white"] * 55))
        with self.assertRaisesRegex(ValueError, "PALETTE_ROLE_DRIFT"):
            assess(image)

    def test_requested_colour_majority_still_passes(self):
        image = blocks(*(["blue"] * 60 + ["red"] * 15 + ["white"] * 25))
        result = assess(image)
        self.assertGreater(result["requested_chromatic_ratio"], result["structural_neutral_ratio"])

    def test_large_blue_and_red_shadows_count_as_requested_families(self):
        for image in (blocks("blue", "red", "navy", "burgundy"),
                      blocks("blue", "red", BLUE_SHADE, RED_SHADE),
                      blocks("blue", "red", BLUE_SHADE, "maroon")):
            with self.subTest(image=image.getpixel((16, 16))):
                result = assess(image)
                self.assertLess(result["off_palette_ratio"], .01)
                self.assertGreater(result["primary_colour_ratio"], .24)
                self.assertGreater(result["secondary_colour_ratio"], .24)

    def test_small_tonal_variation_remains_accepted(self):
        result = assess(blocks("blue", "red", "blue", "red", BLUE_SHADE, RED_SHADE))
        self.assertLess(result["off_palette_ratio"], .01)

    def test_coherent_unrelated_third_colours_still_fail(self):
        for third in ("green", "orange", "purple"):
            with self.subTest(third=third), self.assertRaisesRegex(ValueError, "not requested"):
                assess(blocks("blue", "red", third))

    def test_explicit_strict_palette_retains_stricter_gate(self):
        with self.assertRaisesRegex(ValueError, "not requested"):
            assess(blocks("blue", "red", BLUE_SHADE, RED_SHADE), mode="strict")


if __name__ == "__main__":
    unittest.main()
