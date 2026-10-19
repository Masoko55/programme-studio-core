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

    def test_dark_requested_pigments_keep_their_family(self):
        for primary, shade in (("royal blue", "navy"), ("red", "maroon"),
                               ("green", "forest green")):
            with self.subTest(primary=primary):
                result = validate_palette(blocks(primary, shade, "cream"), primary, "cream",
                                          mode="dominant", quality_result={})
                self.assertLess(result["off_palette_ratio"], .01)
                self.assertGreater(result["primary_ratio"], .6)

    def test_pale_warm_cream_is_credited_before_structural_neutral(self):
        result = validate_palette(blocks("green", (248, 245, 234)), "green", "cream",
                                  mode="dominant", quality_result={})
        self.assertGreater(result["secondary_ratio"], .45)
        self.assertLess(result["structural_neutral_ratio"], .05)

    def test_unrelated_hues_and_strict_palettes_remain_guarded(self):
        for primary, secondary, third in (("green", "cream", "purple"),
                                          ("royal blue", "red", "green")):
            with self.subTest(third=third), self.assertRaises(ValueError) as raised:
                validate_palette(blocks(primary, secondary, third), primary, secondary,
                                 mode="dominant", quality_result={})
            self.assertGreater(raised.exception.palette_metrics["largest_unrequested_region_ratio"], .035)
        with self.assertRaises(ValueError):
            validate_palette(blocks("royal blue", "red", "green"), "royal blue", "red",
                             mode="strict", quality_result={})

    def test_navy_white_architectural_palette_remains_valid(self):
        result = validate_palette(blocks("navy", "white"), "navy", "white",
                                  mode="dominant", quality_result={})
        self.assertLess(result["off_palette_ratio"], .01)
        self.assertGreater(result["primary_ratio"], .4)
        self.assertGreater(result["secondary_ratio"], .4)

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
