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
    def test_distributed_requested_blend_is_not_a_declared_colour(self):
        pixels = np.empty((176, 128, 3), dtype=np.uint8)
        pixels[:, :64] = NAMED_COLOURS["green"]
        pixels[:, 64:] = NAMED_COLOURS["cream"]
        for x in range(4, 124, 12):
            pixels[16:128, x:x + 2] = (162, 195, 146)
        result = validate_palette(Image.fromarray(pixels), "green", "cream",
                                  quality_result={})
        self.assertGreater(result["requested_palette_blend_ratio"], .05)
        self.assertLess(result["off_palette_ratio"], .01)
        self.assertLess(result["primary_ratio"] + result["secondary_ratio"], .96)

    def test_broad_pale_secondary_can_support_chromatic_primary(self):
        image = blocks(*(["green"] * 40 + ["cream"] * 60))
        result = validate_palette(image, "green", "cream", mode="dominant",
                                  relationship="green dominant with cream supporting",
                                  relationship_source="user_explicit", quality_result={})
        self.assertTrue(result["dominance_relationship_passed"])
        self.assertLess(result["primary_ratio"], result["secondary_ratio"])

    def test_tiny_primary_and_similarly_chromatic_reverse_area_still_fail(self):
        for primary, secondary, primary_share in (("green", "cream", 10), ("blue", "red", 40)):
            with self.subTest(primary=primary), self.assertRaisesRegex(ValueError, "PALETTE_ROLE_DRIFT"):
                validate_palette(blocks(*([primary] * primary_share + [secondary] * (100 - primary_share))),
                                 primary, secondary, relationship=f"{primary} dominant with {secondary} supporting",
                                 relationship_source="user_explicit", quality_result={})

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

    def test_muted_requested_pigment_keeps_family_without_admitting_third_hue(self):
        olive_shade = (112, 128, 64)
        result = validate_palette(blocks("green", olive_shade, "cream"), "green", "cream",
                                  mode="dominant", quality_result={})
        self.assertLess(result["off_palette_ratio"], .01)
        self.assertGreater(result["primary_ratio"], .6)
        with self.assertRaises(ValueError) as invalid:
            validate_palette(blocks("green", "orange", "cream"), "green", "cream",
                             mode="dominant", quality_result={})
        self.assertGreater(invalid.exception.palette_metrics["largest_unrequested_region_ratio"], .035)

    def test_darker_muted_pigment_direction_does_not_admit_warm_third_colour(self):
        muted_shade = (116, 124, 75)
        result = validate_palette(blocks("green", muted_shade, "cream"), "green", "cream",
                                  mode="dominant", quality_result={})
        self.assertLess(result["off_palette_ratio"], .01)
        for third in ("brown", "gold", "orange", "peach"):
            with self.subTest(third=third), self.assertRaises(ValueError):
                validate_palette(blocks("green", third, "cream"), "green", "cream",
                                 mode="dominant", quality_result={})

    def test_lit_pigment_and_shaded_light_pigment_follow_requested_chromatic_direction(self):
        image = blocks("green", (144, 160, 96), (160, 160, 112), "cream")
        result = validate_palette(image, "green", "cream", mode="dominant", quality_result={})
        self.assertLess(result["off_palette_ratio"], .01)
        self.assertGreater(result["primary_ratio"], .4)
        self.assertGreater(result["secondary_ratio"], .4)

        for third in ("orange", "peach", "brown", "gold", "purple", "lavender", "cyan", "blue", "red"):
            with self.subTest(third=third), self.assertRaises(ValueError) as rejected:
                validate_palette(blocks("green", "cream", third), "green", "cream",
                                 mode="dominant", quality_result={})
            self.assertGreater(rejected.exception.palette_metrics["largest_unrequested_region_ratio"], .035)

        deeper_cream_shadow = (176, 176, 112)
        deeper = validate_palette(blocks("green", "green", deeper_cream_shadow, "cream"),
                                  "green", "cream", mode="dominant", quality_result={})
        self.assertLess(deeper["off_palette_ratio"], .01)

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
