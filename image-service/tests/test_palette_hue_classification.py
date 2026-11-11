import unittest

from PIL import Image, ImageDraw

from app.services.background_policy import (
    NAMED_COLOURS, _chromatic_requested_family, _rgb_to_hsv,
    _centered_lower_upper_body, validate_palette,
)


class PaletteHueClassificationTests(unittest.TestCase):
    def test_local_warm_white_highlight_is_not_an_unrelated_colour(self):
        image = Image.new("RGB", (128, 176), NAMED_COLOURS["pink"])
        draw = ImageDraw.Draw(image)
        draw.rectangle((57, 0, 127, 175), fill="white")
        draw.rectangle((57, 0, 65, 175), fill=(230, 200, 180))
        result = validate_palette(image, "pink", "white", quality_result={})
        self.assertLessEqual(result["off_palette_ratio"], 0.05)
        self.assertGreater(result["near_palette_derivative_ratio"], 0.06)
        self.assertLess(result["secondary_colour_ratio"], 0.50)

    def test_large_warm_field_and_unrelated_blue_still_fail(self):
        for colour, width in (((230, 200, 180), 40), ((40, 100, 220), 27)):
            with self.subTest(colour=colour):
                image = Image.new("RGB", (128, 176), NAMED_COLOURS["pink"])
                draw = ImageDraw.Draw(image)
                draw.rectangle((57, 0, 127, 175), fill="white")
                draw.rectangle((57, 0, 56 + width, 175), fill=colour)
                with self.assertRaisesRegex(ValueError, "colours that were not requested"):
                    validate_palette(image, "pink", "white", quality_result={})

    def test_requested_tones_and_unrelated_fields_across_palette_modes(self):
        accepted = (
            ("black", "white", (95, 95, 95)),
            ("#7357A4", "#8ED081", (150, 135, 180)),
            ("green", "purple", (110, 160, 120)),
        )
        for primary, secondary, shade in accepted:
            with self.subTest(primary=primary, secondary=secondary):
                image = Image.new("RGB", (128, 176), NAMED_COLOURS.get(primary, (115, 87, 164)))
                draw = ImageDraw.Draw(image)
                draw.rectangle((90, 0, 127, 175), fill=NAMED_COLOURS.get(secondary, (142, 208, 129)))
                draw.rectangle((0, 0, 7, 175), fill=shade)
                self.assertLessEqual(validate_palette(image, primary, secondary, quality_result={})
                                     ["off_palette_ratio"], 0.05)

        rejected = (
            ("black", "white", (220, 45, 45)),
            ("green", "purple", (230, 130, 45)),
            ("pink", "white", (40, 100, 220)),
        )
        for primary, secondary, intruder in rejected:
            with self.subTest(primary=primary, secondary=secondary):
                image = Image.new("RGB", (128, 176), NAMED_COLOURS[primary])
                draw = ImageDraw.Draw(image)
                draw.rectangle((90, 0, 127, 175), fill=NAMED_COLOURS[secondary])
                draw.rectangle((0, 0, 27, 175), fill=intruder)
                with self.assertRaises(ValueError):
                    validate_palette(image, primary, secondary, quality_result={})

    def test_near_derivatives_cannot_replace_requested_primary(self):
        image = Image.new("RGB", (128, 176), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 3, 175), fill=NAMED_COLOURS["pink"])
        draw.rectangle((4, 0, 12, 175), fill=(230, 200, 180))
        with self.assertRaisesRegex(ValueError, "primary colour family"):
            validate_palette(image, "pink", "white", quality_result={})

    def test_central_lower_upper_body_counts_but_edge_ornament_does_not(self):
        self.assertTrue(_centered_lower_upper_body([(398, 962, 100, 82)], 896, 1280))
        self.assertFalse(_centered_lower_upper_body([(131, 1094, 116, 94)], 1024, 1408))

    def test_dark_requested_hues_are_not_neutralised_by_rgb_distance(self):
        targets = []
        for name in ("royal blue", "red"):
            rgb = NAMED_COLOURS[name]
            targets.append({"name": name, "rgb": rgb, "hue": _rgb_to_hsv(rgb)[0]})
        examples = {
            "bright royal blue": ((55, 80, 210), "royal blue"),
            "dark royal blue": ((24, 32, 82), "royal blue"),
            "navy": (NAMED_COLOURS["navy"], "royal blue"),
            "red": (NAMED_COLOURS["red"], "red"),
            "dark red": ((84, 17, 22), "red"),
            "maroon": (NAMED_COLOURS["maroon"], "red"),
            "burgundy": (NAMED_COLOURS["burgundy"], "red"),
            "grey": (NAMED_COLOURS["grey"], None),
            "charcoal": (NAMED_COLOURS["charcoal"], None),
            "teal": (NAMED_COLOURS["teal"], None),
            "cyan": (NAMED_COLOURS["cyan"], None),
        }
        for name, (rgb, expected) in examples.items():
            with self.subTest(name=name):
                self.assertEqual(_chromatic_requested_family(*_rgb_to_hsv(rgb), targets), expected)


if __name__ == "__main__":
    unittest.main()
