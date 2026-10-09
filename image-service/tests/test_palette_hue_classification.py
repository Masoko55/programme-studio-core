import unittest

from app.services.background_policy import (
    NAMED_COLOURS, _chromatic_requested_family, _rgb_to_hsv,
    _centered_lower_upper_body,
)


class PaletteHueClassificationTests(unittest.TestCase):
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
