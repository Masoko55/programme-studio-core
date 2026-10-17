import unittest

from app.services.creative_direction import palette_negative, positive_palette_description


class PalettePromptModeTests(unittest.TestCase):
    def test_dominant_allows_bounded_neutral_detail(self):
        brief = {"primary_colour": "royal blue", "secondary_colour": "red", "palette_mode": "dominant"}
        self.assertIn("bounded achromatic shadows", positive_palette_description(brief))
        self.assertIn("must not become a background colour field", positive_palette_description(brief))
        self.assertIn("Third chromatic", positive_palette_description(brief))
        self.assertNotIn("any neutral", palette_negative(brief))

    def test_strict_excludes_other_colour_families(self):
        brief = {"primary_colour": "royal blue", "secondary_colour": "red", "palette_mode": "strict"}
        self.assertIn("Only the requested colour families", positive_palette_description(brief))
        self.assertIn("any neutral", palette_negative(brief))

    def test_black_white_remains_achromatic(self):
        brief = {"primary_colour": "black", "secondary_colour": "white", "palette_mode": "achromatic"}
        self.assertIn("strictly achromatic", positive_palette_description(brief))
