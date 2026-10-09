import unittest

from PIL import Image, ImageDraw

from app.config.settings import settings
from app.services.visual_quality import validate_visual_quality


class VisualQualityCompositionTests(unittest.TestCase):
    def test_records_dense_central_region_without_rejecting_sound_artwork(self):
        image = Image.new("RGB", (512, 704), "white")
        draw = ImageDraw.Draw(image)

        # Give the outer frame enough intentional structure to pass the
        # decorative-edge check, then fill the title/programme area with a
        # high-contrast grid that must be rejected.
        draw.rectangle((8, 8, 503, 695), outline="black", width=18)
        for x in range(105, 410, 12):
            draw.line((x, 90, x, 600), fill="black", width=5)
        for y in range(90, 600, 12):
            draw.line((105, y, 410, y), fill="black", width=5)

        result = validate_visual_quality(image)

        self.assertTrue(result["visual_quality_passed"])
        self.assertGreater(
            result["center_structural_edge_density"],
            settings.quality_center_max_edge_density,
        )

    def test_accepts_an_edge_led_frame_with_a_calm_centre(self):
        image = Image.new("RGB", (512, 704), "white")
        draw = ImageDraw.Draw(image)

        draw.rectangle((8, 8, 503, 695), outline="black", width=18)
        for offset in range(0, 88, 13):
            draw.arc((10 + offset, 10 + offset, 180, 180), 180, 270, fill="black", width=4)
            draw.arc((332, 524 - offset, 502 - offset, 694 - offset), 0, 90, fill="black", width=4)

        result = validate_visual_quality(image)

        self.assertTrue(result["visual_quality_passed"])
        self.assertLessEqual(
            result["center_structural_edge_density"],
            settings.quality_center_max_edge_density,
        )


if __name__ == "__main__":
    unittest.main()
