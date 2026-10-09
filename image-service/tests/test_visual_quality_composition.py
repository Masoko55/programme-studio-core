import unittest

from PIL import Image, ImageDraw

from app.services.visual_quality import validate_visual_quality


class VisualQualityCompositionTests(unittest.TestCase):
    def test_rejects_dense_central_programme_region(self):
        image = Image.new(
            "RGB",
            (512, 704),
            "white",
        )

        draw = ImageDraw.Draw(
            image
        )

        #
        # Give the outer frame enough intentional structure to satisfy the
        # decorative-background requirement.
        #
        draw.rectangle(
            (8, 8, 503, 695),
            outline="black",
            width=18,
        )

        #
        # Deliberately fill the central title/programme region with a dense
        # high-contrast grid.
        #
        # This represents exactly the kind of technically valid but unusable
        # background that Programme Studio must reject.
        #
        for x in range(
            105,
            410,
            12,
        ):
            draw.line(
                (
                    x,
                    90,
                    x,
                    600,
                ),
                fill="black",
                width=5,
            )

        for y in range(
            90,
            600,
            12,
        ):
            draw.line(
                (
                    105,
                    y,
                    410,
                    y,
                ),
                fill="black",
                width=5,
            )

        with self.assertRaisesRegex(
            ValueError,
            "programme safe region",
        ):
            validate_visual_quality(
                image
            )

    def test_accepts_an_edge_led_frame_with_a_calm_centre(self):
        image = Image.new(
            "RGB",
            (512, 704),
            "white",
        )

        draw = ImageDraw.Draw(
            image
        )

        #
        # Strong decorative outer frame.
        #
        draw.rectangle(
            (8, 8, 503, 695),
            outline="black",
            width=18,
        )

        #
        # Add intentional detail to opposite outer corners while leaving the
        # title/programme region calm.
        #
        for offset in range(
            0,
            88,
            13,
        ):
            draw.arc(
                (
                    10 + offset,
                    10 + offset,
                    180,
                    180,
                ),
                180,
                270,
                fill="black",
                width=4,
            )

            draw.arc(
                (
                    332,
                    524 - offset,
                    502 - offset,
                    694 - offset,
                ),
                0,
                90,
                fill="black",
                width=4,
            )

        result = (
            validate_visual_quality(
                image
            )
        )

        self.assertTrue(
            result[
                "visual_quality_passed"
            ]
        )

        self.assertTrue(
            result[
                "center_safe_region_passed"
            ]
        )

        self.assertLess(
            result[
                "center_structural_edge_density"
            ],
            result[
                "outer_structural_edge_density"
            ],
        )

        self.assertLessEqual(
            result[
                "center_to_outer_edge_ratio"
            ],
            0.72,
        )


if __name__ == "__main__":
    unittest.main()