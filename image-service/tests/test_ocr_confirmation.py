import unittest

from unittest.mock import patch

from PIL import Image

from app.services.comfyui_client import (
    _similar_ocr_tokens,
    detected_text_tokens,
)


class OcrConfirmationTests(
    unittest.TestCase
):
    def test_exact_confirmation_matches(
        self,
    ):
        self.assertTrue(
            _similar_ocr_tokens(
                "title",
                "title",
            )
        )

    def test_one_character_ocr_variation_matches(
        self,
    ):
        self.assertTrue(
            _similar_ocr_tokens(
                "title",
                "titie",
            )
        )

    def test_unrelated_tokens_do_not_match(
        self,
    ):
        self.assertFalse(
            _similar_ocr_tokens(
                "title",
                "city",
            )
        )

    def test_short_noise_is_not_confirmation(
        self,
    ):
        self.assertFalse(
            _similar_ocr_tokens(
                "ti",
                "ti",
            )
        )

    def test_moderate_first_pass_requires_confirmation(
        self,
    ):
        image = Image.new(
            "RGB",
            (
                256,
                256,
            ),
            (
                20,
                30,
                60,
            ),
        )

        first_pass = {
            "text": [
                "tiie",
            ],
            "conf": [
                "52",
            ],
            "left": [
                100,
            ],
            "top": [
                100,
            ],
            "width": [
                21,
            ],
            "height": [
                21,
            ],
        }

        second_pass = {
            "text": [
                "",
            ],
            "conf": [
                "-1",
            ],
            "left": [
                0,
            ],
            "top": [
                0,
            ],
            "width": [
                0,
            ],
            "height": [
                0,
            ],
        }

        with patch(
            "app.services.comfyui_client."
            "pytesseract.image_to_data",
            side_effect=[
                first_pass,
                second_pass,
                second_pass,
            ],
        ):
            tokens = (
                detected_text_tokens(
                    image
                )
            )

        self.assertEqual(
            tokens,
            [],
        )

    def test_moderate_real_text_survives_confirmation(
        self,
    ):
        image = Image.new(
            "RGB",
            (
                256,
                256,
            ),
            (
                20,
                30,
                60,
            ),
        )

        first_pass = {
            "text": [
                "Party",
            ],
            "conf": [
                "60",
            ],
            "left": [
                80,
            ],
            "top": [
                100,
            ],
            "width": [
                60,
            ],
            "height": [
                25,
            ],
        }

        second_pass = {
            "text": [
                "Party",
            ],
            "conf": [
                "88",
            ],
            "left": [
                0,
            ],
            "top": [
                0,
            ],
            "width": [
                100,
            ],
            "height": [
                40,
            ],
        }

        with patch(
            "app.services.comfyui_client."
            "pytesseract.image_to_data",
            side_effect=[
                first_pass,
                second_pass,
            ],
        ):
            tokens = (
                detected_text_tokens(
                    image
                )
            )

        self.assertEqual(
            tokens,
            [
                "Party",
            ],
        )

    def test_high_confidence_long_text_rejects_immediately(
        self,
    ):
        image = Image.new(
            "RGB",
            (
                256,
                256,
            ),
            (
                20,
                30,
                60,
            ),
        )

        first_pass = {
            "text": [
                "BIRTHDAY",
            ],
            "conf": [
                "94",
            ],
            "left": [
                50,
            ],
            "top": [
                80,
            ],
            "width": [
                120,
            ],
            "height": [
                30,
            ],
        }

        with patch(
            "app.services.comfyui_client."
            "pytesseract.image_to_data",
            return_value=(
                first_pass
            ),
        ) as mocked:
            tokens = (
                detected_text_tokens(
                    image
                )
            )

        self.assertEqual(
            tokens,
            [
                "BIRTHDAY",
            ],
        )

        mocked.assert_called_once()


if __name__ == "__main__":
    unittest.main()