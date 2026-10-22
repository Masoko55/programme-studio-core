"""Deterministic acceptance cases for detector decisions, without ComfyUI."""
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageOps

from app.services.background_policy import detect_human_signals
from app.services.subject_adherence import _confirmed_human_answer
from app.services.comfyui_client import _pseudo_text_evidence, detect_text_evidence, detected_text_tokens
from app.services.composition_diversity import boilerplate_metrics


def ocr_data(*items):
    keys = ("text", "conf", "left", "top", "width", "height")
    return {key: [item[index] for item in items] for index, key in enumerate(keys)}


class ExactOcrCases(unittest.TestCase):
    def test_single_pseudo_word_row_with_ocr_hint_is_detected(self):
        artwork = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(artwork)
        x = 60
        for width in (9, 7, 12, 6, 11, 8, 13, 6):
            draw.line((x, 150, x, 163), fill="black", width=2)
            draw.line((x, 150, x + width, 150), fill="black", width=2)
            draw.line((x + width, 150, x + width, 163), fill="black", width=2)
            x += width + 8
        first = ocr_data(("TANSWOR", "74", 60, 145, 130, 22))
        with patch("app.services.comfyui_client.pytesseract.image_to_data",
                   side_effect=[first, ocr_data(), ocr_data(), ocr_data()]), \
             patch("app.services.comfyui_client._confirm_ocr_candidate", return_value=False):
            result = detect_text_evidence(artwork)
        self.assertEqual(result["ocr_diagnostics"]["decision"], "complete")
        self.assertEqual(result["pseudo_text"]["aligned_rows"], 1)
        self.assertTrue(result["text_detected"])
        for hint in (ocr_data(("TANSWOR", "52", 60, 145, 130, 22)),
                     ocr_data(("TANSWOR", "74", 350, 330, 130, 22))):
            with self.subTest(hint=hint), \
                 patch("app.services.comfyui_client.pytesseract.image_to_data",
                       side_effect=[hint, ocr_data(), ocr_data(), ocr_data()]), \
                 patch("app.services.comfyui_client._confirm_ocr_candidate", return_value=False):
                self.assertFalse(detect_text_evidence(artwork)["text_detected"])

    def test_geometry_with_isolated_ocr_hint_is_not_pseudo_text(self):
        for kind in ("lattice", "radial"):
            with self.subTest(kind=kind):
                artwork = Image.new("RGB", (512, 512), "white")
                draw = ImageDraw.Draw(artwork)
                for coordinate in range(0, 512, 24):
                    if kind == "lattice":
                        draw.line((coordinate, 0, coordinate, 511), fill="black", width=2)
                        draw.line((0, coordinate, 511, coordinate), fill="black", width=2)
                    else:
                        draw.line((256, 256, coordinate, 0), fill="black", width=2)
                        draw.line((256, 256, coordinate, 511), fill="black", width=2)
                hint = ocr_data(("TANSWOR", "74", 60, 145, 130, 22))
                with patch("app.services.comfyui_client.pytesseract.image_to_data",
                           side_effect=[hint, ocr_data(), ocr_data(), ocr_data()]), \
                     patch("app.services.comfyui_client._confirm_ocr_candidate", return_value=False):
                    self.assertFalse(detect_text_evidence(artwork)["text_detected"])

    def test_grouped_pseudo_glyphs_without_ocr_and_grid_control(self):
        artwork = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(artwork)
        for y in (150, 195):
            x = 60
            for width in (9, 7, 12, 6, 11, 8, 13, 6):
                draw.line((x, y, x, y + 13), fill="black", width=2)
                draw.line((x, y, x + width, y), fill="black", width=2)
                draw.line((x + width, y, x + width, y + 13), fill="black", width=2)
                x += width + 8
        self.assertTrue(_pseudo_text_evidence(artwork)["detected"])
        self.assertTrue(_pseudo_text_evidence(ImageOps.invert(artwork))["detected"])
        grid = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(grid)
        for coordinate in range(0, 512, 32):
            draw.line((coordinate, 0, coordinate, 511), fill="black", width=2)
            draw.line((0, coordinate, 511, coordinate), fill="black", width=2)
        self.assertFalse(_pseudo_text_evidence(grid)["detected"])
        web = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(web)
        for coordinate in range(0, 512, 32):
            draw.line((256, 256, coordinate, 0), fill="black", width=2)
            draw.line((256, 256, coordinate, 511), fill="black", width=2)
        for radius in (70, 130, 190):
            draw.arc((256-radius, 256-radius, 256+radius, 256+radius), 0, 360, fill="black", width=2)
        self.assertFalse(_pseudo_text_evidence(web)["detected"])

    def test_dense_web_and_window_lattice_are_not_pseudo_words(self):
        web = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(web)
        for coordinate in range(0, 512, 16):
            draw.line((256, 256, coordinate, 0), fill="black", width=2)
            draw.line((256, 256, coordinate, 511), fill="black", width=2)
            draw.line((256, 256, 0, coordinate), fill="black", width=2)
            draw.line((256, 256, 511, coordinate), fill="black", width=2)
        for radius in range(20, 260, 18):
            draw.arc((256-radius, 256-radius, 256+radius, 256+radius),
                     0, 360, fill="black", width=2)
        self.assertFalse(_pseudo_text_evidence(web)["detected"])

        windows = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(windows)
        for y in range(32, 480, 32):
            for x in range(32, 480, 32):
                draw.rectangle((x, y, x + 15, y + 22), outline="black", width=2)
        self.assertFalse(_pseudo_text_evidence(windows)["detected"])

    def evaluate(self, first, repeats=(), crops=False):
        image = Image.new("RGB", (300, 400), "white")
        scaled_repeats = []
        for index, result in enumerate(repeats):
            scaled_repeats.append({key: ([value * (1 if index == 0 else 2) for value in values]
                                          if key in {"left", "top", "width", "height"} else values)
                                   for key, values in result.items()})
        passes = [first, *scaled_repeats]
        passes.extend([ocr_data()] * (4 - len(passes)))
        with patch("app.services.comfyui_client.pytesseract.image_to_data", side_effect=passes), \
             patch("app.services.comfyui_client._confirm_ocr_candidate", return_value=crops):
            return detect_text_evidence(image)

    def test_weak_isolated_architectural_guess(self):
        result = self.evaluate(ocr_data(("XVRTA", "52", 80, 125, 22, 10)))
        self.assertFalse(result["text_detected"])
        self.assertEqual(result["confirmed_tokens"], [])

    def test_scattered_grid_and_web_guesses(self):
        result = self.evaluate(ocr_data(("XI", "65", 10, 20, 12, 9),
                                        ("AE", "60", 220, 160, 13, 9),
                                        ("RT", "70", 55, 340, 11, 8)))
        self.assertFalse(result["text_detected"])

    def test_aligned_gibberish_typography(self):
        line = ocr_data(("DRSANTIO", "82", 18, 100, 70, 19),
                        ("CALEMOR", "80", 100, 100, 70, 19),
                        ("VISUAL", "84", 182, 100, 70, 19))
        result = self.evaluate(line, (ocr_data(), line))
        self.assertTrue(result["text_detected"])
        self.assertTrue(result["text_lines"])

    def test_large_confirmed_word(self):
        word = ocr_data(("MASTER", "90", 50, 60, 150, 42))
        result = self.evaluate(word, (word,))
        self.assertTrue(result["text_detected"])
        self.assertIn("MASTER", result["confirmed_tokens"])

    def test_title_or_sign_phrase(self):
        words = ocr_data(("SUPER", "90", 20, 70, 86, 28),
                         ("HERO", "90", 116, 70, 78, 28),
                         ("BIRTHDAY", "88", 25, 120, 160, 30))
        result = self.evaluate(words, (words,))
        self.assertTrue(result["text_detected"])


class ExactHumanCases(unittest.TestCase):
    def test_located_human_silhouette_is_confirmed_without_anatomy(self):
        self.assertTrue(_confirmed_human_answer(
            True, "Silhouettes of elegantly dressed adults: a couple dancing in the foreground"))
        self.assertIsNone(_confirmed_human_answer(
            True, "Abstract human-like silhouette geometry across the foreground"))

    def evaluate(self, *, hog=(), face=(), profile=(), upper=(), confirmed=()):
        class FakeHog:
            def setSVMDetector(self, value):
                pass
            def detectMultiScale(self, image, **kwargs):
                return [box for box, _ in hog], [weight for _, weight in hog]
        detectors = {"haarcascade_frontalface_default.xml": "face",
                     "haarcascade_profileface.xml": "profile",
                     "haarcascade_upperbody.xml": "upper"}
        boxes = {"face": list(face), "profile": list(profile), "upper": list(upper)}
        with patch("app.services.background_policy.cv2.HOGDescriptor", return_value=FakeHog()), \
             patch("app.services.background_policy.cv2.HOGDescriptor_getDefaultPeopleDetector", return_value=None), \
             patch("app.services.background_policy._load_cascade", side_effect=detectors.get), \
             patch("app.services.background_policy._cascade_detect", side_effect=lambda d, *_: boxes[d]), \
             patch("app.services.background_policy._confirm_cascade_boxes", side_effect=lambda d, *_: list(boxes[d]) if d in confirmed else []):
            return detect_human_signals(Image.new("RGB", (320, 448), "white"))

    def assert_diagnostics(self, result):
        for key in ("strong_signals", "weak_signals", "human_detection_reason", "largest_detection_area_ratio"):
            self.assertIn(key, result)

    def test_weak_face(self):
        result = self.evaluate(face=[(90, 90, 35, 35)])
        self.assertFalse(result["detected"])
        self.assertTrue(result["weak_signals"])
        self.assert_diagnostics(result)

    def test_isolated_upper_body(self):
        result = self.evaluate(upper=[(100, 100, 70, 95)])
        self.assertFalse(result["detected"])
        self.assert_diagnostics(result)

    def test_city_window_noise(self):
        result = self.evaluate(face=[(10, 10, 25, 25)], profile=[(260, 300, 24, 24)],
                               upper=[(130, 190, 45, 60)])
        self.assertFalse(result["detected"])

    def test_web_profile_noise(self):
        self.assertFalse(self.evaluate(profile=[(180, 120, 35, 35)])["detected"])

    def test_strong_hog_person(self):
        result = self.evaluate(hog=[((80, 100, 64, 180), 1.7)])
        self.assertTrue(result["detected"])
        self.assert_diagnostics(result)

    def test_separate_moderate_hog_windows_are_not_independent_confirmation(self):
        result = self.evaluate(hog=[((35, 90, 64, 180), 0.9),
                                    ((210, 210, 64, 180), 1.0)])
        self.assertFalse(result["detected"])
        self.assertEqual(len(result["weak_signals"]), 2)
        self.assert_diagnostics(result)

    def test_repeat_scale_face(self):
        result = self.evaluate(face=[(90, 90, 55, 55)], confirmed=("face",))
        self.assertFalse(result["detected"])
        self.assertTrue(result["ambiguous_face_signal"])

    def test_face_and_upper_body(self):
        result = self.evaluate(face=[(110, 90, 52, 52)], upper=[(80, 85, 110, 160)], confirmed=("face",))
        self.assertTrue(result["detected"])


class ExactPanelCases(unittest.TestCase):
    def designed_background(self):
        image = Image.new("RGB", (400, 600), "#214b86")
        draw = ImageDraw.Draw(image)
        for x in range(0, 400, 48):
            draw.rectangle((x, 310, x + 30, 599), fill="#19385f")
        return image

    def test_bordered_top_title_card(self):
        image = self.designed_background()
        ImageDraw.Draw(image).rectangle((35, 25, 365, 185), fill="white", outline="black", width=7)
        self.assertGreaterEqual(boilerplate_metrics(image)["title_panel_score"], .65)

    def test_floating_cream_card(self):
        image = self.designed_background()
        ImageDraw.Draw(image).rectangle((36, 190, 364, 430), fill="#fff9e9", outline="#252525", width=6)
        self.assertGreaterEqual(boilerplate_metrics(image)["title_panel_score"], .65)

    def test_natural_quiet_sky(self):
        image = self.designed_background()
        self.assertLess(boilerplate_metrics(image)["title_panel_score"], .65)

    def test_atmospheric_region(self):
        image = self.designed_background()
        draw = ImageDraw.Draw(image)
        for y in range(0, 290):
            draw.line((0, y, 399, y), fill=(70 + y // 14, 95 + y // 15, 140 + y // 18))
        self.assertLess(boilerplate_metrics(image)["title_panel_score"], .65)
