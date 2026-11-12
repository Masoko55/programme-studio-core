"""Regression controls for the existing detector evidence contract."""

import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from app.services.background_policy import detect_human_signals
from app.services.prompt_compiler import failure_category


class _Hog:
    def __init__(self, boxes=(), weights=()):
        self.boxes = boxes
        self.weights = weights

    def setSVMDetector(self, value):
        pass

    def detectMultiScale(self, image, **kwargs):
        return self.boxes, self.weights


class HumanEvidenceContractTests(unittest.TestCase):
    def _assess(self, image, boxes=(), weights=(), cascade=None):
        with patch("app.services.background_policy.cv2.HOGDescriptor", return_value=_Hog(boxes, weights)), \
             patch("app.services.background_policy.cv2.HOGDescriptor_getDefaultPeopleDetector", return_value=None), \
             patch("app.services.background_policy._load_cascade", return_value=object() if cascade else None), \
             patch("app.services.background_policy._cascade_detect", side_effect=cascade or (lambda *args: [])), \
             patch("app.services.background_policy._confirm_cascade_boxes", side_effect=(lambda detector, gray, boxes, size: boxes)):
            return detect_human_signals(image)

    def test_strong_person_detector_evidence_rejects_across_sizes(self):
        image = Image.new("RGB", (320, 448), "white")
        for box in ((80, 90, 70, 210), (120, 150, 48, 120), (95, 75, 90, 250)):
            with self.subTest(box=box):
                result = self._assess(image, boxes=[box], weights=[1.7])
                self.assertTrue(result["detected"])
                self.assertTrue(any(item["source"] == "hog" and item["confirmed"]
                                    for item in result["evidence"]))

    def test_visible_face_rejects_when_rescale_confirmed(self):
        image = Image.new("RGB", (320, 448), "white")
        def cascade(detector, gray, size, neighbors):
            return [(105, 95, 55, 55)] if neighbors == 6 else []
        result = self._assess(image, cascade=cascade)
        self.assertTrue(result["detected"])

    def test_spatially_unrelated_cascade_hits_do_not_confirm_human(self):
        image = Image.new("RGB", (896, 1280), "white")
        boxes = [(586, 291, 71, 71), (300, 270, 50, 50), (275, 822, 218, 178)]
        def cascade(detector, gray, size, neighbors):
            return [boxes[0]] if "frontal" in detector else [boxes[1]] if "profile" in detector else [boxes[2]]
        with patch("app.services.background_policy.cv2.HOGDescriptor", return_value=_Hog()), \
             patch("app.services.background_policy.cv2.HOGDescriptor_getDefaultPeopleDetector", return_value=None), \
             patch("app.services.background_policy._load_cascade", side_effect=lambda name: name), \
             patch("app.services.background_policy._cascade_detect", side_effect=cascade), \
             patch("app.services.background_policy._confirm_cascade_boxes", side_effect=lambda detector, gray, boxes, size: boxes), \
             patch("app.services.background_policy._small_face_body_evidence", return_value=[]):
            result = detect_human_signals(image)
        self.assertFalse(result["detected"])
        self.assertFalse(result["strong_upper_body_signal"])
        self.assertEqual(result["strong_signals"], [])

    def test_spatially_coherent_face_and_upper_body_still_confirm_human(self):
        image = Image.new("RGB", (896, 1280), "white")
        face = (365, 800, 50, 50)
        body = (275, 780, 218, 220)
        def cascade(detector, gray, size, neighbors):
            return [face] if "frontal" in detector else [] if "profile" in detector else [body]
        with patch("app.services.background_policy.cv2.HOGDescriptor", return_value=_Hog()), \
             patch("app.services.background_policy.cv2.HOGDescriptor_getDefaultPeopleDetector", return_value=None), \
             patch("app.services.background_policy._load_cascade", side_effect=lambda name: name), \
             patch("app.services.background_policy._cascade_detect", side_effect=cascade), \
             patch("app.services.background_policy._confirm_cascade_boxes", side_effect=lambda detector, gray, boxes, size: boxes), \
             patch("app.services.background_policy._small_face_body_evidence", return_value=[]):
            result = detect_human_signals(image)
        self.assertTrue(result["detected"])
        self.assertTrue(result["strong_upper_body_signal"])

    def test_abstract_forms_do_not_become_human_from_unsupported_evidence(self):
        for kind in ("flowers", "botanical branches", "repeated wing-like motifs", "radial impact marks",
                     "cracks", "draped fabric", "architecture", "empty hall", "geometric burst"):
            with self.subTest(kind=kind):
                image = Image.new("RGB", (320, 448), "white")
                draw = ImageDraw.Draw(image)
                if kind == "flowers":
                    for x, y in ((80, 120), (230, 190), (125, 320)):
                        for dx, dy in ((-25, 0), (25, 0), (0, -25), (0, 25)):
                            draw.ellipse((x + dx - 18, y + dy - 18, x + dx + 18, y + dy + 18), fill="pink")
                elif kind == "botanical branches":
                    draw.line((30, 380, 280, 50), fill="green", width=8)
                    for x, y in ((80, 310), (135, 240), (190, 170), (245, 100)):
                        draw.ellipse((x - 45, y - 35, x, y), fill="green")
                elif kind == "repeated wing-like motifs":
                    for x in (65, 160, 255):
                        draw.polygon(((x, 160), (x - 38, 130), (x - 18, 185), (x, 170),
                                      (x + 18, 185), (x + 38, 130)), fill="black")
                elif kind in ("radial impact marks", "geometric burst"):
                    for x, y in ((10, 30), (310, 40), (25, 420), (300, 405)):
                        draw.line((160, 225, x, y), fill="black", width=4)
                elif kind == "cracks":
                    draw.line((15, 20, 85, 130, 50, 220, 220, 300, 285, 440), fill="black", width=4)
                elif kind == "draped fabric":
                    for x in range(0, 320, 35):
                        draw.polygon(((x, 0), (x + 20, 0), (x + 50, 448), (x + 18, 448)), fill="gray")
                elif kind == "architecture":
                    for x in (40, 120, 200, 280):
                        draw.rectangle((x, 90, x + 20, 400), fill="black")
                    draw.rectangle((35, 80, 305, 95), fill="black")
                result = detect_human_signals(image)
                self.assertFalse(result["detected"])

    def test_unsupported_assertion_is_not_visual_evidence_and_prohibition_is_not_human_failure(self):
        image = Image.new("RGB", (320, 448), "white")
        self.assertFalse(self._assess(image)["detected"])
        self.assertNotEqual(failure_category("no child in costume"), "HUMAN")


if __name__ == "__main__":
    unittest.main()
