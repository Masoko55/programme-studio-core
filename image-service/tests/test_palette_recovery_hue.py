import unittest

import cv2
import numpy as np
from PIL import Image

from app.services.background_policy import constrain_to_requested_palette


class PaletteRecoveryHueTests(unittest.TestCase):
    def test_pale_requested_hue_stays_chromatic_beside_white(self):
        hsv = np.zeros((24, 24, 3), dtype=np.uint8)
        hsv[:, :, 0] = 170
        hsv[:, :, 1] = 30
        hsv[:, :, 2] = 235
        source = Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))

        recovered = constrain_to_requested_palette(source, "pink", "white")
        result = cv2.cvtColor(np.asarray(recovered), cv2.COLOR_RGB2HSV)

        self.assertGreater(float(np.median(result[:, :, 1])), 30)
        self.assertGreater(float(np.median(result[:, :, 0])), 155)

    def test_hueless_white_stays_neutral(self):
        source = Image.new("RGB", (24, 24), (245, 245, 245))

        recovered = constrain_to_requested_palette(source, "pink", "white")
        result = cv2.cvtColor(np.asarray(recovered), cv2.COLOR_RGB2HSV)

        self.assertLess(float(np.max(result[:, :, 1])), 20)
