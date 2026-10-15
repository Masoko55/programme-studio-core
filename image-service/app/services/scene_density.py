"""Spatial density diagnostics for background artwork, independent of subject."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from app.services.visual_hierarchy import density_level


def assess_scene_density(image: Image.Image, preference: str) -> dict:
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(small, 50, 120) > 0
    tiles = edges.reshape(4, 64, 4, 64).mean(axis=(1, 3)).ravel()
    coarse = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3), (64, 64), interpolation=cv2.INTER_AREA)
    coarse_edges = cv2.Canny(coarse, 25, 70) > 0
    level = density_level(preference)
    occupied = float(np.mean(tiles >= .10))
    quiet = float(np.mean(tiles <= .04))
    # A drift requires both broadly occupied tiles and almost no calm tiles.
    # Rich/dense requests remain diagnostic-only until calibrated with labels.
    drift = (level == "sparse" and occupied >= .875 and quiet <= .0625) or (
        level == "moderate" and occupied == 1.0 and quiet == 0.0 and float(np.mean(edges)) >= .22)
    return {
        "requested_density": level, "overall_edge_density": round(float(np.mean(edges)), 4),
        "occupied_tile_fraction": round(occupied, 4),
        "quiet_tile_fraction": round(quiet, 4),
        "safe_region_density": round(float(np.mean(tiles[[5, 6, 9, 10]])), 4),
        "edge_distribution_balance": round(float(np.std(tiles)), 4),
        "macro_form_balance": round(min(1.0, float(np.mean(coarse_edges)) /
                                        max(float(np.mean(edges)), 1e-6)), 4),
        "decision": "DENSITY_DRIFT" if drift else "PASS",
    }
