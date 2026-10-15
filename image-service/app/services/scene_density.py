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
    safe_density = float(np.mean(tiles[[5, 6, 9, 10]]))
    macro_balance = min(1.0, float(np.mean(coarse_edges)) / max(float(np.mean(edges)), 1e-6))
    pressure = (.45 * occupied + .30 * (1 - quiet) +
                .15 * min(1.0, safe_density / .12) + .10 * (1 - macro_balance))
    threshold = {"sparse": .80, "moderate": .85}.get(level)
    # Rich/dense requests remain diagnostic-only until calibrated with labels.
    drift = threshold is not None and pressure >= threshold and quiet <= .0625
    return {
        "requested_density": level, "overall_edge_density": round(float(np.mean(edges)), 4),
        "occupied_tile_fraction": round(occupied, 4),
        "quiet_tile_fraction": round(quiet, 4),
        "safe_region_density": round(safe_density, 4),
        "edge_distribution_balance": round(float(np.std(tiles)), 4),
        "macro_form_balance": round(macro_balance, 4),
        "normalized_measurements": {"density_pressure": round(pressure, 4)},
        "thresholds_used": {"density_pressure": threshold, "max_quiet_tile_fraction": .0625,
                            "occupied_tile_edge_fraction": .10, "quiet_tile_edge_fraction": .04},
        "density_rule_triggered": "spatial_crowding" if drift else None,
        "decision": "DENSITY_DRIFT" if drift else "PASS",
    }
