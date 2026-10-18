"""Spatial density diagnostics for background artwork, independent of subject."""

from __future__ import annotations

import re

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
        "tile_edge_fractions": [round(float(value), 4) for value in tiles],
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


def assess_explicit_safe_regions(density: dict, strategies: dict[str, str]) -> dict:
    """Check the area named by each quiet strategy, independently of global density."""
    regions = {}
    for name, strategy in strategies.items():
        if not re.search(r"\b(?:quiet|calm|low.detail|breathing room)\b", strategy, re.I):
            continue
        tiles = density["tile_edge_fractions"]
        words = strategy.casefold()
        columns = ([0, 1] if re.search(r"\b(?:left|western?)\b", words) else
                   [2, 3] if re.search(r"\b(?:right|eastern?)\b", words) else [1, 2])
        rows = ([0, 1] if re.search(r"\b(?:top|upper)\b", words) else
                [2, 3] if re.search(r"\b(?:bottom|lower)\b", words) else
                [1, 2] if re.search(r"\b(?:middle|mid|center|centre)\b", words) else
                [0, 1] if name == "title" else [2, 3])
        indexes = [row * 4 + column for row in rows for column in columns]
        local = [tiles[index] for index in indexes]
        occupied = sum(value >= .10 for value in local) / len(local)
        quiet = sum(value <= .04 for value in local) / len(local)
        regions[name] = {"decision": "FAIL" if occupied >= .75 and quiet == 0 else "PASS",
                         "tile_indexes": indexes, "occupied_tile_fraction": occupied,
                         "quiet_tile_fraction": quiet}
    return {"decision": "FAIL" if any(region["decision"] == "FAIL" for region in regions.values()) else "PASS",
            "regions": regions}
