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


def _largest_calm_rectangle(calm: np.ndarray) -> tuple[int, int, int]:
    """Find a contiguous area wide enough for a line of overlay text."""
    best = (0, 0, 0)
    for top in range(8):
        for bottom in range(top + 2, 9):
            usable_columns = np.all(calm[top:bottom], axis=0)
            start = 0
            for end in range(9):
                if end < 8 and usable_columns[end]:
                    continue
                width = end - start
                height = bottom - top
                if width >= 3 and width * height > best[0]:
                    best = (width * height, width, height)
                start = end + 1
    return best


def _overlay_interference(image: Image.Image, rows: list[int], columns: list[int],
                          bounds: tuple[float, float, float, float] | None = None) -> dict:
    """Measure sharp detail and large contrasting forms inside one text area."""
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    if bounds is None:
        bounds = (columns[0] / 4, rows[0] / 4,
                  (columns[-1] - columns[0] + 1) / 4,
                  (rows[-1] - rows[0] + 1) / 4)
    x, y, region_width, region_height = bounds
    crop = gray[int(y * height):max(int((y + region_height) * height), int(y * height) + 1),
                int(x * width):max(int((x + region_width) * width), int(x * width) + 1)]
    local = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
    fine_blur = cv2.GaussianBlur(local, (0, 0), 2)
    broad_blur = cv2.GaussianBlur(local, (0, 0), 6)
    high_frequency_mask = np.abs(local.astype(np.float32) - fine_blur) >= 18
    edge_mask = cv2.Canny(local, 40, 100) > 0
    high_frequency = float(np.mean(high_frequency_mask))
    edge_density = float(np.mean(edge_mask))
    detail_tiles = (high_frequency_mask | edge_mask).reshape(8, 16, 8, 16).mean(axis=(1, 3))
    cells = local.reshape(8, 16, 8, 16).transpose(0, 2, 1, 3)
    cell_contrast = np.percentile(cells, 90, axis=(2, 3)) - np.percentile(cells, 10, axis=(2, 3))
    calm_cells = (detail_tiles <= .10) | (cell_contrast < 55)
    calm_fraction = float(np.mean(calm_cells))
    window_area, window_width, window_height = _largest_calm_rectangle(calm_cells)
    contrast = float(np.percentile(local, 90) - np.percentile(local, 10))
    broad_contrast = float(np.percentile(broad_blur, 90) - np.percentile(broad_blur, 10))
    broad_edge_density = float(np.mean(cv2.Canny(broad_blur, 35, 90) > 0))
    dense_detail = contrast >= 65 and (high_frequency >= .18 or
                                       (edge_density >= .13 and high_frequency >= .08))
    broad_form = (broad_contrast >= 80 and broad_edge_density >= .008 and
                  edge_density < .06 and high_frequency < .16)
    # Broad planes and silhouettes can leave ample room for legible type.
    # Hard failure needs sharp detail across nearly the entire named area.
    unusable = dense_detail and window_area == 0
    borderline = dense_detail or broad_form or (contrast >= 65 and edge_density >= .09 and calm_fraction <= .5)
    decision = "FAIL" if unusable else "BORDERLINE" if borderline else "PASS"
    reason = ("dense_high_contrast_detail" if unusable else
              "detail_with_calm_area" if dense_detail else
              "large_contrasting_form" if broad_form else
              "moderate_local_detail" if borderline else None)
    return {"decision": decision, "reason": reason,
            "local_edge_density": round(edge_density, 4),
            "local_contrast": round(contrast, 2),
            "high_frequency_fraction": round(high_frequency, 4),
            "calm_tile_fraction": round(calm_fraction, 4),
            "usability_score": round(calm_fraction, 4),
            "calm_cell_fraction": round(calm_fraction, 4),
            "largest_calm_window_fraction": round(window_area / 64, 4),
            "usable_window_width_fraction": round(window_width / 8, 4),
            "usable_window_height_fraction": round(window_height / 8, 4),
            "broad_contrast": round(broad_contrast, 2),
            "broad_edge_density": round(broad_edge_density, 4)}


def assess_explicit_safe_regions(density: dict, strategies: dict[str, str],
                                 image: Image.Image | None = None,
                                 layout_guidance: dict | None = None) -> dict:
    """Check typography usability in named regions, separately from global density."""
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
        legacy_decision = "FAIL" if occupied >= .75 and quiet == 0 else "PASS"
        zone = (layout_guidance or {}).get(f"{name}_zone") if isinstance(layout_guidance, dict) else None
        bounds = None
        if isinstance(zone, dict):
            try:
                x, y, width, height = (float(zone[key]) for key in ("x", "y", "width", "height"))
                if 0 <= x < 1 and 0 <= y < 1 and width > 0 and height > 0 and x + width <= 1 and y + height <= 1:
                    bounds = (x, y, width, height)
            except (KeyError, TypeError, ValueError):
                pass
        region_bounds = bounds or (columns[0] / 4, rows[0] / 4,
                                   (columns[-1] - columns[0] + 1) / 4,
                                   (rows[-1] - rows[0] + 1) / 4)
        interference = (_overlay_interference(image, rows, columns, bounds) if image is not None else
                        {"decision": legacy_decision, "reason": "occupied_tiles" if legacy_decision == "FAIL" else None})
        regions[name] = {**interference, "tile_indexes": indexes,
                         "occupied_tile_fraction": occupied, "quiet_tile_fraction": quiet,
                         "region_bounds": dict(zip(("x", "y", "width", "height"), region_bounds)),
                         "region_source": "layout_guidance" if bounds else "strategy_fallback"}
    return {"decision": ("FAIL" if any(region["decision"] == "FAIL" for region in regions.values()) else
                         "BORDERLINE" if any(region["decision"] == "BORDERLINE" for region in regions.values()) else "PASS"),
            "evaluation_status": "EVALUATED" if regions else "NOT_EVALUATED",
            "regions": regions}
