"""Deterministic spatial check against a candidate's layout template.

Only coarse structural occupancy is compared. Colours and event motifs are
deliberately absent from this check; existing validators handle those.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.services.template_conditioning import TemplateConditioningSpec


SIZE = (64, 88)


def _region(zone: tuple[float, float, float, float]) -> np.ndarray:
    x, y, width, height = zone
    mask = np.zeros((SIZE[1], SIZE[0]), dtype=bool)
    left, top = round(x * SIZE[0]), round(y * SIZE[1])
    right, bottom = round((x + width) * SIZE[0]), round((y + height) * SIZE[1])
    mask[top:bottom, left:right] = True
    return mask


def _mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(values[mask].mean()) if mask.any() else 0.0


def assess_template_adherence(path: Path, spec: TemplateConditioningSpec) -> dict:
    """Return diagnostic metrics and one actionable retry category."""
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB").resize((256, 352)), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 45, 110).astype(np.float32) / 255.0
    local_mean = cv2.GaussianBlur(gray, (0, 0), 5).astype(np.float32)
    detail = np.abs(gray.astype(np.float32) - local_mean) / 32.0
    structure = np.maximum(edges, np.clip(detail, 0, 1))
    structure = cv2.resize(structure, SIZE, interpolation=cv2.INTER_AREA)
    occupancy = structure >= 0.055

    with Image.open(Path(spec.template_path).with_name("occupancy-map.png")) as source:
        expected = np.asarray(source.convert("L").resize(SIZE), dtype=np.uint8) >= 128
    title = _region(spec.title_zone)
    programme = _region(spec.programme_zone)
    protected = title | programme
    outer = np.zeros_like(expected, dtype=bool)
    side = max(1, round(spec.edge_fraction * SIZE[0] * 1.8))
    outer[:, :side] = True
    outer[:, -side:] = True
    outer[:12, :] = True
    outer[-12:, :] = True
    outer &= ~protected
    center = ~outer & ~protected
    corners = np.zeros_like(expected, dtype=bool)
    corners[:18, :16] = corners[:18, -16:] = True
    corners[-18:, :16] = corners[-18:, -16:] = True
    corners &= ~protected

    title_density = _mean(structure, title)
    programme_density = _mean(structure, programme)
    outer_density = _mean(structure, outer)
    center_density = _mean(structure, center)
    protected_share = float(occupancy[protected].sum() / max(1, occupancy.sum()))
    expected_soft = cv2.GaussianBlur(expected.astype(np.float32), (0, 0), 2)
    observed_soft = cv2.GaussianBlur(occupancy.astype(np.float32), (0, 0), 2)
    agreement = float(1 - np.abs(expected_soft - observed_soft).mean())
    expected_binary = expected_soft >= 0.25
    observed_binary = observed_soft >= 0.25
    union = int((expected_binary | observed_binary).sum())
    iou = float((expected_binary & observed_binary).sum() / union) if union else 1.0
    correlation = float(np.corrcoef(expected_soft.ravel(), observed_soft.ravel())[0, 1]) if expected_soft.std() > 0 and observed_soft.std() > 0 else 0.0
    bias = outer_density / max(center_density, 0.005)

    category = None
    if title_density > 0.11 or programme_density > 0.13:
        category = "PROTECTED_REGION_INTRUSION"
    elif outer_density < 0.028 or _mean(structure, corners) < 0.016:
        category = "WEAK_TEMPLATE_STRUCTURE"
    elif bias < 1.18 or correlation < 0.02:
        category = "TEMPLATE_DRIFT"
    return {
        "template_outer_occupancy_ratio": float(expected[outer].mean()),
        "template_center_occupancy_ratio": float(expected[center].mean()),
        "generated_outer_occupancy_ratio": float(occupancy[outer].mean()),
        "generated_center_occupancy_ratio": float(occupancy[center].mean()),
        "title_zone_edge_density": title_density,
        "programme_zone_edge_density": programme_density,
        "outer_edge_density": outer_density,
        "center_edge_density": center_density,
        "corner_edge_density": _mean(structure, corners),
        "decoration_in_protected_ratio": protected_share,
        "template_structure_iou": iou,
        "template_structure_correlation": correlation,
        "template_structure_agreement": agreement,
        "edge_bias_ratio": bias,
        "template_adherence_passed": category is None,
        "template_failure_category": category,
    }
