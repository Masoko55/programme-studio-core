"""Colour-independent checks for collapsed and repeated compositions.

These checks intentionally ignore the requested palette. They operate on
luminance, local variance and edge geometry so the same rules work for any
colour combination, custom HEX palette, event type or visual theme.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


_STRUCTURE_SIZE = (128, 176)
_EDGE_THRESHOLD = 80.0
_FLAT_VARIANCE_THRESHOLD = 55.0
_MIN_COMPONENT_WIDTH = 18
_MIN_COMPONENT_HEIGHT = 24
_MIN_RECTANGULARITY = 0.80


def _structure(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(
        image.convert("RGB").resize(_STRUCTURE_SIZE),
        dtype=np.uint8,
    )
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(rgb, (5, 5), 0).astype(np.float32)
    dx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.max(np.hypot(dx, dy), axis=2) > _EDGE_THRESHOLD
    return gray, edges


def _flat_mask(gray: np.ndarray, edges: np.ndarray) -> np.ndarray:
    gray_float = gray.astype(np.float32)
    mean = cv2.blur(gray_float, (9, 9))
    variance = cv2.blur(gray_float ** 2, (9, 9)) - mean ** 2
    return (variance < _FLAT_VARIANCE_THRESHOLD) & ~edges


def boilerplate_metrics(image: Image.Image) -> dict[str, float]:
    """Return generic indicators of template-like composition collapse.

    A calm typography-safe region is allowed. The score becomes large when the
    page is dominated by one or more broad, near-uniform rectangular slabs and
    the remaining structure is mostly long axis-aligned separators. No colour
    names or event-specific assumptions are used.
    """
    gray, edges = _structure(image)
    flat = _flat_mask(gray, edges)

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        flat.astype(np.uint8),
        8,
    )

    page_area = float(gray.size)
    largest_rectangle = 0.0
    rectangular_coverage = 0.0
    large_rectangle_count = 0

    for index in range(1, count):
        _, _, width, height, area = stats[index]
        if width < _MIN_COMPONENT_WIDTH or height < _MIN_COMPONENT_HEIGHT:
            continue
        box_area = max(1, width * height)
        rectangularity = float(area / box_area)
        if rectangularity < _MIN_RECTANGULARITY:
            continue
        ratio = float(area / page_area)
        rectangular_coverage += ratio
        largest_rectangle = max(largest_rectangle, ratio)
        if ratio >= 0.08:
            large_rectangle_count += 1

    edge_u8 = edges.astype(np.uint8)
    row_coverage = np.mean(edge_u8, axis=1)
    column_coverage = np.mean(edge_u8, axis=0)
    long_axis_ratio = float(max(
        np.mean(row_coverage >= 0.72),
        np.mean(column_coverage >= 0.72),
    ))

    flat_coverage = float(np.mean(flat))
    multi_slab_score = min(
        1.0,
        rectangular_coverage * 0.78
        + long_axis_ratio * 0.65
        + min(large_rectangle_count, 3) * 0.035,
    )

    score = float(max(largest_rectangle, multi_slab_score))
    return {
        "boilerplate_score": score,
        "largest_flat_rectangle_ratio": float(largest_rectangle),
        "flat_rectangular_coverage_ratio": float(min(1.0, rectangular_coverage)),
        "flat_coverage_ratio": flat_coverage,
        "long_axis_separator_ratio": long_axis_ratio,
        "large_flat_rectangle_count": float(large_rectangle_count),
    }


def boilerplate_score(image: Image.Image) -> float:
    return boilerplate_metrics(image)["boilerplate_score"]


def _regional_edge_signature(edges: np.ndarray, rows: int = 6, columns: int = 4) -> np.ndarray:
    """Summarise edge placement independently of colour and small pixel shifts."""
    height, width = edges.shape
    values: list[float] = []
    for row in range(rows):
        y0 = int(height * row / rows)
        y1 = int(height * (row + 1) / rows)
        for column in range(columns):
            x0 = int(width * column / columns)
            x1 = int(width * (column + 1) / columns)
            block = edges[y0:y1, x0:x1]
            values.append(float(np.mean(block)) if block.size else 0.0)
    return np.asarray(values, dtype=np.float32)


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-8:
        return 0.0
    return float(np.clip(np.dot(first, second) / denominator, 0.0, 1.0))


def direction_similarity(first: Image.Image, second: Image.Image) -> float:
    """Compare composition even when colours, shading or small details differ."""
    _, first_edges = _structure(first)
    _, second_edges = _structure(second)

    first_dilated = cv2.dilate(
        first_edges.astype(np.uint8),
        np.ones((5, 5), np.uint8),
    ) > 0
    second_dilated = cv2.dilate(
        second_edges.astype(np.uint8),
        np.ones((5, 5), np.uint8),
    ) > 0

    union = np.count_nonzero(first_dilated | second_dilated)
    edge_iou = (
        float(np.count_nonzero(first_dilated & second_dilated) / union)
        if union >= 80
        else 0.0
    )

    first_soft = cv2.GaussianBlur(first_edges.astype(np.float32), (0, 0), 3.0).reshape(-1)
    second_soft = cv2.GaussianBlur(second_edges.astype(np.float32), (0, 0), 3.0).reshape(-1)
    soft_similarity = _cosine_similarity(first_soft, second_soft)

    regional_similarity = _cosine_similarity(
        _regional_edge_signature(first_edges),
        _regional_edge_signature(second_edges),
    )

    broad_structure = 0.65 * soft_similarity + 0.35 * regional_similarity
    return float(max(edge_iou, broad_structure))


def compare_directions(candidate: Path, peers: list[Path]) -> float:
    with Image.open(candidate) as image:
        candidate_image = image.convert("RGB")
        scores = []
        for path in peers:
            with Image.open(path) as peer:
                scores.append(direction_similarity(candidate_image, peer.convert("RGB")))
        return max(scores, default=0.0)