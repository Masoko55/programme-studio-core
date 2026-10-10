"""Colour-independent checks for collapsed and repeated compositions."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _structure(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(image.convert("RGB").resize((128, 176)), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(rgb, (5, 5), 0).astype(np.float32)
    dx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.max(np.hypot(dx, dy), axis=2) > 80
    return gray, edges


def boilerplate_score(image: Image.Image) -> float:
    """Fraction of the page occupied by a dominant near-uniform rectangular area."""
    gray, edges = _structure(image)
    # Local variance and edge coverage are independent of the requested hue.
    mean = cv2.blur(gray.astype(np.float32), (9, 9))
    variance = cv2.blur(gray.astype(np.float32) ** 2, (9, 9)) - mean ** 2
    flat = (variance < 55) & ~edges
    count, labels, stats, _ = cv2.connectedComponentsWithStats(flat.astype(np.uint8), 8)
    largest = 0.0
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if width < 30 or height < 40:
            continue
        rectangularity = area / (width * height)
        if rectangularity >= 0.82:
            largest = max(largest, area / gray.size)
    return float(largest)


def direction_similarity(first: Image.Image, second: Image.Image) -> float:
    """Compare geometry even when the two images use different colours."""
    _, first_edges = _structure(first)
    _, second_edges = _structure(second)
    first_edges = cv2.dilate(first_edges.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    second_edges = cv2.dilate(second_edges.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    union = np.count_nonzero(first_edges | second_edges)
    if union < 80:
        return 0.0
    return float(np.count_nonzero(first_edges & second_edges) / union)


def compare_directions(candidate: Path, peers: list[Path]) -> float:
    with Image.open(candidate) as image:
        scores = []
        for path in peers:
            with Image.open(path) as peer:
                scores.append(direction_similarity(image, peer))
        return max(scores, default=0.0)
