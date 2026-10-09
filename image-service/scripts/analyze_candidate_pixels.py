"""Measure generated colours and QA signals without changing a candidate."""

import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import httpx
import numpy as np
from PIL import Image

from app.config.settings import settings
from app.services.background_policy import (
    NAMED_COLOURS, _chromatic_requested_family, _nearest_named_colour,
    _rgb_to_hsv, detect_human_signals, validate_palette,
)
from app.services.comfyui_client import detected_text_tokens
from app.services.prompt_repository import load_prompts_document
from app.services.visual_quality import validate_visual_quality


def _bins(values: np.ndarray, edges: tuple[int, ...]) -> dict[str, float]:
    hist, _ = np.histogram(values, bins=edges)
    return {f"{edges[i]}-{edges[i + 1]}": round(int(count) / len(values), 4)
            for i, count in enumerate(hist)}


def analyze(image: Image.Image, primary: str | None, secondary: str | None) -> dict:
    rgb = np.asarray(image.convert("RGB").resize((128, 176)), dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    rgb_pixels = rgb.reshape(-1, 3)
    hsv_pixels = hsv.reshape(-1, 3)

    targets = []
    for name in (primary, secondary):
        if name in NAMED_COLOURS:
            colour = NAMED_COLOURS[name]
            targets.append({"name": name, "rgb": colour, "hue": _rgb_to_hsv(colour)[0]})
    nearest = Counter()
    assigned = Counter()
    charcoal_hsv = []
    for pixel, (hue, saturation, value) in zip(rgb_pixels, hsv_pixels):
        named = _nearest_named_colour(pixel)
        nearest[named] += 1
        assigned[_chromatic_requested_family(int(hue), int(saturation), int(value), targets)
                 or named] += 1
        if named == "charcoal":
            charcoal_hsv.append((hue, saturation, value))

    cv2.setRNGSeed(0)
    samples = rgb_pixels.astype(np.float32)
    _, labels, centers = cv2.kmeans(samples, 6, None,
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2),
                                   3, cv2.KMEANS_PP_CENTERS)
    clusters = Counter(int(label) for label in labels.reshape(-1))
    total = len(rgb_pixels)
    charcoal = np.asarray(charcoal_hsv, dtype=np.uint8).reshape(-1, 3)
    report = {
        "mean_rgb": np.mean(rgb_pixels, axis=0).round(1).tolist(),
        "median_hsv": np.median(hsv_pixels, axis=0).tolist(),
        "hue_bins": _bins(hsv_pixels[:, 0], (0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180)),
        "saturation_bins": _bins(hsv_pixels[:, 1], (0, 32, 64, 96, 128, 160, 192, 224, 256)),
        "value_bins": _bins(hsv_pixels[:, 2], (0, 32, 64, 96, 128, 160, 192, 224, 256)),
        "dominant_rgb_clusters": [
            {"rgb": centers[index].round().astype(int).tolist(), "ratio": round(count / total, 4)}
            for index, count in clusters.most_common()
        ],
        "rgb_nearest_names": {name: round(count / total, 4) for name, count in nearest.most_common()},
        "hue_first_assignments": {name: round(count / total, 4) for name, count in assigned.most_common()},
        "charcoal_labelled_pixels": {
            "ratio": round(len(charcoal) / total, 4),
            "median_hsv": np.median(charcoal, axis=0).tolist() if len(charcoal) else None,
            "fraction_saturation_at_least_42": round(float(np.mean(charcoal[:, 1] >= 42)), 4)
            if len(charcoal) else 0.0,
        },
    }
    for name, check in (("palette", lambda: validate_palette(image, primary, secondary)),
                        ("visual_quality", lambda: validate_visual_quality(image))):
        try:
            report[name] = check()
        except ValueError as error:
            report[name] = {"passed": False, "reason": str(error)}
    report["human_signals"] = detect_human_signals(image)
    report["ocr_tokens"] = detected_text_tokens(image)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--direction", required=True, choices=("A", "B", "C"))
    args = parser.parse_args()
    record_path = (settings.programme_data_path / args.reference / "backgrounds" /
                   args.engine / f"image-{args.direction.lower()}.json")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    image_path = record_path.with_suffix(".png")
    if image_path.exists():
        image = Image.open(image_path)
    else:
        descriptor = record["remote_image"]
        response = httpx.get(settings.comfyui_base_url.rstrip("/") + "/view",
                             params=descriptor, timeout=30)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
    brief = load_prompts_document(args.reference)["brief"]
    report = analyze(image, brief.get("primary_colour"), brief.get("secondary_colour"))
    report.update(reference=args.reference, engine=args.engine, direction=args.direction,
                  seed=record.get("seed"), spec_sha256=record.get("spec_sha256"),
                  sampling_profile=record.get("sampling_profile"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
