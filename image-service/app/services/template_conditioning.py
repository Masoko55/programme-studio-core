"""Broad, deterministic img2img starting images derived from CandidateSpec.

The template encodes page composition and requested colours, never event motifs.
ComfyUI and the existing validators remain responsible for artwork and quality.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.config.settings import settings
from app.services.background_policy import parse_colour
from app.services.candidate_spec import CandidateSpec
from app.services.prompt_repository import get_job_directory


@dataclass(frozen=True, slots=True)
class TemplateConditioningProfile:
    engine_id: str
    initial_denoise: float
    minimum_denoise: float
    maximum_denoise: float


PROFILES = {
    "sdxl-1-0": TemplateConditioningProfile("sdxl-1-0", 0.76, 0.45, 0.84),
    "sd-3-5-medium": TemplateConditioningProfile("sd-3-5-medium", 0.70, 0.40, 0.80),
    "flux-2": TemplateConditioningProfile("flux-2", 0.70, 0.45, 0.78),
}


@dataclass(frozen=True, slots=True)
class TemplateConditioningSpec:
    candidate_sha256: str
    reference_number: str
    engine_id: str
    direction_id: str
    direction_role: str
    template_path: str
    title_zone: tuple[float, float, float, float]
    programme_zone: tuple[float, float, float, float]
    primary_colour: str
    secondary_colour: str
    denoise: float
    failure_category: str
    retry_stage: str
    edge_fraction: float
    secondary_fraction: float
    quiet_padding: float
    simplified: bool
    attempt: int

    @property
    def conditioning_sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def record(self) -> dict:
        return {**asdict(self), "conditioning_sha256": self.conditioning_sha256}


def persisted_conditioning_spec(record: dict) -> TemplateConditioningSpec:
    """Restore the exact attempt saved before submission; never reapply retry policy."""
    names = {field.name for field in fields(TemplateConditioningSpec)}
    try:
        spec = TemplateConditioningSpec(**{name: record[name] for name in names})
    except (KeyError, TypeError) as error:
        raise ValueError("Persisted template conditioning is incomplete") from error
    if spec.conditioning_sha256 != record.get("attempt_conditioning_sha256", record.get("conditioning_sha256")):
        raise ValueError("Current attempt conditioning metadata SHA mismatch")
    return spec


def verify_conditioning_artifacts(record: dict) -> None:
    """Detect changes to the files belonging to this attempt only."""
    spec = persisted_conditioning_spec(record)
    template = Path(record["template_path"])
    metadata = json.loads(template.with_name("conditioning.json").read_text(encoding="utf-8"))
    if metadata != json.loads(json.dumps(spec.record())):
        raise ValueError("Current attempt conditioning.json changed")
    for name, path in (
        ("template", template),
        ("occupancy_map", template.with_name("occupancy-map.png")),
        ("protected_mask", template.with_name("protected-mask.png")),
        ("generation_mask", template.with_name("generation-mask.png")),
    ):
        expected = record.get(f"{name}_sha256")
        if expected is not None and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Current attempt {name} SHA mismatch")


def _zone(layout: dict, key: str, fallback: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    value = layout.get(key)
    if value is None:
        return fallback
    if not isinstance(value, dict):
        raise ValueError(f"Invalid {key} layout guidance")
    zone = tuple(float(value[field]) for field in ("x", "y", "width", "height"))
    x, y, width, height = zone
    if not (0 <= x < 1 and 0 <= y < 1 and 0 < width <= 1 - x and 0 < height <= 1 - y):
        raise ValueError(f"Invalid {key} layout coordinates")
    return zone


def compile_template_conditioning(
    candidate: CandidateSpec,
    *,
    denoise: float | None = None,
    failure_category: str = "INITIAL",
    retry_stage: str = "normal",
    attempt: int = 1,
) -> TemplateConditioningSpec:
    if candidate.engine_id not in PROFILES or candidate.direction_id not in "ABC":
        raise ValueError("Unsupported engine or direction for template conditioning")
    profile = PROFILES[candidate.engine_id]
    chosen_denoise = profile.initial_denoise if denoise is None else denoise
    if not 0 < chosen_denoise < 1:
        raise ValueError("Img2img denoise must be between zero and one")
    layout = candidate.layout_guidance
    # These are the title/programme text cores measured by visual_quality.py.
    title = _zone(layout, "title_zone", (0.18, 0.09, 0.64, 0.10))
    programme = _zone(layout, "programme_zone", (0.18, 0.51, 0.64, 0.32))
    edge = {"A": 0.13, "B": 0.18, "C": 0.10}[candidate.direction_id]
    secondary = 0.30
    padding = 0.015
    simplified = False
    if failure_category in {"DENSE_SAFE_REGION", "PROTECTED_REGION_INTRUSION"}:
        padding = 0.05
        edge = min(edge, 0.10)
        chosen_denoise -= 0.05 * min(attempt - 1, 4)
    elif failure_category in {"WEAK_OUTER_STRUCTURE", "WEAK_TEMPLATE_STRUCTURE", "WEAK_DECORATIVE_DESIGN", "FLAT"}:
        edge = min(0.22, edge + 0.035)
        if failure_category in {"FLAT", "WEAK_DECORATIVE_DESIGN"}:
            chosen_denoise += 0.03
    elif failure_category == "TEMPLATE_DRIFT":
        chosen_denoise -= 0.05 * min(attempt - 1, 4)
    elif failure_category == "PALETTE_OFF":
        chosen_denoise -= 0.05
    elif failure_category == "PRIMARY_MISSING":
        secondary = 0.20
    elif failure_category == "SECONDARY_MISSING":
        secondary = 0.35
    elif failure_category == "RASTER":
        simplified = True
        chosen_denoise -= 0.05
    chosen_denoise = max(profile.minimum_denoise, min(profile.maximum_denoise, chosen_denoise))
    directory = (get_job_directory(candidate.reference_number) / "conditioning" /
                 candidate.engine_id / candidate.direction_id)
    if attempt > 1:
        directory /= f"attempt-{attempt:02d}"
    path = directory / "template.png"
    return TemplateConditioningSpec(
        candidate.spec_sha256, candidate.reference_number, candidate.engine_id,
        candidate.direction_id, candidate.direction_role, str(path), title, programme,
        candidate.primary_colour, candidate.secondary_colour, round(chosen_denoise, 3),
        failure_category, retry_stage, edge, secondary, padding, simplified, attempt,
    )


def _rectangle(zone: tuple[float, float, float, float], size: tuple[int, int], padding: float) -> tuple[int, int, int, int]:
    x, y, width, height = zone
    return (
        max(0, round((x - padding) * size[0])),
        max(0, round((y - padding) * size[1])),
        min(size[0] - 1, round((x + width + padding) * size[0])),
        min(size[1] - 1, round((y + height + padding) * size[1])),
    )


def render_template(spec: TemplateConditioningSpec, size: tuple[int, int], path: Path | None = None) -> Path:
    """Render broad colour blocks; protected zones stay flat primary colour."""
    if min(size) < 64 or any(dimension % 16 for dimension in size):
        raise ValueError("Template dimensions must be at least 64 and divisible by 16")
    primary, secondary = parse_colour(spec.primary_colour), parse_colour(spec.secondary_colour)
    if primary is None or secondary is None:
        raise ValueError("Template colours must resolve through the existing palette system")
    width, height = size
    image = Image.new("RGB", size, primary)
    draw = ImageDraw.Draw(image)
    occupancy = Image.new("L", size, 0)
    occupancy_draw = ImageDraw.Draw(occupancy)
    band = round(width * spec.edge_fraction * spec.secondary_fraction / 0.30)
    band = min(band, round(width * 0.22))
    corner = round(height * (0.18 if spec.simplified else 0.26))
    if spec.direction_id == "A":
        draw.rectangle((0, 0, band, height), fill=secondary)
        draw.rectangle((width - band, 0, width, height), fill=secondary)
        occupancy_draw.rectangle((0, 0, band, height), fill=255)
        occupancy_draw.rectangle((width - band, 0, width, height), fill=255)
        if not spec.simplified:
            draw.ellipse((0, height - corner, band * 2, height + corner), fill=secondary)
            draw.ellipse((width - band * 2, height - corner, width, height + corner), fill=secondary)
            occupancy_draw.ellipse((0, height - corner, band * 2, height + corner), fill=255)
            occupancy_draw.ellipse((width - band * 2, height - corner, width, height + corner), fill=255)
    elif spec.direction_id == "B":
        draw.rectangle((0, 0, band, height), fill=secondary)
        occupancy_draw.rectangle((0, 0, band, height), fill=255)
        if not spec.simplified:
            draw.ellipse((width - band * 2, height - corner, width, height + corner), fill=secondary)
            occupancy_draw.ellipse((width - band * 2, height - corner, width, height + corner), fill=255)
    else:
        for x in (0, width - band):
            draw.ellipse((x - band, -corner, x + band * 2, corner), fill=secondary)
            occupancy_draw.ellipse((x - band, -corner, x + band * 2, corner), fill=255)
            if not spec.simplified:
                draw.ellipse((x - band, height - corner, x + band * 2, height + corner), fill=secondary)
                occupancy_draw.ellipse((x - band, height - corner, x + band * 2, height + corner), fill=255)
    protected = Image.new("L", size, 0)
    protected_draw = ImageDraw.Draw(protected)
    for zone in (spec.title_zone, spec.programme_zone):
        bounds = _rectangle(zone, size, spec.quiet_padding)
        draw.rectangle(bounds, fill=primary)
        occupancy_draw.rectangle(bounds, fill=0)
        protected_draw.rectangle(bounds, fill=255)
    # White permits diffusion noise. The broad decorative footprint receives
    # full strength; the rest receives only a little. Protected regions are
    # expanded and feathered so no hard rectangular seam is introduced.
    footprint = np.asarray(occupancy, dtype=np.uint8)
    protected_pixels = np.asarray(protected, dtype=np.uint8)
    radius = max(3, round(width * (0.07 if spec.failure_category == "WEAK_DECORATIVE_DESIGN" else 0.055)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    decorative = cv2.dilate(footprint, kernel).astype(np.float32) / 255.0
    decorative = cv2.GaussianBlur(decorative, (0, 0), max(1, width * 0.018))
    extra_protection = (0.035 if spec.failure_category == "PROTECTED_REGION_INTRUSION" else 0.012)
    guard_radius = max(1, round(width * extra_protection))
    guard_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (guard_radius * 2 + 1, guard_radius * 2 + 1))
    guard = cv2.dilate(protected_pixels, guard_kernel).astype(np.float32) / 255.0
    guard = cv2.GaussianBlur(guard, (0, 0), max(1, width * 0.012))
    generation = np.clip((0.18 + 0.82 * decorative) * (1.0 - guard), 0, 1)
    generation_mask = Image.fromarray(np.uint8(np.rint(generation * 255)), mode="L")
    target = path or Path(spec.template_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    artifacts = {
        target: image,
        target.with_name("occupancy-map.png"): occupancy,
        target.with_name("protected-mask.png"): protected,
        target.with_name("generation-mask.png"): generation_mask,
    }
    for destination, artifact in artifacts.items():
        buffer = io.BytesIO()
        artifact.save(buffer, format="PNG")
        data = buffer.getvalue()
        if destination.exists():
            if destination.read_bytes() != data:
                raise ValueError(f"Current attempt conditioning artifact changed: {destination}")
        else:
            destination.write_bytes(data)
    metadata = json.dumps(spec.record(), indent=2, sort_keys=True)
    metadata_path = target.with_name("conditioning.json")
    if metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != spec.record():
            raise ValueError("Current attempt conditioning metadata changed")
    else:
        metadata_path.write_text(metadata, encoding="utf-8")
    return target
