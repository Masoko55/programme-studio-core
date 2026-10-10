"""Broad, deterministic img2img starting images derived from CandidateSpec.

The template encodes page composition and requested colours, never event motifs.
ComfyUI and the existing validators remain responsible for artwork and quality.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

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
    "sdxl-1-0": TemplateConditioningProfile("sdxl-1-0", 0.88, 0.45, 0.95),
    "sd-3-5-medium": TemplateConditioningProfile("sd-3-5-medium", 0.80, 0.40, 0.90),
    "flux-2": TemplateConditioningProfile("flux-2", 0.95, 0.55, 0.97),
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
    elif failure_category in {"WEAK_OUTER_STRUCTURE", "WEAK_TEMPLATE_STRUCTURE", "FLAT"}:
        edge = min(0.22, edge + 0.035)
        if failure_category == "FLAT":
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
    target = path or Path(spec.template_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)
    occupancy.save(target.with_name("occupancy-map.png"))
    protected.save(target.with_name("protected-mask.png"))
    target.with_name("conditioning.json").write_text(
        json.dumps(spec.record(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return target
