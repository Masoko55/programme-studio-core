"""Compile each native-model attempt from the same immutable brief."""

from __future__ import annotations

from dataclasses import dataclass

from app.services.candidate_spec import CandidateSpec


@dataclass(frozen=True, slots=True)
class CompiledPrompt:
    positive: str
    negative: str
    clip_l: str | None
    clip_g: str | None
    t5: str | None
    failure_category: str
    retry_stage: str
    spec_sha256: str

    def record(self) -> dict:
        return {
            "positive": self.positive,
            "negative": self.negative,
            "clip_l": self.clip_l,
            "clip_g": self.clip_g,
            "t5": self.t5,
        }


def failure_category(reason: str) -> str:
    text = reason.casefold()
    if not text:
        return "INITIAL"
    if any(word in text for word in ("human", "person", "face", "silhouette")):
        return "HUMAN"
    if any(word in text for word in ("readable text", "ocr", "lettering", "logo", "watermark")):
        return "TEXT"
    if "secondary colour" in text or "secondary color" in text:
        return "SECONDARY_MISSING"
    if "primary colour" in text or "primary color" in text:
        return "PRIMARY_MISSING"
    if any(word in text for word in ("off-palette", "not requested", "palette", "colour family")):
        return "PALETTE_OFF"
    if any(word in text for word in ("raster", "scanline", "banding", "moire", "repetitive horizontal", "repetitive vertical")):
        return "RASTER"
    if any(word in text for word in ("gradient", "colour wash", "color wash")):
        return "GRADIENT"
    if any(word in text for word in ("noise", "grain")):
        return "NOISE"
    if any(word in text for word in ("outer background", "outer structure", "edge density", "too little useful")):
        return "WEAK_OUTER_STRUCTURE"
    if any(word in text for word in ("central title", "central region", "too visually dense")):
        return "DENSE_SAFE_REGION"
    if any(word in text for word in ("flat", "blank", "insufficient designed structure")):
        return "FLAT"
    if any(word in text for word in ("timeout", "out of memory", "runtime", "execution", "busy")):
        return "RUNTIME"
    return "UNKNOWN"


def retry_stage(attempt: int) -> str:
    if not 1 <= attempt <= 9:
        raise ValueError("One candidate permits attempts 1 through 9 only")
    if attempt <= 3:
        return "normal"
    if attempt <= 6:
        return "targeted_recovery"
    if attempt <= 8:
        return "strict_recovery"
    return "rescue"


def _palette(spec: CandidateSpec) -> str:
    if spec.primary_colour and spec.secondary_colour:
        return (
            f"{spec.primary_colour} dominant, {spec.secondary_colour} supporting; "
            "use only these requested colour families"
        )
    return spec.primary_colour or spec.secondary_colour or "the requested palette"


def _composition(spec: CandidateSpec) -> str:
    zones = []
    for name in ("title_zone", "programme_zone", "headshot_zone", "logo_zone"):
        zone = spec.layout_guidance.get(name)
        if isinstance(zone, dict):
            coords = ", ".join(
                f"{key} {zone[key]}" for key in ("x", "y", "width", "height") if key in zone
            )
            zones.append(f"{name.replace('_', ' ')} at {coords}")
    return (
        "Decorate the outer edges and lower corners while keeping overlay zones calm. "
        + ("Reserve " + "; ".join(zones) + "." if zones else "")
    )


_CORRECTIONS = {
    "HUMAN": "Remove people, faces, silhouettes and humanoid shapes from this same background scene.",
    "TEXT": "Remove readable text, letters, numbers, logos and watermarks from this same design.",
    "PALETTE_OFF": "Render this same design using only the requested colour families; remove unrelated hues and neutral surfaces.",
    "PRIMARY_MISSING": "Make the requested primary colour dominant across substantial surfaces of this same design.",
    "SECONDARY_MISSING": "Increase the requested secondary colour in several substantial visible regions of this same design.",
    "GRADIENT": "Render this same design with discrete matte forms and visible boundaries instead of a smooth colour wash.",
    "FLAT": "Give this same requested scene clearly bounded visual forms; avoid a blank or featureless field.",
    "RASTER": "Render this same scene with broad clean forms; remove repeated fine lines, scanlines and grid artefacts.",
    "NOISE": "Render this same scene with clean surfaces and coherent forms; remove grain and noise texture.",
    "WEAK_OUTER_STRUCTURE": "Increase the requested motifs at the outer edges and lower corners of this same design.",
    "DENSE_SAFE_REGION": "Move detail from the calm overlay zones to the requested outer-edge and lower-corner motifs.",
    "RUNTIME": "Render this same design with fewer, larger forms to reduce generation complexity.",
    "UNKNOWN": "Correct the previous failure while retaining this exact requested scene, direction and palette.",
}


def compile_candidate_prompt(spec: CandidateSpec, attempt: int, reason: str = "") -> CompiledPrompt:
    stage = retry_stage(attempt)
    category = failure_category(reason) if attempt > 1 else "INITIAL"
    if attempt > 1 and category == "INITIAL":
        raise ValueError("A retry requires the previous failure reason")

    subject = spec.background_inspiration or spec.original_positive_prompt
    treatment = spec.theme_reference_treatment
    palette = _palette(spec)
    composition = _composition(spec)
    identity = f"Direction {spec.direction_id}: {spec.direction_role} visual character."
    correction = _CORRECTIONS.get(category, "")
    if stage in {"strict_recovery", "rescue"}:
        correction += (
            " Simplify the rendering to four to six large intentional forms, "
            "while preserving the same subject, inspiration, treatment, "
            "direction, palette and composition."
        )

    semantic_sections = [
        f"Background inspiration: {subject}.",
        f"Theme reference treatment: {treatment}." if treatment else "",
        identity,
        f"Palette: {palette}.",
        f"Composition: {composition}",
        f"Correction for attempt {attempt}: {correction}" if correction else "",
        f"Event: {spec.event_type}; theme: {spec.theme}.",
        f"Creative description: {spec.creative_description}." if spec.creative_description else "",
        f"Direction design: {spec.original_positive_prompt}.",
        "Background artwork only; no people, characters, readable text or logos.",
    ]
    positive = " ".join(section for section in semantic_sections if section)
    negative = "people, faces, characters, readable text, logos, watermarks, unrelated subject"
    if category == "RASTER":
        negative += ", scanlines, raster grid"
    elif category == "NOISE":
        negative += ", grain, noise"

    clip_l = " ".join(
        section for section in (
            subject, identity, f"Palette: {palette}.",
        ) if section
    )
    clip_g = " ".join(
        section for section in (
            identity, f"Subject: {subject}.", f"Palette: {palette}.",
            f"Composition: {composition}",
            f"Correction: {correction}" if correction else "",
            f"Theme treatment: {treatment}." if treatment else "",
        ) if section
    )
    return CompiledPrompt(
        positive=positive,
        negative=negative,
        clip_l=clip_l if spec.engine_id == "sd-3-5-medium" else None,
        clip_g=clip_g if spec.engine_id == "sd-3-5-medium" else None,
        t5=positive if spec.engine_id == "sd-3-5-medium" else None,
        failure_category=category,
        retry_stage=stage,
        spec_sha256=spec.spec_sha256,
    )
