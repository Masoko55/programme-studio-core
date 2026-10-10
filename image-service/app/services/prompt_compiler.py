"""Compile each native-model attempt from the same immutable brief."""

from __future__ import annotations

from dataclasses import dataclass
import re

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
    for category in ("BOILERPLATE_COMPOSITION", "DIRECTION_DUPLICATE", "PROTECTED_REGION_INTRUSION", "WEAK_TEMPLATE_STRUCTURE", "WEAK_DECORATIVE_DESIGN", "TEMPLATE_DRIFT"):
        if category.casefold() in text:
            return category
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

    # Dense-centre failures must be classified before generic edge-density
    # failures. The visual-quality error includes both "center structural
    # edge density" and "outer structural edge density", so checking the
    # generic "edge density" phrase first incorrectly routes a dense centre
    # into WEAK_OUTER_STRUCTURE.
    if any(
        word in text
        for word in (
            "programme safe region",
            "program safe region",
            "central title",
            "central region",
            "too visually dense",
            "too much local contrast",
            "center structural edge density",
            "centre structural edge density",
            "center-to-outer ratio",
            "centre-to-outer ratio",
        )
    ):
        return "DENSE_SAFE_REGION"

    if any(word in text for word in ("outer background", "outer structure", "outer structural edge density", "too little useful")):
        return "WEAK_OUTER_STRUCTURE"
    if any(word in text for word in ("flat", "blank", "insufficient designed structure", "insufficient visible structure")):
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
    design = spec.visual_design
    colours = [str(item["colour"]) for item in design["requested_palette"]]
    if not colours:
        return "the palette described by the brief"
    relationship = design["palette_relationship"] or "respect the requested colour roles"
    return f"{', '.join(colours)}; {relationship}; preserve shading and local detail"


def _composition(spec: CandidateSpec) -> str:
    design = spec.visual_design
    strategy = design["composition_strategy"]
    safe = "; ".join(filter(None, (
        design["title_safe_region_strategy"], design["programme_safe_region_strategy"]
    )))
    return (
        f"{strategy}. Preserve calm, readable space for later typography"
        + (f" according to {safe}" if safe else " where the direction requires it")
        + "; keep key motifs outside those spaces."
    )


def _subject(spec: CandidateSpec) -> str:
    """Put the requested scene ahead of generic direction boilerplate.

    Some stored briefs have no structured inspiration. Their direction prompt
    describes layout at length and names only an "abstract event background",
    while the creative description contains the actual scene. CLIP must see
    that scene early, so use its explicit creation sentence as the subject.
    The complete original direction remains in the positive prompt/spec.
    """
    if spec.background_inspiration:
        return spec.background_inspiration
    sentences = re.split(r"(?<=[.!?])\s+", spec.creative_description)
    for sentence in sentences:
        if re.match(r"\s*(?:create|draw|depict|illustrate|render|feature)\b", sentence, re.I):
            return sentence.strip().rstrip(".!?")
    return _direction_motif(spec)


def _direction_motif(spec: CandidateSpec) -> str:
    """Surface the direction's own visual design before long layout prose."""
    opening = ", ".join(
        part.strip() for part in spec.original_positive_prompt.split(",")[:3]
        if part.strip()
    )
    opening = re.sub(
        r"^(?:an? )?elegant abstract event background,?\s*",
        "",
        opening,
        flags=re.I,
    )
    return " ".join(opening.split()[:35]).rstrip(".!? ")


_CORRECTIONS = {
    "HUMAN": "Remove people, faces, silhouettes and humanoid shapes from this same background scene.",
    "TEXT": "Remove readable text, letters, numbers, logos and watermarks from this same design.",
    "PALETTE_OFF": "Correct the requested colour relationship while preserving this design's texture, shading and details.",
    "PRIMARY_MISSING": "Make the requested primary colour dominant across substantial surfaces of this same design.",
    "SECONDARY_MISSING": "Increase the requested secondary colour in several substantial visible regions of this same design.",
    "GRADIENT": "Render this same design with discrete matte forms and visible boundaries instead of a smooth colour wash.",
    "FLAT": "Give this same requested scene clearly bounded visual forms; avoid a blank or featureless field.",
    "RASTER": "Render this same scene with broad clean forms; remove repeated fine lines, scanlines and grid artefacts.",
    "NOISE": "Render this same scene with clean surfaces and coherent forms; remove grain and noise texture.",
    "WEAK_OUTER_STRUCTURE": "Increase the requested motifs at the outer edges and lower corners of this same design.",
    "DENSE_SAFE_REGION": "Move detail from the calm overlay zones to the requested outer-edge and lower-corner motifs.",
    "PROTECTED_REGION_INTRUSION": "Move all motifs to the allowed edges and corners; leave title and programme areas calm.",
    "TEMPLATE_DRIFT": "Follow the direction's edge-and-corner placement while retaining its requested motifs.",
    "WEAK_TEMPLATE_STRUCTURE": "Strengthen the direction's allowed edge and corner decoration.",
    "WEAK_DECORATIVE_DESIGN": "Add richer requested motifs inside the allowed decorative regions while keeping title and programme areas calm.",
    "BOILERPLATE_COMPOSITION": "Develop a materially different composition with richer subject-specific structure, spatial rhythm and depth.",
    "DIRECTION_DUPLICATE": "Choose a different arrangement, focal treatment and depth strategy while retaining the brief's anchors.",
    "RUNTIME": "Render this same design with fewer, larger forms to reduce generation complexity.",
    "UNKNOWN": "Correct the previous failure while retaining this exact requested scene, direction and palette.",
}


def _join_sections(*sections: str) -> str:
    return " ".join(section for section in sections if section)


def _positive_prompt(
    spec: CandidateSpec, subject: str, motif: str,
    treatment: str, identity: str, palette: str,
    composition: str, correction: str, attempt: int,
) -> str:
    shared = (
        f"Direction artwork: {motif}." if motif else "",
        f"Theme treatment: {treatment}." if treatment else "",
        identity,
        f"Brief anchors: {', '.join(spec.visual_design['visual_anchors'])}.",
        f"Mood and style: {spec.visual_design['mood']}; {spec.visual_design['visual_style']}.",
        f"Palette: {palette}.",
        f"Composition: {composition}",
    )
    safety = "Background artwork only; no people, characters, costumes, masked figures, readable text or logos."
    if spec.engine_id == "sdxl-1-0":
        return _join_sections(
            f"Background inspiration: {subject}." if subject != motif else "",
            *shared,
            f"Correction: {correction}" if correction else "",
            safety,
        )
    return _join_sections(
        shared[0], f"Background inspiration: {subject}.",
        f"Theme reference treatment: {treatment}." if treatment else "",
        *shared[2:],
        f"Correction for attempt {attempt}: {correction}" if correction else "",
        f"Direction design: {spec.original_positive_prompt}.",
        safety,
    )


def _encoder_fields(
    spec: CandidateSpec, subject: str, motif: str,
    encoder_motif: str, treatment: str, identity: str,
    correction: str,
) -> tuple[str, str]:
    direction = f"Direction: {encoder_motif}." if motif != subject else ""
    colour = f"Requested palette: {_palette(spec)}."
    if spec.engine_id == "sdxl-1-0":
        common = (
            f"Visual subject: {subject}.", direction, colour,
            "No lettering or logos.",
            f"Correction: {correction}" if correction else "",
        )
        return (
            _join_sections("Front-facing graphic background filling a flat page with matte forms.", *common),
            _join_sections("Front-facing graphic background artwork on a flat page, with matte forms.", *common),
        )
    common = (f"Visual subject: {subject}.", direction)
    return (
        _join_sections("Front-facing graphic background illustration on a flat page.", *common, colour),
        _join_sections(
            "Front-facing graphic background on a flat page, varied forms and an open centre.",
            *common, identity, colour,
            f"Theme treatment: {treatment}." if treatment else "",
            f"Correction: {correction}" if correction else "",
        ),
    )


def compile_candidate_prompt(spec: CandidateSpec, attempt: int, reason: str = "") -> CompiledPrompt:
    stage = retry_stage(attempt)
    category = failure_category(reason) if attempt > 1 else "INITIAL"
    if attempt > 1 and category == "INITIAL":
        raise ValueError("A retry requires the previous failure reason")

    subject = _subject(spec)
    motif = _direction_motif(spec)
    # The brief already supplies the subject. Keep only the direction's first
    # design clause in the short encoders so a repeated subject does not
    # overwhelm composition and medium instructions.
    encoder_motif = " ".join(motif.split(",", 1)[0].split()[:20])
    treatment = spec.theme_reference_treatment
    palette = _palette(spec)
    composition = _composition(spec)
    identity = f"Direction {spec.direction_id}: {spec.direction_role} visual character."
    correction = _CORRECTIONS.get(category, "")
    if stage in {"strict_recovery", "rescue"}:
        correction += (
            " Replace fine artefacts with coherent, varied visual forms "
            "while preserving the subject, direction, palette and composition."
        )

    positive = _positive_prompt(
        spec, subject, motif, treatment, identity, palette,
        composition, correction, attempt,
    )
    negative = (
        "people, faces, characters, costumes, masked figures, readable text, "
        "letters, typography, logos, watermarks, photographed print, "
        "poster mockup, unrelated subject"
    )
    extra_negative = {
        "RASTER": ", scanlines, raster grid",
        "NOISE": ", grain, noise",
    }
    negative += extra_negative.get(category, "")
    clip_l, clip_g = _encoder_fields(
        spec, subject, motif, encoder_motif, treatment, identity, correction,
    )
    # T5 carries each distinct semantic field once. The original direction
    # prompt remains in CandidateSpec and in the SDXL positive prompt, but it
    # often repeats the inspiration and palette several times. Repeating it
    # in T5 can cross SD3.5 Medium's 256-token training context.
    t5 = " ".join(section for section in (
        f"Background inspiration: {subject}.",
        f"Theme: {spec.theme}." if spec.theme in spec.visual_design["visual_anchors"] else "",
        f"Visual anchors: {', '.join(spec.visual_design['visual_anchors'])}.",
        f"Theme treatment: {treatment}." if treatment else "",
        identity,
        f"Palette: {palette}.",
        f"Composition: {composition}",
        f"Correction: {correction}" if correction else "",
        "No people, characters, costumes, masked figures, readable text or logos.",
    ) if section)
    return CompiledPrompt(
        positive=positive,
        negative=negative,
        clip_l=clip_l,
        clip_g=clip_g,
        t5=t5 if spec.engine_id == "sd-3-5-medium" else None,
        failure_category=category,
        retry_stage=stage,
        spec_sha256=spec.spec_sha256,
    )
