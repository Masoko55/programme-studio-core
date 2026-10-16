"""Compile each native-model attempt from the same immutable brief."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import colorsys
import re

from app.services.candidate_spec import CandidateSpec
from app.services.background_policy import COLOUR_ALIASES, COLOUR_FAMILIES, NAMED_COLOURS, palette_relationship_policy
from app.services.engine_capabilities import engine_capabilities


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
    negative_prompt_channel_active: bool = True
    negative_semantics_preserved: bool = True
    incompatible_treatments: tuple[str, ...] = ()
    semantic_contract_sha256: str = ""
    conditioning_coverage: dict = field(default_factory=dict)
    engine_capabilities: dict = field(default_factory=dict)
    truncated: bool = False
    retry_strength_level: str = "BASE"

    def record(self) -> dict:
        return {
            "positive": self.positive,
            "negative": self.negative,
            "clip_l": self.clip_l,
            "clip_g": self.clip_g,
            "t5": self.t5,
            "negative_prompt_channel_active": self.negative_prompt_channel_active,
            "negative_semantics_preserved": self.negative_semantics_preserved,
            "incompatible_treatments": list(self.incompatible_treatments),
            "semantic_contract_sha256": self.semantic_contract_sha256,
            "conditioning_coverage": self.conditioning_coverage,
            "engine_capabilities": self.engine_capabilities,
            "prompt_budget": {key: {"characters": len(value or ""),
                                    "approx_tokens": len((value or "").split()),
                                    "budget_risk": (len((value or "").split()) > 75 if key in {"clip_l", "clip_g"}
                                                    else len((value or "").split()) > 256 if key == "t5" else False)}
                              for key, value in (("positive", self.positive), ("negative", self.negative),
                                                 ("clip_l", self.clip_l), ("clip_g", self.clip_g), ("t5", self.t5))},
            "truncated": self.truncated,
            "retry_strength_level": self.retry_strength_level,
        }


def failure_category(reason: str) -> str:
    text = reason.casefold()
    if not text:
        return "INITIAL"
    if "TITLE_PANEL" in reason.upper() or "TEMPLATE_PANEL" in reason.upper():
        return "TITLE_PANEL"
    if "SUBJECT_DRIFT" in reason.upper():
        return "SUBJECT_DRIFT"
    for dimension in ("ATMOSPHERE_DRIFT", "MOTIF_DRIFT", "AUDIENCE_DRIFT"):
        if dimension in reason.upper():
            return dimension
    if "BRIEF_FIDELITY_DRIFT" in reason.upper():
        return "BRIEF_FIDELITY_DRIFT"
    if "PALETTE_ROLE_DRIFT" in reason.upper():
        return "PALETTE_ROLE_DRIFT"
    if "DENSITY_DRIFT" in reason.upper():
        return "DENSITY_DRIFT"
    if "VISION_UNAVAILABLE" in reason.upper():
        return "VISION_UNAVAILABLE"
    if "OCR_TIMEOUT" in reason.upper():
        return "OCR_TIMEOUT"
    if "OCR_UNAVAILABLE" in reason.upper() or "OCR_PARTIAL" in reason.upper():
        return "OCR_UNAVAILABLE"
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


def _palette_relationship(spec: CandidateSpec) -> str:
    design = spec.visual_design
    policy = palette_relationship_policy(design["palette_relationship"],
                                         design.get("palette_relationship_source"), spec.secondary_colour)
    if policy["explicit"]:
        return policy["relationship"]
    colours = [str(item["colour"]) for item in design["requested_palette"]]
    return "; ".join(f"{colour} clearly visible in substantial design elements" for colour in colours)


def _palette(spec: CandidateSpec) -> str:
    design = spec.visual_design
    colours = [str(item["colour"]) for item in design["requested_palette"]]
    if not colours:
        return "the palette described by the brief"
    relationship = _palette_relationship(spec)
    mode = design["palette_mode"]
    if mode in {"monochrome", "achromatic"} or set(colours) == {"black", "white"}:
        policy = "strictly achromatic black, white and grey tonal variation; no chromatic colour"
    elif mode == "strict":
        policy = "only requested colour families, apart from microscopic anti-aliasing tolerance"
    else:
        policy = "natural tonal variation within requested families; no large unrequested chromatic region"
    return f"{', '.join(colours)}; {mode} palette; {relationship}; {policy}"


def _encoder_palette(spec: CandidateSpec) -> str:
    design = spec.visual_design
    colours = ", ".join(str(item["colour"]) for item in design["requested_palette"])
    mode = design["palette_mode"]
    if mode in {"monochrome", "achromatic"} or set(colours.split(", ")) == {"black", "white"}:
        policy = "strictly achromatic"
    elif mode == "strict":
        policy = "only requested colours"
    else:
        policy = "no major unrelated chromatic family"
    return f"Requested palette: {colours}; {_palette_relationship(spec)}; {policy}. {_chromatic_boundaries(spec)}".strip()


def _affirmative_palette(spec: CandidateSpec) -> str:
    colours = ", ".join(str(item["colour"]) for item in spec.visual_design["requested_palette"])
    if not colours:
        return ""
    return (f"The visible chromatic design is restricted to the requested families: {colours}. "
            "Tonal variations within those families are allowed. Lighting changes brightness while "
            "preserving those families. " + _chromatic_boundaries(spec))


def _chromatic_boundaries(spec: CandidateSpec) -> str:
    """Name nearby unrequested hues using the validator's colour metadata."""
    requested = [COLOUR_ALIASES.get(str(item["colour"]).casefold(), str(item["colour"]).casefold())
                 for item in spec.visual_design["requested_palette"]]
    allowed = set().union(*(COLOUR_FAMILIES.get(name, {name}) for name in requested))
    clauses = []
    for name in requested:
        rgb = NAMED_COLOURS.get(name)
        if not rgb:
            continue
        hue, saturation, _ = colorsys.rgb_to_hsv(*(channel / 255 for channel in rgb))
        if saturation < .18:
            continue
        alternatives = []
        for other, other_rgb in NAMED_COLOURS.items():
            if other in allowed or other in {"black", "white", "grey", "silver", "charcoal"}:
                continue
            other_hue, other_saturation, _ = colorsys.rgb_to_hsv(*(channel / 255 for channel in other_rgb))
            distance = min(abs(hue - other_hue), 1 - abs(hue - other_hue))
            if other_saturation >= .25 and distance <= .16:
                alternatives.append((distance, other))
        alternatives.sort()
        if alternatives:
            names = [item for _, item in alternatives[:6]]
            clauses.append(f"Keep {name} distinctly {name}, avoiding {', '.join(names)} drift")
    return "; ".join(clauses) + ("." if clauses else "")


def _atmosphere(spec: CandidateSpec) -> str:
    semantics = spec.visual_design["visual_semantics"]
    mood = ", ".join(semantics["atmosphere_descriptors"]) or "event-appropriate"
    audience = ", ".join(semantics["audience_descriptors"])
    return _join_sections(
        f"Event atmosphere: {mood}.",
        f"Audience fit: {audience}." if audience else "",
        f"Scene family: {semantics['scene_family']}." if semantics["scene_family"] != "general" else "",
        "environment-only composition.",
    )


def _motif_structure(spec: CandidateSpec) -> str:
    semantics = spec.visual_design["visual_semantics"]
    structure = semantics["motif_structure_description"] or ", ".join(semantics["motif_descriptors"])
    prominence = spec.visual_design["visual_hierarchy"]["motif_prominence"]
    return f"{prominence.capitalize()} motif topology visibly follows: {structure}." if structure else ""


def _hierarchy(spec: CandidateSpec) -> str:
    hierarchy = spec.visual_design["visual_hierarchy"]
    density = hierarchy["detail_density"]
    amount = {
        "sparse": "few forms, broad calm areas and very limited small detail",
        "moderate": "coherent large forms, limited mid-size forms and concentrated small detail",
        "rich": "layered large and medium forms with rich detail concentrated away from calm areas",
        "dense": "many layered forms with clear size hierarchy and at least one calm visual region",
    }[density]
    return (f"Visual hierarchy: the subject is present without saturating the frame; {amount}. "
            "Large forms establish the subject, medium forms carry motif topology, small details support texture. "
            "Preserve integrated low-detail breathing room and readable depth; avoid uniformly distributed micro-detail. "
            f"Motif prominence is {hierarchy['motif_prominence']}. "
            "Lighting, contrast and edge character should express the requested atmosphere and audience without changing the subject. "
            "Keep the result usable as background artwork rather than a crowded standalone illustration.")


def _scene_tone(spec: CandidateSpec) -> tuple[str, str]:
    """Consume upstream lighting and exclusion semantics without event labels."""
    semantics = spec.visual_design["visual_semantics"]
    lighting = ", ".join(semantics["lighting_descriptors"])
    desired = f"Desired lighting: {lighting}." if lighting else ""
    return desired, ", ".join(semantics["incompatible_treatments"])


def _exclude_incompatible(text: str, spec: CandidateSpec) -> str:
    """Do not repeat explicitly excluded treatments in positive conditioning."""
    for phrase in spec.visual_design["visual_semantics"]["incompatible_treatments"]:
        text = re.sub(r"\b" + re.escape(phrase) + r"\b", "", text, flags=re.I)
    return text


def _composition(spec: CandidateSpec) -> str:
    design = spec.visual_design

    parts = []

    strategy = str(
        design.get("composition_strategy")
        or ""
    ).strip()

    focal = str(
        design.get("focal_strategy")
        or ""
    ).strip()

    depth = str(
        design.get("depth_strategy")
        or ""
    ).strip()

    perimeter = str(
        design.get("border_or_perimeter_strategy")
        or ""
    ).strip()

    title_safe = str(
        design.get("title_safe_region_strategy")
        or ""
    ).strip()

    programme_safe = str(
        design.get("programme_safe_region_strategy")
        or ""
    ).strip()

    if strategy:
        parts.append(strategy)

    if focal:
        parts.append(
            f"Focal strategy: {focal}"
        )

    if depth:
        parts.append(
            f"Depth strategy: {depth}"
        )

    if perimeter:
        parts.append(
            f"Perimeter strategy: {perimeter}"
        )

    safe_regions = [
        value
        for value in (
            title_safe,
            programme_safe,
        )
        if value
    ]

    if safe_regions:
        parts.append(
            "Preserve intentional negative space as integrated low-detail visual breathing room according to "
            + "; ".join(safe_regions)
        )
    else:
        parts.append(
            "Preserve intentional negative space as integrated low-detail visual breathing room "
            "where the composition naturally supports it"
        )

    parts.append(
        "Use natural visual hierarchy, breathing room and spatial balance "
        "with controlled subject concentration"
    )

    return ". ".join(parts).rstrip(". ") + "."


def _design_language(spec: CandidateSpec) -> str:
    design = spec.visual_design
    sections = []
    for key, label in (
        ("mood", "Mood"),
        ("visual_style", "Visual style"),
        ("material_language", "Materials"),
        ("texture_language", "Texture"),
        ("lighting_language", "Lighting"),
        ("abstraction_level", "Abstraction"),
        ("realism_level", "Realism"),
        ("energy_level", "Energy"),
        ("symmetry_preference", "Symmetry"),
        ("density_preference", "Density"),
    ):
        value = str(design.get(key) or "").strip()
        if value:
            sections.append(f"{label}: {value}")

    for key, label in (
        ("symbolic_motifs", "Symbolic motifs"),
        ("environmental_motifs", "Environmental motifs"),
    ):
        values = design.get(key) or []
        if values:
            sections.append(f"{label}: {', '.join(str(v) for v in values)}")

    return ". ".join(sections)

def _subject(spec: CandidateSpec) -> str:
    """Put the requested scene ahead of generic direction boilerplate.

    Some stored briefs have no structured inspiration. Their direction prompt
    describes layout at length and names only an "abstract event background",
    while the creative description contains the actual scene. CLIP must see
    that scene early, so use its explicit creation sentence as the subject.
    The complete original direction remains in the positive prompt/spec.
    """
    if spec.visual_design.get("primary_visual_subject"):
        return spec.visual_design["primary_visual_subject"]
    sentences = re.split(r"(?<=[.!?])\s+", spec.creative_description)
    for sentence in sentences:
        if re.match(r"\s*(?:create|draw|depict|illustrate|render|feature)\b", sentence, re.I):
            return sentence.strip().rstrip(".!?")
    return _direction_motif(spec)


def _direction_motif(spec: CandidateSpec) -> str:
    """Use brief-grounded motifs; stored direction prose may be stale."""
    design = spec.visual_design
    semantics = design["visual_semantics"]
    source = (semantics["motif_structure_description"] or
              ", ".join(semantics["motif_descriptors"]) or
              spec.theme_reference_treatment or
              ", ".join(design["symbolic_motifs"] + design["environmental_motifs"]))
    return " ".join(source.split()[:35]).rstrip(".!? ")


_CORRECTIONS = {
    "HUMAN": "Remove people, faces, silhouettes and humanoid shapes from this same background scene.",
    "TEXT": "Render uninterrupted visual surfaces and coherent image forms across the scene; keep calm areas integrated into the artwork.",
    "TITLE_PANEL": "Integrate the calm area into the scene itself. Remove separate panels, cards, banners, bordered boxes and blank sheets.",
    "SUBJECT_DRIFT": "Depict the requested visual environment with recognizable subject-specific forms and the supplied visual anchors. Retain this direction's composition and palette.",
    "ATMOSPHERE_DRIFT": "Strengthen the supplied atmosphere and lighting descriptors while preserving the same subject, motif, audience and palette.",
    "MOTIF_DRIFT": "Make the supplied motif structure visibly legible while preserving the same subject, atmosphere, audience and palette.",
    "AUDIENCE_DRIFT": "Strengthen the supplied audience-appropriate visual treatment while preserving the same subject, motif, atmosphere and palette.",
    "PALETTE_OFF": "Correct the requested colour relationship while preserving this design's texture, shading and details.",
    "PALETTE_ROLE_DRIFT": "Restore the requested primary and secondary visual-mass relationship without changing subject, motif or structure.",
    "PRIMARY_MISSING": "Make the requested primary colour clearly visible across substantial surfaces of this same design.",
    "SECONDARY_MISSING": "Increase the requested secondary colour in several substantial visible regions of this same design.",
    "GRADIENT": "Give this scene distinct subject-specific forms, tonal variation and visible depth.",
    "FLAT": "Strengthen the requested subject silhouette with coherent medium and large forms, clear depth and a visible focal structure.",
    "RASTER": "Render this scene with coherent forms and clean, varied edge structure.",
    "NOISE": "Render this scene with coherent forms and clear material detail.",
    "WEAK_OUTER_STRUCTURE": "Strengthen coherent medium and large subject forms and depth in the areas permitted by this direction while keeping its quiet regions readable.",
    "DENSE_SAFE_REGION": "Relocate detail away from integrated calm regions while preserving this direction's own composition strategy.",
    "DENSITY_DRIFT": "Restore the requested spatial density and coherent large-to-small form hierarchy; reduce uniformly distributed detail while preserving subject, motif, atmosphere and palette.",
    "PROTECTED_REGION_INTRUSION": "Relocate motifs away from reserved typography regions according to this direction's safe-region strategy.",
    "TEMPLATE_DRIFT": "Restore the direction's requested composition and subject-specific visual structure.",
    "WEAK_TEMPLATE_STRUCTURE": "Strengthen the direction's requested composition using subject-specific forms.",
    "WEAK_DECORATIVE_DESIGN": "Add richer subject-specific visual structure where this direction permits it while preserving usable negative space.",
    "BOILERPLATE_COMPOSITION": "Redistribute visual mass, move the focal structure, vary scale and depth, and integrate breathing room asymmetrically while retaining the requested subject and motifs.",
    "DIRECTION_DUPLICATE": "Change focal location, perspective, visual mass, scale hierarchy, depth, motif placement, symmetry and negative-space geometry while retaining the requested subject, anchors, palette and event meaning.",
    "RUNTIME": "Render this same design with fewer, larger forms to reduce generation complexity.",
    "UNKNOWN": "Correct the previous failure while retaining this exact requested scene, direction and palette.",
}


def _join_sections(*sections: str) -> str:
    return " ".join(section for section in sections if section)


def _first_words(value: str, limit: int) -> str:
    return " ".join(str(value or "").split()[:limit]).rstrip(". ;")


def _background_only(text: str, spec: CandidateSpec) -> str:
    """Keep planning labels out of image-model conditioning."""
    exclusions = [spec.direction_role,
                  "Architectural Master", "Comic Book Cityscape Illustrator",
                  "Creative Direction for A4 Portrait Event Background",
                  *spec.visual_design.get("render_exclusion_text", [])]
    for phrase in sorted(exclusions, key=len, reverse=True):
        phrase = str(phrase).strip()
        if len(phrase) >= 4:
            text = re.sub(r"\b" + re.escape(phrase) + r"\b", "", text, flags=re.I)
    text = re.sub(r"\bDirection\s+[ABC]\b\s*:?", "", text, flags=re.I)
    text = re.sub(r"\bvisual character\b", "", text, flags=re.I)
    text = re.sub(r"\bfor (?:later )?(?:title|programme|program|typography)\b", "", text, flags=re.I)
    text = re.sub(
        r"\b(?:clear|blank|white|empty|bordered|rectangular)?\s*"
        r"(?:title|programme|poster|text)?\s*"
        r"(?:panel|box|card|banner|placard|signboard|rectangle)\b"
        r"|\brectangular negative space\b",
        "visually calm area integrated into the artwork", text, flags=re.I,
    )
    text = re.sub(r"\b(?:title|programme|program|typography|text placement|logos?|schedules?|information|legibility|safe regions?|copy area)\b",
                  "visual calm", text, flags=re.I)
    text = re.sub(r"\b(?:poster|ui)\s+(?:layout|mockup|panel)\b", "continuous scene", text, flags=re.I)
    text = re.sub(r"\bno\s+(?:people|persons|characters)(?:\s+or\s+(?:people|persons|characters))?\b",
                  "environment-only composition", text, flags=re.I)
    text = re.sub(r"\b(?:brief-compatible|no\s+glyph-like\s+marks)\b", "pure visual artwork", text, flags=re.I)
    text = re.sub(r"\b(?:or|and)\s+(?=[.;])", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def _positive_prompt(
    spec: CandidateSpec, subject: str, motif: str,
    treatment: str, palette: str,
    composition: str, correction: str, attempt: int,
) -> str:
    shared = (
        f"Visual motif: {motif}." if motif else "",
        f"Theme treatment: {treatment}." if treatment else "",
        f"Brief anchors: {', '.join(spec.visual_design['visual_anchors'])}.",
        f"Composition: {composition}",
        f"Design language: {_design_language(spec)}." if _design_language(spec) else "",
        f"Palette: {palette}.",
        _hierarchy(spec),
    )
    safety = ("Pure visual background artwork with a clear subject and natural low-detail breathing room. "
              "Continuous scene surfaces and integrated calm areas.")
    if spec.engine_id == "sdxl-1-0":
        return _join_sections(
            f"Background inspiration: {subject}." if subject != motif else "",
            *shared,
            _atmosphere(spec), _scene_tone(spec)[0], _motif_structure(spec),
            f"Correction: {correction}" if correction else "",
            safety,
        )
    return _join_sections(
        f"Background inspiration: {subject}.", shared[0],
        f"Theme reference treatment: {treatment}." if treatment else "",
        *shared[2:],
        _atmosphere(spec), _scene_tone(spec)[0], _motif_structure(spec),
        f"Correction: {correction}" if correction else "",
        "Use only brief-grounded subject and motif content with this direction's spatial strategies.",
        safety,
    )


def _encoder_fields(
    spec: CandidateSpec, subject: str, motif: str,
    encoder_motif: str, treatment: str,
    correction: str,
) -> tuple[str, str]:
    direction_detail = treatment or encoder_motif
    direction = f"Visual motif: {direction_detail}." if direction_detail else ""
    colour = _encoder_palette(spec)
    anchors = spec.visual_design["visual_anchors"]
    distinct_anchors = [item for item in anchors if item.casefold() != subject.casefold()]
    anchor_text = f"Visual anchors: {', '.join(distinct_anchors)}." if distinct_anchors else ""
    motifs = [*spec.visual_design["symbolic_motifs"], *spec.visual_design["environmental_motifs"]]
    motif_text = f"Motifs: {', '.join(motifs)}." if motifs else ""
    design = spec.visual_design
    structure = _join_sections(
        f"Composition: {design['composition_strategy']}.",
        f"Focal: {design['focal_strategy']}.",
        f"Depth: {design['depth_strategy']}.",
        f"Perimeter: {design['border_or_perimeter_strategy']}.",
        f"Quiet areas: {design['title_safe_region_strategy']}; {design['programme_safe_region_strategy']}.",
    )
    material = _join_sections(
        f"Materials: {design['material_language']}." if design['material_language'] else "",
        f"Texture: {design['texture_language']}." if design['texture_language'] else "",
        f"Lighting: {design['lighting_language']}." if design['lighting_language'] else "",
    )
    short_correction = " ".join(correction.split()[:6]) if correction else ""
    if spec.engine_id == "sdxl-1-0":
        subject_line = f"Visual subject: {subject}."
        atmosphere_line = _atmosphere(spec)
        tone_line = _scene_tone(spec)[0]
        structure_line = f"Composition: {_first_words(design['composition_strategy'], 10)}."
        compact_motif = _motif_structure(spec) or f"Motif: {_first_words(direction_detail, 12)}."
        spatial_line = _join_sections(
            structure_line,
            f"Focal: {_first_words(design['focal_strategy'], 7)}.",
            f"Depth: {_first_words(design['depth_strategy'], 7)}.")
        hierarchy_line = (f"{design['visual_hierarchy']['detail_density']} density; "
                          "large forms first, concentrated detail, integrated calm space.")
        return (
            _join_sections("Background artwork.", subject_line,
                           f"Atmosphere: {_first_words(', '.join(design['visual_semantics']['atmosphere_descriptors']), 10)}."
                           if design['visual_semantics']['atmosphere_descriptors'] else "",
                           f"Audience: {_first_words(', '.join(design['visual_semantics']['audience_descriptors']), 8)}."
                           if design['visual_semantics']['audience_descriptors'] else "",
                           compact_motif, _exclude_incompatible(spatial_line, spec), colour,
                           f"Correction: {short_correction}" if short_correction else ""),
            _join_sections("Background artwork.", subject_line, atmosphere_line,
                           compact_motif, _exclude_incompatible(spatial_line, spec),
                           f"Quiet areas: {_first_words(design['title_safe_region_strategy'], 9)}; "
                           f"{_first_words(design['programme_safe_region_strategy'], 9)}.",
                           colour, hierarchy_line,
                           tone_line,
                           f"Correction: {short_correction}" if short_correction else ""),
        )
    if spec.engine_id == "sd-3-5-medium":
        style = str(design.get("visual_style") or "").strip()
        clip_l = _join_sections("Background artwork.", f"Visual subject: {subject}.",
                                f"Anchors: {_first_words(', '.join(anchors), 14)}." if anchors else "")
        clip_g = _join_sections(
            "Background artwork.", f"Visual subject: {subject}.", direction,
            f"Style: {_first_words(style, 8)}." if style else "",
            _atmosphere(spec),
            _scene_tone(spec)[0],
            f"Composition: {_first_words(design['composition_strategy'], 10)}.",
            f"Focal: {_first_words(design['focal_strategy'], 8)}.",
            f"Depth: {_first_words(design['depth_strategy'], 8)}.",
            colour,
            f"Correction: {short_correction}" if short_correction else "",
        )
        return clip_l, clip_g
    common = (f"Visual subject: {subject}.", anchor_text, motif_text, direction,
              _atmosphere(spec), _scene_tone(spec)[0], _motif_structure(spec),
              structure, material)
    return (
        _join_sections("Background artwork.", *common, colour),
        _join_sections(
            "Background artwork.",
            *common, colour,
            f"Correction: {short_correction}" if short_correction else "",
        ),
    )


def compile_candidate_prompt(spec: CandidateSpec, attempt: int, reason: str = "", *, concise: bool = False) -> CompiledPrompt:
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
    correction = _CORRECTIONS.get(category, "")
    if category == "DIRECTION_DUPLICATE":
        recovery = (
            "Emphasize this direction's stated focal location and perimeter rhythm."
            if attempt <= 3 else
            "Separate the largest visual masses and negative-space geometry more clearly from prior candidates."
            if attempt <= 6 else
            "Use a distinct viewpoint and focal band within this direction's spatial intent; keep its subject and palette."
            if attempt <= 8 else
            "Make the strongest valid change in viewpoint, visual mass and motif placement while preserving this direction's identity."
        )
        correction = f"{correction} {recovery}"
    if category in {"DIRECTION_DUPLICATE", "FLAT", "WEAK_OUTER_STRUCTURE"}:
        design = spec.visual_design
        correction += (
            f" Keep the subject grounded in {design['focal_strategy']} "
            f"and {design['depth_strategy']}; retain {design['composition_strategy']}."
        )

    positive = _positive_prompt(
        spec, subject, motif, treatment, palette,
        composition, correction, attempt,
    )
    if spec.engine_id == "sdxl-1-0":
        negative = ("people, faces, characters, readable text, lettering, numbers, logos, watermark, "
                    "pseudo-glyphs, signage, poster mockup, bordered panel, blank card, banner")
        if _scene_tone(spec)[1]:
            negative += ", " + _scene_tone(spec)[1]
    else:
        negative = "people, faces, characters, text, glyphs, logos, title panels"
    extra_negative = {
        "RASTER": ", scanlines, raster grid",
        "NOISE": ", grain, noise",
    }
    negative += extra_negative.get(category, "")
    if spec.engine_id == "sd-3-5-medium":
        if _scene_tone(spec)[1]:
            negative += ", " + _scene_tone(spec)[1]
    negative = ", ".join(dict.fromkeys(part.strip().casefold() for part in negative.split(",") if part.strip()))
    clip_l, clip_g = _encoder_fields(
        spec, subject, motif, encoder_motif, treatment, correction,
    )
    # T5 carries each distinct semantic field once. The original direction
    # prompt remains in CandidateSpec and in the SDXL positive prompt, but it
    # often repeats the inspiration and palette several times. Repeating it
    # in T5 can cross SD3.5 Medium's 256-token training context.
    t5 = " ".join(section for section in (
        f"Visual subject and environment: {subject}.",
        f"Visual anchors: {', '.join(spec.visual_design['visual_anchors'])}.",
        _atmosphere(spec),
        _scene_tone(spec)[0],
        f"Composition and spatial depth: {composition}",
        _motif_structure(spec),
        f"Environmental motifs: {', '.join(spec.visual_design['environmental_motifs'])}." if spec.visual_design['environmental_motifs'] else "",
        f"Symbolic motifs: {', '.join(spec.visual_design['symbolic_motifs'])}." if spec.visual_design['symbolic_motifs'] else "",
        f"Palette: {palette}.",
        (f"{spec.visual_design['visual_hierarchy']['detail_density']} density; "
         "large subject forms, readable motif, concentrated detail, calm visual space."),
        "Decorative background artwork with no typography or glyph-like marks.",
        f"Design language: {_design_language(spec)}." if _design_language(spec) else "",
        f"Theme treatment: {treatment}." if treatment else "",
        f"Correction: {correction}" if correction else "",
    ) if section)
    positive = _background_only(_exclude_incompatible(positive, spec), spec)
    if not engine_capabilities(spec.engine_id)["negative_prompt_channel"]:
        positive = _join_sections(
            positive,
            _affirmative_palette(spec),
            "Environment-only visual artwork with uninterrupted natural image surfaces; "
            "continuous scene composition, calm tonal breathing room, and recognizable subject forms."
        )
    clip_l = _background_only(clip_l, spec)
    clip_g = _background_only(clip_g, spec)
    t5 = _background_only(t5, spec) if spec.engine_id == "sd-3-5-medium" else None
    channels = {"positive": positive, "clip_l": clip_l, "clip_g": clip_g, "t5": t5 or ""}
    contract = spec.semantic_contract
    probes = {
        "subject": contract["primary_visual_subject"],
        "atmosphere": ", ".join(contract["atmosphere_descriptors"]),
        "audience": ", ".join(contract["audience_descriptors"]),
        "motif": contract["motif_structure_description"] or ", ".join(contract["motif_descriptors"]),
        "palette": ", ".join(str(item["colour"]) for item in contract["palette"]),
        "scene_family": contract["scene_family"],
    }
    coverage = {name: [channel for channel, value in channels.items()
                       if probe and probe.casefold() in value.casefold()]
                for name, probe in probes.items()}
    coverage["incompatible_treatments"] = ["negative"] if spec.engine_id != "flux-2" and contract["incompatible_treatments"] else []
    effective = ("positive",) if spec.engine_id == "flux-2" else (
        ("clip_g", "t5") if spec.engine_id == "sd-3-5-medium" else ("clip_g",))
    truncated = any(probes[key] and not any(channel in coverage[key] for channel in effective)
                    for key in ("subject", "atmosphere", "audience", "motif", "palette"))
    result = CompiledPrompt(
        positive=positive,
        negative=negative,
        clip_l=clip_l,
        clip_g=clip_g,
        t5=t5,
        failure_category=category,
        retry_stage=stage,
        spec_sha256=spec.spec_sha256,
        negative_prompt_channel_active=engine_capabilities(spec.engine_id)["negative_prompt_channel"],
        incompatible_treatments=tuple(spec.visual_design["visual_semantics"]["incompatible_treatments"]),
        semantic_contract_sha256=spec.semantic_contract_sha256,
        conditioning_coverage=coverage,
        engine_capabilities=engine_capabilities(spec.engine_id),
        truncated=truncated,
    )
    return _concise_runtime_prompt(result, spec, attempt, reason) if concise else result


def runtime_retry_category(reason: str) -> str:
    """The six correction paths used by production retries."""
    raw = reason.strip().upper().split(":", 1)[0]
    category = raw if raw in {"TEXT", "TEXT_OR_LOGO", "HUMAN", "PALETTE", "PALETTE_OFF",
                              "SUBJECT", "SUBJECT_DRIFT", "GENERATION_FAILURE", "UNKNOWN",
                              "RUNTIME", "DENSITY_DRIFT", "MOTIF_DRIFT",
                              "PALETTE_ROLE_DRIFT", "PRIMARY_MISSING", "SECONDARY_MISSING"} else failure_category(reason)
    if category == "INITIAL":
        return "INITIAL"
    if category in {"HUMAN"}:
        return "HUMAN"
    if category in {"TEXT", "TEXT_OR_LOGO", "OCR_TIMEOUT", "OCR_UNAVAILABLE", "TITLE_PANEL"}:
        return "TEXT_OR_LOGO"
    if category in {"PALETTE", "PALETTE_OFF", "PALETTE_ROLE_DRIFT", "PRIMARY_MISSING", "SECONDARY_MISSING"}:
        return "PALETTE"
    if category in {"SUBJECT", "SUBJECT_DRIFT", "AUDIENCE_DRIFT", "ATMOSPHERE_DRIFT"}:
        return "SUBJECT"
    if category in {"GENERATION_FAILURE", "RUNTIME", "VISION_UNAVAILABLE"}:
        return "GENERATION_FAILURE"
    return "UNKNOWN"


def _spatial_instruction(design: dict) -> str:
    """Translate direction prose into layout only; never import a new object."""
    text = " ".join(str(design.get(key) or "") for key in (
        "composition_strategy", "focal_strategy", "depth_strategy"))
    patterns = (
        (r"\b(?:panoramic|wide[- ]angle)\b", "panoramic viewpoint"),
        (r"\b(?:low[- ]angle|looking up)\b", "low viewpoint"),
        (r"\b(?:aerial|bird.?s[- ]eye)\b", "aerial viewpoint"),
        (r"\b(?:oblique|frontal)\b", "oblique viewpoint"),
        (r"\b(?:left|right|cent(?:er|re)|off[- ]cent(?:er|re))\b", "off-centre focal placement"),
        (r"\bdiagonal\b", "diagonal flow"),
        (r"\b(?:horizontal|panoramic)\b", "horizontal flow"),
        (r"\b(?:vertical|radial|curved)\b", "curved flow"),
        (r"\b(?:receding|perspective)\b", "receding depth"),
        (r"\b(?:layered|foreground|background)\b", "layered depth"),
        (r"\bshallow\b", "shallow depth"),
    )
    choices = [label for pattern, label in patterns if re.search(pattern, text, re.I)]
    return ", ".join(dict.fromkeys(choices[:4])) or "balanced layered composition"


def retry_strength_level(attempt: int) -> str:
    if attempt <= 1:
        return "BASE"
    if attempt == 2:
        return "CORRECTION"
    if attempt == 3:
        return "STRONG_CORRECTION"
    return "STRICT_CORRECTION"


_RETRY_CORRECTIONS = {
    "TEXT_OR_LOGO": (
        "Continuous visual artwork with uninterrupted image surfaces.",
        "Strongly preserve uninterrupted imagery and natural scene forms across every region.",
        "Strictly continuous scene surfaces with natural forms in place of glyphs, signage and artificial marks.",
    ),
    "HUMAN": (
        "Environment-only composition across the canvas.",
        "Strongly preserve a purely environmental scene with natural, non-human forms.",
        "Strict environment-only artwork across focal and incidental scene elements.",
    ),
    "PALETTE": (
        "Keep the requested primary and secondary colour relationship clear.",
        "Strengthen requested colour presence and the explicit relationship while retaining the same scene and shading.",
        "Prioritize requested colour families across major masses in the same scene.",
    ),
    "SUBJECT": (
        "Make the requested subject and anchors recognizable.",
        "Strengthen original subject forms and supplied anchors in this composition.",
        "Make the original environment immediately recognizable in this exact direction.",
    ),
    "GENERATION_FAILURE": (
        "Retry the same requested scene cleanly.",
        "Render the same requested scene with simpler technical execution.",
        "Use cleaner broad forms while retaining subject, composition and palette.",
    ),
    "UNKNOWN": (
        "Retry the same direction faithfully.",
        "Strengthen clarity of the original subject and composition without new content.",
        "Render the original direction literally with the same scene content.",
    ),
}


def _concise_runtime_prompt(compiled: CompiledPrompt, spec: CandidateSpec,
                            attempt: int, reason: str) -> CompiledPrompt:
    design = spec.visual_design
    semantics = design["visual_semantics"]
    subject = _subject(spec)
    anchors = ", ".join(design["visual_anchors"])
    mood = ", ".join(semantics["atmosphere_descriptors"])
    audience = ", ".join(semantics["audience_descriptors"])
    motif = _direction_motif(spec)
    colours = ", ".join(str(item["colour"]) for item in design["requested_palette"])
    palette = f"{colours}; {_palette_relationship(spec)}; no major unrelated chromatic family"
    composition = _spatial_instruction(design)
    strength = retry_strength_level(attempt)
    correction = (_RETRY_CORRECTIONS[runtime_retry_category(reason)][min(attempt - 2, 2)]
                  if attempt > 1 else "")
    if correction:
        correction += (" Preserve original subject, anchors, reference treatment, palette, "
                       "mood, audience, motifs and spatial direction.")
    subject_line = f"Visual subject: {subject}."
    anchor_line = f"Anchors: {_first_words(anchors, 14)}." if anchors else ""
    mood_line = f"Mood: {_first_words(mood, 10)}." if mood else ""
    audience_line = f"Audience: {_first_words(audience, 7)}." if audience else ""
    motif_line = f"Motif: {_first_words(motif, 16)}." if motif else ""
    treatment_line = (f"Reference treatment: {_first_words(spec.theme_reference_treatment, 12)}."
                      if spec.theme_reference_treatment else "")
    style_line = (f"Style: {_first_words(design.get('visual_style'), 8)}."
                  if design.get("visual_style") else "")
    palette_line = f"Palette: {palette}." if colours else ""
    spatial_line = f"Composition: {composition}." if composition else ""
    light_line = f"Lighting: {_first_words(design.get('lighting_language'), 8)}." if design.get("lighting_language") else ""
    clip_l = _join_sections(subject_line, anchor_line)
    clip_g = _join_sections(subject_line, mood_line, audience_line, motif_line,
                           treatment_line, palette_line, spatial_line, style_line, light_line,
                           correction if spec.engine_id != "flux-2" else "")
    positive = _join_sections(clip_g, correction if spec.engine_id == "flux-2" else "",
                             "Continuous background artwork with integrated calm visual space.")
    if spec.engine_id == "flux-2":
        positive = _join_sections(positive, _affirmative_palette(spec),
                                 "Environment-only composition; uninterrupted visual surfaces.")
    t5 = _join_sections(subject_line, anchor_line, mood_line, audience_line, motif_line,
                       treatment_line, palette_line, spatial_line, style_line, light_line, correction,
                       "Recognizable scene with foreground, middle distance and background.")
    def clean(value: str) -> str:
        value = _background_only(value, spec)
        return re.sub(r"\b(?:8k|4k|masterpiece|ultra[- ]detailed|trending on artstation|"
                      r"award[- ]winning|unreal engine|octane render|cinematic masterpiece)\b",
                      "", value, flags=re.I).strip()
    return replace(compiled, positive=clean(positive), clip_l=clean(clip_l),
                   clip_g=clean(clip_g), t5=clean(t5) if spec.engine_id == "sd-3-5-medium" else None,
                   failure_category=runtime_retry_category(reason), retry_stage=compiled.retry_stage,
                   retry_strength_level=strength)