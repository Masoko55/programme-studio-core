"""Compile each native-model attempt from the same immutable brief."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import colorsys
import re

from app.services.candidate_spec import CandidateSpec
from app.services.background_policy import (COLOUR_ALIASES, COLOUR_FAMILIES, NAMED_COLOURS,
                                            NEUTRAL_NAMES, named_colour_is_requested_tone,
                                            palette_negative_contract,
                                            palette_relationship_policy)
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
    if re.match(r"^text(?:\b|_)", text):
        return "TEXT_OR_LOGO" if text.startswith("text_or_logo") else "TEXT"
    for category in ("UNREQUESTED_LITERAL_OBJECT", "DENSE_SAFE_REGION", "TEXT_OR_LOGO",
                     "PALETTE_ROLE_DRIFT", "SUBJECT_DRIFT", "TREATMENT_DRIFT", "CHARACTER", "HUMAN",
                     "VISION_UNAVAILABLE", "VALIDATOR_UNAVAILABLE"):
        if category.casefold() in text:
            return category
    if "grouped glyph-like rows" in text:
        return "TEXT_OR_LOGO"
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
    if "TREATMENT_DRIFT" in reason.upper():
        return "TREATMENT_DRIFT"
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
    if (design["palette_mode"] == "dominant" and colours
            and all(COLOUR_ALIASES.get(colour.casefold(), colour.casefold()) not in NEUTRAL_NAMES
                    for colour in colours)):
        return ("; ".join(f"{colour} clearly visible in substantial design elements" for colour in colours)
                + "; requested colour families own principal surfaces and visual mass; "
                "both requested families remain substantial; neutrals support only fine linework, shading and highlights")
    return "; ".join(f"{colour} clearly visible in substantial design elements" for colour in colours)


def _concise_palette_relationship(spec: CandidateSpec, *, include_explicit: bool = True) -> str:
    """Keep structured colour roles intact in short encoder conditioning."""
    palette = spec.visual_design["requested_palette"]
    roles = "; ".join(f"{item['role'].capitalize()} {item['colour']}" for item in palette)
    policy = palette_relationship_policy(spec.visual_design["palette_relationship"],
                                         spec.visual_design.get("palette_relationship_source"),
                                         spec.secondary_colour)
    dominance = policy["dominance_relationship"]
    if dominance == "BALANCED":
        roles += "; balanced visual mass"
    elif dominance == "SECONDARY_DOMINANT":
        roles += "; secondary controls main mass"
    else:
        roles += "; primary controls main visual mass; secondary remains substantial"
    keep_phrase = policy["explicit"] and (include_explicit or len(policy["relationship"].split()) <= 12)
    return _join_sections(roles + ";",
                          policy["relationship"] + ";" if keep_phrase else "",
                          "neutrals support shading only.")


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


def _nearby_unrequested_hues(spec: CandidateSpec) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Select nearby chromatic drift on either side of each requested family."""
    requested = [COLOUR_ALIASES.get(str(item["colour"]).casefold(), str(item["colour"]).casefold())
                 for item in spec.visual_design["requested_palette"]]
    allowed = set().union(*(COLOUR_FAMILIES.get(name, {name}) for name in requested))
    requested_rgb = [(name, NAMED_COLOURS[name]) for name in requested if name in NAMED_COLOURS]
    boundaries = []
    for name in requested:
        rgb = NAMED_COLOURS.get(name)
        if not rgb:
            continue
        hue, saturation, _ = colorsys.rgb_to_hsv(*(channel / 255 for channel in rgb))
        if saturation < .18:
            continue
        sides = {-1: [], 1: []}
        for other, other_rgb in NAMED_COLOURS.items():
            if (other in allowed or other in NEUTRAL_NAMES or
                    named_colour_is_requested_tone(other_rgb, requested_rgb)):
                continue
            other_hue, other_saturation, _ = colorsys.rgb_to_hsv(*(channel / 255 for channel in other_rgb))
            signed_distance = (other_hue - hue + .5) % 1 - .5
            distance = abs(signed_distance)
            # Near-identical hues can be ordinary tonal derivatives. Exclude
            # only a bounded neighbouring drift, not every named shade.
            if other_saturation >= .25 and .075 <= distance <= .16:
                sides[1 if signed_distance > 0 else -1].append((distance, other))
        names = tuple(min(side)[1] for side in sides.values() if side)
        if names:
            boundaries.append((name, names))
    return tuple(boundaries)


def _chromatic_boundaries(spec: CandidateSpec) -> str:
    """Describe nearby drift using the same metadata as negative conditioning."""
    clauses = [f"Keep {name} distinctly {name}, avoiding {', '.join(neighbours)} drift"
               for name, neighbours in _nearby_unrequested_hues(spec)]
    return "; ".join(clauses) + ("." if clauses else "")


def _palette_negative(spec: CandidateSpec, stage: str, category: str, history) -> str:
    """Supply only exclusions to engines that expose negative conditioning."""
    base = palette_negative_contract(spec.primary_colour, spec.secondary_colour)
    boundaries = _nearby_unrequested_hues(spec)
    # SD3.5 repeats the negative in three encoder fields, so keep one nearby
    # exclusion per requested pigment there; SDXL can carry both sides.
    nearby = (tuple(dict.fromkeys(neighbours[0] for _, neighbours in boundaries))
              if spec.engine_id == "sd-3-5-medium" else
              tuple(dict.fromkeys(neighbour for _, neighbours in boundaries for neighbour in neighbours)))
    if nearby:
        base += ", " + ", ".join(nearby)
    if category in _PALETTE_FAILURES or _persistent_palette_correction(history, category):
        pressure = {
            "normal": "unrelated chromatic accents",
            "targeted_recovery": "independent off-palette chromatic surfaces",
            "strict_recovery": "substantial independent third-colour regions",
            "rescue": "unrelated chromatic accents, third-colour fields on principal chromatic surfaces",
        }[stage]
        base += ", " + pressure
    return base


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


def _event_context(spec: CandidateSpec) -> str:
    """Carry only the supplied event purpose into SD3.5 conditioning."""
    return _first_words(str(spec.visual_design.get("event_context") or "").strip(), 10)


def _audience_context(spec: CandidateSpec) -> str:
    """Prefer structured audience descriptors, then add distinct explicit context."""
    design = spec.visual_design
    descriptors = design["visual_semantics"].get("audience_descriptors") or []
    parts = []
    for value in (*descriptors, design.get("audience_context"), design.get("age_group")):
        text = str(value or "").strip()
        if text and not any(text.casefold() in part.casefold() or part.casefold() in text.casefold()
                            for part in parts):
            parts.append(text)
    return _first_words(", ".join(parts), 14)


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
              design["theme_reference_treatment"] or
              ", ".join(design["symbolic_motifs"] + design["environmental_motifs"]))
    return " ".join(source.split()[:35]).rstrip(".!? ")


def _motif_policy_instruction(spec: CandidateSpec) -> str:
    return ("Motif as abstract pattern structure; no unrequested literal objects."
            if spec.visual_design.get("motif_literalization_policy") == "motif_only" else "")


_CORRECTIONS = {
    "HUMAN": "Remove people, faces, silhouettes and humanoid shapes from this same background scene.",
    "CHARACTER": "Remove depicted fictional, costumed or mascot figures; retain only the requested non-character visual motifs.",
    "TEXT": "Render uninterrupted visual surfaces and coherent image forms across the scene; keep calm areas integrated into the artwork.",
    "TEXT_OR_LOGO": "Render uninterrupted visual surfaces without words, logos or letter-like clusters.",
    "UNREQUESTED_LITERAL_OBJECT": "Keep the requested motif as pattern structure without adding its unrequested literal source object.",
    "TITLE_PANEL": "Integrate the calm area into the scene itself. Remove separate panels, cards, banners, bordered boxes and blank sheets.",
    "SUBJECT_DRIFT": "Depict the requested visual environment with recognizable subject-specific forms and the supplied visual anchors. Retain this direction's composition and palette.",
    "ATMOSPHERE_DRIFT": "Strengthen the supplied atmosphere and lighting descriptors while preserving the same subject, motif, audience and palette.",
    "MOTIF_DRIFT": "Make the supplied motif structure visibly legible while preserving the same subject, atmosphere, audience and palette.",
    "TREATMENT_DRIFT": "Restore the supplied viewpoint, movement and spatial treatment while preserving subject, motif, audience and palette.",
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


def _abstract_background_subject_retry(spec: CandidateSpec) -> bool:
    design = spec.visual_design
    subject = str(design.get("primary_visual_subject") or "")
    return (design.get("characters_prohibited") is True
            and re.search(r"\b(?:background|backdrop)\b", subject, re.I) is not None
            and not str(design.get("authorized_literal_subject") or "").strip())


def _abstract_subject_correction(spec: CandidateSpec) -> str:
    semantics = spec.visual_design["visual_semantics"]
    motif = ("approved motif structure, " if semantics["motif_descriptors"]
             or semantics["motif_structure_description"] else "")
    return (f"Strengthen the {motif}composition, palette and atmosphere as "
            "background artwork; keep people, characters, logos and emblems absent.")


def _join_sections(*sections: str) -> str:
    return " ".join(section for section in sections if section)


def _first_words(value: str, limit: int) -> str:
    return " ".join(str(value or "").split()[:limit]).rstrip(". ;")


def _background_only(text: str, spec: CandidateSpec) -> str:
    """Keep planning labels out of image-model conditioning."""
    exclusions = [spec.direction_role, *spec.visual_design.get("render_exclusion_text", [])]
    for phrase in sorted(exclusions, key=len, reverse=True):
        phrase = str(phrase).strip()
        if len(phrase) >= 4:
            text = re.sub(r"\b" + re.escape(phrase) + r"\b", "", text, flags=re.I)
    # Remove exclusions before replacing planning nouns with positive visual language.
    text = re.sub(
        r"\b(?:no|without)\s+(?:typography|text|logos?|glyph-like marks)"
        r"(?:\s+(?:or|and)\s+(?:typography|text|logos?|glyph-like marks))*\b",
        "continuous image surfaces", text, flags=re.I,
    )
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
        _motif_policy_instruction(spec),
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
        f"Background inspiration: {subject}.", shared[0], shared[1],
        f"Theme reference treatment: {treatment}." if treatment else "",
        *shared[3:],
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
                           compact_motif, f"Treatment: {_first_words(treatment, 22)}." if treatment else "",
                           _exclude_incompatible(spatial_line, spec),
                           f"Quiet areas: {_first_words(design['title_safe_region_strategy'], 9)}; "
                           f"{_first_words(design['programme_safe_region_strategy'], 9)}.",
                           colour, hierarchy_line,
                           tone_line,
                           f"Correction: {short_correction}" if short_correction else ""),
        )
    if spec.engine_id == "sd-3-5-medium":
        style = str(design.get("visual_style") or "").strip()
        event = _event_context(spec)
        audience = _audience_context(spec)
        event_line = f"Event context: {event}." if event else ""
        audience_line = f"Audience: {audience}." if audience else ""
        brief_palette = ", ".join(str(item["colour"]) for item in design["requested_palette"])
        short_colour = (f"Palette: {brief_palette}; {_first_words(_palette_relationship(spec), 10)}."
                        if brief_palette else "")
        clip_l = _join_sections("Background artwork.", f"Visual subject: {subject}.",
                                f"Anchors: {_first_words(', '.join(anchors), 14)}." if anchors else "",
                                event_line, audience_line)
        clip_g = _join_sections(
            "Background artwork.", f"Visual subject: {subject}.", event_line, audience_line, direction,
            f"Style: {_first_words(style, 8)}." if style else "",
            f"Event atmosphere: {', '.join(design['visual_semantics']['atmosphere_descriptors']) or 'event-appropriate'}.",
            _scene_tone(spec)[0],
            f"Treatment: {_first_words(treatment, 22)}." if treatment else "",
            f"Composition: {_first_words(design['composition_strategy'], 10)}.",
            f"Focal: {_first_words(design['focal_strategy'], 8)}.",
            f"Depth: {_first_words(design['depth_strategy'], 8)}.",
            short_colour,
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


def compile_candidate_prompt(spec: CandidateSpec, attempt: int, reason: str = "", *,
                             concise: bool = False, failure_history=None) -> CompiledPrompt:
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
    treatment = spec.visual_design["theme_reference_treatment"]
    palette = _palette(spec)
    composition = _composition(spec)
    correction = _CORRECTIONS.get(category, "")
    if category == "SUBJECT_DRIFT" and _abstract_background_subject_retry(spec):
        correction = _abstract_subject_correction(spec)
    if category == "PALETTE_OFF":
        correction = _join_sections(correction, _palette_retry_focus(stage))
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
        negative = ("people, faces, characters, readable text, words, letters, lettering, numbers, logos, watermark, "
                    "pseudo-glyphs, signage, poster mockup, bordered panel, blank card, banner")
        if _scene_tone(spec)[1]:
            negative += ", " + _scene_tone(spec)[1]
    else:
        negative = "people, faces, characters, text, words, letters, numbers, glyphs, logos, signage, watermark, title panels"
    if spec.visual_design.get("characters_prohibited"):
        negative += ", depicted human figures, humanoid silhouettes, masked or costumed figures, character portraits"
    if engine_capabilities(spec.engine_id)["negative_prompt_channel"]:
        negative += ", " + _palette_negative(spec, stage, category, failure_history)
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
        (_join_sections(f"Event context: {_event_context(spec)}." if _event_context(spec) else "",
                        f"Audience: {_audience_context(spec)}." if _audience_context(spec) else "",
                        f"Event atmosphere: {', '.join(spec.visual_design['visual_semantics']['atmosphere_descriptors']) or 'event-appropriate'}.",
                        "environment-only composition.")
         if spec.engine_id == "sd-3-5-medium" else _atmosphere(spec)),
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
    figure_contract = ("Environment and abstract motif only; depict no human or costumed figures."
                       if spec.visual_design.get("characters_prohibited") else "")
    positive = _join_sections(positive, figure_contract)
    if not engine_capabilities(spec.engine_id)["negative_prompt_channel"]:
        positive = _join_sections(
            positive,
            _affirmative_palette(spec),
            "Environment-only visual artwork with uninterrupted natural image surfaces; "
            "continuous scene composition, calm tonal breathing room, and recognizable subject forms."
        )
    clip_l = _background_only(clip_l, spec)
    clip_g = _join_sections(_background_only(clip_g, spec), figure_contract)
    t5 = _join_sections(_background_only(t5, spec), figure_contract) if spec.engine_id == "sd-3-5-medium" else None
    channels = {"positive": positive, "clip_l": clip_l, "clip_g": clip_g, "t5": t5 or ""}
    contract = spec.semantic_contract
    probes = {
        "subject": subject,
        "atmosphere": ", ".join(contract["atmosphere_descriptors"]),
        "audience": ", ".join(contract["audience_descriptors"]),
        "motif": contract["motif_structure_description"] or ", ".join(contract["motif_descriptors"]),
        "treatment": spec.visual_design["theme_reference_treatment"],
        "palette": ", ".join(str(item["colour"]) for item in contract["palette"]),
        "scene_family": contract["scene_family"],
    }
    if spec.engine_id == "sd-3-5-medium":
        probes["event_context"] = _event_context(spec)
        probes["audience"] = _audience_context(spec)
    coverage = {name: [channel for channel, value in channels.items()
                       if probe and probe.casefold() in value.casefold()]
                for name, probe in probes.items()}
    coverage["incompatible_treatments"] = ["negative"] if spec.engine_id != "flux-2" and contract["incompatible_treatments"] else []
    effective = ("positive",) if spec.engine_id == "flux-2" else (
        ("clip_g", "t5") if spec.engine_id == "sd-3-5-medium" else ("clip_g",))
    truncated = any(probes[key] and not any(channel in coverage[key] for channel in effective)
                    for key in ("subject", "atmosphere", "audience", "motif", "palette", "treatment"))
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
    return _concise_runtime_prompt(result, spec, attempt, reason, failure_history) if concise else result


def runtime_retry_category(reason: str) -> str:
    """Keep the classifier's failed dimension for production retry prompts."""
    raw = reason.strip().upper().split(":", 1)[0]
    category = raw if raw in _RETRY_CORRECTIONS or raw in {"TEXT", "PALETTE", "SUBJECT", "MOTIF", "VISION_UNAVAILABLE"} else failure_category(reason)
    if category == "INITIAL":
        return "INITIAL"
    if category in {"TEXT", "TEXT_OR_LOGO", "OCR_TIMEOUT", "OCR_UNAVAILABLE", "TITLE_PANEL"}:
        return "TEXT_OR_LOGO"
    if category == "PALETTE":
        return "PALETTE_OFF"
    if category == "SUBJECT":
        return "SUBJECT_DRIFT"
    if category == "MOTIF":
        return "MOTIF_DRIFT"
    if category in {"GENERATION_FAILURE", "RUNTIME", "VISION_UNAVAILABLE"}:
        return "GENERATION_FAILURE"
    return category if category in _RETRY_CORRECTIONS else "UNKNOWN"


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
    if attempt <= 3:
        return "CORRECTION"
    if attempt <= 6:
        return "TARGETED_RECOVERY"
    if attempt <= 8:
        return "STRICT_RECOVERY"
    return "RESCUE"


_RETRY_CORRECTIONS = {
    "TEXT_OR_LOGO": "Use continuous visual surfaces without glyph- or signage-like marks.",
    "UNREQUESTED_LITERAL_OBJECT": "Keep the supplied motif abstract; omit its unrequested literal source object.",
    "HUMAN": "Render only the requested non-human environment and motifs, without figures.",
    "CHARACTER": "Render only the requested non-character environment and abstract motifs, without depicted figures or mascots.",
    "PALETTE_OFF": "Strengthen the requested colour families and relationship while retaining shading character.",
    "PALETTE_ROLE_DRIFT": "Restore the requested primary-to-secondary visual-mass relationship.",
    "PRIMARY_MISSING": "Increase the requested primary colour across substantial surfaces without weakening the secondary.",
    "SECONDARY_MISSING": "Increase the requested secondary colour in several substantial regions without weakening the primary.",
    "SUBJECT_DRIFT": "Make the supplied subject forms and visual anchors recognizable.",
    "ATMOSPHERE_DRIFT": "Strengthen the explicitly supplied atmosphere without changing the scene.",
    "AUDIENCE_DRIFT": "Strengthen the supplied audience-appropriate visual treatment without changing the scene.",
    "MOTIF_DRIFT": "Make the supplied motif topology visible without adding related objects or creatures.",
    "TREATMENT_DRIFT": "Restore the supplied viewpoint and movement through the same scene without adding a character or unrelated object.",
    "DENSITY_DRIFT": "Restore the requested density and visual hierarchy; reduce uniformly scattered micro-detail.",
    "FLAT": "Strengthen coherent medium and large subject forms and depth.",
    "GRADIENT": "Restore discrete subject-specific forms, depth and tonal structure.",
    "RASTER": "Use coherent broad forms and varied edges instead of repetitive scanline or grid structure.",
    "NOISE": "Use coherent forms and controlled detail rather than scattered noise.",
    "WEAK_OUTER_STRUCTURE": "Add meaningful subject structure where this direction permits, retaining calm regions.",
    "DENSE_SAFE_REGION": "Move detail away from integrated calm regions while retaining this direction.",
    "DIRECTION_DUPLICATE": "Vary focal placement, viewpoint, visual mass and motif placement within this direction.",
    "PROTECTED_REGION_INTRUSION": "Move motifs away from reserved quiet regions within this direction.",
    "WEAK_TEMPLATE_STRUCTURE": "Strengthen this direction's requested composition with subject-specific forms.",
    "WEAK_DECORATIVE_DESIGN": "Strengthen subject-specific decorative structure outside quiet regions.",
    "TEMPLATE_DRIFT": "Restore this direction's requested composition and subject-specific structure.",
    "BOILERPLATE_COMPOSITION": "Vary scale, depth, focal placement and visual mass within this direction.",
    "RUNTIME": "Retry the same semantic design with simpler technical execution.",
    "GENERATION_FAILURE": "Retry the same semantic design with simpler technical execution.",
    "UNKNOWN": "Retry this same direction faithfully without introducing new content.",
}


def _palette_retry_focus(stage: str) -> str:
    return {
        "normal": "Keep requested colour families on principal design surfaces; suppress unrelated chromatic accents while retaining shading.",
        "targeted_recovery": "Keep subject, motif and composition; confine chromatic design to requested families and remove independent third-colour regions.",
        "strict_recovery": "Keep subject, motif and composition; make requested families the only substantial chromatic fields, with no independent third-colour regions.",
        "rescue": "Requested families must occupy the principal chromatic surfaces; unrelated chromatic accents must be absent or negligible. Retain permitted neutral shading, subject, motif and composition.",
    }[stage]


def _runtime_correction(category: str, stage: str) -> str:
    requirement = _RETRY_CORRECTIONS[category]
    if category == "PALETTE_OFF":
        requirement = _join_sections(requirement, _palette_retry_focus(stage))
    emphasis = {
        "normal": "Correct the failed dimension conservatively:",
        "targeted_recovery": "Make the failed requirement substantially more visible:",
        "strict_recovery": "Simplify competing incidental detail and make the failed requirement unmistakable:",
        "rescue": "Use the simplest semantically faithful rendering of this same direction; make the failed requirement unmistakable:",
    }[stage]
    return f"{emphasis} {requirement} Preserve all original requested semantics and every unaffected visual dimension."


_PALETTE_FAILURES = ("PALETTE_OFF", "PALETTE_ROLE_DRIFT", "PRIMARY_MISSING", "SECONDARY_MISSING")


def _persistent_palette_correction(history, latest: str) -> str:
    """Keep unresolved palette drift alongside a newer, different failure."""
    if history is None or latest in _PALETTE_FAILURES:
        return ""
    counts = getattr(history, "counts", history if isinstance(history, dict) else {})
    if not any(int(counts.get(category, 0)) for category in _PALETTE_FAILURES):
        return ""
    return "Persistent palette recovery: restore the requested colour-family relationship and visual-mass roles."


def _intrusion_contract(history, latest: str) -> tuple[str, str]:
    """Keep a confirmed figure exclusion after a different retry failure."""
    counts = getattr(history, "counts", history if isinstance(history, dict) else {}) or {}
    human = latest == "HUMAN" or int(counts.get("HUMAN", 0)) > 0
    character = latest == "CHARACTER" or int(counts.get("CHARACTER", 0)) > 0
    if not (human or character):
        return "", ""
    positive = ("Keep this same background scene and permitted abstract motifs; "
                "depict no human, mascot or costumed figures.")
    exclusions = []
    if human:
        exclusions.extend(("depicted people", "human figures", "human silhouettes"))
    if character:
        exclusions.extend(("depicted costumed figures", "character portraits", "mascot figures"))
    return positive, ", ".join(exclusions)


def _structural_treatment_features(text: str) -> set[str]:
    text = text.casefold()
    patterns = {
        "viewpoint": r"\b(?:first.person|viewpoint|perspective|point.of.view|eye.level|overhead|aerial)\b",
        "depth": r"\b(?:reced\w*|depth|foreground|distance|vanish\w*|layer\w*)\b",
        "flow": r"\b(?:flow\w*|curv\w*|mov\w*|motion|travers\w*|sweep\w*|leading)\b",
        "viewpoint:first_person": r"\bfirst.person\b|\beye.level\b",
        "viewpoint:aerial": r"\b(?:aerial|overhead|bird.s.eye)\b",
        "depth:receding": r"\b(?:reced\w*|vanish\w*)\b",
        "depth:shallow": r"\bshallow\b",
        "flow:curved": r"\bcurv\w*\b",
        "flow:diagonal": r"\bdiagonal\b",
    }
    return {name for name, pattern in patterns.items() if re.search(pattern, text)}


def _treatment_covered(source: str, channel: str) -> bool:
    required = _structural_treatment_features(source)
    return bool(source and (required <= _structural_treatment_features(channel)
                            if required else _first_words(source, 6).casefold() in channel.casefold()))


def _concise_runtime_prompt(compiled: CompiledPrompt, spec: CandidateSpec,
                            attempt: int, reason: str, failure_history=None) -> CompiledPrompt:
    design = spec.visual_design
    semantics = design["visual_semantics"]
    subject = _subject(spec)
    anchors = ", ".join(design["visual_anchors"])
    mood = ", ".join(semantics["atmosphere_descriptors"])
    audience = ", ".join(semantics["audience_descriptors"])
    sd35_event = _event_context(spec) if spec.engine_id == "sd-3-5-medium" else ""
    sd35_audience = _audience_context(spec) if spec.engine_id == "sd-3-5-medium" else ""
    motif = _direction_motif(spec)
    colours = ", ".join(str(item["colour"]) for item in design["requested_palette"])
    relationship = (_concise_palette_relationship(spec, include_explicit=False) if spec.engine_id != "flux-2"
                    else _first_words(_palette_relationship(spec), 10))
    full_relationship = (_concise_palette_relationship(spec) if spec.engine_id != "flux-2"
                         else relationship)
    composition = _spatial_instruction(design)
    strength = retry_strength_level(attempt)
    retry_category = runtime_retry_category(reason)
    full_correction = (_runtime_correction(retry_category, compiled.retry_stage)
                       if attempt > 1 else "")
    if full_correction and retry_category == "SUBJECT_DRIFT" and _abstract_background_subject_retry(spec):
        full_correction = f"Correct the failed dimension conservatively: {_abstract_subject_correction(spec)}"
    short_correction = (_palette_retry_focus(compiled.retry_stage)
                        if runtime_retry_category(reason) == "PALETTE_OFF" and full_correction else
                        full_correction.split(": ", 1)[-1].split(" Preserve all original", 1)[0]
                        if full_correction else "")
    correction = (f"{compiled.retry_stage}: " + _first_words(short_correction, 18)
                  if full_correction else "")
    persistent_palette = (_persistent_palette_correction(failure_history, runtime_retry_category(reason))
                          if spec.engine_id in {"sdxl-1-0", "sd-3-5-medium"} and attempt > 1 else "")
    intrusion_positive, intrusion_negative = _intrusion_contract(
        failure_history, runtime_retry_category(reason) if attempt > 1 else "INITIAL")
    authorized_subject = str(design.get("authorized_literal_subject") or "")
    figure_terms = r"\b(?:\w*hero|humanoid|character|person|human|mascot|costumed|figure)\b"
    figure_subject_pressure = bool(re.search(
        figure_terms, authorized_subject, re.I)) if design.get("characters_prohibited") else False
    figure_drift = (retry_category == "SUBJECT_DRIFT" and bool(re.search(figure_terms, reason, re.I)))
    sd35_motif_recovery = (spec.engine_id == "sd-3-5-medium" and attempt > 1
                           and design.get("characters_prohibited")
                           and (not authorized_subject or figure_subject_pressure)
                           and bool(motif)
                           and (bool(intrusion_positive)
                                or figure_drift
                                or retry_category == "SUBJECT_DRIFT" and _abstract_background_subject_retry(spec)))
    sdxl_motif_recovery = (spec.engine_id == "sdxl-1-0" and attempt > 1
                           and design.get("characters_prohibited")
                           and (not authorized_subject or figure_subject_pressure)
                           and bool(motif)
                           and (bool(intrusion_positive) or retry_category in _PALETTE_FAILURES
                                or retry_category == "SUBJECT_DRIFT"))
    if sdxl_motif_recovery or sd35_motif_recovery:
        subject = f"Abstract background pattern with {_first_words(motif, 12)}"
    sdxl_topology = (
        "Keep approved motifs visible as distributed, layered background forms, not a single figure silhouette."
        if sdxl_motif_recovery else "")
    sd35_topology = (
        "Approved motifs: distributed layered background forms, never a body silhouette."
        if sd35_motif_recovery else "")
    sdxl_figure_negative = (
        "face, head, torso, body, arms, legs, hands, costume, masked figure, heroic pose"
        if sdxl_motif_recovery and intrusion_positive else "")
    figure_contract = ("Environment and abstract motif only; depict no human or costumed figures."
                       if design.get("characters_prohibited") and not intrusion_positive else "")
    subject_line = f"Visual subject: {subject}."
    short_subject_line = f"Visual subject: {_first_words(subject, 16 if sd35_motif_recovery else 9)}."
    anchor_line = f"Anchors: {_first_words(anchors, 14)}." if anchors and not sd35_motif_recovery else ""
    mood_line = (f"Event atmosphere: {_first_words(mood, 7)}."
                 if spec.engine_id == "sd-3-5-medium" and mood else
                 f"Mood: {_first_words(mood, 7)}." if mood else "")
    if spec.engine_id == "sd-3-5-medium" and not mood_line:
        mood_line = "Event atmosphere: event-appropriate."
    audience_line = f"Audience: {_first_words(audience, 5)}." if audience else ""
    sd35_event_line = f"Event context: {sd35_event}." if sd35_event else ""
    sd35_audience_line = f"Audience: {sd35_audience}." if sd35_audience else ""
    motif_line = f"Motif: {_first_words(motif, 10)}." if motif else ""
    motif_policy_line = (_first_words(_motif_policy_instruction(spec), 12)
                         if design.get("motif_literalization_policy") == "motif_only" else "")
    treatment = design["theme_reference_treatment"]
    if (sdxl_motif_recovery or sd35_motif_recovery) and figure_subject_pressure:
        treatment = re.sub(r"\b(?:\w*hero(?:-like)?|humanoid(?:-like)?|character(?:-like)?|"
                           r"person(?:-like)?|human(?:-like)?|mascot(?:-like)?|costumed?)\b",
                           "", treatment, flags=re.I)
    treatment_line = (f"Treatment: {_first_words(treatment, 22)}."
                      if treatment else "")
    short_treatment = _first_words(treatment, 10)
    for feature, label in (("viewpoint:first_person", "first-person viewpoint"),
                           ("viewpoint:aerial", "aerial viewpoint"),
                           ("depth:receding", "receding depth"),
                           ("depth:shallow", "shallow depth"),
                           ("flow:curved", "curved flow"),
                           ("flow:diagonal", "diagonal flow")):
        if feature in _structural_treatment_features(treatment) and feature not in _structural_treatment_features(short_treatment):
            short_treatment = _join_sections(short_treatment, label)
    for feature, label in (("viewpoint", "viewpoint"), ("depth", "layered depth"),
                           ("flow", "movement flow")):
        if feature in _structural_treatment_features(treatment) and feature not in _structural_treatment_features(short_treatment):
            short_treatment = _join_sections(short_treatment, label)
    short_treatment_line = f"Treatment: {short_treatment}." if short_treatment else ""
    style_line = (f"Style: {_first_words(design.get('visual_style'), 8)}."
                  if design.get("visual_style") else "")
    palette_line = (f"Palette: {full_relationship}" if spec.engine_id != "flux-2" else
                    f"Palette: {colours}; {relationship}; primary visual mass, secondary accents; neutrals support.") if colours else ""
    relationship_words = _palette_relationship(spec).split() if spec.engine_id == "flux-2" else ()
    short_relationship = (" ".join(relationship_words) if len(relationship_words) <= 10 else
                          " ".join((*relationship_words[:4], *relationship_words[-2:]))) if relationship_words else relationship
    short_palette_line = (f"Palette: {colours}; {short_relationship}; "
                          "primary visual mass, secondary accents; neutrals support." if spec.engine_id == "flux-2" else
                          f"Palette: {short_relationship}" if colours else "")
    palette_l_line = (f"Palette: {full_relationship}" if spec.engine_id == "sdxl-1-0" else
                      f"Primary {design['requested_palette'][0]['colour']}; "
                      + (f"secondary {design['requested_palette'][1]['colour']}; "
                         if len(design['requested_palette']) > 1 else "")
                      + f"relationship: {relationship}."
                      if colours else "")
    spatial_line = f"Composition: {composition}." if composition else ""
    light_line = f"Lighting: {_first_words(design.get('lighting_language'), 8)}." if design.get("lighting_language") else ""
    sd35_recovery_correction = (f"Correction: {_first_words(short_correction, 8)}."
                                if sd35_motif_recovery and short_correction else "")
    clip_l = (_join_sections(short_subject_line, palette_l_line, motif_line) if spec.engine_id == "sdxl-1-0"
              else _join_sections(short_subject_line, anchor_line,
                                  sd35_event_line, sd35_audience_line)
              if spec.engine_id == "sd-3-5-medium" else _join_sections(subject_line, anchor_line))
    clip_g = (_join_sections(short_subject_line, mood_line, audience_line, short_treatment_line,
                            spatial_line, motif_policy_line, short_palette_line,
                            persistent_palette, figure_contract, intrusion_positive, sdxl_topology, correction)
              if spec.engine_id == "sdxl-1-0" else
              _join_sections(short_subject_line, short_treatment_line, short_palette_line,
                             sd35_topology, sd35_recovery_correction) if sd35_motif_recovery else
              _join_sections(short_subject_line, sd35_event_line, sd35_audience_line, mood_line,
                             motif_line if sd35_motif_recovery else "", short_treatment_line,
                             spatial_line, short_palette_line, persistent_palette,
                             figure_contract, intrusion_positive, sd35_topology, correction) if spec.engine_id == "sd-3-5-medium" else
              _join_sections(subject_line, mood_line, audience_line, motif_line, motif_policy_line,
                             treatment_line, palette_line, spatial_line, style_line, light_line,
                             figure_contract, intrusion_positive))
    positive_source = clip_g.replace(short_subject_line, subject_line, 1).replace(short_treatment_line, treatment_line, 1) if spec.engine_id != "flux-2" else clip_g
    positive = _join_sections(positive_source.replace(correction, "") if correction else positive_source,
                             motif_line if motif_line not in clip_g else "",
                             mood_line if sd35_motif_recovery else "",
                             (f"Palette relationship: {_palette_relationship(spec)}."
                              if spec.engine_id != "flux-2" and palette_relationship_policy(
                                  design["palette_relationship"], design.get("palette_relationship_source"),
                                  spec.secondary_colour)["explicit"] else ""),
                             full_correction if full_correction else "",
                             "Continuous background artwork with integrated calm visual space.")
    if spec.engine_id == "flux-2":
        positive = _join_sections(positive, _affirmative_palette(spec),
                                 "Environment-only composition; uninterrupted visual surfaces.")
    t5 = _join_sections(subject_line, anchor_line, sd35_event_line, sd35_audience_line,
                       mood_line, audience_line if spec.engine_id != "sd-3-5-medium" else "",
                       motif_line, motif_policy_line,
                       treatment_line, palette_line, spatial_line, style_line, light_line,
                       persistent_palette, figure_contract, intrusion_positive, sd35_topology, correction)
    def clean(value: str) -> str:
        value = _background_only(value, spec)
        return re.sub(r"\b(?:8k|4k|masterpiece|ultra[- ]detailed|trending on artstation|"
                      r"award[- ]winning|unreal engine|octane render|cinematic masterpiece)\b",
                      "", value, flags=re.I).strip()
    cleaned = {name: clean(value) for name, value in
               (("positive", positive), ("clip_l", clip_l), ("clip_g", clip_g), ("t5", t5))}
    if spec.engine_id == "flux-2":
        cleaned["positive"] = _join_sections(
            cleaned["positive"],
            "Purely pictorial background surfaces; no words, letters, numbers, logos, signage or watermarks.",
        )
    coverage = {**compiled.conditioning_coverage,
                "treatment": [channel for channel in (("positive",) if spec.engine_id == "flux-2" else
                                                        ("positive", "clip_g", "t5") if spec.engine_id == "sd-3-5-medium"
                                                        else ("positive", "clip_g"))
                              if _treatment_covered(treatment, cleaned[channel])]}
    if spec.engine_id == "sd-3-5-medium":
        coverage["audience"] = [channel for channel, value in
                                (("clip_l", clip_l), ("clip_g", clip_g), ("t5", t5))
                                if sd35_audience and sd35_audience.casefold() in cleaned[channel].casefold()]
        coverage["event_context"] = [channel for channel, value in
                                     (("clip_l", clip_l), ("clip_g", clip_g), ("t5", t5))
                                     if sd35_event and sd35_event.casefold() in cleaned[channel].casefold()]
    negative = compiled.negative
    if compiled.negative_prompt_channel_active and intrusion_negative:
        negative = f"{negative}, {intrusion_negative}"
    if sdxl_figure_negative:
        negative = f"{negative}, {sdxl_figure_negative}"
    updated: CompiledPrompt = replace(compiled, positive=cleaned["positive"], negative=negative,
                                      clip_l=cleaned["clip_l"],
                                      clip_g=cleaned["clip_g"], t5=cleaned["t5"] if spec.engine_id == "sd-3-5-medium" else None,
                                      failure_category=runtime_retry_category(reason), retry_stage=compiled.retry_stage,
                                      retry_strength_level=strength, conditioning_coverage=coverage,
                                      truncated=bool(treatment and not any(channel in coverage["treatment"] for channel in
                                                   (("positive",) if spec.engine_id == "flux-2" else
                                                    ("clip_g", "t5") if spec.engine_id == "sd-3-5-medium" else
                                                    ("clip_g",)))))
    return updated
