"""Compile each native-model attempt from the same immutable brief."""

from __future__ import annotations

from dataclasses import dataclass
import colorsys
import re

from app.services.candidate_spec import CandidateSpec
from app.services.background_policy import COLOUR_ALIASES, COLOUR_FAMILIES, NAMED_COLOURS


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
    if "TITLE_PANEL" in reason.upper() or "TEMPLATE_PANEL" in reason.upper():
        return "TITLE_PANEL"
    if "SUBJECT_DRIFT" in reason.upper():
        return "SUBJECT_DRIFT"
    if "BRIEF_FIDELITY_DRIFT" in reason.upper():
        return "BRIEF_FIDELITY_DRIFT"
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


def _palette(spec: CandidateSpec) -> str:
    design = spec.visual_design
    colours = [str(item["colour"]) for item in design["requested_palette"]]
    if not colours:
        return "the palette described by the brief"
    relationship = design["palette_relationship"] or "respect the requested colour roles"
    mode = design["palette_mode"]
    if mode in {"monochrome", "achromatic"} or set(colours) == {"black", "white"}:
        policy = "strictly achromatic black, white and grey tonal variation; no chromatic colour"
    elif mode == "strict":
        policy = "only requested colour families, apart from microscopic anti-aliasing tolerance"
    else:
        policy = ("requested colours control the chromatic design; bounded black, charcoal, grey and white "
                  "rendering detail supports linework, shadows, reflections and highlights; "
                  "no meaningful third chromatic design colour")
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
        policy = "requested chromatic colours dominate; neutral shading subordinate"
    return f"Requested palette: {colours}; {policy}. {_chromatic_boundaries(spec)}".strip()


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
    design = spec.visual_design
    source = " ".join(str(part or "") for part in (design.get("mood"), spec.creative_description,
                                                     design.get("event_context"), spec.event_type))
    words = re.findall(r"\b(?:joyful|playful|energetic|high-energy|refined|formal|luxurious|prestigious|contemporary|restrained|dignified|calm|polished|professional|organic|warm|elegant|celebratory|festive)\b", source, re.I)
    mood = ", ".join(dict.fromkeys(word.casefold() for word in words))
    if not mood:
        event = (spec.event_type + " " + spec.theme).casefold()
        if any(word in event for word in ("child", "kid", "school")):
            mood = "playful, joyful, energetic"
        elif any(word in event for word in ("black tie", "black-tie", "gala")):
            mood = "refined, formal, luxurious"
        elif "award" in event:
            mood = "prestigious, contemporary, polished"
        elif "memorial" in event:
            mood = "restrained, dignified, calm"
        elif "corporate" in event or "conference" in event:
            mood = "polished, professional"
        elif "botanical" in event or "garden" in event:
            mood = "organic, warm, elegant"
    audience = str(design.get("audience_context") or "")
    child_safe = any(word in (spec.event_type + " " + audience).casefold() for word in ("child", "kid", "age 7", "age 8", "age 9", "age 10"))
    if child_safe:
        mood = (mood + ", playful child-friendly" if mood else "playful child-friendly")
    if not mood:
        mood = "event-appropriate"
    event_label = str(design.get("event_context") or spec.event_type).casefold()
    if "birthday" in event_label:
        purpose = "birthday celebration"
    elif "conference" in event_label or "corporate" in event_label:
        purpose = "professional gathering"
    elif "memorial" in event_label:
        purpose = "remembrance gathering"
    elif any(word in event_label for word in ("wedding", "bridal", "shower", "gala", "awards")):
        purpose = "celebration gathering"
    else:
        purpose = "event gathering" if event_label else ""
    context = f"Visual purpose: {purpose}; " if purpose else ""
    return f"{context}Event atmosphere: {mood}; no people or characters."


def _motif_structure(spec: CandidateSpec) -> str:
    motifs = [*spec.visual_design["symbolic_motifs"], *spec.visual_design["environmental_motifs"]]
    treatment = spec.theme_reference_treatment
    source = " ".join([treatment, *motifs]).casefold()
    clues = []
    for key, detail in (("web", "interconnected strands, cross-link arcs, network cells and curved connecting segments visibly linking radial strands into web-like geometry"),
                        ("botanic", "leaves, stems and organic branching"),
                        ("floral", "petals and branching stems"),
                        ("fabric", "folds, drape and flowing surfaces"),
                        ("circuit", "panels, circuitry and luminous grids"),
                        ("technolog", "panels, circuitry and luminous grids")):
        if key in source:
            clues.append(detail)
    return f"Motif structure: {', '.join(clues)}." if clues else ""


def _scene_tone(spec: CandidateSpec) -> tuple[str, str]:
    """Translate stated atmosphere into lighting and incompatible treatments."""
    source = " ".join(str(value or "") for value in (
        spec.visual_design.get("mood"), spec.creative_description, spec.event_type
    )).casefold()
    if any(word in source for word in ("joyful", "playful", "high-energy", "cheerful", "festive")):
        return ("Uplifting clean graphic energy, vibrant daylight or bright illustrative lighting, friendly celebratory atmosphere.",
                "ominous, grim, dystopian, horror, post-apocalyptic, burning city, dark noir")
    if any(word in source for word in ("memorial", "dignified", "restrained")):
        return ("Gentle, calm and dignified lighting with restrained visual energy.",
                "neon party, playful, celebratory spectacle")
    if any(word in source for word in ("formal", "luxurious", "black-tie", "black tie")):
        return ("Refined controlled lighting and composed formal atmosphere.",
                "childish, cartoonish, chaotic")
    if "award" in source or "prestigious" in source:
        return ("Polished contemporary lighting and prestigious celebratory atmosphere.",
                "childish, shabby, chaotic")
    if any(word in source for word in ("corporate", "professional", "conference")):
        return ("Clear polished lighting and professional visual atmosphere.",
                "whimsical, childish, fantasy-heavy")
    return "", ""


def _motif_avoidances(spec: CandidateSpec) -> str:
    source = " ".join([spec.theme_reference_treatment,
                       *spec.visual_design["symbolic_motifs"],
                       *spec.visual_design["environmental_motifs"]]).casefold()
    return "explosion, radial sparks, energy burst, disconnected rays, starburst" if "web" in source else ""


def _soften_playful_tone(text: str, spec: CandidateSpec) -> str:
    """Remove aggressive planning language only when the brief asks for play."""
    if not _scene_tone(spec)[0].startswith("Uplifting"):
        return text
    replacements = {
        "dramatic atmospheric perspective": "clear graphic depth",
        "fractured perimeter": "clean shaped perimeter",
        "spikes": "bold shapes",
        "speed-line energy": "celebratory motion",
        "dynamic focal points": "playful focal rhythm",
        "violent burst": "celebratory motion",
        "dark cinematic": "bright illustrative",
    }
    for original, friendly in replacements.items():
        text = re.sub(re.escape(original), friendly, text, flags=re.I)
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
            "Preserve intentional negative space for later typography "
            "according to "
            + "; ".join(safe_regions)
        )
    else:
        parts.append(
            "Preserve intentional negative space for later typography "
            "where the composition naturally supports it"
        )

    parts.append(
        "Use natural visual hierarchy, breathing room and spatial balance "
        "so later information can remain readable"
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
    "TEXT": "Remove readable text and pseudo-text. This is background artwork, not a poster or invitation. Render no glyphs, characters, alphabetic marks, numeric marks, signage or pseudo-typography. Leave the intended typography areas visually empty.",
    "TITLE_PANEL": "Integrate the calm area into the scene itself. Remove separate panels, cards, banners, bordered boxes and blank sheets.",
    "SUBJECT_DRIFT": "Depict the requested visual environment with recognizable subject-specific forms and the supplied visual anchors. Retain this direction's composition and palette.",
    "PALETTE_OFF": "Correct the requested colour relationship while preserving this design's texture, shading and details.",
    "PRIMARY_MISSING": "Make the requested primary colour dominant across substantial surfaces of this same design.",
    "SECONDARY_MISSING": "Increase the requested secondary colour in several substantial visible regions of this same design.",
    "GRADIENT": "Give this scene distinct subject-specific forms, tonal variation and visible depth.",
    "FLAT": "Strengthen the requested subject silhouette with coherent medium and large forms, clear depth and a visible focal structure.",
    "RASTER": "Render this scene with coherent forms and clean, varied edge structure.",
    "NOISE": "Render this scene with coherent forms and clear material detail.",
    "WEAK_OUTER_STRUCTURE": "Strengthen coherent medium and large subject forms and depth in the areas permitted by this direction while keeping its quiet regions readable.",
    "DENSE_SAFE_REGION": "Relocate detail away from the reserved typography regions while preserving this direction's own composition strategy.",
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
    )
    safety = ("Text-free background artwork with a clear visual subject and natural low-detail breathing room. "
              "No typography or glyph-like marks. Quiet areas remain part of the scene; "
              "no separate panels, cards, banners, placards or signboards.")
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
        f"Correction: {correction}" if correction else "",
        f"Visual design: {spec.original_positive_prompt}.",
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
        structure_line = _join_sections(
            f"Composition: {_first_words(design['composition_strategy'], 12)}.",
            f"Focal: {_first_words(design['focal_strategy'], 10)}.",
            f"Depth: {_first_words(design['depth_strategy'], 9)}.")
        global_line = _join_sections(
            f"Style: {_first_words(design.get('visual_style'), 8)}." if design.get('visual_style') else "",
            f"Motifs: {_first_words(', '.join(motifs) or direction_detail, 12)}.")
        return (
            _join_sections("Background artwork.", subject_line, atmosphere_line,
                           _first_words(tone_line, 12), _soften_playful_tone(direction, spec),
                           _soften_playful_tone(structure_line, spec), colour,
                           f"Correction: {short_correction}" if short_correction else ""),
            _join_sections("Background artwork.", subject_line, atmosphere_line,
                           _first_words(tone_line, 12), _soften_playful_tone(direction, spec),
                           _soften_playful_tone(structure_line, spec),
                           _soften_playful_tone(global_line, spec), _motif_structure(spec),
                           f"Quiet areas: {design['title_safe_region_strategy']}; {design['programme_safe_region_strategy']}.", colour,
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
              structure, material)
    return (
        _join_sections("Background artwork.", *common, colour),
        _join_sections(
            "Background artwork.",
            *common, colour,
            f"Correction: {short_correction}" if short_correction else "",
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
        negative = ("people, faces, text, words, letters, numbers, typography, caption, "
                    "title, logo, watermark, signage, glyphs, pseudo-text, poster mockup, "
                    "no blank title panel, no white card, no bordered text box, "
                    "no poster frame, no banner, no placard, no signboard, no empty paper rectangle")
        negative += ", " + ", ".join(value for value in (_scene_tone(spec)[1], _motif_avoidances(spec)) if value)
    else:
        negative = "people, faces, characters, text, glyphs, logos, title panels"
    extra_negative = {
        "RASTER": ", scanlines, raster grid",
        "NOISE": ", grain, noise",
    }
    negative += extra_negative.get(category, "")
    if spec.engine_id == "sd-3-5-medium":
        negative += ", " + ", ".join(value for value in (_scene_tone(spec)[1], _motif_avoidances(spec)) if value)
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
        "Decorative background artwork with no typography or glyph-like marks.",
        f"Design language: {_design_language(spec)}." if _design_language(spec) else "",
        f"Theme treatment: {treatment}." if treatment else "",
        f"Correction: {correction}" if correction else "",
    ) if section)
    return CompiledPrompt(
        positive=_background_only(_soften_playful_tone(positive, spec), spec),
        negative=negative,
        clip_l=_background_only(clip_l, spec),
        clip_g=_background_only(clip_g, spec),
        t5=_background_only(t5, spec) if spec.engine_id == "sd-3-5-medium" else None,
        failure_category=category,
        retry_stage=stage,
        spec_sha256=spec.spec_sha256,
    )
