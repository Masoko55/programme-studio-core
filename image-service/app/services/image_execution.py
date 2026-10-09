from __future__ import annotations

import asyncio
import json
import logging
import re

import httpx

from app.config.settings import settings

from app.services.background_policy import (
    NAMED_COLOURS,
    _colour_family_members,
    requested_palette_names,
)

from app.engines.registry import (
    get_engine,
)

from app.services.comfyui_client import (
    ComfyUIClient,
    ComfyUIError,
    SubmissionUncertain,
)

from app.services.execution_plan import (
    build_execution_plan,
)

from app.services.gpu_lease import (
    GPULease,
)

from app.services.job_persistence import (
    load_image_job_state,
    persist_image_job_state,
    update_output,
)

from app.services.prompt_repository import (
    get_direction,
    load_prompts_document,
)
from app.services.candidate_spec import build_candidate_spec
from app.services.prompt_compiler import compile_candidate_prompt, failure_category, retry_stage


logger = logging.getLogger(
    "uvicorn.error"
)


# ============================================================
# Candidate record helpers
# ============================================================


def _candidate_record(
    reference_number: str,
    engine_id: str,
    direction_id: str,
) -> dict:
    path = (
        settings.programme_data_path
        / reference_number
        / "backgrounds"
        / engine_id
        / (
            f"image-"
            f"{direction_id.lower()}"
            ".json"
        )
    )

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {}


def _candidate_was_rejected(
    reference_number: str,
    engine_id: str,
    direction_id: str,
) -> bool:
    record = (
        _candidate_record(
            reference_number,
            engine_id,
            direction_id,
        )
    )

    return bool(
        record.get(
            "rejection_reason"
        )
    )


def _rejection_reason(
    reference_number: str,
    engine_id: str,
    direction_id: str,
) -> str:
    record = (
        _candidate_record(
            reference_number,
            engine_id,
            direction_id,
        )
    )

    return str(
        record.get(
            "rejection_reason"
        )
        or record.get(
            "error"
        )
        or ""
    )


# ============================================================
# General helpers
# ============================================================


def _normalise_colour(
    value: str | None,
) -> str:
    return (
        str(
            value
            or ""
        )
        .strip()
        .lower()
    )


# ============================================================
# Failure classification
# ============================================================


def _is_palette_failure(
    reason: str,
) -> bool:
    """
    Detect ONLY actual palette failures.

    Important:
    Do not match the generic words "colour" or "color".

    A visual-quality message such as:

        "smooth gradient or colour wash"

    is a quality failure, not a palette failure.
    """

    reason_lower = (
        reason.lower()
    )

    terms = (
        "off-palette",
        "off palette",
        "palette ratio",
        "palette drift",
        "left the requested black-and-white palette",
        "black-and-white palette",
        "monochrome ratio",
        "colours that were not requested",
        "colors that were not requested",
        "unrequested colour",
        "unrequested color",
        "unrequested neutral",
        "requested primary colour family",
        "requested primary color family",
        "requested secondary colour family",
        "requested secondary color family",
        "primary family ratio",
        "secondary family ratio",
    )

    return any(
        term
        in reason_lower
        for term
        in terms
    )


def _is_human_failure(
    reason: str,
) -> bool:
    reason_lower = (
        reason.lower()
    )

    terms = (
        "human",
        "person",
        "people",
        "face",
        "figure",
        "portrait",
        "body",
        "silhouette",
    )

    return any(
        term
        in reason_lower
        for term
        in terms
    )


def _is_text_failure(
    reason: str,
) -> bool:
    reason_lower = (
        reason.lower()
    )

    terms = (
        "ocr",
        "readable text",
        "contains text",
        "letter",
        "word",
        "number",
        "logo",
    )

    return any(
        term
        in reason_lower
        for term
        in terms
    )


def _is_quality_failure(
    reason: str,
) -> bool:
    reason_lower = (
        reason.lower()
    )

    terms = (
        "visual quality",
        "scanline",
        "scan line",
        "raster",
        "banding",
        "moire",
        "glitch",
        "corrupt",
        "degenerate",
        "blank",
        "flat colour",
        "flat color",
        "flat or blank",
        "insufficient visual structure",
        "smooth gradient",
        "colour wash",
        "color wash",
        "high-frequency texture",
        "texture or noise",
        "coherent designed structure",
        "outer background area",
        "decorative structure",
        "central title/programme region",
        "central title/program region",
        "too visually dense",
        "excessive visual contrast",
        "fine repetitive",
    )

    return any(
        term
        in reason_lower
        for term
        in terms
    )


def _is_runtime_failure(
    reason: str,
) -> bool:
    reason_lower = (
        reason.lower()
    )

    terms = (
        "comfyui execution failed",
        "execution_error",
        "execution interrupted",
        "execution_interrupted",
        "generation timed out",
        "cuda",
        "out of memory",
        "allocation",
        "expected exactly one persisted",
        "comfyui rejected the workflow",
    )

    return any(
        term
        in reason_lower
        for term
        in terms
    )


# ============================================================
# Retry strength
# ============================================================


def _retry_strength_instruction(
    effective_retry: int,
) -> str:
    if (
        effective_retry
        >= 7
    ):
        return (
            "VERY STRICT RETRY. "
            "Several previous generations failed deterministic QA. "
            "Use the simplest coherent professional event background "
            "that satisfies every requirement literally. "
            "Prefer large intentional shapes, controlled composition "
            "and minimal unnecessary detail."
        )

    if (
        effective_retry
        >= 4
    ):
        return (
            "STRICT RETRY. "
            "Strengthen compliance beyond the previous attempt. "
            "Reduce unnecessary complexity and make every required "
            "constraint visually obvious."
        )

    if (
        effective_retry
        >= 2
    ):
        return (
            "ENHANCED RETRY. "
            "The previous candidate failed QA. "
            "Apply the required correction substantially more strongly."
        )

    return (
        "RETRY CORRECTION. "
        "The previous candidate failed deterministic QA. "
        "Correct the identified violation explicitly."
    )


# ============================================================
# Palette helpers
# ============================================================


def _allowed_palette_names(
    primary_colour: str | None,
    secondary_colour: str | None,
) -> set[str]:
    allowed = set()

    for name in (
        requested_palette_names(
            primary_colour,
            secondary_colour,
        )
    ):
        allowed.update(
            _colour_family_members(
                name
            )
        )

    return (
        allowed
    )


def _forbidden_palette_names(
    primary_colour: str | None,
    secondary_colour: str | None,
) -> list[str]:
    allowed = (
        _allowed_palette_names(
            primary_colour,
            secondary_colour,
        )
    )

    return sorted(
        name
        for name
        in NAMED_COLOURS
        if (
            name
            not in allowed
        )
    )


def _palette_negative_prompt(
    primary_colour: str | None,
    secondary_colour: str | None,
) -> str:
    forbidden = (
        _forbidden_palette_names(
            primary_colour,
            secondary_colour,
        )
    )

    if not forbidden:
        return (
            "colour drift, third colour family, "
            "multicolour palette, rainbow palette"
        )

    return (
        "unrequested colour family, colour drift, "
        "third colour family, multicolour palette, "
        "rainbow palette, "
        + ", ".join(
            forbidden
        )
    )


# ============================================================
# Initial engine prompts
# ============================================================


def _bounded_engine_prompt(
    prompt: str,
    limit: int,
) -> str:
    """Keep diffusion recovery prompts decisive instead of cumulative."""
    compact = " ".join(str(prompt or "").split())
    if len(compact) <= limit:
        return compact

    clauses = [
        clause.strip()
        for clause in re.split(r"(?<=[.!?])\s+|,\s*", compact)
        if clause.strip()
    ]
    selected = []
    seen = set()
    size = 0
    for clause in clauses:
        key = clause.casefold()
        if key in seen:
            continue
        separator = 2 if selected else 0
        if selected and size + separator + len(clause) > limit:
            continue
        if not selected and len(clause) > limit:
            selected.append(clause[:limit].rsplit(" ", 1)[0])
            break
        selected.append(clause)
        seen.add(key)
        size += separator + len(clause)

    # A separator is added after the size calculation above. Trim once more
    # here so the contract remains true even when the input contains unusual
    # punctuation or an overlong first clause.
    bounded = ", ".join(selected)
    if len(bounded) <= limit:
        return bounded

    return bounded[:limit].rsplit(" ", 1)[0].rstrip(",. ")


def _compact_engine_subject(
    prompt: str,
) -> str:
    """Keep the creative subject without overloading SDXL or SD3.5."""
    compact = " ".join(
        str(prompt or "").split()
    )

    if len(compact) <= 420:
        return compact

    return compact[:420].rsplit(" ", 1)[0]


def _native_visual_subject(
    prompt: str,
) -> str:
    """Extract the visual idea before translating it for a native model.

    Creative Direction appends layout and safety contracts to its general
    prompt. FLUX can follow that prose directly, while SDXL and SD3.5 produce
    more coherent images when their subject channel contains only the
    recognisable visual idea.
    """
    implementation_clauses = (
        "a4 portrait",
        "colour palette",
        "color palette",
        "primary colour",
        "primary color",
        "secondary colour",
        "secondary color",
        "central field",
        "outer edges",
        "lower corners",
        "layout contract",
        "theme references",
        "abstract geometry",
        "decoration at",
    )
    clauses = [
        " ".join(clause.split())
        for clause in re.split(r"[,;.!?]+", str(prompt or ""))
        if clause.strip()
    ]
    visual = [
        clause
        for clause in clauses
        if not any(
            marker in clause.casefold()
            for marker in implementation_clauses
        )
    ]
    subject = ", ".join(visual[:3])
    return _bounded_engine_prompt(
        subject or _compact_engine_subject(prompt),
        300,
    )


def _neutral_palette_structure_instruction(
    primary_colour: str | None,
    secondary_colour: str | None,
) -> str:
    """Prevent neutral palettes from degenerating into a blank colour wash."""
    colours = {
        _normalise_colour(primary_colour),
        _normalise_colour(secondary_colour),
    }
    if not (colours & {"black", "white"}):
        return ""

    if colours == {"black", "white"}:
        return (
            "Use crisp black architectural or geometric edge structures on "
            "clean white surfaces. Create hard visible boundaries at both side "
            "edges and lower corners. Do not create a plain grey or white wash. "
        )

    return (
        "Render white as clean bright paper-like shapes with clear boundaries, "
        "not cream, champagne, beige or a blank wash. Use the non-white requested "
        "colour for large edge-anchored structural forms and lower-corner detail. "
    )


def _initial_engine_prompts(
    engine_id: str,
    base_positive_prompt: str,
    base_negative_prompt: str,
    primary_colour: str | None,
    secondary_colour: str | None,
) -> tuple[
    str,
    str,
]:
    """Compile the shared creative brief into engine-specific conditioning."""
    if engine_id == settings.engine_1_id:
        return (
            base_positive_prompt,
            base_negative_prompt,
        )

    primary = _normalise_colour(primary_colour)
    secondary = _normalise_colour(secondary_colour)
    subject = _native_visual_subject(base_positive_prompt)
    palette = (
        f"Restricted {primary} and {secondary} palette. "
        f"{primary} dominant, with substantial {secondary} supporting forms. "
        if primary and secondary
        else "Restricted requested palette only. "
    )

    structure = (
        "Portrait decorative background with a deliberate perimeter "
        "composition: several large connected motifs at the side edges and "
        "lower corners, plus generous calm open space through the interior. "
        "Clean matte graphic illustration, stable silhouettes and clear "
        "shape boundaries."
    )

    if engine_id == settings.engine_3_id:
        # SD3.5 is sensitive to stacked prohibitions. Its encoders receive a
        # compact affirmative description ordered from subject to palette to
        # composition, so every conditioning channel sees the same intent.
        positive = (
            f"{subject}. {palette}{structure} "
            "Broad discrete forms; no fine texture."
        )
        negative = ""
    else:
        # SDXL benefits from a compact scene description and a conventional
        # negative tag list. Do not give it FLUX's long product-rule prose.
        positive = (
            f"{subject}. "
            f"{palette}"
            + _neutral_palette_structure_instruction(
                primary_colour,
                secondary_colour,
            )
            + structure
            + " Polished editorial graphic design with broad intentional forms."
        )
        negative = (
            _palette_negative_prompt(primary_colour, secondary_colour)
            + ", people, faces, characters, text, typography, logos, "
            "watermarks, metallic, chrome, photographic lighting, glow, "
            "bloom, scanlines, raster lines, moire, glitch, grain, "
            "noise field, full-page gradient, dense central detail, fine hatching"
        )

    return positive, negative


# ============================================================
# Palette-specific recovery
# ============================================================


def _standard_palette_instruction(
    primary_colour: str | None,
    secondary_colour: str | None,
    retry_number: int,
) -> str:
    primary = (
        _normalise_colour(
            primary_colour
        )
    )

    secondary = (
        _normalise_colour(
            secondary_colour
        )
    )

    if (
        primary
        and secondary
    ):
        if (
            retry_number
            >= 7
        ):
            return (
                "EXTREME PALETTE ENFORCEMENT. "
                f"Use only {primary} and {secondary}. "
                f"Make {primary} occupy most major surfaces. "
                f"Make {secondary} occupy clearly visible supporting "
                "regions in several areas. "
                "Every neutral or chromatic colour family that was not "
                "requested is forbidden. "
                "All shadows and highlights must remain tonal variations "
                "inside the requested colour families."
            )

        if (
            retry_number
            >= 4
        ):
            return (
                "STRICT PALETTE RECOVERY. "
                f"Use {primary} as the dominant visible colour across "
                "most large surfaces. "
                f"Use {secondary} as a substantial supporting colour. "
                "Do not introduce any third colour family. "
                "Do not use unrequested neutral highlights or shadows."
            )

        return (
            "STRICT COLOUR CORRECTION. "
            f"The primary colour is {primary}. "
            f"The secondary colour is {secondary}. "
            f"Make {primary} visibly dominant and {secondary} "
            "clearly visible in repeated supporting regions. "
            "Every additional colour family is forbidden."
        )

    if primary:
        return (
            "STRICT COLOUR CORRECTION. "
            f"Use {primary} as the only requested visible "
            "colour family."
        )

    return (
        "STRICT COLOUR CORRECTION. "
        "Obey the requested palette exactly."
    )


def _sd35_palette_instruction(
    primary_colour: str | None,
    secondary_colour: str | None,
    retry_number: int,
) -> str:
    primary = (
        _normalise_colour(
            primary_colour
        )
    )

    secondary = (
        _normalise_colour(
            secondary_colour
        )
    )

    if (
        primary
        and secondary
    ):
        if (
            retry_number
            >= 6
        ):
            return (
                "SD3.5 EMERGENCY PALETTE RECOVERY. "
                "Create a simple coherent event background with large "
                "clean colour regions. "
                f"Approximately 65 percent should read as {primary}. "
                f"Approximately 25 to 35 percent should read as {secondary}. "
                f"Use only {primary} and {secondary}. "
                "No third colour family. "
                "No unrequested neutral fills."
            )

        if (
            retry_number
            >= 3
        ):
            return (
                "SD3.5 STRICT TWO-COLOUR RECOVERY. "
                f"Make {primary} cover most major surfaces. "
                f"Make {secondary} clearly visible across multiple "
                "substantial supporting regions. "
                f"Use only {primary} and {secondary}."
            )

        return (
            "SD3.5 PALETTE CORRECTION. "
            f"Use a strong {primary} dominant base and clearly visible "
            f"{secondary} supporting regions."
        )

    if primary:
        return (
            "SD3.5 PALETTE CORRECTION. "
            f"Use the {primary} colour family."
        )

    return (
        "SD3.5 PALETTE CORRECTION. "
        "Use only the explicitly requested palette."
    )


# ============================================================
# Human / text recovery
# ============================================================


def _human_instruction(
    retry_number: int,
) -> tuple[
    str,
    str,
]:
    if (
        retry_number
        >= 4
    ):
        positive = (
            "STRICT HUMAN-FREE RECOVERY. "
            "Use only abstract geometry, environmental forms, patterns, "
            "architecture and ornamental shapes. "
            "Do not arrange shapes into heads, faces, torsos, limbs, "
            "bodies, silhouettes or character-like structures."
        )

    else:
        positive = (
            "STRICT HUMAN-FREE CORRECTION. "
            "Background only with abstract or environmental structure."
        )

    negative = (
        "person, people, human, human figure, man, woman, child, "
        "face, portrait, silhouette, body, head, arms, hands, legs, "
        "clothing, mannequin, character, hero, superhero, humanoid"
    )

    return (
        positive,
        negative,
    )


def _text_instruction() -> tuple[
    str,
    str,
]:
    return (
        (
            "STRICT TEXT-FREE CORRECTION. "
            "Use visual artwork only with no readable symbols."
        ),
        (
            "text, typography, words, letters, numbers, writing, "
            "logo, watermark, signage, labels, pseudo-text"
        ),
    )


# ============================================================
# Quality recovery
# ============================================================


def _quality_instruction(
    effective_retry: int,
) -> tuple[
    str,
    str,
]:
    if (
        effective_retry
        >= 7
    ):
        positive = (
            "MAXIMUM VISUAL-QUALITY RECOVERY. "
            "Use large coherent designed shapes around the outside "
            "of the page. "
            "Preserve a simple calm centre. "
            "Avoid micro-detail, fine repetitive line work, gradients "
            "and texture fields."
        )

    elif (
        effective_retry
        >= 4
    ):
        positive = (
            "STRICT VISUAL-QUALITY RECOVERY. "
            "Create meaningful decorative structure around the outer "
            "edges and lower portion of the page. "
            "Keep the central overlay region visually simple."
        )

    else:
        positive = (
            "VISUAL-QUALITY CORRECTION. "
            "Render a coherent designed event background with useful "
            "outer structure and a calm central region."
        )

    negative = (
        "scanlines, scan lines, horizontal scanlines, vertical scanlines, "
        "raster banding, repetitive horizontal lines, repetitive vertical "
        "lines, full-frame stripes, fine stripe field, digital interference, "
        "moire, glitch, corrupted texture, broken render, blank image, "
        "empty flat field, smooth full-page gradient, plain colour wash, "
        "featureless gradient, texture-only image, grain-only image, "
        "pixel noise, random noise field, unstructured noise, "
        "dense central architecture, dense central pattern, "
        "full-page decoration"
    )

    return (
        positive,
        negative,
    )


def _structural_quality_recovery_prompt(
    *,
    base_positive_prompt: str,
    reason: str,
    primary_colour: str | None,
    secondary_colour: str | None,
    engine_id: str,
    effective_retry: int,
) -> tuple[
    str,
    str,
]:
    """
    Replace the failed creative prompt with a deterministic
    composition-oriented recovery prompt.

    Routing priority is deliberate:

    1. raster / scanline / noise
    2. gradient / colour wash
    3. dense centre
    4. weak outer structure
    5. generic structural failure

    More specific failure modes must be checked before generic
    phrases such as "insufficient designed structure".
    """

    reason_lower = (
        reason.lower()
    )

    primary = (
        _normalise_colour(
            primary_colour
        )
        or "the requested primary colour"
    )

    secondary = (
        _normalise_colour(
            secondary_colour
        )
        or "the requested secondary colour"
    )

    engine_label = (
        "SDXL"
        if (
            engine_id
            == settings.engine_2_id
        )
        else "SD3.5"
    )

    palette_contract = (
        f"Use only the requested {primary} and {secondary} "
        "colour families. "
        f"Use {primary} as the dominant field and {secondary} "
        "as clearly visible supporting structural regions. "
    )

    common = (
        f"{_native_visual_subject(base_positive_prompt)}. "
        f"{engine_label} STRUCTURAL QUALITY RECOVERY. "
        "Create an A4 portrait decorative event background only. "
        f"{palette_contract}"
        "Use broad matte graphic forms. "
        "Keep all people, faces, characters, text, lettering, "
        "numbers and logos completely absent. "
        "Do not create metallic reflections, glow, bloom or "
        "photographic lighting. "
    )

    # ========================================================
    # 1. Raster / scanline / noise
    # ========================================================

    if (
        "scanline"
        in reason_lower
        or "scan line"
        in reason_lower
        or "raster"
        in reason_lower
        or "fine repetitive"
        in reason_lower
        or "high-frequency texture"
        in reason_lower
        or "texture or noise"
        in reason_lower
        or "moire"
        in reason_lower
        or "glitch"
        in reason_lower
        or "corrupted texture"
        in reason_lower
    ):
        positive = (
            common
            + (
                "RASTER AND NOISE RECOVERY. "
                "Use only large clean connected forms. "
                "Use broad polygonal architecture and large graphic "
                "decorative regions. "
                "Do not use repeated parallel lines. "
                "Do not use fine hatching. "
                "Do not use grain, pixel texture, noise, scanlines, "
                "raster effects, print-screen texture or moire. "
                "All major visual details must remain visible after "
                "slight blur. "
                "Keep the central region calm."
            )
        )

    # ========================================================
    # 2. Gradient / colour wash
    # ========================================================

    elif (
        "smooth gradient"
        in reason_lower
        or "colour wash"
        in reason_lower
        or "color wash"
        in reason_lower
        or "featureless gradient"
        in reason_lower
    ):
        positive = (
            common
            + (
                "GRADIENT FAILURE RECOVERY. "
                "Do not create a smooth blended full-page background. "
                "Build the image from large discrete matte shapes. "
                "Use clearly bounded architectural silhouettes, "
                "polygonal blocks, angular comic-inspired geometry and "
                "large edge-anchored forms. "
                "Create obvious visual structure along the left edge, "
                "right edge and lower corners. "
                "Use flat or gently tonal surfaces rather than continuous "
                "full-page colour blending. "
                "The centre should remain a simple stable field, not a "
                "featureless gradient. "
                "Use visible large-scale forms around the perimeter."
            )
        )

    # ========================================================
    # 3. Dense central overlay region
    # ========================================================

    elif (
        "central title/programme region"
        in reason_lower
        or "central title/program region"
        in reason_lower
        or "too visually dense"
        in reason_lower
        or "excessive visual contrast"
        in reason_lower
    ):
        positive = (
            common
            + (
                "CENTRAL SAFE-ZONE RECOVERY. "
                "Reserve the middle approximately 60 percent of the page "
                "width as a calm low-detail field. "
                "From roughly 20 percent to 80 percent of the page width, "
                "use mostly simple uninterrupted primary-colour surfaces. "
                "Do not place buildings, web geometry, speed lines, "
                "ornament, dense patterns or architectural detail "
                "through the central safe zone. "
                "Concentrate the designed structure inside narrow left "
                "and right edge zones and the lower 18 to 22 percent "
                "of the page. "
                "Create several large edge-anchored architectural or "
                "geometric forms rather than fine texture. "
                "The centre must immediately read as calm and available "
                "for later title and programme overlays."
            )
        )

    # ========================================================
    # 4. Weak outer structure
    # ========================================================

    elif (
        "outer background area"
        in reason_lower
        or "too little useful decorative structure"
        in reason_lower
        or "outer structural edge density"
        in reason_lower
    ):
        positive = (
            common
            + (
                "OUTER STRUCTURE RECOVERY. "
                "Add large clearly visible decorative forms to both "
                "side edges and both lower corners. "
                "Use several substantial architectural silhouettes, "
                "polygonal structures, comic-inspired geometry or "
                "large ornamental forms. "
                "The side-edge structures must extend vertically through "
                "a meaningful part of the page. "
                "The lower region must contain visible grounded structure. "
                "Do not solve this using a gradient or subtle texture. "
                "Use discrete shapes with clear boundaries. "
                "Keep the central region comparatively simple."
            )
        )

    # ========================================================
    # 5. Generic structural failure
    # ========================================================

    elif (
        "insufficient designed structure"
        in reason_lower
        or "insufficient visual structure"
        in reason_lower
        or "coherent designed structure"
        in reason_lower
    ):
        positive = (
            common
            + (
                "GENERAL STRUCTURAL RECOVERY. "
                "Create clearly visible designed structure using "
                "large geometric, architectural or ornamental forms. "
                "Place substantial decorative structure around both "
                "side edges and the lower region. "
                "Keep the central region simple and usable for overlays. "
                "Do not use only gradients, noise, raster textures or "
                "fine repetitive patterns."
            )
        )

    else:
        positive = (
            common
            + (
                "GENERAL STRUCTURAL RECOVERY. "
                "Create clearly designed outer decorative structure "
                "using large geometric or architectural forms. "
                "Keep the central region simple and usable for overlays. "
                "Avoid gradients, noise, raster textures and dense "
                "full-page patterns."
            )
        )

    if (
        effective_retry
        >= 6
    ):
        positive += (
            " Use fewer elements than previous attempts. "
            "Prefer approximately four to eight large intentional "
            "decorative structures instead of many small details."
        )

    negative = (
        ", ".join(
            value
            for value
            in (
                _palette_negative_prompt(
                    primary_colour,
                    secondary_colour,
                ),
                (
                    "person, people, human, face, body, silhouette, "
                    "character, superhero, text, typography, letters, "
                    "numbers, logo, watermark, signage, "
                    "smooth full-page gradient, plain colour wash, "
                    "featureless gradient, grain, grain-only image, "
                    "noise-only image, scanlines, raster lines, "
                    "horizontal line field, vertical line field, "
                    "moire, glitch, corrupted texture, "
                    "dense central architecture, dense central pattern, "
                    "full-page web pattern, full-page speed lines, "
                    "metallic, chrome, steel, reflective surface, "
                    "glow, bloom, photographic lighting"
                ),
            )
            if value
        )
    )

    return (
        positive,
        negative,
    )


# ============================================================
# Literal palette recovery
# ============================================================


def _minimal_two_colour_recovery_prompt(
    *,
    base_positive_prompt: str,
    primary_colour: str | None,
    secondary_colour: str | None,
    engine_id: str,
    effective_retry: int,
) -> tuple[
    str,
    str,
]:
    primary = (
        _normalise_colour(
            primary_colour
        )
        or "the requested primary colour"
    )

    secondary = (
        _normalise_colour(
            secondary_colour
        )
        or "the requested secondary colour"
    )

    engine_label = (
        "SDXL"
        if (
            engine_id
            == settings.engine_2_id
        )
        else "SD3.5"
    )

    positive = (
        f"{_native_visual_subject(base_positive_prompt)}. "
        f"{engine_label} STRICT LITERAL PALETTE RECOVERY. "
        "Create a clean A4 portrait abstract decorative event "
        "background with a calm central region and meaningful "
        "visual detail around the outer edges and lower corners. "
        f"Use {primary} across approximately 65 to 75 percent "
        "of the visible image. "
        f"Use {secondary} across approximately 20 to 30 percent "
        "of the visible image. "
        f"Create multiple large solid {secondary} areas. "
        f"Do not reduce {secondary} to tiny accents, thin lines, "
        "reflections or hidden shadows. "
        f"The central quiet region must stay inside the {primary} "
        "colour family. "
        "Use broad matte surfaces, simple graphic geometry and "
        "coherent architectural or environmental shapes. "
        "The outer edges and lower area must visibly contain "
        "designed structural forms. "
        "Do not generate only a smooth gradient. "
        "Do not generate only grain, texture or scanlines. "
        "Shadows must be darker tonal members of the requested families. "
        "Highlights must be lighter tonal members of the requested families. "
        "No third colour family. "
        "No unrequested neutral colour. "
        "No metallic surfaces. "
        "No chrome. "
        "No coloured illumination. "
        "No glow. "
        "No bloom. "
        "No iridescence. "
        "No photographic lighting. "
        "No people, faces, characters, humanoids, text, letters, "
        "numbers, logos or readable symbols."
    )

    negative = (
        ", ".join(
            value
            for value
            in (
                _palette_negative_prompt(
                    primary_colour,
                    secondary_colour,
                ),
                (
                    "metallic, chrome, steel, coloured lighting, "
                    "colourless reflection, reflective material, "
                    "glow, bloom, iridescent, holographic, "
                    "pearlescent, person, people, face, portrait, "
                    "human, character, superhero, text, typography, "
                    "letters, numbers, logo, watermark, signage, "
                    "scanlines, raster lines, raster field, moire, glitch, "
                    "corrupted texture, smooth full-page gradient, "
                    "plain colour wash, featureless gradient, "
                    "texture-only image, grain-only image, noise-only image"
                ),
            )
            if value
        )
    )

    return (
        positive,
        negative,
    )


# ============================================================
# Prompt strengthening
# ============================================================


def _strengthen_prompts(
    base_positive_prompt: str,
    base_negative_prompt: str,
    reason: str,
    primary_colour: str | None,
    secondary_colour: str | None,
    retry_number: int,
    engine_id: str,
) -> tuple[
    str,
    str,
]:
    (
        positive_prompt,
        negative_prompt,
    ) = (
        _initial_engine_prompts(
            engine_id,
            base_positive_prompt,
            base_negative_prompt,
            primary_colour,
            secondary_colour,
        )
    )

    positive_additions = []

    negative_additions = []

    palette_failure = (
        _is_palette_failure(
            reason
        )
    )

    human_failure = (
        _is_human_failure(
            reason
        )
    )

    text_failure = (
        _is_text_failure(
            reason
        )
    )

    quality_failure = (
        _is_quality_failure(
            reason
        )
    )

    runtime_failure = (
        _is_runtime_failure(
            reason
        )
    )

    effective_retry = (
        retry_number
    )

    positive_additions.append(
        _retry_strength_instruction(
            effective_retry
        )
    )

    # ========================================================
    # Structural quality recovery has priority.
    #
    # If the validator says the image is structurally bad,
    # replace the creative prompt with a deterministic structural
    # recovery prompt.
    #
    # Do NOT incorrectly convert a quality failure into palette
    # recovery merely because its error text mentions "colour".
    # ========================================================

    structural_recovery = bool(
        quality_failure
    )

    if (
        structural_recovery
    ):
        (
            positive_prompt,
            structural_negative,
        ) = (
            _structural_quality_recovery_prompt(
                base_positive_prompt=base_positive_prompt,
                reason=(
                    reason
                ),
                primary_colour=(
                    primary_colour
                ),
                secondary_colour=(
                    secondary_colour
                ),
                engine_id=(
                    engine_id
                ),
                effective_retry=(
                    effective_retry
                ),
            )
        )

        recovery_base_negative = (
            negative_prompt
            if engine_id in {settings.engine_2_id, settings.engine_3_id}
            else base_negative_prompt
        )
        negative_prompt = (
            ", ".join(
                value
                for value
                in (
                    recovery_base_negative.strip(
                        ", "
                    ),
                    structural_negative,
                )
                if value
            )
        )

        logger.info(
            "event=structural_quality_recovery "
            "engine=%s "
            "effective_retry=%s "
            "reason=%s",
            engine_id,
            effective_retry,
            reason,
        )

    # ========================================================
    # Literal palette recovery only when this is genuinely a
    # palette failure and NOT primarily a visual-quality failure.
    # ========================================================

    literal_palette_recovery = bool(
        palette_failure
        and not quality_failure
        and effective_retry
        >= 4
        and primary_colour
        and secondary_colour
    )

    if (
        literal_palette_recovery
    ):
        (
            positive_prompt,
            strict_negative_prompt,
        ) = (
            _minimal_two_colour_recovery_prompt(
                base_positive_prompt=(
                    base_positive_prompt
                ),
                primary_colour=(
                    primary_colour
                ),
                secondary_colour=(
                    secondary_colour
                ),
                engine_id=(
                    engine_id
                ),
                effective_retry=(
                    effective_retry
                ),
            )
        )

        recovery_base_negative = (
            negative_prompt
            if engine_id in {settings.engine_2_id, settings.engine_3_id}
            else base_negative_prompt
        )
        negative_prompt = (
            ", ".join(
                value
                for value
                in (
                    recovery_base_negative.strip(
                        ", "
                    ),
                    strict_negative_prompt,
                )
                if value
            )
        )

        logger.info(
            "event=literal_palette_recovery "
            "engine=%s "
            "effective_retry=%s "
            "primary=%s "
            "secondary=%s",
            engine_id,
            effective_retry,
            primary_colour,
            secondary_colour,
        )

    # ========================================================
    # Palette correction
    # ========================================================

    if (
        palette_failure
    ):
        if (
            engine_id
            == settings.engine_3_id
        ):
            positive_additions.append(
                _sd35_palette_instruction(
                    primary_colour,
                    secondary_colour,
                    effective_retry,
                )
            )

        else:
            positive_additions.append(
                _standard_palette_instruction(
                    primary_colour,
                    secondary_colour,
                    effective_retry,
                )
            )

        negative_additions.append(
            _palette_negative_prompt(
                primary_colour,
                secondary_colour,
            )
        )

    # ========================================================
    # SDXL-specific palette correction
    # ========================================================

    if (
        engine_id
        == settings.engine_2_id
        and palette_failure
        and not structural_recovery
    ):
        if (
            effective_retry
            >= 3
            and not literal_palette_recovery
        ):
            positive_additions.append(
                (
                    "SDXL STRICT FLAT-COLOUR RECOVERY. "
                    "Remove realistic materials, reflective surfaces, "
                    "metallic appearance, cinematic illumination, "
                    "photographic shading and colourless highlights. "
                    "Use large matte colour regions with clean graphic "
                    "geometry."
                )
            )

        else:
            positive_additions.append(
                (
                    "SDXL PALETTE RECOVERY. "
                    "Strengthen the requested colours and remove "
                    "neutral highlights, neutral shading and metallic tones."
                )
            )

    # ========================================================
    # Human correction
    # ========================================================

    if (
        human_failure
    ):
        (
            positive_human,
            negative_human,
        ) = (
            _human_instruction(
                effective_retry
            )
        )

        positive_additions.append(
            positive_human
        )

        negative_additions.append(
            negative_human
        )

    # ========================================================
    # Text correction
    # ========================================================

    if (
        text_failure
    ):
        (
            positive_text,
            negative_text,
        ) = (
            _text_instruction()
        )

        positive_additions.append(
            positive_text
        )

        negative_additions.append(
            negative_text
        )

    # ========================================================
    # Quality correction
    #
    # The replacement prompt above handles the main recovery.
    # This addition further reinforces the correction.
    # ========================================================

    if (
        quality_failure
    ):
        (
            positive_quality,
            negative_quality,
        ) = (
            _quality_instruction(
                effective_retry
            )
        )

        positive_additions.append(
            positive_quality
        )

        negative_additions.append(
            negative_quality
        )

    # ========================================================
    # Runtime correction
    # ========================================================

    if (
        runtime_failure
    ):
        positive_additions.append(
            (
                "RUNTIME RECOVERY ATTEMPT. "
                "Use a simpler coherent composition with fewer "
                "intricate micro-details and larger clean regions."
            )
        )

        if (
            engine_id
            == settings.engine_3_id
        ):
            positive_additions.append(
                (
                    "SD3.5 RUNTIME RECOVERY. "
                    "Prefer broad geometric forms, restrained layering "
                    "and uncomplicated surfaces."
                )
            )

    # ========================================================
    # Specific missing secondary colour
    # ========================================================

    reason_lower = (
        reason.lower()
    )

    if (
        "secondary"
        in reason_lower
        and secondary_colour
    ):
        secondary = (
            _normalise_colour(
                secondary_colour
            )
        )

        positive_additions.append(
            (
                "MANDATORY SECONDARY COLOUR REQUIREMENT. "
                f"Render {secondary} as actual visible surface colour "
                "in several substantial separate regions. "
                f"Do not hide {secondary} inside shadows, reflections, "
                "thin outlines or tiny accents. "
                f"Several large visible areas must immediately read "
                f"as {secondary}."
            )
        )

        if (
            effective_retry
            >= 6
            and engine_id
            in {
                settings.engine_2_id,
                settings.engine_3_id,
            }
        ):
            positive_additions.append(
                (
                    "SECONDARY COLOUR AREA RECOVERY. "
                    f"Dedicate roughly 25 to 35 percent of the image "
                    f"to large solid {secondary} surfaces. "
                    "Use separate blocks or structural regions rather "
                    "than thin lines."
                )
            )

    # ========================================================
    # Specific missing primary colour
    # ========================================================

    if (
        "primary"
        in reason_lower
        and primary_colour
    ):
        primary = (
            _normalise_colour(
                primary_colour
            )
        )

        positive_additions.append(
            (
                "MANDATORY PRIMARY COLOUR REQUIREMENT. "
                f"Render {primary} across large visible surfaces."
            )
        )

    # ========================================================
    # Final combined prompt
    # ========================================================

    positive_prompt = (
        positive_prompt.rstrip(
            ". "
        )
        + ". "
        + " ".join(
            positive_additions
        )
    )

    negative_prompt = (
        ", ".join(
            value
            for value
            in (
                negative_prompt.strip(
                    ", "
                ),
                *negative_additions,
            )
            if value
        )
    )

    if engine_id == settings.engine_1_id:
        # Flux also loses the active correction when retry text accumulates.
        positive_prompt = _bounded_engine_prompt(positive_prompt, 1400)
        negative_prompt = _bounded_engine_prompt(negative_prompt, 1000)
    elif engine_id == settings.engine_3_id:
        # SD3.5's three text encoders are particularly sensitive to long,
        # repetitive correction stacks. A concise composition brief gives it
        # materially better structure than another layer of prohibitions.
        positive_prompt = _bounded_engine_prompt(positive_prompt, 900)
        negative_prompt = _bounded_engine_prompt(negative_prompt, 650)
    elif engine_id == settings.engine_2_id:
        # SDXL also needs a bounded current correction rather than history.
        positive_prompt = _bounded_engine_prompt(positive_prompt, 1600)
        negative_prompt = _bounded_engine_prompt(negative_prompt, 1400)

    return (
        positive_prompt,
        negative_prompt,
    )


# ============================================================
# Engine preparation
# ============================================================


async def _prepare_engine(
    client: ComfyUIClient,
    reference_number: str,
    output,
    current_engine: str | None,
) -> str:
    if (
        current_engine
        == output.engine_id
    ):
        return (
            current_engine
        )

    queue = (
        await client.request(
            "GET",
            "/queue",
        )
    ).json()

    entries = (
        queue.get(
            "queue_running",
            [],
        )
        + queue.get(
            "queue_pending",
            [],
        )
    )

    own_path = (
        settings.programme_data_path
        / reference_number
        / "backgrounds"
        / output.engine_id
        / (
            f"image-"
            f"{output.direction_id.lower()}"
            ".json"
        )
    )

    own = (
        json.loads(
            own_path.read_text(
                encoding="utf-8"
            )
        )
        if (
            own_path.exists()
        )
        else {}
    )

    if entries:
        if (
            not own
            or any(
                (
                    entry[
                        3
                    ]
                    .get(
                        "programme_request_id"
                    )
                    != own.get(
                        "request_id"
                    )
                )
                for entry
                in entries
                if (
                    len(
                        entry
                    )
                    > 3
                )
            )
        ):
            raise RuntimeError(
                "ComfyUI has another active job; "
                "retry later."
            )

    else:
        await (
            client.release_models()
        )

    return (
        output.engine_id
    )


# ============================================================
# Candidate generation
# ============================================================


def _resume_retry_context(existing: dict) -> tuple[int, str]:
    """Return zero-based retry and the failure it corrects after a restart."""
    status = existing.get("status")
    persisted_attempts = int(existing.get("attempt_count", 0))
    if status in {"rejected", "runtime_failed", "failed"}:
        start_retry = persisted_attempts
        reason = existing.get("rejection_reason") or existing.get("error") or ""
    elif status in {"prepared", "submitted"}:
        start_retry = max(0, persisted_attempts - 1)
        history = existing.get("rejected_attempts", []) + existing.get("failed_attempts", [])
        last = max(history, key=lambda item: item.get("attempt_count", 0)) if history else {}
        reason = last.get("rejection_reason") or last.get("error") or ""
        if start_retry and not reason:
            raise ValueError("In-flight retry has no previous failure context")
    else:
        start_retry, reason = 0, ""
    return start_retry, reason


async def _attempt_output_wave(
    *,
    client: ComfyUIClient,
    state,
    document: dict,
    output,
    reference_number: str,
    primary_colour: str | None,
    secondary_colour: str | None,
) -> tuple[
    bool,
    int,
]:
    direction = (
        get_direction(
            document,
            output.direction_id,
        )
    )

    spec = build_candidate_spec(document, output.engine_id, output.direction_id)
    existing = _candidate_record(reference_number, output.engine_id, output.direction_id)
    if existing.get("spec_sha256") and existing["spec_sha256"] != spec.spec_sha256:
        raise ValueError("Candidate specification changed across attempts; use a new reference")
    start_retry, previous_reason = _resume_retry_context(existing)
    if start_retry >= min(settings.max_candidate_retries, 8) + 1:
        update_output(state, output.engine_id, output.direction_id, "failed",
                      error="Candidate attempt budget exhausted")
        return False, 0
    logger.info(
        "event=candidate_spec_created reference=%s engine=%s direction=%s "
        "direction_role=%s spec_sha256=%s",
        reference_number, output.engine_id, output.direction_id,
        spec.direction_role, spec.spec_sha256,
    )

    base_positive_prompt = (
        direction[
            "positive_prompt"
        ]
    )

    base_negative_prompt = (
        direction.get(
            "negative_prompt",
            "",
        )
    )

    if output.engine_id == settings.engine_1_id:
        positive_prompt, negative_prompt = _initial_engine_prompts(
            output.engine_id,
            base_positive_prompt,
            base_negative_prompt,
            primary_colour,
            secondary_colour,
        )
    else:
        # Native prompts are compiled afresh from CandidateSpec per attempt.
        positive_prompt, negative_prompt = "", ""

    generated = 0

    for retry_count in range(start_retry, min(settings.max_candidate_retries, 8) + 1):
        attempt = retry_count + 1
        compiled = None
        if output.engine_id in {settings.engine_2_id, settings.engine_3_id}:
            compiled = compile_candidate_prompt(spec, attempt, previous_reason)
            positive_prompt, negative_prompt = compiled.positive, compiled.negative
        elif retry_count > start_retry or start_retry > 0:
            positive_prompt, negative_prompt = _strengthen_prompts(
                base_positive_prompt, base_negative_prompt, previous_reason,
                primary_colour, secondary_colour, retry_count, output.engine_id,
            )
        category = compiled.failure_category if compiled else failure_category(previous_reason)
        stage = compiled.retry_stage if compiled else retry_stage(attempt)
        logger.info(
            "event=spec_lock_verified reference=%s engine=%s direction=%s "
            "attempt=%s direction_role=%s spec_sha256=%s",
            reference_number, output.engine_id, output.direction_id,
            attempt, spec.direction_role, spec.spec_sha256,
        )
        logger.info(
            "event=candidate_prompt_compiled reference=%s engine=%s direction=%s "
            "attempt=%s direction_role=%s spec_sha256=%s failure_category=%s retry_stage=%s",
            reference_number, output.engine_id, output.direction_id,
            attempt, spec.direction_role, spec.spec_sha256, category, stage,
        )
        try:
            result = (
                await get_engine(
                    output.engine_id
                ).generate(
                    reference_number=(
                        reference_number
                    ),
                    direction_id=(
                        output.direction_id
                    ),
                    positive_prompt=(
                        positive_prompt
                    ),
                    negative_prompt=(
                        negative_prompt
                    ),
                    compiled_prompt=compiled,
                    spec_sha256=spec.spec_sha256,
                    direction_role=spec.direction_role,
                    retry_stage=stage,
                    failure_category=category,
                )
            )

            output.seed = (
                result[
                    "seed"
                ]
            )

            output.prompt_id = (
                result[
                    "prompt_id"
                ]
            )

            output.workflow_sha256 = (
                result[
                    "workflow_sha256"
                ]
            )

            update_output(
                state,
                output.engine_id,
                output.direction_id,
                "complete",
                result[
                    "output_path"
                ],
                result[
                    "sha256"
                ],
            )

            logger.info(
                "event=candidate_complete "
                "reference=%s "
                "engine=%s "
                "direction=%s "
                "reused=%s",
                reference_number,
                output.engine_id,
                output.direction_id,
                result[
                    "reused"
                ],
            )

            if not (
                result[
                    "reused"
                ]
            ):
                generated = 1

            return (
                True,
                generated,
            )

        except SubmissionUncertain:
            raise

        except ComfyUIError as error:
            reason = (
                _rejection_reason(
                    reference_number,
                    output.engine_id,
                    output.direction_id,
                )
                or str(
                    error
                )
            )

            retryable = (
                _candidate_was_rejected(
                    reference_number,
                    output.engine_id,
                    output.direction_id,
                )
                or _is_runtime_failure(
                    reason
                )
            )

            if not retryable:
                raise

            if (
                retry_count
                >= settings.max_candidate_retries
            ):
                update_output(
                    state,
                    output.engine_id,
                    output.direction_id,
                    "failed",
                    error=(
                        reason
                    ),
                )

                logger.error(
                    "event=candidate_exhausted "
                    "reference=%s "
                    "engine=%s "
                    "direction=%s "
                    "attempts=%s "
                    "reason=%s",
                    reference_number,
                    output.engine_id,
                    output.direction_id,
                    (
                        settings.max_candidate_retries
                        + 1
                    ),
                    reason,
                )

                return (
                    False,
                    0,
                )

            next_retry = (
                retry_count
                + 1
            )
            previous_reason = reason
            logger.info(
                "event=retry_strategy_selected reference=%s engine=%s direction=%s "
                "attempt=%s direction_role=%s spec_sha256=%s "
                "failure_category=%s retry_stage=%s",
                reference_number, output.engine_id, output.direction_id,
                next_retry + 1, spec.direction_role, spec.spec_sha256,
                failure_category(reason), retry_stage(next_retry + 1),
            )

            delay_seconds = min(
                2
                ** retry_count,
                settings
                .max_candidate_retry_delay_seconds,
            )

            palette_failure = (
                _is_palette_failure(
                    reason
                )
            )

            human_failure = (
                _is_human_failure(
                    reason
                )
            )

            text_failure = (
                _is_text_failure(
                    reason
                )
            )

            quality_failure = (
                _is_quality_failure(
                    reason
                )
            )

            runtime_failure = (
                _is_runtime_failure(
                    reason
                )
            )

            logger.warning(
                "event=candidate_retry "
                "reference=%s "
                "engine=%s "
                "direction=%s "
                "retry=%s/%s "
                "effective_retry=%s "
                "delay_seconds=%s "
                "reason=%s",
                reference_number,
                output.engine_id,
                output.direction_id,
                next_retry,
                settings.max_candidate_retries,
                next_retry,
                delay_seconds,
                reason,
            )

            logger.info(
                "event=prompt_strengthened "
                "reference=%s "
                "engine=%s "
                "direction=%s "
                "effective_retry=%s "
                "palette_failure=%s "
                "human_failure=%s "
                "text_failure=%s "
                "quality_failure=%s "
                "runtime_failure=%s",
                reference_number,
                output.engine_id,
                output.direction_id,
                next_retry,
                palette_failure,
                human_failure,
                text_failure,
                quality_failure,
                runtime_failure,
            )

            if (
                output.engine_id
                == settings.engine_2_id
                and (
                    palette_failure
                    or quality_failure
                )
            ):
                logger.info(
                    "event=sdxl_recovery "
                    "reference=%s "
                    "direction=%s "
                    "effective_retry=%s "
                    "palette_failure=%s "
                    "quality_failure=%s",
                    reference_number,
                    output.direction_id,
                    next_retry,
                    palette_failure,
                    quality_failure,
                )

            if (
                output.engine_id
                == settings.engine_3_id
            ):
                logger.info(
                    "event=sd35_recovery "
                    "reference=%s "
                    "direction=%s "
                    "effective_retry=%s "
                    "runtime_failure=%s "
                    "palette_failure=%s "
                    "quality_failure=%s",
                    reference_number,
                    output.direction_id,
                    next_retry,
                    runtime_failure,
                    palette_failure,
                    quality_failure,
                )

            await asyncio.sleep(
                delay_seconds
            )

    return (
        False,
        0,
    )


# ============================================================
# Job execution
# ============================================================


async def execute_image_job(
    reference_number: str,
    max_outputs: int | None = None,
):
    async with GPULease():
        state = (
            load_image_job_state(
                reference_number
            )
        )

        document = (
            load_prompts_document(
                reference_number
            )
        )

        brief = (
            document.get(
                "brief",
                {},
            )
        )

        primary_colour = (
            brief.get(
                "primary_colour"
            )
        )

        secondary_colour = (
            brief.get(
                "secondary_colour"
            )
        )

        expected = [
            (
                step[
                    "engine_id"
                ],
                step[
                    "direction"
                ],
            )
            for step
            in (
                build_execution_plan(
                    document
                )[
                    "execution_order"
                ]
            )
        ]

        actual = [
            (
                output.engine_id,
                output.direction_id,
            )
            for output
            in state.outputs
        ]

        if (
            actual
            != expected
        ):
            raise ValueError(
                "Historical engine plan differs from the "
                "current configuration. Preserve this job "
                "and create a new reference number."
            )

        state.status = (
            "processing"
        )

        state.current_stage = (
            "generating_backgrounds"
        )

        state.error = None

        persist_image_job_state(
            state
        )

        try:
            # ====================================================
            # Ollama must release GPU memory first
            # ====================================================

            async with httpx.AsyncClient(
                timeout=60
            ) as http:
                response = (
                    await http.get(
                        settings
                        .ollama_base_url
                        + "/api/ps"
                    )
                )

                response.raise_for_status()

                if (
                    response.json()
                    .get(
                        "models"
                    )
                ):
                    raise RuntimeError(
                        "Ollama has resident models; "
                        "unload them before image generation."
                    )

            async with (
                ComfyUIClient()
                as client
            ):
                generated = 0

                current_engine = None

                stopped_for_max_outputs = (
                    False
                )

                try:
                    # ============================================
                    # SINGLE WAVE ONLY
                    #
                    # 1 initial attempt
                    # + 8 retries
                    # = 9 attempts maximum
                    #
                    # No second +9 recovery cycle.
                    # ============================================

                    for output in (
                        state.outputs
                    ):
                        if (
                            output.status
                            in {
                                "complete",
                                "failed",
                            }
                        ):
                            continue

                        current_engine = (
                            await _prepare_engine(
                                client,
                                reference_number,
                                output,
                                current_engine,
                            )
                        )

                        state.current_stage = (
                            f"{output.engine_id}:"
                            f"{output.direction_id}"
                        )

                        persist_image_job_state(
                            state
                        )

                        (
                            _,
                            newly_generated,
                        ) = (
                            await _attempt_output_wave(
                                client=(
                                    client
                                ),
                                state=(
                                    state
                                ),
                                document=(
                                    document
                                ),
                                output=(
                                    output
                                ),
                                reference_number=(
                                    reference_number
                                ),
                                primary_colour=(
                                    primary_colour
                                ),
                                secondary_colour=(
                                    secondary_colour
                                ),
                            )
                        )

                        generated += (
                            newly_generated
                        )

                        if (
                            max_outputs
                            is not None
                            and generated
                            >= max_outputs
                        ):
                            stopped_for_max_outputs = (
                                True
                            )

                            break

                finally:
                    queue = (
                        await client.request(
                            "GET",
                            "/queue",
                        )
                    ).json()

                    if (
                        not queue.get(
                            "queue_running"
                        )
                        and not queue.get(
                            "queue_pending"
                        )
                    ):
                        await (
                            client.release_models()
                        )

            completed = (
                state.completed_outputs
            )

            failed = sum(
                1
                for output
                in state.outputs
                if (
                    output.status
                    == "failed"
                )
            )

            pending = sum(
                1
                for output
                in state.outputs
                if (
                    output.status
                    == "pending"
                )
            )

            if (
                stopped_for_max_outputs
                and pending
                > 0
            ):
                state.status = (
                    "processing"
                )

                state.current_stage = (
                    "generation_paused"
                )

                state.error = (
                    f"Generation paused after "
                    f"{generated} new outputs; "
                    f"{completed}/"
                    f"{state.total_outputs} "
                    "candidates complete."
                )

            elif (
                completed
                > 0
            ):
                state.status = (
                    "awaiting_selection"
                )

                if (
                    completed
                    == state.total_outputs
                ):
                    state.current_stage = (
                        "backgrounds_complete"
                    )

                    state.error = None

                else:
                    state.current_stage = (
                        "backgrounds_complete_with_failures"
                    )

                    state.error = (
                        f"{completed}/"
                        f"{state.total_outputs} "
                        "background candidates completed; "
                        f"{failed} failed; "
                        f"{pending} remain pending."
                    )

            else:
                state.status = (
                    "failed"
                )

                state.current_stage = (
                    "generation_failed"
                )

                state.error = (
                    "No valid background candidates "
                    "were produced."
                )

            persist_image_job_state(
                state
            )

            return (
                state
            )

        except Exception as error:
            state.status = (
                "failed"
            )

            state.current_stage = (
                "generation_failed"
            )

            state.error = (
                str(
                    error
                )
            )

            persist_image_job_state(
                state
            )

            raise
