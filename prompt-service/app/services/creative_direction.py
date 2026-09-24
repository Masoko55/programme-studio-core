import json
import logging
import re

from app.config.settings import settings
from app.schemas.creative_direction import (
    CreativeDirectionOutput,
    LayoutGuidance,
    LayoutZone,
)
from app.services.ollama import generate_text


logger = logging.getLogger("uvicorn.error")


POSITIVE_PROMPT_FORBIDDEN_TERMS = (
    "typography",
    "lettering",
    "watermark",
    "readable text",
    "written text",
    "text",
    "words",
    "writing",
    "logo",
    "portrait",
    "person",
    "people",
    "human",
    "face",
    "figure",
    # These terms are usually evidence that a model is describing the composed
    # programme rather than a background.  The Composer is the only component
    # allowed to place the title or agenda on the final page.
    "title",
    "programme",
    "program",
    "agenda",
    "schedule",
    "display",
    "displayed",
    "organize",
    "organized",
    "organised",
)


def fallback_visual_prompt(brief: dict, role: str) -> str:
    """Produce a usable background-only prompt when a model echoes instructions."""
    palette = ", ".join(
        colour
        for colour in (
            brief.get("primary_colour"),
            brief.get("secondary_colour"),
        )
        if colour
    ) or "refined neutral"
    theme = brief.get("theme") or "celebratory event"
    return (
        f"{role} A4 portrait abstract event background, {palette} palette, "
        f"{theme} atmosphere, refined border motifs, quiet central composition, "
        "subtle tactile paper texture, soft cinematic lighting, generous open space"
    )


def sanitize_positive_prompt(
    positive_prompt: str,
    brief: dict,
    role: str,
) -> str:
    """Remove model instructions that would put generated lettering into art.

    The Composer owns every visible character and supplied asset.  A model may
    still echo a prohibition despite structured-output instructions, so retain
    only visual clauses and use a deterministic visual fallback if needed.
    """
    clauses = re.split(r"[,;.!?]+", positive_prompt)
    kept = [
        clause.strip()
        for clause in clauses
        if clause.strip()
        and not any(
            re.search(rf"\b{re.escape(term)}\b", clause, re.IGNORECASE)
            for term in POSITIVE_PROMPT_FORBIDDEN_TERMS
        )
    ]
    sanitized = ", ".join(kept)
    if len(sanitized) < 40 or sanitized.lower().count("do not") >= 2:
        return fallback_visual_prompt(brief, role)
    return sanitized


def build_safe_layout(brief: dict) -> LayoutGuidance:
    """Return print-safe, non-overlapping zones for deterministic composition.

    LLMs describe the visual direction; they do not calculate page geometry.
    Every value keeps at least the required 10 mm margin on a 2480 x 3508 page.
    """
    headshot_supplied = bool(brief.get("headshot_path"))
    logo_supplied = bool(brief.get("logo_path"))

    return LayoutGuidance(
        title_zone=LayoutZone(x=0.10, y=0.08, width=0.80, height=0.12),
        programme_zone=LayoutZone(x=0.10, y=0.47, width=0.80, height=0.40),
        headshot_zone=(
            LayoutZone(x=0.10, y=0.24, width=0.22, height=0.18)
            if headshot_supplied else None
        ),
        logo_zone=(
            LayoutZone(x=0.68, y=0.24, width=0.22, height=0.18)
            if logo_supplied else None
        ),
    )


def get_direction_configuration(
    direction_id: str,
) -> tuple[str, str]:
    normalized_id = direction_id.lower()

    if normalized_id not in {"a", "b", "c"}:
        raise ValueError(
            f"Unsupported creative direction: {direction_id}"
        )

    model = getattr(
        settings,
        f"direction_{normalized_id}_model",
    )

    role = getattr(
        settings,
        f"direction_{normalized_id}_role",
    )

    return model, role


def build_creative_direction_prompt(
    brief: dict,
    direction_id: str,
    role: str,
    correction_error: str | None = None,
) -> str:
    output_schema = CreativeDirectionOutput.model_json_schema()

    requirements = [
        "Use the supplied event brief as the source of truth.",
        "Design only the visual background and composition concept.",
        "Do not render the actual event programme text into the background.",
        "Do not request text, typography, lettering, words, logos, watermarks, "
        "writing, or any readable characters in the positive_prompt.",
        "Do not invent logos.",
        "Do not invent portraits.",
        "Do not generate people, human figures, faces, or portraits in the background.",
        "Return all layout zones as normalized coordinates between 0 and 1.",
        "x and y represent the top-left position of a zone.",
        "width and height represent the size of the zone.",
        "No zone may extend beyond the page boundary.",
        "The title zone and programme zone must not overlap.",
        "Only include a headshot zone when a headshot is supplied.",
        "Only include a logo zone when a logo is supplied.",
        "If no headshot is supplied, headshot_zone must be null.",
        "If no logo is supplied, logo_zone must be null.",
        "Keep sufficient visual space around all reserved content zones.",
        "Keep every reserved zone within a 10 mm safe margin on all page edges.",
        (
            "The positive_prompt must be a clean diffusion/image-generation "
            "prompt describing visual style, composition, atmosphere, "
            "palette, materials, lighting, and background treatment."
        ),
        (
            "Do not copy these instructions into positive_prompt or "
            "negative_prompt."
        ),
        (
            "The positive_prompt must describe the final desired visual, "
            "not explain the task."
        ),
        "Return data matching the supplied output schema.",
    ]

    instruction = {
        "task": (
            "Create one visual creative direction "
            "for an A4 portrait event programme."
        ),
        "direction_id": direction_id.upper(),
        "creative_role": role,
        "requirements": requirements,
        "event_brief": brief,
        "output_schema": output_schema,
    }

    if correction_error is not None:
        instruction["correction_required"] = {
            "previous_validation_error": correction_error,
            "instruction": (
                "Regenerate the direction and correct the validation error. "
                "Return a fresh valid direction. "
                "Do not repeat the previous invalid content."
            ),
        }

    return json.dumps(
        instruction,
        ensure_ascii=False,
        indent=2,
    )


async def generate_creative_direction(
    direction_id: str,
    brief: dict,
    correction_error: str | None = None,
) -> CreativeDirectionOutput:
    model, role = get_direction_configuration(
        direction_id
    )

    if correction_error is None:
        logger.info(
            "Generating direction %s using model %s",
            direction_id.upper(),
            model,
        )
    else:
        logger.info(
            "Regenerating direction %s using model %s because: %s",
            direction_id.upper(),
            model,
            correction_error,
        )

    prompt = build_creative_direction_prompt(
        brief=brief,
        direction_id=direction_id,
        role=role,
        correction_error=correction_error,
    )

    result = await generate_text(
        model=model,
        prompt=prompt,
        response_format=CreativeDirectionOutput.model_json_schema(),
    )

    direction = CreativeDirectionOutput.model_validate_json(
        result["response"]
    )

    sanitized_prompt = sanitize_positive_prompt(
        direction.positive_prompt,
        brief,
        role,
    )
    if sanitized_prompt != direction.positive_prompt:
        logger.info(
            "Sanitized non-visual instruction from direction %s positive prompt",
            direction_id.upper(),
        )
        direction.positive_prompt = sanitized_prompt

    # Geometry is a production constraint, not a creative choice.  Replacing
    # model-proposed zones prevents invalid coordinates from exhausting retries.
    direction.layout_guidance = build_safe_layout(brief)

    if direction.direction_id.upper() != direction_id.upper():
        raise ValueError(
            "Generated direction ID does not match requested direction."
        )

    if direction.role != role:
        raise ValueError(
            f"Generated role '{direction.role}' "
            f"does not match configured role '{role}'."
        )

    logger.info(
        "Completed direction %s using model %s",
        direction_id.upper(),
        model,
    )

    return direction
