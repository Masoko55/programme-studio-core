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


logger = logging.getLogger(
    "uvicorn.error"
)


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
    "title",
    "programme",
    "program",
    "agenda",
    "schedule",
    "display",
)


COLOUR_ALIASES = {
    "gray": "grey",
    "golden": "gold",
}


KNOWN_COLOURS = (
    "black",
    "white",
    "grey",
    "silver",
    "red",
    "blue",
    "green",
    "pink",
    "purple",
    "violet",
    "orange",
    "yellow",
    "gold",
    "brown",
    "beige",
    "cream",
    "navy",
    "teal",
    "cyan",
    "maroon",
    "burgundy",
)


NEUTRAL_COLOURS = {
    "black",
    "white",
    "grey",
    "silver",
}


def normalize_colour(
    value: str | None,
) -> str | None:
    if not value:
        return None

    normalized = (
        value
        .strip()
        .lower()
    )

    return (
        COLOUR_ALIASES.get(
            normalized,
            normalized,
        )
    )


def requested_palette(
    brief: dict,
) -> list[str]:
    colours = []

    for key in (
        "primary_colour",
        "secondary_colour",
    ):
        colour = normalize_colour(
            brief.get(
                key
            )
        )

        if (
            colour
            and colour not in colours
        ):
            colours.append(
                colour
            )

    return colours


def positive_palette_description(
    brief: dict,
) -> str:
    colours = requested_palette(
        brief
    )

    if not colours:
        return (
            "A restrained cohesive colour palette "
            "with balanced tonal variation."
        )

    primary = colours[
        0
    ]

    secondary = (
        colours[
            1
        ]
        if len(
            colours
        ) > 1
        else None
    )

    if (
        set(
            colours
        )
        == {
            "black",
            "white",
        }
    ):
        return (
            "A strictly achromatic palette of black, "
            "white and neutral grayscale, with black "
            "as the dominant visual colour and white "
            "providing clean tonal contrast."
        )

    if secondary:
        return (
            f"A controlled {primary} and {secondary} palette, "
            f"with {primary} clearly dominant throughout the "
            f"composition and {secondary} used as the supporting "
            "colour."
        )

    return (
        f"A controlled {primary} palette with {primary} "
        "clearly dominant throughout the composition."
    )


def palette_negative(
    brief: dict,
) -> str:
    allowed = set(
        requested_palette(
            brief
        )
    )

    banned = []

    for colour in KNOWN_COLOURS:
        normalized = normalize_colour(
            colour
        )

        if (
            normalized not in allowed
            and normalized not in banned
        ):
            banned.append(
                normalized
            )

    return (
        "off-palette colours, colour drift, "
        "unrequested accent colours, multicolour palette, "
        + ", ".join(
            banned
        )
    )


def safe_layout() -> LayoutGuidance:
    return LayoutGuidance(
        title_zone=LayoutZone(
            x=0.10,
            y=0.08,
            width=0.80,
            height=0.12,
        ),
        programme_zone=LayoutZone(
            x=0.10,
            y=0.47,
            width=0.80,
            height=0.40,
        ),
        headshot_zone=None,
        logo_zone=None,
    )


def direction_configuration(
    direction_id: str,
) -> tuple[
    str,
    str,
]:
    normalized = (
        direction_id
        .lower()
    )

    if normalized not in {
        "a",
        "b",
        "c",
    }:
        raise ValueError(
            "Unsupported creative direction: "
            f"{direction_id}"
        )

    model = getattr(
        settings,
        f"direction_{normalized}_model",
    )

    role = getattr(
        settings,
        f"direction_{normalized}_role",
    )

    return (
        model,
        role,
    )


def fallback_prompt(
    brief: dict,
    role: str,
) -> str:
    theme = (
        brief.get(
            "theme"
        )
        or "celebration"
    )

    palette = (
        positive_palette_description(
            brief
        )
    )

    return (
        f"{role} abstract A4 portrait decorative event background, "
        f"{theme} atmosphere, "
        f"{palette} "
        "refined layered materials, "
        "soft controlled lighting, "
        "elegant abstract forms, "
        "balanced border details, "
        "subtle depth and texture, "
        "generous visual breathing room, "
        "quiet upper and central regions reserved for later composition."
    )


def sanitize_positive_prompt(
    value: str,
    brief: dict,
    role: str,
) -> str:
    clauses = re.split(
        r"[,;.!?]+",
        value,
    )

    clean = []

    for clause in clauses:
        clause = (
            clause.strip()
        )

        if not clause:
            continue

        blocked = any(
            re.search(
                rf"\b{re.escape(term)}\b",
                clause,
                re.IGNORECASE,
            )
            for term
            in POSITIVE_PROMPT_FORBIDDEN_TERMS
        )

        negative_instruction = bool(
            re.search(
                r"\b(?:do not|don't|no|without|avoid)\b",
                clause,
                re.IGNORECASE,
            )
        )

        if (
            not blocked
            and not negative_instruction
        ):
            clean.append(
                clause
            )

    result = (
        ", ".join(
            clean
        )
    )

    if (
        len(
            result
        )
        < 40
    ):
        return fallback_prompt(
            brief,
            role,
        )

    palette = (
        positive_palette_description(
            brief
        )
    )

    return (
        result.rstrip(
            ". "
        )
        + ". "
        + palette
    )


def build_creative_direction_prompt(
    brief: dict,
    direction_id: str,
    role: str,
    correction_error: str | None = None,
) -> str:
    requirements = [
        (
            "Use the supplied event brief as the source "
            "of truth."
        ),
        (
            "Create visual background artwork rather than "
            "a finished poster."
        ),
        (
            "Keep the positive_prompt purely descriptive."
        ),
        (
            "Put exclusions and unwanted content only in "
            "negative_prompt."
        ),
        (
            "The positive_prompt must describe visual style, "
            "composition, materials, lighting, atmosphere and "
            "the requested colour palette."
        ),
        (
            "The requested primary colour must be clearly "
            "visible and dominant."
        ),
        (
            "The requested secondary colour must support "
            "the primary colour."
        ),
        (
            "The negative_prompt must exclude people, faces, "
            "human figures, portraits, silhouettes, body parts, "
            "mannequins, clothing, text, words, letters, numbers, "
            "logos and watermarks."
        ),
        (
            "The negative_prompt must exclude colour families "
            "outside the requested palette."
        ),
        (
            "Leave visually quiet regions suitable for title "
            "and programme overlays added later."
        ),
        (
            "Return JSON matching the supplied output schema."
        ),
    ]

    payload = {
        "task": (
            "Create one creative direction for an "
            "A4 portrait event background."
        ),
        "direction_id": (
            direction_id.upper()
        ),
        "creative_role": (
            role
        ),
        "event_brief": (
            brief
        ),
        "positive_palette_description": (
            positive_palette_description(
                brief
            )
        ),
        "negative_palette_constraints": (
            palette_negative(
                brief
            )
        ),
        "requirements": (
            requirements
        ),
        "output_schema": (
            CreativeDirectionOutput
            .model_json_schema()
        ),
    }

    if correction_error:
        payload[
            "correction_required"
        ] = {
            "previous_validation_error": (
                correction_error
            ),
            "instruction": (
                "Return a corrected creative direction that "
                "satisfies the schema and validation requirements."
            ),
        }

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


async def generate_creative_direction(
    direction_id: str,
    brief: dict,
    correction_error: str | None = None,
) -> CreativeDirectionOutput:
    model, role = (
        direction_configuration(
            direction_id
        )
    )

    logger.info(
        "Generating direction %s using model %s",
        direction_id.upper(),
        model,
    )

    prompt = (
        build_creative_direction_prompt(
            brief=brief,
            direction_id=(
                direction_id
            ),
            role=role,
            correction_error=(
                correction_error
            ),
        )
    )

    result = (
        await generate_text(
            model=model,
            prompt=prompt,
            response_format=(
                CreativeDirectionOutput
                .model_json_schema()
            ),
        )
    )

    direction = (
        CreativeDirectionOutput
        .model_validate_json(
            result[
                "response"
            ]
        )
    )

    direction.direction_id = (
        direction_id.upper()
    )

    direction.role = (
        role
    )

    direction.positive_prompt = (
        sanitize_positive_prompt(
            direction.positive_prompt,
            brief,
            role,
        )
    )

    generated_negative = (
        direction.negative_prompt
        or ""
    )

    direction.negative_prompt = (
        ", ".join(
            value
            for value in (
                generated_negative.strip(
                    ", "
                ),
                (
                    "person, people, human, human figure, "
                    "man, woman, child, face, portrait, "
                    "silhouette, body, head, hands, arms, "
                    "legs, clothing, mannequin, character"
                ),
                (
                    "text, typography, lettering, words, "
                    "letters, numbers, writing, calligraphy, "
                    "signature, logo, watermark, signage"
                ),
                palette_negative(
                    brief
                ),
            )
            if value
        )
    )

    direction.layout_guidance = (
        safe_layout()
    )

    logger.info(
        "Completed direction %s using model %s",
        direction_id.upper(),
        model,
    )

    return direction