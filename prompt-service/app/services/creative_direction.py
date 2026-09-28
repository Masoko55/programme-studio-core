"""Generate creative directions from a completed Grill-Me brief."""

import json
import logging
import re

from app.config.settings import (
    settings,
)

from app.schemas.creative_direction import (
    CreativeDirectionOutput,
    LayoutGuidance,
    LayoutZone,
)

from app.services.ollama import (
    generate_text,
)


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


def normalize_colour(
    value: str | None,
) -> str | None:
    if not value:
        return None

    value = (
        value
        .strip()
        .lower()
    )

    return (
        COLOUR_ALIASES.get(
            value,
            value,
        )
    )


def requested_palette(
    brief: dict,
) -> list[str]:
    result = []

    for key in (
        "primary_colour",
        "secondary_colour",
    ):
        colour = (
            normalize_colour(
                brief.get(
                    key
                )
            )
        )

        if (
            colour
            and colour not in result
        ):
            result.append(
                colour
            )

    return result


def palette_contract(
    brief: dict,
) -> str:
    colours = (
        requested_palette(
            brief
        )
    )

    if not colours:
        return (
            "Use one restrained coherent palette."
        )

    primary = (
        colours[
            0
        ]
    )

    secondary = (
        colours[
            1
        ]
        if len(
            colours
        ) > 1
        else None
    )

    allowed = (
        primary
        if secondary is None
        else (
            f"{primary} and {secondary}"
        )
    )

    text = [
        (
            "STRICT COLOUR PALETTE CONTRACT: "
            f"use {allowed}."
        ),
        (
            f"{primary} is the primary colour "
            "and must remain clearly visible and dominant."
        ),
        (
            "Do not substitute the requested colours "
            "with theme-associated alternatives."
        ),
        (
            "Do not introduce unrelated accent colours."
        ),
    ]

    if secondary:
        text.append(
            (
                f"{secondary} is the secondary colour."
            )
        )

    if set(
        colours
    ) == {
        "black",
        "white",
    }:
        text.append(
            (
                "Use only black, white and neutral grayscale. "
                "No chromatic colour."
            )
        )

    return " ".join(
        text
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

    for colour in (
        KNOWN_COLOURS
    ):
        normalized = (
            normalize_colour(
                colour
            )
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
        "unrequested accent colours, "
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
        (
            f"direction_"
            f"{normalized}_model"
        ),
    )

    role = getattr(
        settings,
        (
            f"direction_"
            f"{normalized}_role"
        ),
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

    return (
        f"{role} abstract A4 portrait event background. "
        f"{theme} atmosphere. "
        "Decorative abstract texture. "
        "Refined border treatment. "
        "Generous negative space. "
        "No people. No text. "
        f"{palette_contract(brief)}"
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

        if not blocked:
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
        < 30
    ):
        result = (
            fallback_prompt(
                brief,
                role,
            )
        )

    return (
        result.rstrip(
            ". "
        )
        + ". "
        + palette_contract(
            brief
        )
    )


def build_creative_direction_prompt(
    brief: dict,
    direction_id: str,
    role: str,
    correction_error: str | None = None,
) -> str:
    requirements = [
        (
            "Use the supplied event brief as "
            "the source of truth."
        ),

        (
            "Design background artwork only."
        ),

        (
            "Do not render event title, venue, programme "
            "or other event details into the image."
        ),

        (
            "Do not generate text, typography, words, "
            "letters, numbers, logos or watermarks."
        ),

        (
            "Do not generate people, human figures, "
            "faces, silhouettes, portraits, body parts, "
            "mannequins or human-shaped focal objects."
        ),

        (
            "The requested primary and secondary colours "
            "are strict production requirements."
        ),

        (
            "Do not replace the user's palette with "
            "colours that merely suit the event theme."
        ),

        (
            "Leave visually quiet space for title "
            "and programme overlays added later."
        ),

        (
            "Return JSON matching the supplied schema."
        ),
    ]

    instruction = {
        "task": (
            "Create one creative direction for "
            "an A4 portrait event background."
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

        "palette_contract": (
            palette_contract(
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
        instruction[
            "correction_required"
        ] = {
            "previous_validation_error": (
                correction_error
            ),

            "instruction": (
                "Correct the previous validation error "
                "without violating the event brief or "
                "palette contract."
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
            brief,
            direction_id,
            role,
            correction_error,
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

    direction.positive_prompt = (
        sanitize_positive_prompt(
            direction.positive_prompt,
            brief,
            role,
        )
    )

    base_negative = (
        direction.negative_prompt
        or ""
    )

    direction.negative_prompt = (
        ", ".join(
            value
            for value in (
                base_negative.strip(
                    ", "
                ),
                (
                    "people, person, human, human figure, "
                    "face, portrait, silhouette, body, "
                    "hands, arms, legs, mannequin, clothing"
                ),
                (
                    "text, typography, words, letters, "
                    "numbers, writing, logo, watermark"
                ),
                palette_negative(
                    brief
                ),
            )
            if value
        )
    )

    # We do not trust model-generated coordinates.
    direction.layout_guidance = (
        safe_layout()
    )

    if (
        direction.direction_id.upper()
        != direction_id.upper()
    ):
        raise ValueError(
            "Generated direction ID does not "
            "match requested direction."
        )

    if (
        direction.role
        != role
    ):
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