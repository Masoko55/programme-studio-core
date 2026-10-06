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
    "title",
    "programme",
    "program",
    "agenda",
    "schedule",
    "display",
    "character",
    "characters",
    "hero",
    "heroes",
    "superhero",
    "superheroes",
    "spiderman",
    "spider-man",
    "masked person",
    "masked character",
    "costume",
    "costumed",
)


COLOUR_ALIASES = {
    "gray": "grey",
    "golden": "gold",
    "navy blue": "navy",
    "royal": "royal blue",
    "sky": "sky blue",
    "baby blue": "sky blue",
    "light blue": "sky blue",
    "dark blue": "navy",
    "hotpink": "hot pink",
    "rose pink": "rose",
    "rose-pink": "rose",
    "fuchsia": "magenta",
    "lilac": "lavender",
    "wine": "burgundy",
    "wine red": "burgundy",
    "rose-gold": "rose gold",
}


KNOWN_COLOURS = (
    "black",
    "white",
    "grey",
    "silver",
    "charcoal",
    "red",
    "orange",
    "yellow",
    "lime",
    "green",
    "forest green",
    "mint green",
    "teal",
    "turquoise",
    "cyan",
    "aqua",
    "blue",
    "sky blue",
    "royal blue",
    "navy",
    "purple",
    "violet",
    "lavender",
    "magenta",
    "pink",
    "hot pink",
    "rose",
    "coral",
    "peach",
    "maroon",
    "burgundy",
    "brown",
    "beige",
    "cream",
    "champagne",
    "gold",
    "rose gold",
)


def normalize_colour(
    value: str | None,
) -> str | None:
    if not value:
        return None

    normalized = (
        value
        .strip()
        .lower()
        .replace(
            "_",
            " ",
        )
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return COLOUR_ALIASES.get(
        normalized,
        normalized,
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

    primary = colours[0]

    secondary = (
        colours[1]
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
            "A strictly achromatic palette of black and white, "
            "with tonal variation only inside those requested "
            "neutral colours."
        )

    if secondary:
        return (
            f"A controlled {primary} and {secondary} palette, "
            f"with {primary} clearly dominant throughout the "
            f"composition and {secondary} used as the supporting "
            "colour. No third colour family is allowed."
        )

    return (
        f"A controlled {primary} palette with {primary} clearly "
        "dominant throughout the composition. "
        "No additional colour family is allowed."
    )


def palette_negative(
    brief: dict,
) -> str:
    colours = requested_palette(
        brief
    )

    if not colours:
        return (
            "off-palette colours, colour drift, "
            "unrequested accent colours, multicolour palette, "
            "unrelated hues"
        )

    return (
        "off-palette colours, colour drift, "
        "unrequested accent colours, multicolour palette, "
        "unrelated hues, any neutral or chromatic colour family "
        "outside the requested palette. "
        "The only requested colour families are: "
        + ", ".join(
            colours
        )
    )


def safe_layout() -> LayoutGuidance:
    return LayoutGuidance(
        title_zone=(
            LayoutZone(
                x=0.10,
                y=0.08,
                width=0.80,
                height=0.12,
            )
        ),
        programme_zone=(
            LayoutZone(
                x=0.10,
                y=0.47,
                width=0.80,
                height=0.40,
            )
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
        direction_id.lower()
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


def composition_variant(
    direction_id: str,
) -> str:
    variants = {
        "a": (
            "Create a balanced border-led composition with detail "
            "around the outer edges and a calm central field."
        ),
        "b": (
            "Create an asymmetric corner-led composition with visual "
            "weight anchored at opposite corners and a calm overlay field."
        ),
        "c": (
            "Create a structured side-led composition with vertical "
            "or diagonal detail at the edges and a calm central overlay field."
        ),
    }

    return variants[
        direction_id.lower()
    ]


def background_design_summary(
    brief: dict,
) -> str:
    return (
        "Main subject: "
        + str(
            brief.get(
                "background_subject"
            )
            or "abstract event motifs"
        )
        + ". Style: "
        + str(
            brief.get(
                "background_style"
            )
            or "refined event artwork"
        )
        + ". Motifs: "
        + str(
            brief.get(
                "background_motifs"
            )
            or "subtle decorative forms"
        )
        + ". Avoid: "
        + str(
            brief.get(
                "background_exclusions"
            )
            or "unrelated imagery"
        )
        + ". Requested composition: "
        + str(
            brief.get(
                "background_composition"
            )
            or "quiet upper and lower overlay zones"
        )
        + "."
    )


def background_positive_requirements(
    brief: dict,
) -> str:
    return (
        "Main visual subject: "
        + str(
            brief.get(
                "background_subject"
            )
            or "abstract event motifs"
        )
        + ". Visual style: "
        + str(
            brief.get(
                "background_style"
            )
            or "refined event artwork"
        )
        + ". Required motifs: "
        + str(
            brief.get(
                "background_motifs"
            )
            or "subtle decorative forms"
        )
        + ". Composition: "
        + str(
            brief.get(
                "background_composition"
            )
            or "quiet upper and lower overlay zones"
        )
        + "."
    )


def background_negative_constraints(
    brief: dict,
) -> str:
    return str(
        brief.get(
            "background_exclusions"
        )
        or "unrelated generic imagery"
    ).strip()


def fallback_prompt(
    brief: dict,
    role: str,
) -> str:
    palette = (
        positive_palette_description(
            brief
        )
    )

    return (
        f"{role} abstract A4 portrait decorative event background, "
        f"{palette} "
        "strong tonal contrast, "
        "clear separation between light and dark visual regions, "
        "high-contrast composition suitable for readable overlays, "
        "refined layered materials, "
        "controlled lighting, "
        "abstract geometric and environmental motifs, "
        "balanced border details, "
        "subtle depth and texture, "
        "generous visual breathing room, "
        "quiet upper and central regions reserved for later composition. "
        + background_design_summary(
            brief
        )
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

    result = ", ".join(
        clean
    )

    if len(
        result
    ) < 40:
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
        + " "
        + background_positive_requirements(
            brief
        )
        + " Use theme references only as abstract visual motifs. "
        + "Never depict or describe a named person, fictional "
        + "character, hero, superhero, mascot or humanoid subject."
    )


def build_creative_direction_prompt(
    brief: dict,
    direction_id: str,
    role: str,
    correction_error: str | None = None,
) -> str:
    requirements = [
        "Use the supplied event brief as the source of truth.",
        "Create visual background artwork rather than a finished poster.",
        "Keep the positive_prompt purely descriptive.",
        "Put exclusions and unwanted content only in negative_prompt.",
        (
            "If the event brief mentions a named person, fictional "
            "character, superhero, mascot, celebrity or franchise "
            "character, translate that reference into abstract motifs, "
            "shapes, textures, colours, patterns, architecture or "
            "atmosphere only."
        ),
        (
            "Never place the named character, person, hero, superhero, "
            "mascot or humanoid subject itself in positive_prompt."
        ),
        (
            "For comic or superhero-inspired themes, use abstract comic "
            "energy, geometric web patterns, speed lines, city geometry "
            "and colour relationships without depicting any hero or character."
        ),
        (
            "The positive_prompt must describe visual style, composition, "
            "materials, lighting, atmosphere and the requested colour palette."
        ),
        (
            "The requested primary colour must be clearly visible and dominant."
        ),
        (
            "The requested secondary colour must support the primary colour."
        ),
        (
            "Do not introduce any visible colour family that is not explicitly "
            "requested. Tonal variation should remain inside the requested "
            "colour families."
        ),
        (
            "The background must use strong tonal contrast with clearly "
            "separated light and dark values without inventing "
            "unrequested colours for contrast."
        ),
        (
            "The background must remain visually readable when adaptive "
            "dark or light programme text is placed over it later."
        ),
        (
            "Do not make the whole image uniformly pale, uniformly dark, "
            "washed out or low contrast."
        ),
        (
            "The negative_prompt must exclude people, faces, human figures, "
            "portraits, silhouettes, body parts, mannequins, clothing, "
            "characters, heroes, superheroes, mascots, costumes, humanoids, "
            "text, words, letters, numbers, logos and watermarks."
        ),
        (
            "The negative_prompt must exclude every colour family outside "
            "the requested palette without accidentally banning the requested "
            "colour names themselves."
        ),
        (
            "Follow the requested background subject, style, motifs, "
            "exclusions and composition exactly; do not substitute "
            "unrelated generic artwork."
        ),
        (
            "Apply this deliberate composition variation: "
            + composition_variant(
                direction_id
            )
        ),
        (
            "Leave visually quiet regions suitable for title and "
            "programme overlays added later."
        ),
        "Return JSON matching the supplied output schema.",
    ]

    payload = {
        "task": (
            "Create one creative direction for an "
            "A4 portrait event background."
        ),
        "direction_id": (
            direction_id.upper()
        ),
        "composition_variant": (
            composition_variant(
                direction_id
            )
        ),
        "background_design_summary": (
            background_design_summary(
                brief
            )
        ),
        "background_positive_requirements": (
            background_positive_requirements(
                brief
            )
        ),
        "background_negative_constraints": (
            background_negative_constraints(
                brief
            )
        ),
        "creative_role": role,
        "event_brief": brief,
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
        "automatic_rendering_policy": {
            "high_contrast": True,
            "readable_overlay_regions": True,
            "background_only": True,
            "characters_allowed": False,
            "people_allowed": False,
        },
        "requirements": requirements,
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
            direction_id=direction_id,
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

    direction.role = role

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
            for value
            in (
                generated_negative.strip(
                    ", "
                ),
                (
                    "person, people, human, human figure, "
                    "man, woman, child, face, portrait, "
                    "silhouette, body, head, hands, arms, "
                    "legs, clothing, mannequin, character, "
                    "fictional character, hero, superhero, "
                    "masked character, mascot, costume, "
                    "costumed figure, humanoid"
                ),
                (
                    "spiderman, spider-man, spider man"
                ),
                (
                    "text, typography, lettering, words, "
                    "letters, numbers, writing, calligraphy, "
                    "signature, logo, watermark, signage"
                ),
                palette_negative(
                    brief
                ),
                background_negative_constraints(
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