import json
import logging
import re

from app.config.settings import settings

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
    "headshot",
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
        colour = (
            normalize_colour(
                brief.get(
                    key
                )
            )
        )

        if (
            colour
            and colour
            not in colours
        ):
            colours.append(
                colour
            )

    return colours


def positive_palette_description(
    brief: dict,
) -> str:
    colours = (
        requested_palette(
            brief
        )
    )

    if not colours:
        return (
            "A restrained cohesive colour palette "
            "with balanced tonal variation."
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
    colours = (
        requested_palette(
            brief
        )
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


def _asset_zone(
    placement: str,
) -> LayoutZone:
    normalized = (
        str(
            placement
            or "center"
        )
        .strip()
        .lower()
    )

    aliases = {
        "centre": "center",
        "middle": "center",
        "top-left": "left",
        "top left": "left",
        "top-center": "center",
        "top centre": "center",
        "top center": "center",
        "top-centre": "center",
        "top-right": "right",
        "top right": "right",
    }

    normalized = (
        aliases.get(
            normalized,
            normalized,
        )
    )

    positions = {
        "left": 0.08,
        "center": 0.39,
        "right": 0.70,
    }

    x = (
        positions.get(
            normalized,
            0.39,
        )
    )

    return (
        LayoutZone(
            x=x,
            y=0.045,
            width=0.22,
            height=0.16,
        )
    )


def asset_layout_description(
    brief: dict,
) -> str:
    """
    This description is for the LLM's layout reasoning only.

    IMPORTANT:
    It must never be appended directly to positive_prompt because
    it contains layout terminology such as title/programme/text,
    which the image-prompt validator intentionally forbids.
    """

    asset_type = (
        str(
            brief.get(
                "asset_type"
            )
            or "none"
        )
        .strip()
        .lower()
    )

    if (
        asset_type
        not in {
            "headshot",
            "logo",
        }
    ):
        return (
            "No uploaded asset is enabled. "
            "Use the standard layout zones."
        )

    placement = (
        str(
            brief.get(
                "asset_placement"
            )
            or "center"
        )
        .strip()
        .lower()
    )

    return (
        f"The {asset_type} occupies a reserved top-{placement} "
        "layout zone. All other content zones begin underneath "
        "the reserved asset area."
    )


def safe_layout(
    brief: dict,
) -> LayoutGuidance:
    """
    Layout coordinates are deterministic.

    We do not trust an LLM to calculate asset placement coordinates.
    """

    asset_type = (
        str(
            brief.get(
                "asset_type"
            )
            or "none"
        )
        .strip()
        .lower()
    )

    asset_placement = (
        str(
            brief.get(
                "asset_placement"
            )
            or "center"
        )
        .strip()
        .lower()
    )

    if (
        asset_type
        in {
            "headshot",
            "logo",
        }
    ):
        asset_zone = (
            _asset_zone(
                asset_placement
            )
        )

        #
        # Asset:
        # y = 0.045 -> 0.205
        #
        # Content begins below it.
        #
        title_zone = (
            LayoutZone(
                x=0.10,
                y=0.245,
                width=0.80,
                height=0.12,
            )
        )

        programme_zone = (
            LayoutZone(
                x=0.10,
                y=0.43,
                width=0.80,
                height=0.47,
            )
        )

        if (
            asset_type
            == "headshot"
        ):
            return (
                LayoutGuidance(
                    title_zone=(
                        title_zone
                    ),
                    programme_zone=(
                        programme_zone
                    ),
                    headshot_zone=(
                        asset_zone
                    ),
                    logo_zone=None,
                )
            )

        return (
            LayoutGuidance(
                title_zone=(
                    title_zone
                ),
                programme_zone=(
                    programme_zone
                ),
                headshot_zone=None,
                logo_zone=(
                    asset_zone
                ),
            )
        )

    return (
        LayoutGuidance(
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

    if (
        normalized
        not in {
            "a",
            "b",
            "c",
        }
    ):
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
            "weight anchored at opposite corners and a calm central field."
        ),
        "c": (
            "Create a structured side-led composition with vertical "
            "or diagonal detail at the edges and a calm central field."
        ),
    }

    return (
        variants[
            direction_id.lower()
        ]
    )


def _background_inspiration(brief: dict) -> str:
    """Return the user's plain-language visual idea for direction prompts."""
    return str(
        brief.get("background_inspiration")
        or brief.get("grill_me_context", {}).get("background_generation_summary")
        or "an elegant abstract event background"
    ).strip()


def reference_visual_language(
    brief: dict,
) -> str:
    """Return safe abstract traits for recognizable-reference themes."""
    reference = " ".join(
        str(brief.get(field) or "")
        for field in ("theme", "theme_reference_treatment")
    ).casefold()

    if any(term in reference for term in ("spiderman", "spider-man", "spider man")):
        return (
            "Use radial web geometry, diagonal web strands, angular skyline "
            "silhouettes and kinetic comic-book framing. "
        )

    return ""


def background_design_summary(
    brief: dict,
) -> str:
    """Describe the user's visual request for the creative-direction model."""
    return (
        "User background inspiration: "
        + _background_inspiration(brief)
        + ". "
        + reference_visual_language(brief)
        + "Preserve a calm open central field and keep decorative detail "
        "toward the outer edges. Layout contract: "
        + asset_layout_description(brief)
        + "."
    )


def background_positive_requirements(
    brief: dict,
) -> str:
    """Safe positive-prompt material derived from the Grill-Me answer."""
    return (
        "Make the visual subject clearly recognizable as: "
        + _background_inspiration(brief)
        + ". "
        + reference_visual_language(brief)
        + "Keep decoration at the outer edges and lower corners, "
        "with a calm open central field."
    )


def background_negative_constraints(
    brief: dict,
) -> str:
    return "unrelated generic imagery, crowded centre, readable text, people, characters"


def fallback_prompt(
    brief: dict,
    role: str,
) -> str:
    palette = positive_palette_description(brief)
    return (
        f"{role} A4 portrait decorative event background, {palette} "
        "strong tonal contrast, refined layered materials, controlled lighting, "
        "balanced border details, generous visual breathing room, "
        "calm upper and central regions, "
        + _background_inspiration(brief)
        + "."
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
        result = (
            fallback_prompt(
                brief,
                role,
            )
        )

    palette = (
        positive_palette_description(
            brief
        )
    )

    result = (
        result.rstrip(
            ". "
        )
        + ". "
        + palette
        + " "
        + background_positive_requirements(
            brief
        )
        + " Theme references are represented through abstract "
        + "geometry, patterns, materials, architecture, atmosphere "
        + "and colour relationships."
    )

    #
    # Final defensive sanitisation.
    #
    # Because deterministic content is appended above, run another
    # pass to make sure forbidden rendering concepts were not
    # accidentally reintroduced.
    #
    clauses = re.split(
        r"[,;.!?]+",
        result,
    )

    clean_final = []
    seen_clauses = set()

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

        key = re.sub(
            r"\s+",
            " ",
            clause,
        ).casefold()

        if (
            not blocked
            and key not in seen_clauses
        ):
            clean_final.append(
                clause
            )
            seen_clauses.add(
                key
            )

    final_prompt = (
        ", ".join(
            clean_final
        )
        .strip()
    )

    if (
        len(
            final_prompt
        )
        < 40
    ):
        raise ValueError(
            "Sanitised positive prompt became too short."
        )

    return (
        final_prompt
    )


def deduplicate_negative_prompt(
    value: str,
) -> str:
    """Keep model exclusions compact without dropping any unique restriction."""
    clauses = []
    seen = set()

    for clause in str(value or "").split(","):
        cleaned = " ".join(clause.split())
        key = cleaned.casefold()
        if cleaned and key not in seen:
            clauses.append(cleaned)
            seen.add(key)

    return ", ".join(clauses)


def build_creative_direction_prompt(
    brief: dict,
    direction_id: str,
    role: str,
    correction_error: str | None = None,
) -> str:
    asset_type = (
        str(
            brief.get(
                "asset_type"
            )
            or "none"
        )
        .lower()
    )

    asset_enabled = (
        asset_type
        in {
            "headshot",
            "logo",
        }
    )

    requirements = [
        (
            "Use the supplied event brief as the source of truth."
        ),
        (
            "Create visual background artwork rather than a finished poster."
        ),
        (
            "Keep positive_prompt purely visual and descriptive."
        ),
        (
            "Never mention title, programme, text, logo, headshot, "
            "schedule, agenda or typography inside positive_prompt."
        ),
        (
            "Put exclusions and unwanted rendering concepts only "
            "inside negative_prompt."
        ),
        (
            "If the event brief uses a recognizable person, fictional "
            "character, superhero, mascot, celebrity or franchise, "
            "translate the reference into abstract geometry, materials, "
            "patterns, architecture, atmosphere and colour relationships."
        ),
        (
            "Never place the referenced person or character itself "
            "inside positive_prompt."
        ),
        (
            "The requested primary colour must be clearly visible "
            "and dominant."
        ),
        (
            "The requested secondary colour must visibly support "
            "the primary colour."
        ),
        (
            "Do not introduce visible colour families that were not "
            "explicitly requested."
        ),
        (
            "Use strong tonal contrast without inventing an "
            "unrequested colour family."
        ),
        (
            "Follow background subject, style, motifs, exclusions "
            "and composition exactly."
        ),
        (
            "Apply this composition variation: "
            + composition_variant(
                direction_id
            )
        ),
        (
            "Preserve calm visual regions through the composition."
        ),
        (
            "Layout coordinates must follow the supplied layout contract."
        ),
        (
            "Return JSON matching the supplied output schema."
        ),
    ]

    if asset_enabled:
        requirements.extend(
            [
                (
                    "A reserved top layout zone is required for "
                    f"the selected {asset_type}."
                ),
                (
                    "All non-asset layout zones must begin below "
                    "the reserved top asset zone."
                ),
                (
                    "Do not describe the reserved asset inside "
                    "positive_prompt; it belongs only in layout_guidance."
                ),
            ]
        )

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
        "asset_layout_contract": {
            "enabled": (
                asset_enabled
            ),
            "asset_type": (
                asset_type
            ),
            "horizontal_placement": (
                brief.get(
                    "asset_placement"
                )
            ),
            "vertical_placement": (
                brief.get(
                    "asset_vertical_position"
                )
            ),
            "flow": (
                brief.get(
                    "text_flow"
                )
            ),
            "description": (
                asset_layout_description(
                    brief
                )
            ),
        },
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
        "automatic_rendering_policy": {
            "high_contrast": True,
            "readable_overlay_regions": True,
            "background_only": True,
            "characters_allowed": False,
            "people_allowed": False,
            "asset_enabled": (
                asset_enabled
            ),
            "asset_position_is_top": (
                asset_enabled
            ),
            "content_below_asset": (
                asset_enabled
            ),
        },
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
                "Return a corrected creative direction. "
                "Fix the exact validation problem while preserving "
                "the original event brief."
            ),
        }

    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
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
        deduplicate_negative_prompt(
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
                    "spiderman, spider-man, spider man",
                    (
                        "text, typography, lettering, words, "
                        "letters, numbers, writing, calligraphy, "
                        "signature, logo, watermark, signage"
                    ),
                    palette_negative(brief),
                    background_negative_constraints(brief),
                )
                if value
            )
        )
    )

    #
    # LLM layout output is ignored.
    #
    # The user's Grill-Me placement answer is the source of truth.
    #
    direction.layout_guidance = (
        safe_layout(
            brief
        )
    )

    logger.info(
        "Completed direction %s using model %s",
        direction_id.upper(),
        model,
    )

    return (
        direction
    )
