import re

from app.schemas.creative_direction import (
    CreativeDirectionOutput,
    LayoutZone,
)


FORBIDDEN_PROMPT_PHRASES = (
    "use the supplied event brief as the source of truth",
    "return all layout zones as normalized coordinates",
    "x and y represent the top-left position",
    "width and height represent the size of the zone",
    "return data matching the supplied output schema",
    "do not invent logos",
    "do not invent portraits",
    "only include a headshot zone",
    "only include a logo zone",
)


FORBIDDEN_BACKGROUND_CONTENT = (
    "typography",
    "lettering",
    "watermark",
    "readable text",
    "written text",
    "text",
    "words",
    "writing",
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


SAFE_MARGIN_X = 119 / 2480
SAFE_MARGIN_Y = 119 / 3508


def zones_overlap(
    first: LayoutZone,
    second: LayoutZone,
) -> bool:
    first_right = first.x + first.width
    first_bottom = first.y + first.height

    second_right = second.x + second.width
    second_bottom = second.y + second.height

    horizontal_overlap = (
        first.x < second_right
        and first_right > second.x
    )

    vertical_overlap = (
        first.y < second_bottom
        and first_bottom > second.y
    )

    return horizontal_overlap and vertical_overlap


def validate_direction_layout(
    direction: CreativeDirectionOutput,
) -> None:
    layout = direction.layout_guidance

    required_zones = [
        ("title_zone", layout.title_zone),
        ("programme_zone", layout.programme_zone),
    ]

    optional_zones = []

    if layout.headshot_zone is not None:
        optional_zones.append(
            ("headshot_zone", layout.headshot_zone)
        )

    if layout.logo_zone is not None:
        optional_zones.append(
            ("logo_zone", layout.logo_zone)
        )

    all_zones = required_zones + optional_zones

    for name, zone in all_zones:
        if (
            zone.x < SAFE_MARGIN_X
            or zone.y < SAFE_MARGIN_Y
            or zone.x + zone.width > 1 - SAFE_MARGIN_X
            or zone.y + zone.height > 1 - SAFE_MARGIN_Y
        ):
            raise ValueError(
                f"{direction.direction_id}: {name} violates "
                "the 10 mm safe margin."
            )

    for index, (
        first_name,
        first_zone,
    ) in enumerate(all_zones):
        for (
            second_name,
            second_zone,
        ) in all_zones[index + 1:]:
            if zones_overlap(
                first_zone,
                second_zone,
            ):
                raise ValueError(
                    f"{direction.direction_id}: "
                    f"{first_name} overlaps "
                    f"{second_name}."
                )


def validate_direction_assets(
    direction: CreativeDirectionOutput,
    brief: dict,
) -> None:
    layout = direction.layout_guidance

    headshot_supplied = bool(
        brief.get("headshot_path")
    )

    logo_supplied = bool(
        brief.get("logo_path")
    )

    if (
        not headshot_supplied
        and layout.headshot_zone is not None
    ):
        raise ValueError(
            f"{direction.direction_id}: "
            "headshot_zone was returned even "
            "though no headshot was supplied."
        )

    if (
        not logo_supplied
        and layout.logo_zone is not None
    ):
        raise ValueError(
            f"{direction.direction_id}: "
            "logo_zone was returned even "
            "though no logo was supplied."
        )


def validate_positive_prompt_quality(
    direction: CreativeDirectionOutput,
) -> None:
    prompt = (
        direction.positive_prompt
        .strip()
        .lower()
    )

    if len(prompt) < 40:
        raise ValueError(
            f"{direction.direction_id}: "
            "positive_prompt is too short."
        )

    for phrase in FORBIDDEN_PROMPT_PHRASES:
        if phrase in prompt:
            raise ValueError(
                f"{direction.direction_id}: "
                "positive_prompt appears to echo "
                "system instructions instead of "
                "describing the intended visual."
            )

    for term in FORBIDDEN_BACKGROUND_CONTENT:
        if re.search(rf"\b{re.escape(term)}\b", prompt):
            raise ValueError(
                f"{direction.direction_id}: positive_prompt "
                f"must not request {term}; the Composer "
                "renders all programme text and assets."
            )

    if prompt.count("do not") >= 2:
        raise ValueError(
            f"{direction.direction_id}: "
            "positive_prompt contains too many "
            "negative/instructional constraints."
        )

    word_count = len(
        re.findall(
            r"\b\w+\b",
            prompt,
        )
    )

    if word_count < 8:
        raise ValueError(
            f"{direction.direction_id}: "
            "positive_prompt is not descriptive enough."
        )


def validate_negative_prompt_quality(
    direction: CreativeDirectionOutput,
) -> None:
    prompt = (
        direction.negative_prompt
        .strip()
    )

    if len(prompt) < 10:
        raise ValueError(
            f"{direction.direction_id}: "
            "negative_prompt is too short."
        )


def validate_creative_direction(
    direction: CreativeDirectionOutput,
    brief: dict,
) -> None:
    validate_direction_layout(
        direction
    )

    validate_direction_assets(
        direction,
        brief,
    )

    validate_positive_prompt_quality(
        direction
    )

    validate_negative_prompt_quality(
        direction
    )
