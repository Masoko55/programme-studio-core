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


SAFE_MARGIN_X = (
    119
    / 2480
)

SAFE_MARGIN_Y = (
    119
    / 3508
)


def zones_overlap(
    first: LayoutZone,
    second: LayoutZone,
) -> bool:
    first_right = (
        first.x
        + first.width
    )

    first_bottom = (
        first.y
        + first.height
    )

    second_right = (
        second.x
        + second.width
    )

    second_bottom = (
        second.y
        + second.height
    )

    horizontal_overlap = (
        first.x
        < second_right
        and first_right
        > second.x
    )

    vertical_overlap = (
        first.y
        < second_bottom
        and first_bottom
        > second.y
    )

    return (
        horizontal_overlap
        and vertical_overlap
    )


def validate_direction_layout(
    direction: CreativeDirectionOutput,
) -> None:
    layout = (
        direction.layout_guidance
    )

    required_zones = [
        (
            "title_zone",
            layout.title_zone,
        ),
        (
            "programme_zone",
            layout.programme_zone,
        ),
    ]

    optional_zones = []

    if (
        layout.headshot_zone
        is not None
    ):
        optional_zones.append(
            (
                "headshot_zone",
                layout.headshot_zone,
            )
        )

    if (
        layout.logo_zone
        is not None
    ):
        optional_zones.append(
            (
                "logo_zone",
                layout.logo_zone,
            )
        )

    all_zones = (
        required_zones
        + optional_zones
    )

    for (
        name,
        zone,
    ) in all_zones:
        if (
            zone.x
            < SAFE_MARGIN_X
            or zone.y
            < SAFE_MARGIN_Y
            or (
                zone.x
                + zone.width
                > (
                    1
                    - SAFE_MARGIN_X
                )
            )
            or (
                zone.y
                + zone.height
                > (
                    1
                    - SAFE_MARGIN_Y
                )
            )
        ):
            raise ValueError(
                f"{direction.direction_id}: "
                f"{name} violates the "
                "10 mm safe margin."
            )

    for index, (
        first_name,
        first_zone,
    ) in enumerate(
        all_zones
    ):
        for (
            second_name,
            second_zone,
        ) in (
            all_zones[
                index
                + 1:
            ]
        ):
            if (
                zones_overlap(
                    first_zone,
                    second_zone,
                )
            ):
                raise ValueError(
                    f"{direction.direction_id}: "
                    f"{first_name} overlaps "
                    f"{second_name}."
                )


def _normalise_asset_type(
    brief: dict,
) -> str:
    value = (
        brief.get(
            "asset_type"
        )
        or "none"
    )

    return (
        str(
            value
        )
        .strip()
        .lower()
    )


def _normalise_placement(
    value,
) -> str | None:
    if value is None:
        return None

    normalized = (
        str(
            value
        )
        .strip()
        .lower()
    )

    aliases = {
        "centre": "center",
        "middle": "center",
        "top left": "left",
        "top-left": "left",
        "top centre": "center",
        "top-center": "center",
        "top center": "center",
        "top-centre": "center",
        "top right": "right",
        "top-right": "right",
    }

    return (
        aliases.get(
            normalized,
            normalized,
        )
    )


def _expected_asset_x(
    placement: str,
) -> float:
    positions = {
        "left": 0.08,
        "center": 0.39,
        "right": 0.70,
    }

    return (
        positions[
            placement
        ]
    )


def _validate_absent_asset(direction: CreativeDirectionOutput) -> None:
    layout = direction.layout_guidance
    if (
        layout.headshot_zone
        is not None
    ):
        raise ValueError(
            f"{direction.direction_id}: "
            "headshot_zone exists even though "
            "asset_type is none."
        )

    if (
        layout.logo_zone
        is not None
    ):
        raise ValueError(
            f"{direction.direction_id}: "
            "logo_zone exists even though "
            "asset_type is none."
        )


def _asset_zone(direction: CreativeDirectionOutput, asset_type: str):
    layout = direction.layout_guidance
    if asset_type == "headshot":
        if layout.headshot_zone is None:
            raise ValueError(
                f"{direction.direction_id}: "
                "asset_type=headshot requires "
                "headshot_zone."
            )
        if layout.logo_zone is not None:
            raise ValueError(
                f"{direction.direction_id}: "
                "logo_zone must be null when "
                "asset_type=headshot."
            )
        return layout.headshot_zone

    if layout.logo_zone is None:
        raise ValueError(
            f"{direction.direction_id}: "
            "asset_type=logo requires "
            "logo_zone."
        )
    if layout.headshot_zone is not None:
        raise ValueError(
            f"{direction.direction_id}: "
            "headshot_zone must be null when "
            "asset_type=logo."
        )
    return layout.logo_zone


def validate_direction_assets(
    direction: CreativeDirectionOutput,
    brief: dict,
) -> None:
    """Validate the Grill-Me asset type, placement, and reserved layout zone."""

    layout = direction.layout_guidance
    asset_type = _normalise_asset_type(brief)
    placement = _normalise_placement(brief.get("asset_placement"))

    if asset_type == "none":
        _validate_absent_asset(direction)
        return

    if (
        asset_type
        not in {
            "headshot",
            "logo",
        }
    ):
        raise ValueError(
            f"{direction.direction_id}: "
            f"unsupported asset_type "
            f"'{asset_type}'."
        )

    if (
        placement
        not in {
            "left",
            "center",
            "right",
        }
    ):
        raise ValueError(
            f"{direction.direction_id}: "
            "asset placement must be "
            "left, center or right."
        )

    zone = _asset_zone(direction, asset_type)

    #
    # Asset must remain at the top.
    #
    if (
        zone.y
        > 0.10
    ):
        raise ValueError(
            f"{direction.direction_id}: "
            "asset zone must remain at "
            "the top of the page."
        )

    expected_x = (
        _expected_asset_x(
            placement
        )
    )

    if (
        abs(
            zone.x
            - expected_x
        )
        > 0.02
    ):
        raise ValueError(
            f"{direction.direction_id}: "
            f"asset zone does not match "
            f"requested {placement} placement."
        )

    #
    # Every content zone must begin below asset.
    #
    asset_bottom = (
        zone.y
        + zone.height
    )

    if (
        layout.title_zone.y
        <= asset_bottom
    ):
        raise ValueError(
            f"{direction.direction_id}: "
            "title_zone must begin below "
            "the top asset."
        )

    if (
        layout.programme_zone.y
        <= asset_bottom
    ):
        raise ValueError(
            f"{direction.direction_id}: "
            "programme_zone must begin below "
            "the top asset."
        )


def validate_positive_prompt_quality(
    direction: CreativeDirectionOutput,
) -> None:
    prompt = (
        direction
        .positive_prompt
        .strip()
        .lower()
    )

    if (
        len(
            prompt
        )
        < 40
    ):
        raise ValueError(
            f"{direction.direction_id}: "
            "positive_prompt is too short."
        )

    for phrase in (
        FORBIDDEN_PROMPT_PHRASES
    ):
        if (
            phrase
            in prompt
        ):
            raise ValueError(
                f"{direction.direction_id}: "
                "positive_prompt appears to echo "
                "system instructions instead of "
                "describing the intended visual."
            )

    for term in (
        FORBIDDEN_BACKGROUND_CONTENT
    ):
        if re.search(
            rf"\b{re.escape(term)}\b",
            prompt,
        ):
            raise ValueError(
                f"{direction.direction_id}: "
                f"positive_prompt must not request "
                f"'{term}'; downstream composition "
                "handles content overlays separately."
            )

    if (
        prompt.count(
            "do not"
        )
        >= 2
    ):
        raise ValueError(
            f"{direction.direction_id}: "
            "positive_prompt contains too many "
            "negative or instructional constraints."
        )

    word_count = len(
        re.findall(
            r"\b\w+\b",
            prompt,
        )
    )

    if (
        word_count
        < 8
    ):
        raise ValueError(
            f"{direction.direction_id}: "
            "positive_prompt is not descriptive enough."
        )


def validate_negative_prompt_quality(
    direction: CreativeDirectionOutput,
) -> None:
    prompt = (
        direction
        .negative_prompt
        .strip()
    )

    if (
        len(
            prompt
        )
        < 10
    ):
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
