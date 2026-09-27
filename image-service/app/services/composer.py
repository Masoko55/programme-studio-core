"""Deterministic final programme composition.

Diffusion supplies the background only.

After a user selects one generated background, this module adds event
information and approved assets. Text colours dynamically adapt to the
actual brightness behind each content panel while remaining related to
the user's primary/secondary event palette.
"""

import hashlib
import io
import re
from pathlib import Path

from PIL import (
    Image,
    ImageDraw,
    ImageFilter,
    ImageFont,
    ImageStat,
)

from app.config.settings import (
    settings,
)
from app.schemas.composition import (
    CompositionResult,
)
from app.services.atomic import (
    write_bytes,
)


CANVAS_WIDTH = 2480
CANVAS_HEIGHT = 3508

SAFE_MARGIN_X = 119
SAFE_MARGIN_Y = 119


NAMED_COLOURS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),

    "red": (220, 45, 45),
    "blue": (45, 95, 220),
    "green": (45, 150, 80),

    "pink": (235, 115, 170),
    "purple": (125, 75, 185),
    "violet": (120, 80, 185),

    "orange": (230, 130, 45),
    "yellow": (230, 195, 45),

    "gold": (195, 150, 50),
    "golden": (195, 150, 50),

    "silver": (175, 180, 190),

    "grey": (125, 125, 125),
    "gray": (125, 125, 125),

    "navy": (25, 45, 100),
    "teal": (35, 135, 135),
    "cyan": (55, 170, 190),

    "brown": (120, 80, 50),
    "beige": (210, 190, 150),
    "cream": (240, 225, 190),

    "maroon": (115, 30, 50),
    "burgundy": (120, 35, 60),
}


def normalized_zone_to_pixels(
    zone: dict,
) -> tuple[
    int,
    int,
    int,
    int,
]:
    x = float(
        zone["x"]
    )

    y = float(
        zone["y"]
    )

    width = float(
        zone["width"]
    )

    height = float(
        zone["height"]
    )

    if (
        min(
            x,
            y,
        )
        < 0
        or min(
            width,
            height,
        )
        <= 0
        or x + width
        > 1.000001
        or y + height
        > 1.000001
    ):
        raise ValueError(
            "Invalid layout zone: "
            "outside page."
        )

    result = (
        round(
            x
            * CANVAS_WIDTH
        ),
        round(
            y
            * CANVAS_HEIGHT
        ),
        round(
            (
                x
                + width
            )
            * CANVAS_WIDTH
        ),
        round(
            (
                y
                + height
            )
            * CANVAS_HEIGHT
        ),
    )

    if (
        result[0]
        < SAFE_MARGIN_X
        or result[1]
        < SAFE_MARGIN_Y
        or result[2]
        > CANVAS_WIDTH
        - SAFE_MARGIN_X
        or result[3]
        > CANVAS_HEIGHT
        - SAFE_MARGIN_Y
    ):
        raise ValueError(
            "Layout zone violates "
            "the 10 mm safe margin."
        )

    return result


def load_font(
    size: int,
    bold: bool = False,
):
    suffix = (
        "-Bold"
        if bold
        else ""
    )

    candidates = [
        (
            "/usr/share/fonts/"
            "truetype/dejavu/"
            "DejaVuSans"
        ),
        (
            "/usr/share/fonts/"
            "truetype/liberation2/"
            "LiberationSans"
        ),
    ]

    for base in candidates:
        path = Path(
            base
            + suffix
            + ".ttf"
        )

        if path.exists():
            return (
                ImageFont.truetype(
                    str(path),
                    size=size,
                )
            )

    raise RuntimeError(
        "Required scalable fonts "
        "are missing."
    )


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    max_width: int,
) -> list[str]:
    lines = []

    for paragraph in (
        str(text).split("\n")
    ):
        current = ""

        for word in (
            paragraph.split(" ")
        ):
            candidate = (
                current
                + (
                    " "
                    if current
                    else ""
                )
                + word
            )

            if (
                draw.textlength(
                    candidate,
                    font=font,
                )
                <= max_width
            ):
                current = (
                    candidate
                )

            else:
                if current:
                    lines.append(
                        current
                    )

                current = word

                if (
                    draw.textlength(
                        word,
                        font=font,
                    )
                    > max_width
                ):
                    raise ValueError(
                        "CONTENT_TOO_LONG: "
                        "a word does not fit "
                        "its reserved zone."
                    )

        if current:
            lines.append(
                current
            )

    return lines


def parse_colour(
    value: str | None,
) -> tuple[
    int,
    int,
    int,
] | None:
    if not value:
        return None

    normalized = (
        value
        .strip()
        .lower()
    )

    if normalized in NAMED_COLOURS:
        return (
            NAMED_COLOURS[
                normalized
            ]
        )

    hexadecimal = re.fullmatch(
        r"#?([0-9a-f]{6})",
        normalized,
    )

    if hexadecimal:
        raw = (
            hexadecimal.group(1)
        )

        return (
            int(
                raw[0:2],
                16,
            ),
            int(
                raw[2:4],
                16,
            ),
            int(
                raw[4:6],
                16,
            ),
        )

    for name, rgb in (
        NAMED_COLOURS.items()
    ):
        if re.search(
            rf"\b{re.escape(name)}\b",
            normalized,
        ):
            return rgb

    return None


def srgb_channel(
    value: int,
) -> float:
    channel = (
        value
        / 255.0
    )

    if channel <= 0.04045:
        return (
            channel
            / 12.92
        )

    return (
        (
            (
                channel
                + 0.055
            )
            / 1.055
        )
        ** 2.4
    )


def relative_luminance(
    colour: tuple[
        int,
        int,
        int,
    ],
) -> float:
    red = srgb_channel(
        colour[0]
    )

    green = srgb_channel(
        colour[1]
    )

    blue = srgb_channel(
        colour[2]
    )

    return (
        0.2126
        * red
        + 0.7152
        * green
        + 0.0722
        * blue
    )


def contrast_ratio(
    first: tuple[
        int,
        int,
        int,
    ],
    second: tuple[
        int,
        int,
        int,
    ],
) -> float:
    first_luminance = (
        relative_luminance(
            first
        )
    )

    second_luminance = (
        relative_luminance(
            second
        )
    )

    lighter = max(
        first_luminance,
        second_luminance,
    )

    darker = min(
        first_luminance,
        second_luminance,
    )

    return (
        (
            lighter
            + 0.05
        )
        / (
            darker
            + 0.05
        )
    )


def mix_colour(
    first: tuple[
        int,
        int,
        int,
    ],
    second: tuple[
        int,
        int,
        int,
    ],
    amount: float,
) -> tuple[
    int,
    int,
    int,
]:
    amount = max(
        0.0,
        min(
            1.0,
            amount,
        ),
    )

    return tuple(
        round(
            first[index]
            * (
                1.0
                - amount
            )
            + second[index]
            * amount
        )
        for index in range(
            3
        )
    )


def average_region_colour(
    image: Image.Image,
    box: tuple[
        int,
        int,
        int,
        int,
    ],
) -> tuple[
    int,
    int,
    int,
]:
    """Measure the actual selected background behind a content panel."""

    region = (
        image
        .convert(
            "RGB"
        )
        .crop(
            box
        )
        .resize(
            (
                32,
                32,
            ),
            Image.Resampling.LANCZOS,
        )
    )

    statistics = (
        ImageStat.Stat(
            region
        )
    )

    return tuple(
        round(value)
        for value in (
            statistics.mean[:3]
        )
    )


def sample_background_accent(
    image: Image.Image,
) -> tuple[
    int,
    int,
    int,
]:
    sample = (
        image
        .convert(
            "RGB"
        )
        .resize(
            (
                80,
                112,
            ),
            Image.Resampling.LANCZOS,
        )
    )

    candidates = {}

    for (
        red,
        green,
        blue,
    ) in sample.getdata():
        maximum = max(
            red,
            green,
            blue,
        )

        minimum = min(
            red,
            green,
            blue,
        )

        lightness = (
            maximum
            + minimum
        ) / 510

        saturation = (
            maximum
            - minimum
        )

        if (
            0.16
            <= lightness
            <= 0.84
            and saturation
            >= 24
        ):
            colour = (
                red
                // 16
                * 16,
                green
                // 16
                * 16,
                blue
                // 16
                * 16,
            )

            candidates[
                colour
            ] = (
                candidates.get(
                    colour,
                    0,
                )
                + saturation
            )

    if not candidates:
        return (
            90,
            90,
            90,
        )

    return max(
        candidates,
        key=candidates.get,
    )


def theme_colours(
    image: Image.Image,
    brief: dict,
) -> list[
    tuple[
        int,
        int,
        int,
    ]
]:
    colours = []

    for key in (
        "primary_colour",
        "secondary_colour",
    ):
        parsed = parse_colour(
            brief.get(
                key
            )
        )

        if (
            parsed is not None
            and parsed not in colours
        ):
            colours.append(
                parsed
            )

    if not colours:
        colours.append(
            sample_background_accent(
                image
            )
        )

    return colours


def themed_text_colour(
    image: Image.Image,
    brief: dict,
    background_colour: tuple[
        int,
        int,
        int,
    ],
) -> tuple[
    int,
    int,
    int,
    int,
]:
    """Select readable text while respecting the event palette.

    Dark artwork:
        Prefer a lightened primary/secondary colour.

    Light artwork:
        Prefer a darkened primary/secondary colour.

    The final candidate must have sufficient contrast against the
    measured local background. Otherwise a palette-tinted near-white
    or near-black fallback is used.
    """

    palette = theme_colours(
        image,
        brief,
    )

    background_luminance = (
        relative_luminance(
            background_colour
        )
    )

    background_is_dark = (
        background_luminance
        < 0.34
    )

    candidates = []

    if background_is_dark:
        for colour in palette:
            candidates.extend(
                [
                    mix_colour(
                        colour,
                        (
                            255,
                            255,
                            255,
                        ),
                        0.72,
                    ),
                    mix_colour(
                        colour,
                        (
                            255,
                            255,
                            255,
                        ),
                        0.84,
                    ),
                    mix_colour(
                        colour,
                        (
                            255,
                            255,
                            255,
                        ),
                        0.92,
                    ),
                ]
            )

        palette_average = tuple(
            round(
                sum(
                    colour[index]
                    for colour
                    in palette
                )
                / len(
                    palette
                )
            )
            for index in range(
                3
            )
        )

        candidates.append(
            mix_colour(
                palette_average,
                (
                    255,
                    255,
                    255,
                ),
                0.90,
            )
        )

        candidates.append(
            (
                248,
                248,
                246,
            )
        )

    else:
        for colour in palette:
            candidates.extend(
                [
                    mix_colour(
                        colour,
                        (
                            0,
                            0,
                            0,
                        ),
                        0.62,
                    ),
                    mix_colour(
                        colour,
                        (
                            0,
                            0,
                            0,
                        ),
                        0.74,
                    ),
                    mix_colour(
                        colour,
                        (
                            0,
                            0,
                            0,
                        ),
                        0.84,
                    ),
                ]
            )

        palette_average = tuple(
            round(
                sum(
                    colour[index]
                    for colour
                    in palette
                )
                / len(
                    palette
                )
            )
            for index in range(
                3
            )
        )

        candidates.append(
            mix_colour(
                palette_average,
                (
                    0,
                    0,
                    0,
                ),
                0.88,
            )
        )

        candidates.append(
            (
                24,
                24,
                28,
            )
        )

    best = max(
        candidates,
        key=lambda colour: (
            contrast_ratio(
                colour,
                background_colour,
            )
        ),
    )

    return (
        best[0],
        best[1],
        best[2],
        255,
    )


def panel_accent_colour(
    image: Image.Image,
    brief: dict,
    background_colour: tuple[
        int,
        int,
        int,
    ],
) -> tuple[
    int,
    int,
    int,
    int,
]:
    palette = theme_colours(
        image,
        brief,
    )

    # Prefer the theme colour that is visually most distinct
    # from the local background.
    accent = max(
        palette,
        key=lambda colour: (
            contrast_ratio(
                colour,
                background_colour,
            )
        ),
    )

    return (
        accent[0],
        accent[1],
        accent[2],
        150,
    )


def liquid_glass_panel(
    image: Image.Image,
    box: tuple[
        int,
        int,
        int,
        int,
    ],
    background_colour: tuple[
        int,
        int,
        int,
    ],
    accent: tuple[
        int,
        int,
        int,
        int,
    ],
) -> None:
    """Render an almost-transparent adaptive glass panel."""

    (
        left,
        top,
        right,
        bottom,
    ) = box

    width = (
        right
        - left
    )

    height = (
        bottom
        - top
    )

    if (
        width <= 0
        or height <= 0
    ):
        return

    radius = 28

    source_region = (
        image
        .crop(
            box
        )
        .filter(
            ImageFilter.GaussianBlur(
                radius=15
            )
        )
    )

    blur_mask = Image.new(
        "L",
        (
            width,
            height,
        ),
        0,
    )

    ImageDraw.Draw(
        blur_mask
    ).rounded_rectangle(
        (
            0,
            0,
            width - 1,
            height - 1,
        ),
        radius=radius,
        fill=115,
    )

    blur_layer = Image.new(
        "RGBA",
        image.size,
        (
            0,
            0,
            0,
            0,
        ),
    )

    blur_layer.paste(
        source_region,
        (
            left,
            top,
        ),
        blur_mask,
    )

    image.alpha_composite(
        blur_layer
    )

    background_is_dark = (
        relative_luminance(
            background_colour
        )
        < 0.34
    )

    # Almost-transparent glass:
    #
    # dark backgrounds get a tiny light veil;
    # light backgrounds get a tiny dark veil.
    if background_is_dark:
        glass_fill = (
            255,
            255,
            255,
            34,
        )

        highlight = (
            255,
            255,
            255,
            90,
        )

    else:
        glass_fill = (
            20,
            20,
            24,
            24,
        )

        highlight = (
            255,
            255,
            255,
            70,
        )

    glass = Image.new(
        "RGBA",
        image.size,
        (
            0,
            0,
            0,
            0,
        ),
    )

    draw = ImageDraw.Draw(
        glass,
        "RGBA",
    )

    draw.rounded_rectangle(
        box,
        radius=radius,
        fill=glass_fill,
        outline=accent,
        width=4,
    )

    draw.line(
        (
            left
            + radius,
            top
            + 3,
            right
            - radius,
            top
            + 3,
        ),
        fill=highlight,
        width=2,
    )

    image.alpha_composite(
        glass
    )


def draw_panel(
    image: Image.Image,
    texts: list[
        tuple[
            str,
            bool,
        ]
    ],
    zone: dict,
    brief: dict,
    max_size: int,
    min_size: int = 42,
) -> None:
    (
        left,
        top,
        right,
        bottom,
    ) = normalized_zone_to_pixels(
        zone
    )

    padding = 32

    draw = ImageDraw.Draw(
        image
    )

    available_width = (
        right
        - left
        - (
            2
            * padding
        )
    )

    runs = []

    for size in range(
        max_size,
        min_size - 1,
        -2,
    ):
        candidate_runs = []

        try:
            for (
                text,
                bold,
            ) in texts:
                font = load_font(
                    size,
                    bold,
                )

                ascent, descent = (
                    font.getmetrics()
                )

                line_height = (
                    ascent
                    + descent
                    + 10
                )

                for line in wrap_text(
                    draw,
                    str(text),
                    font,
                    available_width,
                ):
                    candidate_runs.append(
                        (
                            line,
                            font,
                            line_height,
                        )
                    )

            if (
                sum(
                    run[2]
                    for run
                    in candidate_runs
                )
                <= (
                    bottom
                    - top
                    - (
                        2
                        * padding
                    )
                )
            ):
                runs = (
                    candidate_runs
                )

                break

        except ValueError:
            continue

    if not runs:
        raise ValueError(
            "CONTENT_TOO_LONG: "
            "full text cannot fit "
            "inside its reserved zone."
        )

    content_height = sum(
        run[2]
        for run in runs
    )

    panel_bottom = min(
        bottom,
        top
        + content_height
        + (
            2
            * padding
        ),
    )

    panel_box = (
        left,
        top,
        right,
        panel_bottom,
    )

    local_background = (
        average_region_colour(
            image,
            panel_box,
        )
    )

    text_colour = (
        themed_text_colour(
            image,
            brief,
            local_background,
        )
    )

    accent = (
        panel_accent_colour(
            image,
            brief,
            local_background,
        )
    )

    liquid_glass_panel(
        image,
        panel_box,
        local_background,
        accent,
    )

    draw = ImageDraw.Draw(
        image
    )

    y = (
        top
        + padding
    )

    for (
        line,
        font,
        line_height,
    ) in runs:
        draw.text(
            (
                left
                + padding,
                y,
            ),
            line,
            font=font,
            fill=text_colour,
            anchor="lt",
        )

        y += line_height


def overlay_asset(
    canvas: Image.Image,
    asset_path: str | None,
    zone: dict | None,
) -> None:
    if not asset_path:
        return

    if not zone:
        raise ValueError(
            "Supplied asset has no "
            "reserved layout zone."
        )

    (
        left,
        top,
        right,
        bottom,
    ) = normalized_zone_to_pixels(
        zone
    )

    with Image.open(
        asset_path
    ) as source:
        if source.format not in {
            "JPEG",
            "PNG",
        }:
            raise ValueError(
                "Supplied asset must "
                "be JPEG or PNG."
            )

        asset = (
            source
            .convert(
                "RGBA"
            )
        )

    asset.thumbnail(
        (
            right
            - left,
            bottom
            - top,
        ),
        Image.Resampling.LANCZOS,
    )

    if (
        zone.get(
            "shape"
        )
        == "circle"
    ):
        side = min(
            asset.width,
            asset.height,
        )

        asset = asset.crop(
            (
                (
                    asset.width
                    - side
                )
                // 2,
                (
                    asset.height
                    - side
                )
                // 2,
                (
                    asset.width
                    + side
                )
                // 2,
                (
                    asset.height
                    + side
                )
                // 2,
            )
        )

        mask = Image.new(
            "L",
            asset.size,
            0,
        )

        ImageDraw.Draw(
            mask
        ).ellipse(
            (
                0,
                0,
                asset.width - 1,
                asset.height - 1,
            ),
            fill=255,
        )

        asset.putalpha(
            mask
        )

    elif (
        zone.get(
            "shape"
        )
        == "rounded"
    ):
        mask = Image.new(
            "L",
            asset.size,
            0,
        )

        ImageDraw.Draw(
            mask
        ).rounded_rectangle(
            (
                0,
                0,
                asset.width - 1,
                asset.height - 1,
            ),
            radius=(
                min(
                    asset.width,
                    asset.height,
                )
                // 10
            ),
            fill=255,
        )

        asset.putalpha(
            mask
        )

    canvas.alpha_composite(
        asset,
        dest=(
            left
            + (
                right
                - left
                - asset.width
            )
            // 2,
            top
            + (
                bottom
                - top
                - asset.height
            )
            // 2,
        ),
    )


def validate_layout(
    layout_guidance: dict,
) -> None:
    boxes = [
        normalized_zone_to_pixels(
            zone
        )
        for zone in (
            layout_guidance.values()
        )
        if (
            isinstance(
                zone,
                dict,
            )
            and "x" in zone
        )
    ]

    for index, first in enumerate(
        boxes
    ):
        for second in (
            boxes[
                index + 1:
            ]
        ):
            if (
                first[0]
                < second[2]
                and first[2]
                > second[0]
                and first[1]
                < second[3]
                and first[3]
                > second[1]
            ):
                raise ValueError(
                    "Reserved layout zones overlap."
                )


def compose_programme(
    reference_number: str,
    engine_id: str,
    direction_id: str,
    background_path: str,
    brief: dict,
    layout_guidance: dict,
    output_path: Path | None = None,
) -> CompositionResult:
    """Compose the selected background into the single final programme."""

    validate_layout(
        layout_guidance
    )

    with Image.open(
        background_path
    ) as source:
        source.load()

        canvas = (
            source
            .convert(
                "RGBA"
            )
            .resize(
                (
                    CANVAS_WIDTH,
                    CANVAS_HEIGHT,
                ),
                Image.Resampling.LANCZOS,
            )
        )

    title = [
        (
            brief[
                "title"
            ],
            True,
        )
    ]

    details = [
        str(
            brief[key]
        )
        for key in (
            "event_date",
            "start_time",
            "timezone",
            "venue",
        )
        if brief.get(
            key
        )
    ]

    if details:
        title.append(
            (
                " | ".join(
                    details
                ),
                False,
            )
        )

    draw_panel(
        canvas,
        title,
        layout_guidance[
            "title_zone"
        ],
        brief,
        max_size=80,
    )

    programme = (
        brief.get(
            "programme"
        )
        or []
    )

    if len(programme) > 15:
        raise ValueError(
            "A programme may contain "
            "at most 15 rows."
        )

    rows = []

    for item in programme:
        time = (
            item.get(
                "time"
            )
            or item.get(
                "start_time",
                "",
            )
        )

        label = (
            item.get(
                "title"
            )
            or item.get(
                "item",
                "",
            )
        )

        row_text = (
            f"{time}  {label}"
            if time
            else str(label)
        )

        rows.append(
            (
                row_text,
                True,
            )
        )

        for key in (
            "speaker",
            "description",
        ):
            if item.get(
                key
            ):
                rows.append(
                    (
                        str(
                            item[key]
                        ),
                        False,
                    )
                )

        if (
            item.get(
                "duration_minutes"
            )
            is not None
        ):
            rows.append(
                (
                    (
                        f"{item['duration_minutes']} "
                        "min"
                    ),
                    False,
                )
            )

    if rows:
        draw_panel(
            canvas,
            rows,
            layout_guidance[
                "programme_zone"
            ],
            brief,
            max_size=52,
        )

    for name in (
        "headshot",
        "logo",
    ):
        overlay_asset(
            canvas,
            brief.get(
                name
                + "_path"
            ),
            layout_guidance.get(
                name
                + "_zone"
            ),
        )

    path = (
        output_path
        or (
            settings.programme_data_path
            / reference_number
            / "final"
            / "programme.png"
        )
    )

    buffer = io.BytesIO()

    canvas.convert(
        "RGB"
    ).save(
        buffer,
        format="PNG",
        dpi=(
            settings.final_image_dpi,
            settings.final_image_dpi,
        ),
    )

    data = (
        buffer.getvalue()
    )

    write_bytes(
        path,
        data,
    )

    return CompositionResult(
        reference_number=(
            reference_number
        ),
        engine_id=(
            engine_id
        ),
        direction_id=(
            direction_id
        ),
        background_path=(
            background_path
        ),
        output_path=str(
            path
        ),
        sha256=(
            hashlib.sha256(
                data
            ).hexdigest()
        ),
        width=(
            CANVAS_WIDTH
        ),
        height=(
            CANVAS_HEIGHT
        ),
        dpi=(
            settings.final_image_dpi
        ),
    )