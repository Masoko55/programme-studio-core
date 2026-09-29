import hashlib
import io
from pathlib import Path

import cv2
import numpy as np

from PIL import (
    Image,
    ImageDraw,
    ImageFilter,
    ImageFont,
    ImageOps,
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


def normalized_zone_to_pixels(
    zone: dict,
) -> tuple[
    int,
    int,
    int,
    int,
]:
    x = float(
        zone[
            "x"
        ]
    )

    y = float(
        zone[
            "y"
        ]
    )

    width = float(
        zone[
            "width"
        ]
    )

    height = float(
        zone[
            "height"
        ]
    )

    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > 1.000001
        or y + height > 1.000001
    ):
        raise ValueError(
            "Invalid layout zone: outside page."
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
        result[
            0
        ]
        < SAFE_MARGIN_X
        or result[
            1
        ]
        < SAFE_MARGIN_Y
        or result[
            2
        ]
        > (
            CANVAS_WIDTH
            - SAFE_MARGIN_X
        )
        or result[
            3
        ]
        > (
            CANVAS_HEIGHT
            - SAFE_MARGIN_Y
        )
    ):
        raise ValueError(
            "Layout zone violates the 10 mm "
            "safe margin."
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

    bases = [
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

    for base in bases:
        path = Path(
            base
            + suffix
            + ".ttf"
        )

        if path.exists():
            return (
                ImageFont.truetype(
                    str(
                        path
                    ),
                    size=size,
                )
            )

    raise RuntimeError(
        "Required scalable fonts are missing."
    )


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    max_width: int,
) -> list[str]:
    lines = []

    for paragraph in (
        str(
            text
        ).split(
            "\n"
        )
    ):
        current = ""

        for word in (
            paragraph.split(
                " "
            )
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
                current = candidate

            else:
                if current:
                    lines.append(
                        current
                    )

                current = word

        if current:
            lines.append(
                current
            )

    return lines


def _srgb_channel(
    value: float,
) -> float:
    value = (
        value
        / 255.0
    )

    if (
        value
        <= 0.04045
    ):
        return (
            value
            / 12.92
        )

    return (
        (
            (
                value
                + 0.055
            )
            / 1.055
        )
        ** 2.4
    )


def relative_luminance(
    rgb: tuple[
        int,
        int,
        int,
    ],
) -> float:
    red = _srgb_channel(
        rgb[
            0
        ]
    )

    green = _srgb_channel(
        rgb[
            1
        ]
    )

    blue = _srgb_channel(
        rgb[
            2
        ]
    )

    return (
        0.2126
        * red
        + 0.7152
        * green
        + 0.0722
        * blue
    )


def region_statistics(
    image: Image.Image,
    box: tuple[
        int,
        int,
        int,
        int,
    ],
) -> dict:
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
                64,
                64,
            ),
            Image.Resampling.LANCZOS,
        )
    )

    statistics = (
        ImageStat.Stat(
            region
        )
    )

    mean_rgb = tuple(
        round(
            value
        )
        for value
        in statistics.mean[
            :3
        ]
    )

    luminances = []

    for pixel in (
        region.getdata()
    ):
        luminances.append(
            relative_luminance(
                pixel
            )
        )

    average_luminance = (
        sum(
            luminances
        )
        / len(
            luminances
        )
    )

    variance = (
        sum(
            (
                value
                - average_luminance
            )
            ** 2
            for value
            in luminances
        )
        / len(
            luminances
        )
    )

    dark_fraction = (
        sum(
            1
            for value
            in luminances
            if value < 0.30
        )
        / len(
            luminances
        )
    )

    light_fraction = (
        sum(
            1
            for value
            in luminances
            if value > 0.65
        )
        / len(
            luminances
        )
    )

    return {
        "mean_rgb": (
            mean_rgb
        ),
        "average_luminance": (
            average_luminance
        ),
        "variance": (
            variance
        ),
        "dark_fraction": (
            dark_fraction
        ),
        "light_fraction": (
            light_fraction
        ),
    }


def choose_panel_style(
    statistics: dict,
) -> dict:
    luminance = (
        statistics[
            "average_luminance"
        ]
    )

    variance = (
        statistics[
            "variance"
        ]
    )

    dark_fraction = (
        statistics[
            "dark_fraction"
        ]
    )

    light_fraction = (
        statistics[
            "light_fraction"
        ]
    )

    highly_mixed = (
        variance > 0.07
        or (
            dark_fraction > 0.22
            and light_fraction > 0.22
        )
    )

    if (
        luminance >= 0.52
    ):
        return {
            "text": (
                18,
                18,
                20,
                255,
            ),
            "panel_fill": (
                255,
                255,
                255,
                (
                    150
                    if highly_mixed
                    else 72
                ),
            ),
            "border": (
                15,
                15,
                18,
                150,
            ),
            "highlight": (
                255,
                255,
                255,
                110,
            ),
        }

    return {
        "text": (
            246,
            246,
            246,
            255,
        ),
        "panel_fill": (
            0,
            0,
            0,
            (
                150
                if highly_mixed
                else 76
            ),
        ),
        "border": (
            245,
            245,
            245,
            145,
        ),
        "highlight": (
            255,
            255,
            255,
            100,
        ),
    }


def liquid_glass_panel(
    image: Image.Image,
    box: tuple[
        int,
        int,
        int,
        int,
    ],
    style: dict,
) -> None:
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

    radius = 28

    background_region = (
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

    mask = Image.new(
        "L",
        (
            width,
            height,
        ),
        0,
    )

    ImageDraw.Draw(
        mask
    ).rounded_rectangle(
        (
            0,
            0,
            width - 1,
            height - 1,
        ),
        radius=radius,
        fill=125,
    )

    blurred_layer = Image.new(
        "RGBA",
        image.size,
        (
            0,
            0,
            0,
            0,
        ),
    )

    blurred_layer.paste(
        background_region,
        (
            left,
            top,
        ),
        mask,
    )

    image.alpha_composite(
        blurred_layer
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
        fill=(
            style[
                "panel_fill"
            ]
        ),
        outline=(
            style[
                "border"
            ]
        ),
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
        fill=(
            style[
                "highlight"
            ]
        ),
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
    max_size: int,
    min_size: int = 42,
) -> None:
    (
        left,
        top,
        right,
        bottom,
    ) = (
        normalized_zone_to_pixels(
            zone
        )
    )

    padding = 32

    available_width = (
        right
        - left
        - (
            padding
            * 2
        )
    )

    draw = ImageDraw.Draw(
        image
    )

    runs = []

    for size in range(
        max_size,
        min_size - 1,
        -2,
    ):
        candidate_runs = []

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

            lines = wrap_text(
                draw,
                str(
                    text
                ),
                font,
                available_width,
            )

            for line in lines:
                candidate_runs.append(
                    (
                        line,
                        font,
                        line_height,
                    )
                )

        total_height = sum(
            item[
                2
            ]
            for item
            in candidate_runs
        )

        if (
            total_height
            <= (
                bottom
                - top
                - (
                    padding
                    * 2
                )
            )
        ):
            runs = (
                candidate_runs
            )

            break

    if not runs:
        raise ValueError(
            "CONTENT_TOO_LONG: text does not fit "
            "inside the reserved zone."
        )

    content_height = sum(
        item[
            2
        ]
        for item
        in runs
    )

    panel_bottom = min(
        bottom,
        (
            top
            + content_height
            + (
                padding
                * 2
            )
        ),
    )

    panel_box = (
        left,
        top,
        right,
        panel_bottom,
    )

    statistics = (
        region_statistics(
            image,
            panel_box,
        )
    )

    style = (
        choose_panel_style(
            statistics
        )
    )

    liquid_glass_panel(
        image,
        panel_box,
        style,
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
            fill=(
                style[
                    "text"
                ]
            ),
            anchor="lt",
        )

        y += (
            line_height
        )


def remove_logo_edge_matte(
    asset: Image.Image,
) -> Image.Image:
    rgba = (
        asset
        .convert(
            "RGBA"
        )
    )

    array = np.array(
        rgba,
        dtype=np.uint8,
    )

    rgb = (
        array[
            :,
            :,
            :3
        ]
    )

    minimum = (
        rgb.min(
            axis=2
        )
    )

    maximum = (
        rgb.max(
            axis=2
        )
    )

    spread = (
        maximum
        - minimum
    )

    candidate = (
        (
            minimum >= 235
        )
        & (
            spread <= 28
        )
    ).astype(
        np.uint8
    )

    count, labels = (
        cv2.connectedComponents(
            candidate,
            connectivity=8,
        )
    )

    if count <= 1:
        return rgba

    border_labels = set(
        np.unique(
            np.concatenate(
                [
                    labels[
                        0,
                        :
                    ],
                    labels[
                        -1,
                        :
                    ],
                    labels[
                        :,
                        0
                    ],
                    labels[
                        :,
                        -1
                    ],
                ]
            )
        ).tolist()
    )

    border_labels.discard(
        0
    )

    if not border_labels:
        return rgba

    matte = np.isin(
        labels,
        list(
            border_labels
        ),
    )

    alpha = (
        array[
            :,
            :,
            3
        ]
    )

    alpha[
        matte
    ] = 0

    array[
        :,
        :,
        3
    ] = alpha

    return Image.fromarray(
        array,
        mode="RGBA",
    )


def mask_asset_shape(
    asset: Image.Image,
    shape: str | None,
) -> Image.Image:
    if (
        shape
        not in {
            "circle",
            "rounded",
        }
    ):
        return asset

    mask = Image.new(
        "L",
        asset.size,
        0,
    )

    draw = ImageDraw.Draw(
        mask
    )

    if (
        shape
        == "circle"
    ):
        draw.ellipse(
            (
                0,
                0,
                asset.width - 1,
                asset.height - 1,
            ),
            fill=255,
        )

    else:
        draw.rounded_rectangle(
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
                // 8
            ),
            fill=255,
        )

    current_alpha = (
        asset.getchannel(
            "A"
        )
    )

    combined_array = np.minimum(
        np.array(
            current_alpha
        ),
        np.array(
            mask
        ),
    ).astype(
        np.uint8
    )

    combined = Image.fromarray(
        combined_array,
        mode="L",
    )

    result = (
        asset.copy()
    )

    result.putalpha(
        combined
    )

    return result


def draw_asset_glass(
    canvas: Image.Image,
    box: tuple[
        int,
        int,
        int,
        int,
    ],
) -> None:
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

    radius = 28

    background_region = (
        canvas
        .crop(
            box
        )
        .filter(
            ImageFilter.GaussianBlur(
                radius=5
            )
        )
    )

    mask = Image.new(
        "L",
        (
            width,
            height,
        ),
        0,
    )

    ImageDraw.Draw(
        mask
    ).rounded_rectangle(
        (
            0,
            0,
            width - 1,
            height - 1,
        ),
        radius=radius,
        fill=38,
    )

    blurred_layer = Image.new(
        "RGBA",
        canvas.size,
        (
            0,
            0,
            0,
            0,
        ),
    )

    blurred_layer.paste(
        background_region,
        (
            left,
            top,
        ),
        mask,
    )

    canvas.alpha_composite(
        blurred_layer
    )

    statistics = (
        region_statistics(
            canvas,
            box,
        )
    )

    style = (
        choose_panel_style(
            statistics
        )
    )

    glass = Image.new(
        "RGBA",
        canvas.size,
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
        fill=(
            style[
                "panel_fill"
            ][
                :3
            ]
            + (
                14,
            )
        ),
        outline=(
            style[
                "border"
            ][
                :3
            ]
            + (
                28,
            )
        ),
        width=2,
    )

    draw.line(
        (
            left
            + radius,
            top
            + 2,
            right
            - radius,
            top
            + 2,
        ),
        fill=(
            255,
            255,
            255,
            22,
        ),
        width=1,
    )

    canvas.alpha_composite(
        glass
    )


def overlay_asset(
    canvas: Image.Image,
    asset_path: str | None,
    zone: dict | None,
    asset_kind: str,
) -> None:
    if not asset_path:
        return

    if not zone:
        raise ValueError(
            "Supplied asset has no reserved layout zone."
        )

    (
        left,
        top,
        right,
        bottom,
    ) = (
        normalized_zone_to_pixels(
            zone
        )
    )

    zone_width = (
        right
        - left
    )

    zone_height = (
        bottom
        - top
    )

    with Image.open(
        asset_path
    ) as source:
        if (
            source.format
            not in {
                "JPEG",
                "PNG",
            }
        ):
            raise ValueError(
                "Supplied asset must be JPEG or PNG."
            )

        asset = (
            source
            .convert(
                "RGBA"
            )
        )

    if (
        asset_kind
        == "headshot"
    ):
        shape = (
            zone.get(
                "shape"
            )
            or "rounded"
        )

        target_width = (
            zone_width
        )

        target_height = (
            zone_height
        )

        if (
            shape
            == "circle"
        ):
            side = min(
                target_width,
                target_height,
            )

            target_width = side
            target_height = side

        asset = ImageOps.fit(
            asset,
            (
                target_width,
                target_height,
            ),
            method=(
                Image.Resampling.LANCZOS
            ),
            centering=(
                0.5,
                0.42,
            ),
        )

        asset = (
            mask_asset_shape(
                asset,
                shape,
            )
        )

    elif (
        asset_kind
        == "logo"
    ):
        asset = (
            remove_logo_edge_matte(
                asset
            )
        )

        inner_width = int(
            zone_width
            * 0.86
        )

        inner_height = int(
            zone_height
            * 0.82
        )

        asset.thumbnail(
            (
                inner_width,
                inner_height,
            ),
            Image.Resampling.LANCZOS,
        )

        draw_asset_glass(
            canvas,
            (
                left,
                top,
                right,
                bottom,
            ),
        )

    else:
        raise ValueError(
            f"Unsupported asset kind: {asset_kind}"
        )

    destination = (
        left
        + (
            zone_width
            - asset.width
        )
        // 2,
        top
        + (
            zone_height
            - asset.height
        )
        // 2,
    )

    canvas.alpha_composite(
        asset,
        dest=destination,
    )


def validate_layout(
    layout_guidance: dict,
) -> None:
    boxes = [
        normalized_zone_to_pixels(
            zone
        )
        for zone
        in layout_guidance.values()
        if (
            isinstance(
                zone,
                dict,
            )
            and "x"
            in zone
        )
    ]

    for index, first in enumerate(
        boxes
    ):
        for second in (
            boxes[
                index
                + 1:
            ]
        ):
            overlaps = (
                first[
                    0
                ]
                < second[
                    2
                ]
                and first[
                    2
                ]
                > second[
                    0
                ]
                and first[
                    1
                ]
                < second[
                    3
                ]
                and first[
                    3
                ]
                > second[
                    1
                ]
            )

            if overlaps:
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

    title_content = [
        (
            brief[
                "title"
            ],
            True,
        )
    ]

    metadata = [
        str(
            brief[
                key
            ]
        )
        for key in (
            "event_date",
            "start_time",
            "venue",
        )
        if (
            brief.get(
                key
            )
        )
    ]

    if metadata:
        title_content.append(
            (
                " | ".join(
                    metadata
                ),
                False,
            )
        )

    draw_panel(
        canvas,
        title_content,
        layout_guidance[
            "title_zone"
        ],
        max_size=80,
    )

    programme = (
        brief.get(
            "programme"
        )
        or []
    )

    if (
        len(
            programme
        )
        > 15
    ):
        raise ValueError(
            "A programme may contain at most 15 rows."
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

        text = (
            f"{time}  {label}"
            if time
            else str(
                label
            )
        )

        rows.append(
            (
                text,
                True,
            )
        )

        if (
            item.get(
                "speaker"
            )
        ):
            rows.append(
                (
                    str(
                        item[
                            "speaker"
                        ]
                    ),
                    False,
                )
            )

        if (
            item.get(
                "description"
            )
        ):
            rows.append(
                (
                    str(
                        item[
                            "description"
                        ]
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
                        f"{item['duration_minutes']} min"
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
            asset_kind=name,
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

    buffer = (
        io.BytesIO()
    )

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
            )
            .hexdigest()
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