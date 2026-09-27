"""Deterministic final composition.

Diffusion supplies only the selected background.

After the user selects one background, this module adds:
- event title
- date/time/venue
- programme rows
- optional approved assets

Text is rendered over translucent liquid-glass panels. The visible colour
treatment derives from the event theme and selected background rather than
using hardcoded white/gold styling.
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
)

from app.config.settings import settings
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

    "red": (220, 40, 40),
    "blue": (40, 90, 210),
    "green": (45, 145, 80),

    "pink": (235, 110, 165),
    "purple": (125, 70, 180),
    "violet": (115, 75, 180),

    "orange": (225, 125, 40),
    "yellow": (225, 185, 40),

    "gold": (190, 145, 45),
    "golden": (190, 145, 45),

    "silver": (165, 170, 180),
    "grey": (120, 120, 120),
    "gray": (120, 120, 120),

    "navy": (25, 45, 95),
    "teal": (35, 125, 125),
    "cyan": (50, 165, 185),

    "brown": (115, 75, 45),
    "beige": (205, 185, 145),
    "cream": (235, 220, 180),

    "maroon": (110, 30, 45),
    "burgundy": (115, 35, 55),
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
        "Required scalable fonts are missing. "
        "Install fonts-dejavu-core."
    )


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    max_width: int,
) -> list[str]:
    lines: list[str] = []

    for paragraph in (
        text.split("\n")
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
    """Parse common colour names and hexadecimal colours."""

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
        value = (
            hexadecimal.group(1)
        )

        return (
            int(
                value[0:2],
                16,
            ),
            int(
                value[2:4],
                16,
            ),
            int(
                value[4:6],
                16,
            ),
        )

    # Allow phrases such as:
    #
    #   dark blue
    #   blush pink
    #   black and white
    #
    # by looking for a known colour token.
    for name, rgb in (
        NAMED_COLOURS.items()
    ):
        if re.search(
            rf"\b{re.escape(name)}\b",
            normalized,
        ):
            return rgb

    return None


def relative_luminance(
    colour: tuple[
        int,
        int,
        int,
    ],
) -> float:
    red, green, blue = colour

    return (
        0.2126
        * red
        + 0.7152
        * green
        + 0.0722
        * blue
    )


def darken_colour(
    colour: tuple[
        int,
        int,
        int,
    ],
    factor: float = 0.32,
) -> tuple[
    int,
    int,
    int,
    int,
]:
    """Produce a dark version of a theme colour."""

    red, green, blue = colour

    result = (
        max(
            8,
            round(
                red
                * factor
            ),
        ),
        max(
            8,
            round(
                green
                * factor
            ),
        ),
        max(
            8,
            round(
                blue
                * factor
            ),
        ),
        255,
    )

    return result


def sample_background_accent(
    image: Image.Image,
) -> tuple[
    int,
    int,
    int,
]:
    """Find a useful mid-tone from the selected background."""

    sample = (
        image
        .convert("RGB")
        .resize(
            (
                80,
                112,
            ),
            Image.Resampling.LANCZOS,
        )
    )

    candidates: dict[
        tuple[
            int,
            int,
            int,
        ],
        int,
    ] = {}

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
                red // 16 * 16,
                green // 16 * 16,
                blue // 16 * 16,
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
            75,
            75,
            75,
        )

    return max(
        candidates,
        key=candidates.get,
    )


def theme_accent_colour(
    image: Image.Image,
    brief: dict,
) -> tuple[
    int,
    int,
    int,
]:
    """Prefer the user's theme colours; fall back to the image."""

    primary = parse_colour(
        brief.get(
            "primary_colour"
        )
    )

    secondary = parse_colour(
        brief.get(
            "secondary_colour"
        )
    )

    if primary is not None:
        # A pure white primary colour would make
        # a weak outline, so prefer the secondary
        # colour if it has stronger visual identity.
        if (
            relative_luminance(
                primary
            )
            > 230
            and secondary
            is not None
        ):
            return secondary

        return primary

    if secondary is not None:
        return secondary

    return (
        sample_background_accent(
            image
        )
    )


def theme_text_colour(
    image: Image.Image,
    brief: dict,
) -> tuple[
    int,
    int,
    int,
    int,
]:
    """Choose text darker than the theme.

    Examples:

    blue theme  -> dark navy
    pink theme  -> dark berry/maroon
    red theme   -> dark burgundy
    black theme -> near black

    Text never becomes a light theme colour because legibility
    takes priority over matching the palette literally.
    """

    accent = (
        theme_accent_colour(
            image,
            brief,
        )
    )

    # White / very light themes still require dark text.
    if (
        relative_luminance(
            accent
        )
        > 215
    ):
        return (
            28,
            28,
            32,
            255,
        )

    text = darken_colour(
        accent,
        factor=0.30,
    )

    # Prevent unusually bright derived colours.
    if (
        relative_luminance(
            text[:3]
        )
        > 95
    ):
        return (
            35,
            35,
            40,
            255,
        )

    return text


def liquid_glass_panel(
    image: Image.Image,
    box: tuple[
        int,
        int,
        int,
        int,
    ],
    accent: tuple[
        int,
        int,
        int,
    ],
) -> None:
    """Render a subtle translucent liquid-glass panel.

    The selected artwork remains clearly visible through the panel.
    There is no opaque white rectangle.
    """

    left, top, right, bottom = box

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

    # Create a softly blurred version of the artwork
    # behind the panel.
    region = image.crop(
        (
            left,
            top,
            right,
            bottom,
        )
    )

    blurred = (
        region
        .filter(
            ImageFilter.GaussianBlur(
                radius=14
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

    mask_draw = (
        ImageDraw.Draw(
            mask
        )
    )

    mask_draw.rounded_rectangle(
        (
            0,
            0,
            width - 1,
            height - 1,
        ),
        radius=radius,
        fill=90,
    )

    blurred_layer = (
        Image.new(
            "RGBA",
            image.size,
            (
                0,
                0,
                0,
                0,
            ),
        )
    )

    blurred_layer.paste(
        blurred,
        (
            left,
            top,
        ),
        mask,
    )

    image.alpha_composite(
        blurred_layer
    )

    # Near-transparent glass tint.
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

    glass_draw = (
        ImageDraw.Draw(
            glass,
            "RGBA",
        )
    )

    glass_draw.rounded_rectangle(
        (
            left,
            top,
            right,
            bottom,
        ),
        radius=radius,
        fill=(
            255,
            255,
            255,
            48,
        ),
        outline=(
            accent[0],
            accent[1],
            accent[2],
            125,
        ),
        width=4,
    )

    # Thin highlight across the upper edge adds the
    # glass-like reflective effect without becoming
    # a solid white container.
    glass_draw.line(
        (
            left + radius,
            top + 3,
            right - radius,
            top + 3,
        ),
        fill=(
            255,
            255,
            255,
            105,
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
    brief: dict,
    max_size: int,
    min_size: int = 42,
) -> None:
    left, top, right, bottom = (
        normalized_zone_to_pixels(
            zone
        )
    )

    padding = 32

    draw = ImageDraw.Draw(
        image
    )

    available_width = (
        right
        - left
        - 2
        * padding
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

                wrapped = wrap_text(
                    draw,
                    str(text),
                    font,
                    available_width,
                )

                for line in wrapped:
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
                    - 2
                    * padding
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
        + 2
        * padding,
    )

    accent = (
        theme_accent_colour(
            image,
            brief,
        )
    )

    text_colour = (
        theme_text_colour(
            image,
            brief,
        )
    )

    liquid_glass_panel(
        image,
        (
            left,
            top,
            right,
            panel_bottom,
        ),
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
        height,
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

        y += height


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

        image = (
            source
            .convert(
                "RGBA"
            )
        )

    image.thumbnail(
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
            image.width,
            image.height,
        )

        image = image.crop(
            (
                (
                    image.width
                    - side
                )
                // 2,
                (
                    image.height
                    - side
                )
                // 2,
                (
                    image.width
                    + side
                )
                // 2,
                (
                    image.height
                    + side
                )
                // 2,
            )
        )

        mask = Image.new(
            "L",
            image.size,
            0,
        )

        ImageDraw.Draw(
            mask
        ).ellipse(
            (
                0,
                0,
                image.width - 1,
                image.height - 1,
            ),
            fill=255,
        )

        image.putalpha(
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
            image.size,
            0,
        )

        ImageDraw.Draw(
            mask
        ).rounded_rectangle(
            (
                0,
                0,
                image.width - 1,
                image.height - 1,
            ),
            radius=(
                min(
                    image.width,
                    image.height,
                )
                // 10
            ),
            fill=255,
        )

        image.putalpha(
            mask
        )

    destination = (
        left
        + (
            right
            - left
            - image.width
        )
        // 2,
        top
        + (
            bottom
            - top
            - image.height
        )
        // 2,
    )

    canvas.alpha_composite(
        image,
        dest=destination,
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
            overlaps = (
                first[0]
                < second[2]
                and first[2]
                > second[0]
                and first[1]
                < second[3]
                and first[3]
                > second[1]
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
    """Compose text/assets only after a background has been selected."""

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

    rows = []

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

        rows.append(
            (
                (
                    f"{time}  {label}"
                    if time
                    else str(label)
                ),
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