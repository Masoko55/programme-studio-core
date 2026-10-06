from __future__ import annotations

import re

import cv2
import numpy as np

from PIL import Image


NAMED_COLOURS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "grey": (128, 128, 128),
    "silver": (185, 185, 190),
    "charcoal": (54, 69, 79),
    "red": (220, 45, 45),
    "orange": (230, 130, 45),
    "yellow": (230, 195, 45),
    "lime": (125, 220, 55),
    "green": (45, 150, 80),
    "forest green": (34, 100, 55),
    "mint green": (152, 255, 152),
    "teal": (35, 135, 135),
    "turquoise": (64, 190, 180),
    "cyan": (55, 170, 190),
    "aqua": (65, 210, 205),
    "blue": (45, 95, 220),
    "sky blue": (90, 175, 235),
    "royal blue": (55, 80, 210),
    "navy": (25, 45, 100),
    "purple": (125, 75, 185),
    "violet": (120, 80, 185),
    "lavender": (180, 150, 220),
    "magenta": (210, 55, 175),
    "pink": (235, 115, 170),
    "hot pink": (245, 70, 155),
    "rose": (215, 90, 130),
    "coral": (235, 110, 100),
    "peach": (240, 170, 130),
    "maroon": (115, 30, 50),
    "burgundy": (120, 35, 60),
    "brown": (120, 80, 50),
    "beige": (210, 190, 150),
    "cream": (240, 225, 190),
    "champagne": (230, 210, 165),
    "gold": (195, 150, 50),
    "rose gold": (190, 120, 115),
}


COLOUR_ALIASES = {
    "gray": "grey",
    "light gray": "grey",
    "light grey": "grey",
    "dark gray": "charcoal",
    "dark grey": "charcoal",
    "off white": "white",
    "off-white": "white",
    "ivory": "cream",
    "golden": "gold",
    "golden yellow": "gold",
    "lime green": "lime",
    "lime-green": "lime",
    "electric lime": "lime",
    "electric lime green": "lime",
    "mint": "mint green",
    "forest": "forest green",
    "turquoise blue": "turquoise",
    "aqua blue": "aqua",
    "sky": "sky blue",
    "royal": "royal blue",
    "dark blue": "navy",
    "navy blue": "navy",
    "light blue": "sky blue",
    "baby blue": "sky blue",
    "hotpink": "hot pink",
    "rose pink": "rose",
    "rose-pink": "rose",
    "light pink": "pink",
    "dark pink": "hot pink",
    "deep pink": "hot pink",
    "fuchsia": "magenta",
    "lilac": "lavender",
    "deep purple": "purple",
    "light purple": "lavender",
    "wine": "burgundy",
    "wine red": "burgundy",
    "rose-gold": "rose gold",
}


COLOUR_MODIFIERS = {
    "neon",
    "fluorescent",
    "electric",
    "pastel",
    "bright",
    "vibrant",
    "muted",
    "soft",
    "dark",
    "deep",
    "light",
    "metallic",
    "glowing",
    "glow",
    "luminous",
    "radiant",
}


NEUTRAL_NAMES = {
    "black",
    "white",
    "grey",
    "silver",
    "charcoal",
}


DARK_CHROMATIC_NAMES = {
    "navy",
    "forest green",
    "maroon",
    "burgundy",
    "brown",
}


MAX_OFF_PALETTE_RATIO = 0.12
MIN_PRIMARY_COLOUR_RATIO = 0.05
MIN_SECONDARY_COLOUR_RATIO = 0.008
MIN_BLACK_WHITE_RATIO = 0.90
MAX_HUE_DISTANCE = 20
DARK_MAX_HUE_DISTANCE = 26
DARK_MIN_SATURATION = 22
DARK_MIN_VALUE = 10
STANDARD_MIN_SATURATION = 42
STANDARD_MIN_VALUE = 32

MIN_HOG_PERSON_WEIGHT = 0.75
MIN_FACE_SIZE = 42
MIN_PROFILE_SIZE = 46
MIN_UPPER_BODY_SIZE = 70


BACKGROUND_ONLY_NEGATIVE = (
    "(person:2.4), "
    "(people:2.4), "
    "(human:2.4), "
    "(human figure:2.4), "
    "(character:2.4), "
    "(fictional character:2.4), "
    "(hero:2.4), "
    "(superhero:2.4), "
    "(masked character:2.4), "
    "(costumed figure:2.4), "
    "(mascot:2.4), "
    "(humanoid:2.4), "
    "(spiderman:2.5), "
    "(spider-man:2.5), "
    "(spider man:2.5), "
    "(person:2.2), "
    "(people:2.2), "
    "(human:2.2), "
    "(human figure:2.2), "
    "(man:2.2), "
    "(woman:2.2), "
    "(child:2.2), "
    "(face:2.2), "
    "(portrait:2.2), "
    "(body:2.2), "
    "(silhouette:2.2), "
    "(character:2.2), "
    "(head:2.2), "
    "(hands:2.2), "
    "(arms:2.2), "
    "(legs:2.2), "
    "(clothing:2.2), "
    "(mannequin:2.2), "
    "(human-shaped object:2.2), "
    "(text:2.0), "
    "(words:2.0), "
    "(letters:2.0), "
    "(typography:2.0), "
    "(writing:2.0), "
    "(calligraphy:2.0), "
    "(signature:2.0), "
    "(logo:2.0), "
    "(watermark:2.0), "
    "(poster:2.0), "
    "(card:2.0), "
    "(document:2.0), "
    "(menu:2.0), "
    "(certificate:2.0), "
    "(signage:2.0), "
    "numbers, labels, invitation"
)


BACKGROUND_ONLY_SUFFIX = (
    "A4 portrait decorative event background only. "
    "Pure abstract nonrepresentational visual artwork. "
    "No central subject. "
    "No focal character. "
    "No narrative human scene. "
    "No photographic person. "
    "No human silhouette. "
    "No human-shaped geometry. "
    "No body-shaped form. "
    "No portrait composition. "
    "Absolutely no people, human figures, faces, portraits, "
    "characters, mannequins, heads, bodies, hands, arms, legs, "
    "clothing or human-like subjects. "
    "Absolutely no words, lettering, typography, logos, signatures, "
    "watermarks, labels, numbers or readable characters. "
    "Do not create a finished poster, invitation, card, certificate, "
    "menu, document or signage. "
    "Use abstract patterns, materials, textures, ornament, lines, "
    "gradients, shapes, architecture, lighting and environmental "
    "design elements only. "
    "Preserve visually quiet regions for later title and programme text."
)


FORBIDDEN_PROMPT_TERMS = (
    "typography",
    "lettering",
    "watermark",
    "readable text",
    "written text",
    "text",
    "words",
    "writing",
    "person",
    "people",
    "human",
    "face",
    "figure",
    "portrait",
    "character",
    "characters",
    "hero",
    "heroes",
    "superhero",
    "superheroes",
    "spiderman",
    "spider-man",
    "spider man",
    "masked character",
    "costume",
    "costumed",
    "mascot",
    "humanoid",
)


def _clean_colour_text(
    value: str | None,
) -> str | None:
    if not value:
        return None

    cleaned = (
        value.strip()
        .lower()
        .replace("_", " ")
    )

    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned,
    )

    return cleaned or None


def _extract_hex_colour(
    value: str | None,
) -> tuple[int, int, int] | None:
    if not value:
        return None

    match = re.fullmatch(
        r"#?([0-9a-fA-F]{6})",
        value.strip(),
    )

    if not match:
        return None

    raw = match.group(1)

    return (
        int(raw[0:2], 16),
        int(raw[2:4], 16),
        int(raw[4:6], 16),
    )


def _remove_modifiers(
    value: str,
) -> tuple[str, list[str]]:
    words = value.split()

    modifiers = []
    remaining = []

    for word in words:
        if word in COLOUR_MODIFIERS:
            modifiers.append(
                word
            )
        else:
            remaining.append(
                word
            )

    return (
        " ".join(
            remaining
        ).strip(),
        modifiers,
    )


def colour_descriptor(
    value: str | None,
) -> dict:
    cleaned = (
        _clean_colour_text(
            value
        )
    )

    if not cleaned:
        return {
            "raw": value,
            "cleaned": None,
            "base_name": None,
            "rgb": None,
            "modifiers": [],
            "modifier_only": False,
            "is_hex": False,
        }

    hex_rgb = (
        _extract_hex_colour(
            cleaned
        )
    )

    if (
        hex_rgb
        is not None
    ):
        return {
            "raw": value,
            "cleaned": cleaned,
            "base_name": cleaned,
            "rgb": hex_rgb,
            "modifiers": [],
            "modifier_only": False,
            "is_hex": True,
        }

    alias_direct = (
        COLOUR_ALIASES.get(
            cleaned
        )
    )

    if alias_direct:
        return {
            "raw": value,
            "cleaned": cleaned,
            "base_name": alias_direct,
            "rgb": (
                NAMED_COLOURS.get(
                    alias_direct
                )
            ),
            "modifiers": [],
            "modifier_only": False,
            "is_hex": False,
        }

    (
        base_candidate,
        modifiers,
    ) = (
        _remove_modifiers(
            cleaned
        )
    )

    if not base_candidate:
        return {
            "raw": value,
            "cleaned": cleaned,
            "base_name": None,
            "rgb": None,
            "modifiers": modifiers,
            "modifier_only": bool(
                modifiers
            ),
            "is_hex": False,
        }

    base_candidate = (
        COLOUR_ALIASES.get(
            base_candidate,
            base_candidate,
        )
    )

    if (
        base_candidate
        in NAMED_COLOURS
    ):
        return {
            "raw": value,
            "cleaned": cleaned,
            "base_name": base_candidate,
            "rgb": (
                NAMED_COLOURS[
                    base_candidate
                ]
            ),
            "modifiers": modifiers,
            "modifier_only": False,
            "is_hex": False,
        }

    candidates = sorted(
        set(
            list(
                NAMED_COLOURS.keys()
            )
            + list(
                COLOUR_ALIASES.keys()
            )
        ),
        key=len,
        reverse=True,
    )

    for candidate in candidates:
        if re.search(
            rf"\b{re.escape(candidate)}\b",
            cleaned,
        ):
            canonical = (
                COLOUR_ALIASES.get(
                    candidate,
                    candidate,
                )
            )

            if (
                canonical
                in NAMED_COLOURS
            ):
                remaining_words = (
                    cleaned.replace(
                        candidate,
                        " ",
                        1,
                    )
                    .strip()
                    .split()
                )

                detected_modifiers = [
                    word
                    for word
                    in remaining_words
                    if (
                        word
                        in COLOUR_MODIFIERS
                    )
                ]

                return {
                    "raw": value,
                    "cleaned": cleaned,
                    "base_name": canonical,
                    "rgb": (
                        NAMED_COLOURS[
                            canonical
                        ]
                    ),
                    "modifiers": (
                        modifiers
                        + detected_modifiers
                    ),
                    "modifier_only": False,
                    "is_hex": False,
                }

    return {
        "raw": value,
        "cleaned": cleaned,
        "base_name": None,
        "rgb": None,
        "modifiers": modifiers,
        "modifier_only": bool(
            modifiers
        ),
        "is_hex": False,
    }


def normalize_colour_name(
    value: str | None,
) -> str | None:
    return (
        colour_descriptor(
            value
        )[
            "base_name"
        ]
    )


def parse_colour(
    value: str | None,
) -> tuple[int, int, int] | None:
    return (
        colour_descriptor(
            value
        )[
            "rgb"
        ]
    )


def _prompt_colour_label(
    value: str | None,
) -> str | None:
    descriptor = (
        colour_descriptor(
            value
        )
    )

    if not (
        descriptor[
            "cleaned"
        ]
    ):
        return None

    return (
        descriptor[
            "cleaned"
        ]
    )


def requested_palette_names(
    primary_colour: str | None,
    secondary_colour: str | None,
) -> list[str]:
    colours = []

    for value in (
        primary_colour,
        secondary_colour,
    ):
        descriptor = (
            colour_descriptor(
                value
            )
        )

        name = (
            descriptor[
                "base_name"
            ]
        )

        if (
            name
            and name not in colours
        ):
            colours.append(
                name
            )

    return colours


def palette_prompt_contract(
    primary_colour: str | None,
    secondary_colour: str | None,
) -> str:
    primary_descriptor = (
        colour_descriptor(
            primary_colour
        )
    )

    secondary_descriptor = (
        colour_descriptor(
            secondary_colour
        )
    )

    primary_label = (
        _prompt_colour_label(
            primary_colour
        )
    )

    secondary_label = (
        _prompt_colour_label(
            secondary_colour
        )
    )

    primary_base = (
        primary_descriptor[
            "base_name"
        ]
    )

    secondary_base = (
        secondary_descriptor[
            "base_name"
        ]
    )

    statements = [
        (
            "STRICT COLOUR PALETTE CONTRACT."
        ),
        (
            "Do not substitute the requested colours "
            "with colours that merely fit the event theme."
        ),
        (
            "Do not introduce unrelated chromatic accent colours."
        ),
    ]

    if (
        primary_label
        and primary_base
    ):
        statements.append(
            (
                f"The primary colour is {primary_label}. "
                "It must be clearly visible and visually dominant."
            )
        )

    elif (
        primary_descriptor[
            "modifier_only"
        ]
    ):
        modifier_text = (
            " ".join(
                primary_descriptor[
                    "modifiers"
                ]
            )
        )

        statements.append(
            (
                f"The requested primary treatment is {modifier_text}. "
                "This is a colour treatment rather than a specific hue. "
                "Apply this treatment only to the explicitly requested "
                "colour family and do not invent unrelated hues."
            )
        )

    if (
        secondary_label
        and secondary_base
    ):
        statements.append(
            (
                f"The secondary colour is {secondary_label}. "
                "It must be visibly present as a supporting colour."
            )
        )

    elif (
        secondary_descriptor[
            "modifier_only"
        ]
    ):
        modifier_text = (
            " ".join(
                secondary_descriptor[
                    "modifiers"
                ]
            )
        )

        statements.append(
            (
                f"The requested secondary treatment is {modifier_text}. "
                "Treat it as a visual finish rather than a hue."
            )
        )

    measurable_names = (
        requested_palette_names(
            primary_colour,
            secondary_colour,
        )
    )

    if (
        set(
            measurable_names
        )
        == {
            "black",
            "white",
        }
    ):
        statements.extend(
            [
                (
                    "This design must remain strictly achromatic."
                ),
                (
                    "Use black, white and neutral grey only."
                ),
                (
                    "No coloured lighting or coloured tint."
                ),
            ]
        )

    if (
        primary_descriptor[
            "modifier_only"
        ]
        and secondary_base
    ):
        statements.append(
            (
                f"Use {secondary_label or secondary_base} as the "
                "measurable chromatic colour family and apply the "
                f"{' '.join(primary_descriptor['modifiers'])} "
                "treatment to it."
            )
        )

    if (
        secondary_descriptor[
            "modifier_only"
        ]
        and primary_base
    ):
        statements.append(
            (
                f"Use {primary_label or primary_base} as the "
                "measurable chromatic colour family and apply the "
                f"{' '.join(secondary_descriptor['modifiers'])} "
                "treatment where appropriate."
            )
        )

    return (
        " ".join(
            statements
        )
    )


def palette_negative_contract(
    primary_colour: str | None,
    secondary_colour: str | None,
) -> str:
    descriptors = [
        colour_descriptor(
            primary_colour
        ),
        colour_descriptor(
            secondary_colour
        ),
    ]

    measurable = [
        descriptor[
            "base_name"
        ]
        for descriptor
        in descriptors
        if (
            descriptor[
                "base_name"
            ]
        )
    ]

    if (
        set(
            measurable
        )
        == {
            "black",
            "white",
        }
    ):
        return (
            "coloured lighting, coloured tint, chromatic accents, "
            "off-palette colours, colour drift, "
            "unrequested accent colours, multicolour palette"
        )

    if measurable:
        allowed_text = (
            ", ".join(
                measurable
            )
        )

        return (
            "off-palette colours, colour drift, "
            "unrequested accent colours, multicolour palette, "
            "rainbow colours, unrelated hues. "
            f"Keep chromatic colours inside the requested "
            f"colour families: {allowed_text}"
        )

    return (
        "off-palette colours, colour drift, "
        "unrequested accent colours, multicolour palette"
    )


def sanitize_background_prompt(
    positive_prompt: str,
) -> str:
    clauses = re.split(
        r"[,;.!?]+",
        positive_prompt,
    )

    kept = []

    for clause in clauses:
        clause = (
            clause.strip()
        )

        if not clause:
            continue

        contains_forbidden = any(
            re.search(
                rf"\b{re.escape(term)}\b",
                clause,
                re.IGNORECASE,
            )
            for term
            in FORBIDDEN_PROMPT_TERMS
        )

        if not (
            contains_forbidden
        ):
            kept.append(
                clause
            )

    return (
        ", ".join(
            kept
        )
        or (
            "refined abstract decorative background "
            "with generous negative space"
        )
    )


def build_engine_prompt(
    positive_prompt: str,
    primary_colour: str | None,
    secondary_colour: str | None,
) -> str:
    visual = (
        sanitize_background_prompt(
            positive_prompt
        )
    )

    palette = (
        palette_prompt_contract(
            primary_colour,
            secondary_colour,
        )
    )

    return (
        "Abstract nonrepresentational event background. "
        "Decorative background surface only. "
        "No human subject and no human narrative scene. "
        f"{palette} "
        f"{visual.rstrip('. ')}. "
        f"{BACKGROUND_ONLY_SUFFIX}"
    )


def _rgb_to_hsv(
    colour: tuple[
        int,
        int,
        int,
    ],
) -> tuple[
    int,
    int,
    int,
]:
    pixel = np.array(
        [
            [
                colour
            ]
        ],
        dtype=np.uint8,
    )

    converted = (
        cv2.cvtColor(
            pixel,
            cv2.COLOR_RGB2HSV,
        )
    )

    hue, saturation, value = (
        converted[
            0,
            0,
        ]
    )

    return (
        int(
            hue
        ),
        int(
            saturation
        ),
        int(
            value
        ),
    )


def _hue_distance(
    first: int,
    second: int,
) -> int:
    raw = abs(
        first
        - second
    )

    return min(
        raw,
        180 - raw,
    )


def _is_dark_target(
    name: str,
    rgb: tuple[
        int,
        int,
        int,
    ],
) -> bool:
    if (
        name
        in DARK_CHROMATIC_NAMES
    ):
        return True

    _, _, value = (
        _rgb_to_hsv(
            rgb
        )
    )

    return (
        value
        <= 125
    )


def _target_hue_limit(
    name: str,
    rgb: tuple[
        int,
        int,
        int,
    ],
) -> int:
    if (
        _is_dark_target(
            name,
            rgb,
        )
    ):
        return (
            DARK_MAX_HUE_DISTANCE
        )

    return (
        MAX_HUE_DISTANCE
    )


def _pixel_can_match_target(
    saturation: int,
    value: int,
    name: str,
    rgb: tuple[
        int,
        int,
        int,
    ],
) -> bool:
    if (
        _is_dark_target(
            name,
            rgb,
        )
    ):
        return (
            saturation
            >= DARK_MIN_SATURATION
            and value
            >= DARK_MIN_VALUE
        )

    return (
        saturation
        >= STANDARD_MIN_SATURATION
        and value
        >= STANDARD_MIN_VALUE
    )


def validate_palette(
    image: Image.Image,
    primary_colour: str | None,
    secondary_colour: str | None,
) -> dict:
    primary_descriptor = (
        colour_descriptor(
            primary_colour
        )
    )

    secondary_descriptor = (
        colour_descriptor(
            secondary_colour
        )
    )

    primary_name = (
        primary_descriptor[
            "base_name"
        ]
    )

    secondary_name = (
        secondary_descriptor[
            "base_name"
        ]
    )

    primary_rgb = (
        primary_descriptor[
            "rgb"
        ]
    )

    secondary_rgb = (
        secondary_descriptor[
            "rgb"
        ]
    )

    requested = [
        (
            primary_name,
            primary_rgb,
        ),
        (
            secondary_name,
            secondary_rgb,
        ),
    ]

    requested = [
        item
        for item
        in requested
        if (
            item[
                0
            ]
            and item[
                1
            ]
            is not None
        )
    ]

    if not requested:
        return {
            "palette_checked": False,
            "reason": (
                "No measurable colour hue was supplied."
            ),
        }

    sample = (
        image
        .convert(
            "RGB"
        )
        .resize(
            (
                128,
                176,
            ),
            Image.Resampling.LANCZOS,
        )
    )

    rgb_array = (
        np.array(
            sample,
            dtype=np.uint8,
        )
    )

    hsv_array = (
        cv2.cvtColor(
            rgb_array,
            cv2.COLOR_RGB2HSV,
        )
    )

    pixels = (
        hsv_array.reshape(
            -1,
            3,
        )
    )

    total = len(
        pixels
    )

    requested_names = {
        name
        for (
            name,
            _,
        )
        in requested
    }

    if (
        requested_names
        == {
            "black",
            "white",
        }
    ):
        saturation = (
            pixels[
                :,
                1
            ]
        )

        monochrome_ratio = float(
            np.mean(
                saturation
                <= 30
            )
        )

        if (
            monochrome_ratio
            < MIN_BLACK_WHITE_RATIO
        ):
            raise ValueError(
                "Generated background left the requested "
                "black-and-white palette "
                f"(monochrome ratio "
                f"{monochrome_ratio:.2f})."
            )

        return {
            "palette_checked": True,
            "palette_mode": (
                "black-white"
            ),
            "palette_match_ratio": (
                monochrome_ratio
            ),
            "off_palette_ratio": (
                1.0
                - monochrome_ratio
            ),
            "primary_colour_ratio": None,
            "secondary_colour_ratio": None,
        }

    chromatic_targets = []

    for (
        name,
        rgb_value,
    ) in requested:
        if (
            name
            not in NEUTRAL_NAMES
        ):
            hue, _, _ = (
                _rgb_to_hsv(
                    rgb_value
                )
            )

            chromatic_targets.append(
                {
                    "name": name,
                    "rgb": rgb_value,
                    "hue": hue,
                    "dark": (
                        _is_dark_target(
                            name,
                            rgb_value,
                        )
                    ),
                    "hue_limit": (
                        _target_hue_limit(
                            name,
                            rgb_value,
                        )
                    ),
                }
            )

    chromatic_counts = {
        target[
            "name"
        ]: 0
        for target
        in chromatic_targets
    }

    neutral_count = 0
    off_palette_count = 0

    for (
        hue,
        saturation,
        value,
    ) in pixels:
        hue = int(
            hue
        )

        saturation = int(
            saturation
        )

        value = int(
            value
        )

        eligible_targets = [
            target
            for target
            in chromatic_targets
            if (
                _pixel_can_match_target(
                    saturation,
                    value,
                    target[
                        "name"
                    ],
                    target[
                        "rgb"
                    ],
                )
            )
        ]

        if not eligible_targets:
            neutral_count += 1
            continue

        best_target = None
        best_distance = None

        for target in (
            eligible_targets
        ):
            distance = (
                _hue_distance(
                    hue,
                    target[
                        "hue"
                    ],
                )
            )

            if (
                best_distance
                is None
                or distance
                < best_distance
            ):
                best_target = target
                best_distance = distance

        if (
            best_target
            is not None
            and best_distance
            is not None
            and best_distance
            <= best_target[
                "hue_limit"
            ]
        ):
            chromatic_counts[
                best_target[
                    "name"
                ]
            ] += 1

        else:
            off_palette_count += 1

    off_palette_ratio = (
        off_palette_count
        / total
    )

    if (
        off_palette_ratio
        > MAX_OFF_PALETTE_RATIO
    ):
        raise ValueError(
            "Generated background contains too much "
            "off-palette colour "
            f"(off-palette ratio "
            f"{off_palette_ratio:.2f})."
        )

    primary_ratio = None

    if (
        primary_name
        and primary_name
        not in NEUTRAL_NAMES
        and primary_rgb
        is not None
    ):
        primary_ratio = (
            chromatic_counts.get(
                primary_name,
                0,
            )
            / total
        )

        if (
            primary_ratio
            < MIN_PRIMARY_COLOUR_RATIO
        ):
            raise ValueError(
                "Generated background does not visibly "
                "contain enough of the requested primary "
                f"colour '{primary_name}' "
                f"(primary ratio "
                f"{primary_ratio:.2f})."
            )

    secondary_ratio = None

    if (
        secondary_name
        and secondary_name
        not in NEUTRAL_NAMES
        and secondary_rgb
        is not None
    ):
        secondary_ratio = (
            chromatic_counts.get(
                secondary_name,
                0,
            )
            / total
        )

        minimum_secondary_ratio = (
            MIN_SECONDARY_COLOUR_RATIO
        )

        if (
            primary_descriptor[
                "modifier_only"
            ]
            and primary_rgb
            is None
        ):
            minimum_secondary_ratio = (
                MIN_PRIMARY_COLOUR_RATIO
            )

        if (
            secondary_ratio
            < minimum_secondary_ratio
        ):
            raise ValueError(
                "Generated background does not visibly "
                "contain enough of the requested secondary "
                f"colour '{secondary_name}' "
                f"(secondary ratio "
                f"{secondary_ratio:.2f})."
            )

    palette_match_ratio = (
        (
            neutral_count
            + sum(
                chromatic_counts.values()
            )
        )
        / total
    )

    return {
        "palette_checked": True,
        "palette_mode": (
            "strict-theme"
        ),
        "palette_match_ratio": (
            palette_match_ratio
        ),
        "off_palette_ratio": (
            off_palette_ratio
        ),
        "primary_colour_ratio": (
            primary_ratio
        ),
        "secondary_colour_ratio": (
            secondary_ratio
        ),
        "chromatic_counts": (
            chromatic_counts
        ),
        "primary_descriptor": {
            "raw": (
                primary_descriptor[
                    "raw"
                ]
            ),
            "base_name": (
                primary_descriptor[
                    "base_name"
                ]
            ),
            "modifiers": (
                primary_descriptor[
                    "modifiers"
                ]
            ),
            "modifier_only": (
                primary_descriptor[
                    "modifier_only"
                ]
            ),
        },
        "secondary_descriptor": {
            "raw": (
                secondary_descriptor[
                    "raw"
                ]
            ),
            "base_name": (
                secondary_descriptor[
                    "base_name"
                ]
            ),
            "modifiers": (
                secondary_descriptor[
                    "modifiers"
                ]
            ),
            "modifier_only": (
                secondary_descriptor[
                    "modifier_only"
                ]
            ),
        },
    }


def _load_cascade(
    filename: str,
):
    path = (
        cv2.data.haarcascades
        + filename
    )

    classifier = (
        cv2.CascadeClassifier(
            path
        )
    )

    if (
        classifier.empty()
    ):
        return None

    return classifier


def _cascade_detect(
    classifier,
    gray: np.ndarray,
    minimum_size: tuple[
        int,
        int,
    ],
    minimum_neighbors: int,
) -> list[
    tuple[
        int,
        int,
        int,
        int,
    ]
]:
    if (
        classifier
        is None
    ):
        return []

    results = (
        classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=(
                minimum_neighbors
            ),
            minSize=(
                minimum_size
            ),
        )
    )

    return [
        tuple(
            int(
                value
            )
            for value
            in rectangle
        )
        for rectangle
        in results
    ]


def detect_human_signals(
    image: Image.Image,
) -> dict:
    rgb = (
        np.array(
            image.convert(
                "RGB"
            )
        )
    )

    bgr = (
        cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR,
        )
    )

    maximum_dimension = max(
        bgr.shape[
            :2
        ]
    )

    if (
        maximum_dimension
        > 1280
    ):
        scale = (
            1280.0
            / maximum_dimension
        )

        bgr = (
            cv2.resize(
                bgr,
                None,
                fx=scale,
                fy=scale,
                interpolation=(
                    cv2.INTER_AREA
                ),
            )
        )

    gray = (
        cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2GRAY,
        )
    )

    hog = (
        cv2.HOGDescriptor()
    )

    hog.setSVMDetector(
        cv2.HOGDescriptor_getDefaultPeopleDetector()
    )

    (
        rectangles,
        weights,
    ) = (
        hog.detectMultiScale(
            bgr,
            winStride=(
                8,
                8,
            ),
            padding=(
                16,
                16,
            ),
            scale=1.05,
        )
    )

    people = []
    people_weights = []

    for (
        rectangle,
        weight,
    ) in zip(
        rectangles,
        weights,
    ):
        numeric_weight = (
            float(
                weight
            )
        )

        if (
            numeric_weight
            >= MIN_HOG_PERSON_WEIGHT
        ):
            people.append(
                tuple(
                    int(
                        value
                    )
                    for value
                    in rectangle
                )
            )

            people_weights.append(
                numeric_weight
            )

    frontal_detector = (
        _load_cascade(
            "haarcascade_frontalface_default.xml"
        )
    )

    profile_detector = (
        _load_cascade(
            "haarcascade_profileface.xml"
        )
    )

    upper_body_detector = (
        _load_cascade(
            "haarcascade_upperbody.xml"
        )
    )

    faces = (
        _cascade_detect(
            frontal_detector,
            gray,
            (
                MIN_FACE_SIZE,
                MIN_FACE_SIZE,
            ),
            6,
        )
    )

    profiles = (
        _cascade_detect(
            profile_detector,
            gray,
            (
                MIN_PROFILE_SIZE,
                MIN_PROFILE_SIZE,
            ),
            6,
        )
    )

    upper_bodies = (
        _cascade_detect(
            upper_body_detector,
            gray,
            (
                MIN_UPPER_BODY_SIZE,
                MIN_UPPER_BODY_SIZE,
            ),
            7,
        )
    )

    strong_upper_body_signal = (
        len(
            upper_bodies
        )
        >= 2
    )

    detected = bool(
        people
        or faces
        or profiles
        or strong_upper_body_signal
    )

    return {
        "people": people,
        "people_weights": (
            people_weights
        ),
        "faces": faces,
        "profiles": profiles,
        "upper_bodies": (
            upper_bodies
        ),
        "strong_upper_body_signal": (
            strong_upper_body_signal
        ),
        "detected": detected,
    }