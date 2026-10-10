from __future__ import annotations

import math
import re

import cv2
import numpy as np

from PIL import Image, ImageColor

from app.services.visual_quality import (
    validate_visual_quality,
)


FOREST_GREEN = "forest green"
MINT_GREEN = "mint green"
SKY_BLUE = "sky blue"
ROYAL_BLUE = "royal blue"
HOT_PINK = "hot pink"
ROSE_GOLD = "rose gold"


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
    FOREST_GREEN: (34, 100, 55),
    MINT_GREEN: (152, 255, 152),
    "teal": (35, 135, 135),
    "turquoise": (64, 190, 180),
    "cyan": (55, 170, 190),
    "aqua": (65, 210, 205),
    "blue": (45, 95, 220),
    SKY_BLUE: (90, 175, 235),
    ROYAL_BLUE: (55, 80, 210),
    "navy": (25, 45, 100),
    "purple": (125, 75, 185),
    "violet": (120, 80, 185),
    "lavender": (180, 150, 220),
    "magenta": (210, 55, 175),
    "pink": (235, 115, 170),
    HOT_PINK: (245, 70, 155),
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
    ROSE_GOLD: (190, 120, 115),
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
    "mint": MINT_GREEN,
    "forest": FOREST_GREEN,
    "turquoise blue": "turquoise",
    "aqua blue": "aqua",
    "sky": SKY_BLUE,
    "royal": ROYAL_BLUE,
    "dark blue": "navy",
    "navy blue": "navy",
    "light blue": SKY_BLUE,
    "baby blue": SKY_BLUE,
    "hotpink": HOT_PINK,
    "rose pink": "rose",
    "rose-pink": "rose",
    "light pink": "pink",
    "dark pink": HOT_PINK,
    "deep pink": HOT_PINK,
    "fuchsia": "magenta",
    "lilac": "lavender",
    "deep purple": "purple",
    "light purple": "lavender",
    "wine": "burgundy",
    "wine red": "burgundy",
    "rose-gold": ROSE_GOLD,
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
    FOREST_GREEN,
    "maroon",
    "burgundy",
    "brown",
}


COLOUR_FAMILIES = {
    "red": {
        "red",
        "maroon",
        "burgundy",
    },
    "maroon": {
        "red",
        "maroon",
        "burgundy",
    },
    "burgundy": {
        "red",
        "maroon",
        "burgundy",
    },

    "blue": {
        "blue",
        ROYAL_BLUE,
        "navy",
    },
    ROYAL_BLUE: {
        "blue",
        ROYAL_BLUE,
        "navy",
    },
    "navy": {
        "blue",
        ROYAL_BLUE,
        "navy",
    },

    "green": {
        "green",
        FOREST_GREEN,
    },
    FOREST_GREEN: {
        "green",
        FOREST_GREEN,
    },

    "pink": {
        "pink",
        HOT_PINK,
        "rose",
    },
    HOT_PINK: {
        "pink",
        HOT_PINK,
        "rose",
    },
    "rose": {
        "pink",
        HOT_PINK,
        "rose",
    },

    "purple": {
        "purple",
        "violet",
    },
    "violet": {
        "purple",
        "violet",
    },

    "turquoise": {
        "turquoise",
        "teal",
    },
    "teal": {
        "turquoise",
        "teal",
    },

    "cyan": {
        "cyan",
        "aqua",
    },
    "aqua": {
        "cyan",
        "aqua",
    },

    "gold": {
        "gold",
        "champagne",
    },
    "champagne": {
        "gold",
        "champagne",
    },

    "orange": {
        "orange",
    },
    "yellow": {
        "yellow",
    },
    "lime": {
        "lime",
    },
    MINT_GREEN: {
        MINT_GREEN,
    },
    SKY_BLUE: {
        SKY_BLUE,
    },
    "lavender": {
        "lavender",
    },
    "magenta": {
        "magenta",
    },
    "coral": {
        "coral",
    },
    "peach": {
        "peach",
    },
    "brown": {
        "brown",
    },
    "beige": {
        "beige",
    },
    "cream": {
        "cream",
    },
    ROSE_GOLD: {
        ROSE_GOLD,
    },
}


MAX_OFF_PALETTE_RATIO = 0.05

MIN_PRIMARY_COLOUR_RATIO = 0.05

MIN_SECONDARY_COLOUR_RATIO = 0.008

MIN_PRIMARY_DOMINANCE_MARGIN = 0.05

MIN_BLACK_WHITE_RATIO = 0.90


MAX_HUE_DISTANCE = 20

DARK_MAX_HUE_DISTANCE = 26


DARK_MIN_SATURATION = 22

DARK_MIN_VALUE = 10


STANDARD_MIN_SATURATION = 42

STANDARD_MIN_VALUE = 32


NEUTRAL_MAX_SATURATION = 34


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
    "(man:2.2), "
    "(woman:2.2), "
    "(child:2.2), "
    "(face:2.2), "
    "(portrait:2.2), "
    "(body:2.2), "
    "(silhouette:2.2), "
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
    "(scanlines:2.0), "
    "(raster banding:2.0), "
    "(horizontal stripes:1.8), "
    "(vertical stripes:1.8), "
    "(glitch:2.0), "
    "(moire:2.0), "
    "(corrupted image:2.0), "
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
    "Do not generate full-frame scanlines, raster banding, repetitive "
    "horizontal or vertical line corruption, moire, glitch artefacts, "
    "broken image textures or blank flat image fields. "
    "Do not generate a featureless full-page gradient, plain colour wash, "
    "grain-only texture or noise-only image. "
    "Use coherent abstract patterns, ornament, lines, shapes, architecture "
    "and environmental design elements. "
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
        .replace(
            "_",
            " ",
        )
    )

    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned,
    )

    return (
        cleaned
        or None
    )


def _extract_hex_colour(
    value: str | None,
) -> tuple[
    int,
    int,
    int,
] | None:
    if not value:
        return None

    match = re.fullmatch(
        r"#?([0-9a-fA-F]{6})",
        value.strip(),
    )

    if not match:
        return None

    raw = (
        match.group(
            1
        )
    )

    return (
        int(
            raw[
                0:2
            ],
            16,
        ),
        int(
            raw[
                2:4
            ],
            16,
        ),
        int(
            raw[
                4:6
            ],
            16,
        ),
    )


def _remove_modifiers(
    value: str,
) -> tuple[
    str,
    list[str],
]:
    modifiers = []
    remaining = []

    for word in (
        value.split()
    ):
        if (
            word
            in COLOUR_MODIFIERS
        ):
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

    rgb_match = re.fullmatch(r"rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)", cleaned)
    if rgb_match:
        channels = tuple(map(int, rgb_match.groups()))
        if all(channel <= 255 for channel in channels):
            return {
                "raw": value, "cleaned": cleaned, "base_name": cleaned,
                "rgb": channels, "modifiers": [], "modifier_only": False, "is_hex": False,
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
            "base_name": (
                base_candidate
            ),
            "rgb": (
                NAMED_COLOURS[
                    base_candidate
                ]
            ),
            "modifiers": (
                modifiers
            ),
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
        if not re.search(
            rf"\b{re.escape(candidate)}\b",
            cleaned,
        ):
            continue

        canonical = (
            COLOUR_ALIASES.get(
                candidate,
                candidate,
            )
        )

        if (
            canonical
            not in NAMED_COLOURS
        ):
            continue

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
            "base_name": (
                canonical
            ),
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

    try:
        css_rgb = ImageColor.getrgb(cleaned)
    except ValueError:
        css_rgb = None
    if css_rgb is not None:
        return {
            "raw": value, "cleaned": cleaned, "base_name": cleaned,
            "rgb": css_rgb[:3], "modifiers": [], "modifier_only": False, "is_hex": False,
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


def _restore_chromatic_primary_dominance(target_index, source_saturation, target_names: list[str], usable: list[dict], primary_index: int) -> None:
    #
    # Chromatic primary dominance recovery
    # ============================================================
    #
    # Preserve the existing contract for two distinct chromatic requested
    # families. Only the minimum number of secondary boundary pixels is moved
    # into the primary family.
    #
    if (
        len(usable) > 1
        and target_names[primary_index]
        not in NEUTRAL_NAMES
    ):
        secondary_indices = [
            index
            for index in range(
                len(usable)
            )
            if (
                index != primary_index
                and target_names[index]
                not in NEUTRAL_NAMES
            )
        ]

        if secondary_indices:
            primary_count = int(
                np.count_nonzero(
                    target_index
                    == primary_index
                )
            )

            chromatic_indices_array = np.asarray(
                [
                    primary_index,
                    *secondary_indices,
                ],
                dtype=np.intp,
            )

            chromatic_mask = np.isin(
                target_index,
                chromatic_indices_array,
            )

            chromatic_total = int(
                np.count_nonzero(
                    chromatic_mask
                )
            )

            recovery_dominance_margin = 0.08

            required_primary_count = int(
                math.ceil(
                    (
                        chromatic_total
                        * (
                            1.0
                            + recovery_dominance_margin
                        )
                    )
                    / 2.0
                )
            )

            pixels_to_reassign = max(
                0,
                required_primary_count
                - primary_count,
            )

            if pixels_to_reassign > 0:
                secondary_mask = np.isin(
                    target_index,
                    np.asarray(
                        secondary_indices,
                        dtype=np.intp,
                    ),
                ).astype(
                    np.uint8
                )

                boundary_distance = cv2.distanceTransform(
                    secondary_mask,
                    cv2.DIST_L2,
                    3,
                )

                candidate_positions = np.flatnonzero(
                    secondary_mask.reshape(-1)
                )

                if candidate_positions.size > 0:
                    candidate_distances = (
                        boundary_distance
                        .reshape(-1)[
                            candidate_positions
                        ]
                    )

                    candidate_saturation = (
                        source_saturation
                        .reshape(-1)[
                            candidate_positions
                        ]
                    )

                    order = np.lexsort(
                        (
                            candidate_saturation,
                            candidate_distances,
                        )
                    )

                    take = min(
                        pixels_to_reassign,
                        candidate_positions.size,
                    )

                    chosen = candidate_positions[
                        order[:take]
                    ]

                    flat_target_index = (
                        target_index.reshape(-1)
                    )

                    flat_target_index[
                        chosen
                    ] = primary_index



def constrain_to_requested_palette(
    image: Image.Image,
    primary_colour: str | None,
    secondary_colour: str | None,
) -> Image.Image:
    """Constrain artwork to any requested palette while preserving structure.

    Recovery is palette-agnostic. It supports:

    - chromatic + chromatic palettes;
    - chromatic + neutral palettes;
    - neutral + chromatic palettes;
    - neutral + neutral palettes;
    - single-colour palettes.

    The transform preserves source shading and local structure. Hue/family
    assignment is deterministic, neutral targets preserve luminance variation,
    and the normal strict validator remains the final authority.
    """

    descriptors = [
        colour_descriptor(value)
        for value in (
            primary_colour,
            secondary_colour,
        )
    ]

    usable = [
        descriptor
        for descriptor in descriptors
        if descriptor["rgb"] is not None
    ]

    if not usable:
        return image.convert("RGB")

    source_rgb = np.asarray(
        image.convert("RGB"),
        dtype=np.uint8,
    )

    source_hsv = cv2.cvtColor(
        source_rgb,
        cv2.COLOR_RGB2HSV,
    ).astype(np.float32)

    source_hue = source_hsv[:, :, 0]
    source_saturation = source_hsv[:, :, 1]
    source_value = source_hsv[:, :, 2]

    target_rgb = np.asarray(
        [
            descriptor["rgb"]
            for descriptor in usable
        ],
        dtype=np.uint8,
    )

    target_hsv = cv2.cvtColor(
        target_rgb.reshape(1, -1, 3),
        cv2.COLOR_RGB2HSV,
    ).reshape(-1, 3).astype(np.float32)

    target_hue = target_hsv[:, 0]
    target_saturation = target_hsv[:, 1]
    target_value = target_hsv[:, 2]

    target_names = [
        descriptor["base_name"]
        for descriptor in usable
    ]

    primary_index = 0

    neutral_indices = [
        index
        for index, name in enumerate(target_names)
        if name in NEUTRAL_NAMES
    ]

    chromatic_indices = [
        index
        for index, name in enumerate(target_names)
        if name not in NEUTRAL_NAMES
    ]

    #
    # Family assignment
    # ============================================================
    #
    # Chromatic source pixels are assigned by circular hue distance.
    # Neutral-like source pixels are assigned to a requested neutral only
    # when hue evidence for a requested chromatic family is weak.
    #
    # This prevents warm cream/champagne/beige drift from being forced into
    # pink/red/etc. merely because a chromatic target exists, while also
    # preventing legitimately pale requested colours from being erased into
    # white/grey.
    #
    if chromatic_indices:
        chromatic_hues = target_hue[
            np.asarray(
                chromatic_indices,
                dtype=np.intp,
            )
        ]

        hue_delta = np.abs(
            source_hue[:, :, None]
            - chromatic_hues[None, None, :]
        )

        hue_delta = np.minimum(
            hue_delta,
            180.0 - hue_delta,
        )

        nearest_chromatic_slot = np.argmin(
            hue_delta,
            axis=2,
        )

        chromatic_lookup = np.asarray(
            chromatic_indices,
            dtype=np.intp,
        )

        target_index = chromatic_lookup[
            nearest_chromatic_slot
        ]

        minimum_chromatic_hue_delta = np.min(
            hue_delta,
            axis=2,
        )

        if neutral_indices:
            #
            # Saturation below 42 has little reliable hue information.
            #
            # Moderately desaturated pixels are also neutral candidates when
            # their hue is not close to any requested chromatic family. This
            # catches warm near-whites and cool greys generically without
            # special-casing any named palette pair.
            #
            # Pale requested colours can have low saturation while retaining
            # a stable hue. Keep those pixels in their chromatic family;
            # otherwise pastel petals and washes are erased into white.
            reliable_requested_hue = (
                (source_saturation >= 12.0)
                & (minimum_chromatic_hue_delta <= 12.0)
            )
            neutral_candidate = (
                (source_saturation < 42.0)
                | (
                    (source_saturation < 96.0)
                    & (minimum_chromatic_hue_delta > 20.0)
                )
            ) & ~reliable_requested_hue

            neutral_lookup = np.asarray(
                neutral_indices,
                dtype=np.intp,
            )

            neutral_values = target_value[
                neutral_lookup
            ]

            neutral_value_delta = np.abs(
                source_value[:, :, None]
                - neutral_values[None, None, :]
            )

            nearest_neutral_slot = np.argmin(
                neutral_value_delta,
                axis=2,
            )

            nearest_neutral_index = neutral_lookup[
                nearest_neutral_slot
            ]

            target_index[
                neutral_candidate
            ] = nearest_neutral_index[
                neutral_candidate
            ]

        else:
            #
            # A palette containing only chromatic colours cannot retain
            # unrequested greys/whites. Give hue-less pixels to the primary
            # requested family and preserve their source luminance later.
            #
            target_index[
                source_saturation < 28.0
            ] = primary_index

    else:
        #
        # Neutral-only palette. Pick the requested neutral whose luminance is
        # closest to each source pixel, then remove chroma while preserving
        # local value differences inside that neutral family.
        #
        neutral_lookup = np.asarray(
            neutral_indices,
            dtype=np.intp,
        )

        neutral_values = target_value[
            neutral_lookup
        ]

        neutral_value_delta = np.abs(
            source_value[:, :, None]
            - neutral_values[None, None, :]
        )

        nearest_neutral_slot = np.argmin(
            neutral_value_delta,
            axis=2,
        )

        target_index = neutral_lookup[
            nearest_neutral_slot
        ]

    #
    # Local family stabilization
    # ============================================================
    #
    # The strict validator evaluates a reduced image. Very fine alternating
    # target families can blend into an unrequested intermediate hue during
    # Lanczos downsampling. A small full-resolution majority filter removes
    # that unstable micro-alternation without creating coarse rectangular
    # blocks.
    #
    if len(usable) > 1:
        family_scores = []

        for family_index in range(
            len(usable)
        ):
            family_mask = (
                target_index
                == family_index
            ).astype(
                np.float32
            )

            family_score = cv2.boxFilter(
                family_mask,
                ddepth=-1,
                ksize=(7, 7),
                normalize=True,
                borderType=cv2.BORDER_REFLECT101,
            )

            family_scores.append(
                family_score
            )

        target_index = np.argmax(
            np.stack(
                family_scores,
                axis=2,
            ),
            axis=2,
        ).astype(
            np.intp
        )

    _restore_chromatic_primary_dominance(
        target_index, source_saturation, target_names, usable, primary_index,
    )

    output_hsv = source_hsv.copy()

    selected_hue = target_hue[
        target_index
    ]

    selected_saturation = target_saturation[
        target_index
    ]

    output_hsv[:, :, 0] = selected_hue

    #
    # Saturation recovery
    # ============================================================
    #
    # Chromatic targets preserve source saturation variation while receiving
    # enough chroma to classify inside the requested family.
    #
    # Neutral targets remove chroma deterministically, but do not flatten
    # luminance or geometry.
    #
    blended_saturation = (
        (source_saturation * 0.62)
        + (selected_saturation * 0.38)
    )

    minimum_requested_saturation = (
        selected_saturation
        * 0.60
    )

    blended_saturation = np.maximum(
        blended_saturation,
        minimum_requested_saturation,
    )

    for neutral_index in neutral_indices:
        neutral_mask = (
            target_index
            == neutral_index
        )

        blended_saturation[
            neutral_mask
        ] = np.minimum(
            source_saturation[
                neutral_mask
            ]
            * 0.25,
            18.0,
        )

    output_hsv[:, :, 1] = np.clip(
        blended_saturation,
        0.0,
        255.0,
    )

    #
    # Value/luminance recovery
    # ============================================================
    #
    # Chromatic targets preserve source luminance with only the minimum floor
    # required by the strict classifier.
    #
    selected_min_value = np.where(
        np.asarray(
            [
                _is_dark_target(
                    name,
                    tuple(rgb),
                )
                for name, rgb
                in zip(
                    target_names,
                    target_rgb,
                )
            ],
            dtype=bool,
        )[
            target_index
        ],
        DARK_MIN_VALUE,
        STANDARD_MIN_VALUE,
    ).astype(
        np.float32
    )

    preserved_value = np.maximum(
        source_value,
        selected_min_value + 4.0,
    )

    #
    # Neutral targets use luminance ranges that sit inside the validator's
    # accepted neutral bands. Source luminance is linearly mapped into the
    # range instead of being clamped to one fixed brightness, so gradients,
    # borders, texture and structural edges survive recovery.
    #
    neutral_value_ranges = {
        "black": (0.0, 55.0),
        "charcoal": (35.0, 105.0),
        "grey": (80.0, 185.0),
        "silver": (145.0, 225.0),
        "white": (190.0, 255.0),
    }

    for neutral_index in neutral_indices:
        neutral_name = target_names[
            neutral_index
        ]

        low, high = neutral_value_ranges.get(
            neutral_name,
            (0.0, 255.0),
        )

        neutral_mask = (
            target_index
            == neutral_index
        )

        source_fraction = (
            source_value[
                neutral_mask
            ]
            / 255.0
        )

        preserved_value[
            neutral_mask
        ] = (
            low
            + (
                source_fraction
                * (
                    high - low
                )
            )
        )

    output_hsv[:, :, 2] = np.clip(
        preserved_value,
        0.0,
        255.0,
    )

    constrained_rgb = cv2.cvtColor(
        output_hsv.astype(
            np.uint8
        ),
        cv2.COLOR_HSV2RGB,
    )

    return Image.fromarray(
        constrained_rgb,
        mode="RGB",
    )

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
) -> tuple[
    int,
    int,
    int,
] | None:
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

    return (
        descriptor[
            "cleaned"
        ]
        or None
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
        name = (
            colour_descriptor(
                value
            )[
                "base_name"
            ]
        )

        if (
            name
            and name
            not in colours
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

    measurable_names = (
        requested_palette_names(
            primary_colour,
            secondary_colour,
        )
    )

    requested_names = set(
        measurable_names
    )

    statements = [
        (
            "STRICT COLOUR PALETTE CONTRACT."
        ),
        (
            "Use only the explicitly requested colour families."
        ),
        (
            "Do not substitute the requested colours with colours "
            "that merely fit the event theme."
        ),
        (
            "Do not introduce any additional chromatic colour "
            "or neutral colour that was not requested."
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
        modifier_text = " ".join(
            primary_descriptor[
                "modifiers"
            ]
        )

        statements.append(
            (
                f"The requested primary treatment is {modifier_text}. "
                "Apply that treatment only to an explicitly requested "
                "colour family and do not invent a new hue."
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
        modifier_text = " ".join(
            secondary_descriptor[
                "modifiers"
            ]
        )

        statements.append(
            (
                f"The requested secondary treatment is {modifier_text}. "
                "Treat it as a finish rather than a new hue."
            )
        )

    if measurable_names:
        statements.append(
            (
                "No other visible colour family is allowed. "
                "The only allowed named colour families are: "
                + ", ".join(
                    measurable_names
                )
                + "."
            )
        )

    if (
        requested_names
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
                "measurable colour family and apply the "
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
                "measurable colour family and apply the "
                f"{' '.join(secondary_descriptor['modifiers'])} "
                "treatment where appropriate."
            )
        )

    return " ".join(
        statements
    )


def palette_negative_contract(
    primary_colour: str | None,
    secondary_colour: str | None,
) -> str:
    measurable = (
        requested_palette_names(
            primary_colour,
            secondary_colour,
        )
    )

    if not measurable:
        return (
            "off-palette colours, colour drift, "
            "unrequested accent colours, multicolour palette, "
            "rainbow colours, unrelated hues"
        )

    return (
        "off-palette colours, colour drift, "
        "unrequested accent colours, multicolour palette, "
        "rainbow colours, unrelated hues. "
        "Do not add any neutral or chromatic colour family "
        "that was not requested. "
        "Keep every visible colour inside the requested "
        "colour families: "
        + ", ".join(
            measurable
        )
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

    (
        hue,
        saturation,
        value,
    ) = (
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
        180
        - raw,
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

    (
        _,
        _,
        value,
    ) = (
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


def _matches_requested_neutral(
    saturation: int,
    value: int,
    requested_neutrals: set[str],
) -> str | None:
    if (
        saturation
        > NEUTRAL_MAX_SATURATION
    ):
        return None

    checks = (
        (
            "black",
            value <= 55,
        ),
        (
            "charcoal",
            35
            <= value
            <= 105,
        ),
        (
            "grey",
            80
            <= value
            <= 185,
        ),
        (
            "silver",
            145
            <= value
            <= 225,
        ),
        # Requested white includes bright low-saturation warm paper tones.
        # They are normal lighting variants of white, not a third palette.
        (
            "white",
            value >= 190 and saturation <= 78,
        ),
    )

    for (
        name,
        matches,
    ) in checks:
        if (
            name
            in requested_neutrals
            and matches
        ):
            return (
                name
            )

    return None


def _nearest_named_colour(
    rgb: np.ndarray,
) -> str:
    pixel = (
        rgb.astype(
            np.int32
        )
    )

    best_name = (
        "unknown"
    )

    best_distance = None

    for (
        name,
        reference,
    ) in (
        NAMED_COLOURS.items()
    ):
        ref = np.asarray(
            reference,
            dtype=np.int32,
        )

        distance = int(
            np.sum(
                (
                    pixel
                    - ref
                )
                ** 2
            )
        )

        if (
            best_distance
            is None
            or distance
            < best_distance
        ):
            best_distance = (
                distance
            )

            best_name = (
                name
            )

    return (
        best_name
    )


def _colour_family_members(
    name: str | None,
) -> set[str]:
    if not name:
        return set()

    if (
        name
        in NEUTRAL_NAMES
    ):
        return {
            name
        }

    return set(
        COLOUR_FAMILIES.get(
            name,
            {
                name
            },
        )
    )


def _nearest_requested_family(
    nearest_name: str,
    pixel_rgb: np.ndarray,
    requested_targets: list[dict],
) -> str | None:
    candidates = [
        target
        for target
        in requested_targets
        if (
            nearest_name
            in target[
                "family_members"
            ]
        )
    ]

    if not candidates:
        return None

    pixel = (
        pixel_rgb.astype(
            np.int32
        )
    )

    best_name = None

    best_distance = None

    for target in candidates:
        target_rgb = np.asarray(
            target[
                "rgb"
            ],
            dtype=np.int32,
        )

        distance = int(
            np.sum(
                (
                    pixel
                    - target_rgb
                )
                ** 2
            )
        )

        if (
            best_distance
            is None
            or distance
            < best_distance
        ):
            best_distance = (
                distance
            )

            best_name = (
                target[
                    "name"
                ]
            )

    return (
        best_name
    )


def _chromatic_requested_family(
    hue: int,
    saturation: int,
    value: int,
    requested_targets: list[dict],
) -> str | None:
    """Classify dark coloured pixels by hue before RGB-nearest neutral names.

    Euclidean RGB distance calls a dark royal-blue pixel "charcoal" because
    its low value dominates the distance, even when its saturation is high.
    Twelve OpenCV hue units separate the blue family from teal and violet in
    the named palette while still covering navy and royal-blue shadows.
    """
    candidates = []
    for target in requested_targets:
        if not _pixel_can_match_target(saturation, value, target["name"], target["rgb"]):
            continue
        distance = _hue_distance(hue, target["hue"])
        if distance <= min(12, _target_hue_limit(target["name"], target["rgb"])):
            candidates.append((distance, target["name"]))
    return min(candidates)[1] if candidates else None


def _classify_palette_pixel(
    hue: int,
    saturation: int,
    value: int,
    pixel_rgb,
    requested_neutrals: set[str],
    requested_targets: list[dict],
) -> tuple[str | None, str]:
    neutral = _matches_requested_neutral(saturation, value, requested_neutrals)
    if neutral is not None:
        return neutral, neutral

    chromatic = _chromatic_requested_family(hue, saturation, value, requested_targets)
    if chromatic is not None:
        return chromatic, chromatic

    nearest = _nearest_named_colour(pixel_rgb)
    family = _nearest_requested_family(nearest, pixel_rgb, requested_targets)
    if family is not None:
        return family, nearest

    best_target = None
    best_distance = None
    for target in requested_targets:
        if not target["descriptor"]["is_hex"]:
            continue
        if not _pixel_can_match_target(saturation, value, target["name"], target["rgb"]):
            continue
        distance = _hue_distance(hue, target["hue"])
        if distance > _target_hue_limit(target["name"], target["rgb"]):
            continue
        if best_distance is None or distance < best_distance:
            best_target = target
            best_distance = distance

    return (best_target["name"] if best_target is not None else None), nearest


def _validate_palette_balance(primary_name, primary_rgb, primary_descriptor, secondary_name, secondary_rgb, requested_counts: dict, total: int) -> tuple[float | None, float | None]:
    primary_ratio = None

    if (
        primary_name
        and primary_rgb
        is not None
    ):
        primary_ratio = (
            requested_counts.get(
                primary_name,
                0,
            )
            / total
        )

        if (
            primary_name
            not in NEUTRAL_NAMES
            and primary_ratio
            < MIN_PRIMARY_COLOUR_RATIO
        ):
            raise ValueError(
                "Generated background does not visibly "
                "contain enough of the requested primary "
                f"colour family '{primary_name}' "
                f"(primary family ratio "
                f"{primary_ratio:.2f})."
            )

    secondary_ratio = None

    if (
        secondary_name
        and secondary_rgb
        is not None
    ):
        secondary_ratio = (
            requested_counts.get(
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
            secondary_name
            not in NEUTRAL_NAMES
            and secondary_ratio
            < minimum_secondary_ratio
        ):
            raise ValueError(
                "Generated background does not visibly "
                "contain enough of the requested secondary "
                f"colour family '{secondary_name}' "
                f"(secondary family ratio "
                f"{secondary_ratio:.2f})."
            )

    #
    # When both requested colours are chromatic, the declared primary colour
    # must be meaningfully dominant rather than merely present.
    #
    if (
        primary_ratio
        is not None
        and secondary_ratio
        is not None
        and primary_name
        not in NEUTRAL_NAMES
        and secondary_name
        not in NEUTRAL_NAMES
        and primary_name
        != secondary_name
        and primary_ratio
        < (
            secondary_ratio
            + MIN_PRIMARY_DOMINANCE_MARGIN
        )
    ):
        raise ValueError(
            "Generated background does not keep the requested "
            f"primary colour family '{primary_name}' meaningfully dominant "
            f"over the requested secondary colour family '{secondary_name}' "
            f"(primary family ratio {primary_ratio:.2f}, "
            f"secondary family ratio {secondary_ratio:.2f}; "
            f"required dominance margin "
            f"{MIN_PRIMARY_DOMINANCE_MARGIN:.2f})."
        )

    return primary_ratio, secondary_ratio


def validate_palette(
    image: Image.Image,
    primary_colour: str | None,
    secondary_colour: str | None,
    *,
    palette: list[dict] | None = None,
    relationship: str | None = None,
) -> dict:
    quality_result = (
        validate_visual_quality(
            image
        )
    )

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

    if palette:
        requested = [
            (descriptor["base_name"], descriptor["rgb"], descriptor)
            for item in palette if isinstance(item, dict)
            for descriptor in [colour_descriptor(item.get("colour"))]
        ]
    else:
        requested = [
            (primary_name, primary_rgb, primary_descriptor),
            (secondary_name, secondary_rgb, secondary_descriptor),
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
            **quality_result,
            "palette_checked": False,
            "reason": (
                "No measurable colour hue was supplied."
            ),
        }

    sample = (
        image.convert(
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

    rgb_array = np.array(
        sample,
        dtype=np.uint8,
    )

    hsv_array = (
        cv2.cvtColor(
            rgb_array,
            cv2.COLOR_RGB2HSV,
        )
    )

    hsv_pixels = (
        hsv_array.reshape(
            -1,
            3,
        )
    )

    rgb_pixels = (
        rgb_array.reshape(
            -1,
            3,
        )
    )

    total = len(
        hsv_pixels
    )

    requested_names = {
        name
        for (
            name,
            _,
            _,
        )
        in requested
    }

    requested_neutrals = (
        requested_names
        & NEUTRAL_NAMES
    )

    if (
        requested_names
        == {
            "black",
            "white",
        }
    ):
        saturation = (
            hsv_pixels[
                :,
                1
            ]
        )

        monochrome_ratio = float(
            np.mean(
                saturation
                <= NEUTRAL_MAX_SATURATION
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
            **quality_result,
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
            "requested_colour_counts": {},
            "requested_colour_family_counts": {},
            "accepted_tonal_breakdown": {},
            "unrequested_colour_breakdown": {},
        }

    requested_targets = []

    for (
        name,
        rgb_value,
        descriptor,
    ) in requested:
        if (
            name
            in NEUTRAL_NAMES
        ):
            continue

        (
            hue,
            _,
            _,
        ) = (
            _rgb_to_hsv(
                rgb_value
            )
        )

        requested_targets.append(
            {
                "name": (
                    name
                ),
                "rgb": (
                    rgb_value
                ),
                "hue": (
                    hue
                ),
                "descriptor": (
                    descriptor
                ),
                "family_members": (
                    _colour_family_members(
                        name
                    )
                ),
            }
        )

    requested_counts = dict.fromkeys(
        (name for name, _, _ in requested),
        0,
    )

    accepted_tonal_counts: dict[
        str,
        int,
    ] = {}

    off_palette_count = 0

    off_palette_named_counts: dict[
        str,
        int,
    ] = {}

    for index, (hue, saturation, value) in enumerate(hsv_pixels):
        family, tonal_name = _classify_palette_pixel(
            int(hue), int(saturation), int(value), rgb_pixels[index],
            requested_neutrals, requested_targets,
        )
        if family is None:
            off_palette_count += 1
            off_palette_named_counts[tonal_name] = off_palette_named_counts.get(tonal_name, 0) + 1
            continue
        requested_counts[family] = requested_counts.get(family, 0) + 1
        accepted_tonal_counts[tonal_name] = accepted_tonal_counts.get(tonal_name, 0) + 1

    off_palette_ratio = (
        off_palette_count
        / total
    )

    breakdown = {
        name: (
            count
            / total
        )
        for (
            name,
            count,
        )
        in sorted(
            off_palette_named_counts.items(),
            key=(
                lambda item:
                item[
                    1
                ]
            ),
            reverse=True,
        )
    }

    tonal_breakdown = {
        name: (
            count
            / total
        )
        for (
            name,
            count,
        )
        in sorted(
            accepted_tonal_counts.items(),
            key=(
                lambda item:
                item[
                    1
                ]
            ),
            reverse=True,
        )
    }

    if (
        off_palette_ratio
        > MAX_OFF_PALETTE_RATIO
    ):
        top_unrequested = [
            (
                f"{name}="
                f"{ratio:.2f}"
            )
            for (
                name,
                ratio,
            )
            in list(
                breakdown.items()
            )[
                :6
            ]
        ]

        details = (
            ", ".join(
                top_unrequested
            )
            or "unclassified"
        )

        raise ValueError(
            "Generated background contains colours "
            "that were not requested "
            f"(off-palette ratio "
            f"{off_palette_ratio:.2f}; "
            f"strongest matches: {details})."
        )

    primary_ratio, secondary_ratio = (None, None)
    if not palette or "dominant" in (relationship or "").casefold():
        primary_ratio, secondary_ratio = _validate_palette_balance(
            primary_name, primary_rgb, primary_descriptor, secondary_name,
            secondary_rgb, requested_counts, total,
        )

    palette_match_ratio = (
        1.0
        - off_palette_ratio
    )

    return {
        **quality_result,

        "palette_checked": True,

        "palette_mode": (
            "strict-requested-colour-families"
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

        "requested_colour_counts": (
            requested_counts
        ),

        "requested_colour_family_counts": (
            requested_counts
        ),

        "accepted_tonal_breakdown": (
            tonal_breakdown
        ),

        "unrequested_colour_breakdown": (
            breakdown
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
            "family_members": sorted(
                _colour_family_members(
                    primary_name
                )
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
            "family_members": sorted(
                _colour_family_members(
                    secondary_name
                )
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

    return (
        classifier
    )


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


def _centered_lower_upper_body(
    upper_bodies: list[tuple[int, int, int, int]], image_width: int, image_height: int,
) -> bool:
    return any(
        0.35 <= (x + width / 2) / image_width <= 0.65
        and 0.60 <= (y + height / 2) / image_height <= 0.90
        for x, y, width, height in upper_bodies
    )


def detect_human_signals(
    image: Image.Image,
) -> dict:
    rgb = np.array(
        image.convert(
            "RGB"
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

        bgr = cv2.resize(
            bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=(
                cv2.INTER_AREA
            ),
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
        numeric_weight = float(
            weight
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

    # A single upper-body cascade hit in the lower centre is significant for
    # a background whose centre must be clear. The SD3.5 E9EBE8 probe has a
    # visible central silhouette here; HOG and face cascades miss it. An
    # isolated hit at the outer decoration (as in FLUX B) remains insufficient.
    image_height, image_width = gray.shape[:2]
    centered_lower_body = _centered_lower_upper_body(
        upper_bodies, image_width, image_height,
    )
    strong_upper_body_signal = len(upper_bodies) >= 2 or centered_lower_body

    detected = bool(
        people
        or faces
        or profiles
        or strong_upper_body_signal
    )

    return {
        "people": (
            people
        ),
        "people_weights": (
            people_weights
        ),
        "faces": (
            faces
        ),
        "profiles": (
            profiles
        ),
        "upper_bodies": (
            upper_bodies
        ),
        "strong_upper_body_signal": (
            strong_upper_body_signal
        ),
        "detected": (
            detected
        ),
    }
