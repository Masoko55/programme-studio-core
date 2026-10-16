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
    "emerald": "green",
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
    "(masked character:2.4), "
    "(costumed figure:2.4), "
    "(mascot:2.4), "
    "(humanoid:2.4), "
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
    "A4 portrait event-background artwork only. "
    "Faithfully render the requested visual subject, whether representational, "
    "illustrative, environmental, ornamental, graphic or abstract. "
    "Do not introduce a human focal subject unless the product explicitly allows it. "
    "No photographic person, human silhouette, human-shaped figure or body-shaped form. "
    "Absolutely no people, human figures, faces, portraits, characters, mannequins, "
    "heads, bodies, hands, arms, legs, clothing or human-like subjects. "
    "Absolutely no words, lettering, typography, logos, signatures, watermarks, "
    "labels, numbers or readable characters. "
    "Do not create a finished poster, invitation, card, certificate, menu, document or signage. "
    "Do not generate full-frame scanlines, raster banding, repetitive horizontal or "
    "vertical line corruption, moire, glitch artefacts, broken image textures or blank flat fields. "
    "Do not generate a featureless full-page gradient, plain colour wash, grain-only texture "
    "or noise-only image. Use coherent subject-specific forms, materials, motifs, architecture, "
    "environmental elements or abstract structure as requested by the brief. "
    "Preserve intentional visually quiet regions for later title and programme typography."
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
    palette_mode: str = "strict",
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

    # A mild recovery must leave already-correct material shading untouched.
    # Rewriting these pixels with one target hue/saturation destroys fine
    # contrast even though the requested family was already present.
    keep_original = np.zeros(source_hue.shape, dtype=bool)
    for target_h, target_s in zip(target_hue, target_saturation):
        if target_s < 35:
            continue
        distance = np.abs(source_hue - target_h)
        distance = np.minimum(distance, 180.0 - distance)
        keep_original |= (distance <= 9.0) & (source_saturation >= 35.0)
    if palette_mode == "dominant":
        keep_original |= source_saturation < 25.0

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
    # Hue/saturation recovery should not rewrite the scene's luminance edges.
    # Preserve the original Lab lightness for chromatic targets. Neutral
    # targets keep their bounded tonal mapping because black/white roles need
    # different absolute lightness, while still retaining local variation.
    chromatic_mask = np.isin(target_index, chromatic_indices)
    if np.any(chromatic_mask):
        original_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB)
        recovered_lab = cv2.cvtColor(constrained_rgb, cv2.COLOR_RGB2LAB)
        recovered_lab[:, :, 0][chromatic_mask] = original_lab[:, :, 0][chromatic_mask]
        constrained_rgb = cv2.cvtColor(recovered_lab, cv2.COLOR_LAB2RGB)
    constrained_rgb[keep_original] = source_rgb[keep_original]

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
            "REQUESTED COLOUR PALETTE CONTRACT."
        ),
        (
            "Keep the requested colour families dominant while preserving natural tonal variation."
        ),
        (
            "Do not substitute the requested colours with colours "
            "that merely fit the event theme."
        ),
        (
            "Avoid a large unrelated third-colour design region."
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
                "The requested dominant colour families are: "
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
        "Avoid large unrelated third-colour regions. "
        "Keep the design grounded in the requested colour families: "
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

    # A pale tint or a shaded pigment may be closer in RGB to a named neutral
    # than to its source pigment. Hue is still informative above very low
    # saturation; classify that relationship before the finite colour table.
    if saturation >= 12:
        candidates = [
            (_hue_distance(hue, target["hue"]), target["name"])
            for target in requested_targets
            if _hue_distance(hue, target["hue"]) <= min(10, _target_hue_limit(target["name"], target["rgb"]))
        ]
        if candidates:
            _, family = min(candidates)
            return family, family

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


def _perceptual_palette_family(pixel_lab, targets: list[dict], pixel_hue: int) -> str | None:
    """Match a pigment across shading while keeping unrelated hue separate."""
    light, a, b = (float(value) for value in pixel_lab)
    chroma = float(np.hypot(a, b))
    matches = []
    for target in targets:
        target_lab = target["lab"]
        target_chroma = float(np.hypot(target_lab[1], target_lab[2]))
        if chroma < 8 or target_chroma < 8:
            continue
        cosine = float(np.clip((a * target_lab[1] + b * target_lab[2]) /
                               (chroma * target_chroma), -1, 1))
        angle = float(np.degrees(np.arccos(cosine)))
        # A low-chroma petal shadow has less reliable hue than a saturated
        # third colour. It still needs a perceptual connection to the pigment.
        limit = 43 if chroma < 26 else 29
        if angle <= limit:
            distance = float(np.linalg.norm(pixel_lab - target_lab))
            # LAB hue angle alone groups saturated neighbouring pigments.
            # Pale shading has uncertain hue; strongly chromatic pixels do not.
            if (chroma >= 26 and distance > 12
                    and _hue_distance(pixel_hue, target["hue"]) > 6):
                continue
            matches.append((angle + distance * 0.2, target["name"]))
    return min(matches)[1] if matches else None


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


def palette_relationship_policy(relationship: str | None, source: str | None,
                                secondary_name: str | None = None) -> dict:
    """Interpret relationship semantics once for validation and conditioning."""
    phrase = str(relationship or "").strip()
    text = phrase.casefold()
    source = str(source or "").casefold()
    generic = text in {"", "requested colours dominate with natural tonal variation",
                       "respect the requested colour roles"}
    semantic = bool(re.search(
        r"\b(balanced|equal|equally|even split|monochrom\w*|only|dominant|dominate|"
        r"majority|largest|leading|main|mostly|subtle|accent\w*|support\w*|small\s+\w+\s+details)\b", text))
    explicit = (not generic and semantic and source in {
        "", "user", "user_explicit", "grill_me_clarified"})
    relation = "FLEXIBLE"
    if explicit and re.search(r"\b(balanced|equal|equally|even split)\b", text):
        relation = "BALANCED"
    elif explicit and re.search(r"\b(dominant|dominate|majority|largest|leading|main|mostly|subtle|accent\w*|support\w*|small)\b", text):
        relation = "PRIMARY_DOMINANT"
        if secondary_name and re.search(
            rf"(?:\bmostly\s+{re.escape(secondary_name)}\b|\b{re.escape(secondary_name)}\s+"
            r"(?:(?:should|must|is|be|the)\s+)*(?:dominant|dominate|main|leading)\b)", text):
            relation = "SECONDARY_DOMINANT"
    return {"explicit": explicit, "relationship": phrase if explicit else "",
            "dominance_relationship": relation,
            "source": "user_explicit" if explicit else "default"}


def _palette_role_metrics(primary_name: str | None, secondary_name: str | None,
                          counts: dict, total: int, relationship: str | None,
                          mode: str | None, palette: list[dict] | None,
                          relationship_source: str | None = None) -> dict:
    """Check declared colour roles separately from family membership."""
    phrase = str(relationship or "").casefold()
    roles = [str(item.get("role") or "").casefold() for item in (palette or []) if isinstance(item, dict)]
    primary_role = roles[0] if roles else "primary"
    secondary_role = roles[1] if len(roles) > 1 else "secondary"
    monochromatic = mode in {"monochrome", "achromatic"} or bool(re.search(r"\bmonochrom", phrase)) or not secondary_name
    optional = "optional" in secondary_role or bool(re.search(r"\b(optional|if present|may appear)\b", phrase))
    balanced = bool(re.search(r"\b(balanced|equal|equally|even split)\b", phrase))
    policy = palette_relationship_policy(relationship, relationship_source, secondary_name)
    explicit = policy["explicit"]
    relation = "MONOCHROMATIC" if monochromatic else policy["dominance_relationship"]
    primary_ratio = counts.get(primary_name, 0) / total if primary_name else None
    secondary_ratio = counts.get(secondary_name, 0) / total if secondary_name else None
    strong_secondary = "strong" in secondary_role or bool(re.search(r"\b(strong|substantial|prominent)\s+(?:\w+\s+)?(?:secondary|accent)", phrase))
    secondary_minimum = (0 if optional or monochromatic else .05 if strong_secondary else
                         .02 if balanced or "support" in secondary_role else MIN_SECONDARY_COLOUR_RATIO)
    presence = secondary_ratio is None or secondary_ratio >= secondary_minimum
    dominance = (True if relation in {"FLEXIBLE", "MONOCHROMATIC"} or primary_ratio is None or secondary_ratio is None else
                 abs(primary_ratio - secondary_ratio) <= .20 if relation == "BALANCED" else
                 secondary_ratio + MIN_PRIMARY_DOMINANCE_MARGIN >= primary_ratio if relation == "SECONDARY_DOMINANT" else
                 primary_ratio + MIN_PRIMARY_DOMINANCE_MARGIN >= secondary_ratio)
    metrics = {"primary_ratio": primary_ratio, "secondary_ratio": secondary_ratio,
               "primary_family": primary_name, "secondary_family": secondary_name,
               "primary_role": primary_role,
               "primary_secondary_ratio": (primary_ratio / secondary_ratio if secondary_ratio else None),
               "secondary_presence": presence, "dominance_relationship": relation,
               "dominance_relationship_passed": dominance, "secondary_role": secondary_role,
               "secondary_minimum": secondary_minimum,
               "palette_relationship_source": policy["source"],
               "palette_role_warning": ("PALETTE_ROLE_WARNING" if not explicit and primary_ratio is not None
                                        and secondary_ratio is not None and primary_ratio < secondary_ratio else None)}
    if primary_name and primary_name not in NEUTRAL_NAMES and primary_ratio is not None and primary_ratio < MIN_PRIMARY_COLOUR_RATIO:
        error = ValueError("PALETTE_ROLE_DRIFT: requested primary colour is not meaningfully present")
    elif not presence:
        error = ValueError("PALETTE_ROLE_DRIFT: required secondary colour is absent or too small")
    elif explicit and not dominance:
        error = ValueError("PALETTE_ROLE_DRIFT: requested colour dominance relationship is reversed")
    else:
        return metrics
    error.palette_metrics = metrics
    raise error


def validate_palette(
    image: Image.Image,
    primary_colour: str | None,
    secondary_colour: str | None,
    *,
    palette: list[dict] | None = None,
    relationship: str | None = None,
    relationship_source: str | None = None,
    mode: str | None = None,
    quality_result: dict | None = None,
) -> dict:
    if quality_result is None:
        quality_result = validate_visual_quality(image)

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
    lab_pixels = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    lab_pixels[:, 0] *= 100.0 / 255.0
    lab_pixels[:, 1:] -= 128.0

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
        chroma = np.linalg.norm(lab_pixels[:, 1:], axis=1)
        luminance = lab_pixels[:, 0]
        achromatic = chroma <= 16
        monochrome_ratio = float(np.mean(achromatic))
        black_ratio = float(np.mean(achromatic & (luminance <= 35)))
        white_ratio = float(np.mean(achromatic & (luminance >= 65)))
        dynamic_range = float(np.percentile(luminance, 90) - np.percentile(luminance, 10))
        achromatic_metrics = {
            "achromatic_ratio": monochrome_ratio,
            "chromatic_contamination_ratio": 1.0 - monochrome_ratio,
            "dark_tone_ratio": black_ratio,
            "light_tone_ratio": white_ratio,
            "luminance_dynamic_range": dynamic_range,
            "off_palette_ratio": 1.0 - monochrome_ratio,
            "palette_mode": mode or "achromatic",
        }

        minimum_monochrome = 0.99 if mode == "strict" else MIN_BLACK_WHITE_RATIO
        if monochrome_ratio < minimum_monochrome:
            error = ValueError(
                "Generated background left the requested "
                "black-and-white palette "
                f"(monochrome ratio "
                f"{monochrome_ratio:.2f})."
            )
            error.palette_metrics = achromatic_metrics
            raise error
        if black_ratio < 0.02 or white_ratio < 0.02 or dynamic_range < 30:
            error = ValueError(
                "Generated background lacks useful black-and-white tonal separation "
                f"(dark {black_ratio:.2f}, light {white_ratio:.2f}, range {dynamic_range:.1f})."
            )
            error.palette_metrics = achromatic_metrics
            raise error

        try:
            role_metrics = _palette_role_metrics(primary_name, secondary_name,
                {"black": round(black_ratio * total), "white": round(white_ratio * total)},
                total, relationship, mode, palette, relationship_source)
        except ValueError as error:
            error.palette_metrics = {**achromatic_metrics, **getattr(error, "palette_metrics", {})}
            raise

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
            **achromatic_metrics,
            **role_metrics,
            "palette_mode": "black-white",
            "primary_colour_ratio": black_ratio if primary_name == "black" else white_ratio,
            "secondary_colour_ratio": white_ratio if secondary_name == "white" else black_ratio,
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
                "lab": cv2.cvtColor(np.uint8([[rgb_value]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32),
            }
        )
        requested_targets[-1]["lab"][0] *= 100.0 / 255.0
        requested_targets[-1]["lab"][1:] -= 128.0

    requested_counts = dict.fromkeys(
        (name for name, _, _ in requested),
        0,
    )

    accepted_tonal_counts: dict[
        str,
        int,
    ] = {}

    off_palette_count = 0
    incidental_neutral_count = 0
    incidental_neutral_mask = np.zeros(total, dtype=np.uint8)
    unrelated_chromatic_mask = np.zeros(total, dtype=np.uint8)

    off_palette_named_counts: dict[
        str,
        int,
    ] = {}
    exclusive_palette = (mode in {"strict", "monochrome"}) or bool(re.search(
        r"\b(?:only (?:these |the )?colou?rs|no third colou?r|strictly monochrome|exclusive palette)\b",
        relationship or "", re.I,
    ))

    tonal_derivative_count = 0
    unrelated_hue_count = 0
    for index, (hue, saturation, value) in enumerate(hsv_pixels):
        lab = lab_pixels[index]
        chroma = float(np.linalg.norm(lab[1:]))
        family = None
        if chroma <= 18:
            if "white" in requested_neutrals and lab[0] >= (80 if exclusive_palette else 65):
                family = "white"
            elif "black" in requested_neutrals and lab[0] <= 45:
                family = "black"
            elif "grey" in requested_neutrals or "gray" in requested_neutrals:
                family = "grey" if "grey" in requested_neutrals else "gray"
        if family is None:
            family = _perceptual_palette_family(lab, requested_targets, int(hue))
        tonal_name = family or _nearest_named_colour(rgb_pixels[index])
        if family in requested_neutrals and tonal_name == "white" and lab[0] < 90:
            tonal_derivative_count += 1
        elif family in {target["name"] for target in requested_targets}:
            target = next(target for target in requested_targets if target["name"] == family)
            if np.linalg.norm(lab - target["lab"]) > 12:
                tonal_derivative_count += 1
        if family is None:
            if chroma <= 18:
                incidental_neutral_count += 1
                incidental_neutral_mask[index] = 1
            else:
                unrelated_hue_count += 1
                unrelated_chromatic_mask[index] = 1
            off_palette_count += 1
            off_palette_named_counts[tonal_name] = off_palette_named_counts.get(tonal_name, 0) + 1
            continue
        requested_counts[family] = requested_counts.get(family, 0) + 1
        accepted_tonal_counts[tonal_name] = accepted_tonal_counts.get(tonal_name, 0) + 1

    # A bounded amount of near-neutral shadow or reflection is rendering,
    # not a newly chosen design colour. Explicit exclusive palettes do not
    # receive this allowance; chromatic third colours never receive it.
    _, _, neutral_components, _ = cv2.connectedComponentsWithStats(
        incidental_neutral_mask.reshape(176, 128), 8,
    )
    largest_neutral_region = (
        int(np.max(neutral_components[1:, cv2.CC_STAT_AREA])) / total
        if len(neutral_components) > 1 else 0.0
    )
    _, _, chromatic_components, _ = cv2.connectedComponentsWithStats(
        unrelated_chromatic_mask.reshape(176, 128), 8,
    )
    largest_unrequested_region = (
        int(np.max(chromatic_components[1:, cv2.CC_STAT_AREA])) / total
        if len(chromatic_components) > 1 else 0.0
    )
    structural_neutral_ratio = incidental_neutral_count / total
    requested_chromatic_ratio = sum(
        requested_counts.get(target["name"], 0) for target in requested_targets
    ) / total
    unrequested_chromatic_ratio = unrelated_hue_count / total
    off_palette_ratio = (
        (off_palette_count if exclusive_palette else unrelated_hue_count) / total
    )
    perceptual_metrics = {
        "perceptual_in_family_ratio": 1.0 - off_palette_ratio,
        "unrelated_hue_ratio": unrelated_hue_count / total,
        "requested_chromatic_ratio": requested_chromatic_ratio,
        "structural_neutral_ratio": structural_neutral_ratio,
        "unrequested_chromatic_ratio": unrequested_chromatic_ratio,
        "largest_unrequested_region_ratio": largest_unrequested_region,
        "primary_ratio": requested_counts.get(primary_name, 0) / total,
        "secondary_ratio": requested_counts.get(secondary_name, 0) / total,
        "requested_palette_mode": "strict" if exclusive_palette else "dominant",
        "tonal_derivative_ratio": tonal_derivative_count / total,
        "primary_presence_ratio": requested_counts.get(primary_name, 0) / total,
        "secondary_presence_ratio": requested_counts.get(secondary_name, 0) / total,
    }

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

    neutral_takeover = (
        not exclusive_palette and bool(requested_targets)
        and structural_neutral_ratio > 0.55 and requested_chromatic_ratio < 0.35
    )
    palette_limit = 0.01 if exclusive_palette else MAX_OFF_PALETTE_RATIO
    if (off_palette_ratio > palette_limit or largest_unrequested_region > 0.035
            or neutral_takeover):
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

        error = ValueError(
            "Generated background contains colours "
            "that were not requested "
            f"(off-palette ratio "
            f"{off_palette_ratio:.2f}; "
            f"strongest matches: {details}; neutral ratio {structural_neutral_ratio:.2f}; "
            f"largest unrelated region {largest_unrequested_region:.2f})."
        )
        error.palette_metrics = {
            "off_palette_ratio": off_palette_ratio,
            **perceptual_metrics,
            "requested_colour_family_counts": requested_counts,
            "accepted_tonal_breakdown": tonal_breakdown,
            "unrequested_colour_breakdown": breakdown,
        }
        raise error

    try:
        role_metrics = _palette_role_metrics(primary_name, secondary_name, requested_counts,
                                              total, relationship, mode, palette, relationship_source)
    except ValueError as error:
        error.palette_metrics = {"off_palette_ratio": off_palette_ratio, **perceptual_metrics,
                                 **getattr(error, "palette_metrics", {}), "palette_mode": mode or "dominant"}
        raise
    primary_ratio, secondary_ratio = role_metrics["primary_ratio"], role_metrics["secondary_ratio"]

    palette_match_ratio = (
        1.0
        - off_palette_ratio
    )

    return {
        **quality_result,

        "palette_checked": True,

        "palette_mode": (
            "strict-requested-colour-families" if exclusive_palette else "dominant-requested-colour-families"
        ),
        "incidental_neutral_ratio": incidental_neutral_count / total,
        "largest_incidental_neutral_region_ratio": largest_neutral_region,

        "palette_match_ratio": (
            palette_match_ratio
        ),

        "off_palette_ratio": (
            off_palette_ratio
        ),
        **perceptual_metrics,
        **role_metrics,

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


def _boxes_overlap(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[0] + first[2], second[0] + second[2])
    y1 = min(first[1] + first[3], second[1] + second[3])
    return x1 > x0 and y1 > y0


def _confirm_cascade_boxes(
    detector, gray: np.ndarray, boxes: list[tuple[int, int, int, int]],
    minimum_size: int,
) -> list[tuple[int, int, int, int]]:
    if not boxes or detector is None:
        return []
    enlarged = cv2.resize(gray, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_LINEAR)
    repeated = _cascade_detect(detector, enlarged,
                               (int(minimum_size * 1.25), int(minimum_size * 1.25)), 8)
    scaled = [(round(x / 1.25), round(y / 1.25), round(w / 1.25), round(h / 1.25))
              for x, y, w, h in repeated]
    return [box for box in boxes if any(_boxes_overlap(box, other) for other in scaled)]


def _coherent_upper_signal(body: tuple, signal: tuple) -> bool:
    """Require containment and head/upper-body placement, not a box touching."""
    x, y, width, height = body
    sx, sy, sw, sh = signal
    intersection = max(0, min(x + width, sx + sw) - max(x, sx)) * max(0, min(y + height, sy + sh) - max(y, sy))
    return (sw > 0 and sh > 0 and width > 0 and height > 0
            and intersection >= .8 * sw * sh
            and x <= sx + sw / 2 <= x + width
            and y <= sy + sh / 2 <= y + .65 * height
            and sw <= 1.2 * width and sw * sh < .8 * width * height)


def _small_face_body_evidence(gray, face_detector, body_detector) -> list[dict]:
    """A separate small-person scan requires repeat face and body agreement."""
    faces = _cascade_detect(face_detector, gray, (MIN_FACE_SIZE // 2, MIN_FACE_SIZE // 2), 6)
    faces = [box for box in faces if box[2] < MIN_FACE_SIZE and box[3] < MIN_FACE_SIZE]
    if not faces:
        return []
    bodies = _cascade_detect(body_detector, gray, (MIN_UPPER_BODY_SIZE // 2, MIN_UPPER_BODY_SIZE // 2), 7)
    repeated = _confirm_cascade_boxes(face_detector, gray, faces, MIN_FACE_SIZE // 2)
    area = float(gray.shape[0] * gray.shape[1])
    return [{"source": "small_face_body", "bbox": body, "face_bbox": face,
             "confirmed": True, "area_ratio": body[2] * body[3] / area,
             "confirmation": "small repeat-scale face coherently contained in independent upper-body detection"}
            for face in repeated for body in bodies
            if body[3] >= 1.5 * face[3] and body[2] >= 1.3 * face[2]
            and _coherent_upper_signal(body, face)]


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
    small_people = []
    evidence = []
    image_area = float(bgr.shape[0] * bgr.shape[1])

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

        box = tuple(int(value) for value in rectangle)
        _, _, box_width, box_height = box
        plausible = (
            numeric_weight >= MIN_HOG_PERSON_WEIGHT
            and box_height > 0
            and 0.22 <= box_width / box_height <= 0.90
            and box_width * box_height / image_area >= 0.003
        )
        evidence.append({"source": "hog", "bbox": box, "weight": numeric_weight,
                         "area_ratio": box_width * box_height / image_area,
                         "confirmed": plausible,
                         "confirmation": "weight, human aspect and meaningful area" if plausible else "insufficient shape or confidence"})
        if (not plausible and numeric_weight >= MIN_HOG_PERSON_WEIGHT and box_height > 0
                and 0.22 <= box_width / box_height <= 0.90
                and 0.0005 <= box_width * box_height / image_area < 0.003):
            small_people.append((box, numeric_weight))
        if plausible:
            people.append(
                box
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

    corroborators = faces + profiles + upper_bodies
    for box, weight in small_people:
        if any(_coherent_upper_signal(box, signal) for signal in corroborators):
            people.append(box)
            people_weights.append(weight)
            for item in evidence:
                if item["source"] == "hog" and item["bbox"] == box:
                    item.update(confirmed=True, small_person_corroborated=True,
                                confirmation="small upright HOG and coherent independent face/profile/upper-body evidence")
    multiple_distinct_people = any(
        not _boxes_overlap(first, second)
        for index, first in enumerate(people)
        for second in people[index + 1:]
    )
    credible_people = []
    credible_weights = []
    for box, weight in zip(people, people_weights):
        corroborated = (weight >= 1.5 or multiple_distinct_people
                        or any(_boxes_overlap(box, other) for other in corroborators))
        for item in evidence:
            if item["source"] == "hog" and item["bbox"] == box:
                item["confirmed"] = corroborated
                item["confirmation"] = item["confirmation"] if item.get("small_person_corroborated") else (
                    "strong HOG weight, distinct people, or independent detector overlap"
                    if corroborated else "moderate HOG pattern without independent corroboration"
                )
        if corroborated:
            credible_people.append(box)
            credible_weights.append(weight)
    people = credible_people
    people_weights = credible_weights

    confirmed_faces = _confirm_cascade_boxes(frontal_detector, gray, faces, MIN_FACE_SIZE)
    confirmed_profiles = _confirm_cascade_boxes(profile_detector, gray, profiles, MIN_PROFILE_SIZE)
    for source, boxes, confirmed in (
        ("frontal_face", faces, confirmed_faces),
        ("profile_face", profiles, confirmed_profiles),
    ):
        for box in boxes:
            area_ratio = box[2] * box[3] / image_area
            body_overlap = any(_boxes_overlap(box, body) for body in upper_bodies + people)
            other_faces = profiles if source == "frontal_face" else faces
            independent_face = any(_boxes_overlap(box, other) for other in other_faces)
            # A repeat at another scale of the same cascade is one detector,
            # not independent evidence. Size alone cannot confirm a face.
            corroborated = box in confirmed and (body_overlap or independent_face)
            plausible = 0.55 <= box[2] / max(1, box[3]) <= 1.8 and area_ratio >= 0.0005
            accepted = corroborated and plausible
            evidence.append({"source": source, "bbox": box, "weight": None,
                             "area_ratio": area_ratio, "body_overlap": body_overlap,
                             "independent_face_overlap": independent_face,
                             "confirmed": accepted,
                             "confirmation": "independent detector overlap" if accepted else "isolated cascade pattern"})

    for box in upper_bodies:
        evidence.append({"source": "upper_body", "bbox": box, "weight": None,
                         "confirmed": False, "confirmation": "supporting evidence only"})

    # Upper-body cascades alone frequently match architectural and floral
    # patterns; they corroborate a person or face but never reject alone.
    image_height, image_width = gray.shape[:2]
    centered_lower_body = _centered_lower_upper_body(
        upper_bodies, image_width, image_height,
    )
    # The upper-body cascade often matches flowers, curtains and ornament.
    # A lone central hit is diagnostic evidence, not confirmation of a human.
    strong_upper_body_signal = centered_lower_body and bool(people or confirmed_faces or confirmed_profiles)

    detected = bool(
        people
        or any(item["confirmed"] and item["source"] == "frontal_face" for item in evidence)
        or any(item["confirmed"] and item["source"] == "profile_face" for item in evidence)
        or strong_upper_body_signal
    )
    if not detected:
        small_evidence = _small_face_body_evidence(gray, frontal_detector, upper_body_detector)
        evidence.extend(small_evidence)
        detected = bool(small_evidence)
    strong_signals = [item for item in evidence if item["confirmed"]]
    weak_signals = [item for item in evidence if not item["confirmed"]]

    return {
        "people": (
            people
        ),
        "evidence": evidence,
        "strong_signals": strong_signals,
        "weak_signals": weak_signals,
        "overlaps": [item for item in evidence if item.get("body_overlap")],
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
        "ambiguous_face_signal": bool((faces or profiles) and not detected),
        "detected": (
            detected
        ),
        "human_detection_confidence": "high" if detected else "low",
        "human_detection_reason": (
            "; ".join(dict.fromkeys(item["confirmation"] for item in strong_signals)) if detected
            else "isolated detector patterns without independent confirmation"
        ),
        "largest_detection_area_ratio": max(
            (item["bbox"][2] * item["bbox"][3] / image_area for item in evidence),
            default=0.0,
        ),
    }
