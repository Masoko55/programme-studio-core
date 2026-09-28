"""Deterministic policy for generated programme backgrounds.

This module contains requirements that must remain consistent regardless
of which diffusion model or creative-direction model is being used.

Responsibilities:
- enforce the user's requested primary/secondary colour palette
- strengthen prompts so generated images remain background-only
- prevent people, faces, silhouettes and human subjects
- validate generated candidates before they become selectable
"""

from __future__ import annotations

import re

import cv2
import numpy as np

from PIL import Image


NAMED_COLOURS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),

    "grey": (128, 128, 128),
    "gray": (128, 128, 128),
    "silver": (185, 185, 190),

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

    "navy": (25, 45, 100),
    "teal": (35, 135, 135),
    "cyan": (55, 170, 190),

    "brown": (120, 80, 50),
    "beige": (210, 190, 150),
    "cream": (240, 225, 190),

    "maroon": (115, 30, 50),
    "burgundy": (120, 35, 60),
}


NEUTRAL_NAMES = {
    "black",
    "white",
    "grey",
    "gray",
    "silver",
}


BACKGROUND_ONLY_NEGATIVE = (
    "(person:2.0), "
    "(people:2.0), "
    "(human:2.0), "
    "(human figure:2.0), "
    "(man:2.0), "
    "(woman:2.0), "
    "(child:2.0), "
    "(face:2.0), "
    "(portrait:2.0), "
    "(body:2.0), "
    "(silhouette:2.0), "
    "(character:2.0), "
    "(head:2.0), "
    "(hands:2.0), "
    "(arms:2.0), "
    "(legs:2.0), "
    "(clothing:2.0), "
    "(mannequin:2.0), "
    "(human-shaped object:2.0), "
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
    "No narrative scene. "
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
    "gradients, shapes and lighting only. "
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
)


COLOUR_ALIASES = {
    "gray": "grey",
    "golden": "gold",
}


def normalize_colour_name(
    value: str | None,
) -> str | None:
    if not value:
        return None

    value = (
        value
        .strip()
        .lower()
    )

    return (
        COLOUR_ALIASES.get(
            value,
            value,
        )
    )


def parse_colour(
    value: str | None,
) -> tuple[int, int, int] | None:
    if not value:
        return None

    normalized = (
        value
        .strip()
        .lower()
    )

    if normalized in NAMED_COLOURS:
        return NAMED_COLOURS[
            normalized
        ]

    hexadecimal = re.fullmatch(
        r"#?([0-9a-f]{6})",
        normalized,
    )

    if hexadecimal:
        raw = hexadecimal.group(
            1
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

    for (
        name,
        rgb,
    ) in NAMED_COLOURS.items():
        if re.search(
            rf"\b{re.escape(name)}\b",
            normalized,
        ):
            return rgb

    return None


def requested_palette_names(
    primary_colour: str | None,
    secondary_colour: str | None,
) -> list[str]:
    colours = []

    for value in (
        primary_colour,
        secondary_colour,
    ):
        normalized = (
            normalize_colour_name(
                value
            )
        )

        if (
            normalized
            and normalized not in colours
        ):
            colours.append(
                normalized
            )

    return colours


def palette_prompt_contract(
    primary_colour: str | None,
    secondary_colour: str | None,
) -> str:
    colours = (
        requested_palette_names(
            primary_colour,
            secondary_colour,
        )
    )

    if not colours:
        return (
            "Use one restrained and coherent colour palette."
        )

    primary = colours[
        0
    ]

    secondary = (
        colours[
            1
        ]
        if len(
            colours
        ) > 1
        else None
    )

    if secondary:
        palette_text = (
            f"{primary} and {secondary}"
        )
    else:
        palette_text = primary

    statements = [
        (
            "STRICT COLOUR PALETTE CONTRACT: "
            f"use only {palette_text} as the intended palette."
        ),
        (
            f"The primary colour is {primary}. "
            "It must be clearly visible and visually dominant."
        ),
        (
            "Do not substitute the requested colours with colours "
            "that merely fit the event theme."
        ),
        (
            "Do not introduce unrelated accent colours."
        ),
    ]

    if secondary:
        statements.append(
            (
                f"The secondary colour is {secondary}. "
                "It supports the primary colour."
            )
        )

    if set(
        colours
    ) == {
        "black",
        "white",
    }:
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

    elif (
        secondary
        in {
            "white",
            "black",
            "grey",
        }
    ):
        statements.append(
            (
                "Any chromatic colour must remain inside "
                f"the {primary} colour family."
            )
        )

    return " ".join(
        statements
    )


def palette_negative_contract(
    primary_colour: str | None,
    secondary_colour: str | None,
) -> str:
    allowed = set(
        requested_palette_names(
            primary_colour,
            secondary_colour,
        )
    )

    banned = []

    for colour in NAMED_COLOURS:
        normalized = (
            normalize_colour_name(
                colour
            )
        )

        if (
            normalized
            and normalized not in allowed
            and normalized not in banned
        ):
            banned.append(
                normalized
            )

    if not banned:
        return (
            "off-palette colour, colour drift, "
            "unrequested accent colours"
        )

    return (
        "off-palette colour, colour drift, "
        "unrequested accent colours, multicolour palette, "
        + ", ".join(
            banned
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

        if not contains_forbidden:
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
        "No subject and no narrative scene. "
        f"{palette} "
        f"{visual.rstrip('. ')}. "
        f"{BACKGROUND_ONLY_SUFFIX}"
    )


def _rgb_to_hsv(
    colour: tuple[int, int, int],
) -> tuple[int, int, int]:
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
        180
        - raw,
    )


def validate_palette(
    image: Image.Image,
    primary_colour: str | None,
    secondary_colour: str | None,
) -> dict:
    """Validate actual generated pixels against the requested palette."""

    primary_name = (
        normalize_colour_name(
            primary_colour
        )
    )

    secondary_name = (
        normalize_colour_name(
            secondary_colour
        )
    )

    primary_rgb = (
        parse_colour(
            primary_colour
        )
    )

    secondary_rgb = (
        parse_colour(
            secondary_colour
        )
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
        for item in requested
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
            _
        ) in requested
    }

    if requested_names == {
        "black",
        "white",
    }:
        saturation = (
            pixels[
                :,
                1
            ]
        )

        monochrome_ratio = float(
            np.mean(
                saturation <= 30
            )
        )

        if (
            monochrome_ratio
            < 0.94
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
                (
                    name,
                    hue,
                )
            )

    chromatic_counts = {
        name: 0
        for (
            name,
            _
        ) in chromatic_targets
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

        # Neutral regions include white, grey and black.
        #
        # Neutral shading is permitted for depth, but it
        # does not satisfy the requirement that a chromatic
        # primary colour must actually be visible.
        if (
            saturation <= 42
            or value <= 32
        ):
            neutral_count += 1
            continue

        best_name = None
        best_distance = None

        for (
            name,
            target_hue,
        ) in chromatic_targets:
            distance = (
                _hue_distance(
                    hue,
                    target_hue,
                )
            )

            if (
                best_distance
                is None
                or distance
                < best_distance
            ):
                best_name = name
                best_distance = distance

        if (
            best_name is not None
            and best_distance is not None
            and best_distance <= 12
        ):
            chromatic_counts[
                best_name
            ] += 1

        else:
            off_palette_count += 1

    off_palette_ratio = (
        off_palette_count
        / total
    )

    # 5% tolerance handles antialiasing, shadows and
    # minor diffusion noise without permitting a second
    # unrelated colour scheme.
    if (
        off_palette_ratio
        > 0.05
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
    ):
        primary_ratio = (
            chromatic_counts.get(
                primary_name,
                0,
            )
            / total
        )

        # Prevent a virtually white/grey image from passing
        # a pink+white, blue+white, red+white etc. brief.
        if (
            primary_ratio
            < 0.10
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
    ):
        secondary_ratio = (
            chromatic_counts.get(
                secondary_name,
                0,
            )
            / total
        )

        if (
            secondary_ratio
            < 0.04
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

    if classifier.empty():
        return None

    return classifier


def _cascade_detect(
    classifier,
    gray: np.ndarray,
    minimum_size: tuple[int, int],
) -> list[
    tuple[int, int, int, int]
]:
    if classifier is None:
        return []

    results = (
        classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
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
            for value in rectangle
        )
        for rectangle
        in results
    ]


def detect_human_signals(
    image: Image.Image,
) -> dict:
    """Detect multiple kinds of human content."""

    rgb = np.array(
        image.convert(
            "RGB"
        )
    )

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
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

    gray = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2GRAY,
    )

    hog = (
        cv2.HOGDescriptor()
    )

    hog.setSVMDetector(
        cv2.HOGDescriptor_getDefaultPeopleDetector()
    )

    rectangles, weights = (
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

    for (
        rectangle,
        weight,
    ) in zip(
        rectangles,
        weights,
    ):
        if (
            float(
                weight
            )
            >= 0.40
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
                36,
                36,
            ),
        )
    )

    profiles = (
        _cascade_detect(
            profile_detector,
            gray,
            (
                40,
                40,
            ),
        )
    )

    upper_bodies = (
        _cascade_detect(
            upper_body_detector,
            gray,
            (
                55,
                55,
            ),
        )
    )

    detected = bool(
        people
        or faces
        or profiles
        or upper_bodies
    )

    return {
        "people": people,
        "faces": faces,
        "profiles": profiles,
        "upper_bodies": upper_bodies,
        "detected": detected,
    }