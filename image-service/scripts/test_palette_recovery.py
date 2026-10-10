from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from PIL import Image

from app.services.background_policy import (
    DARK_MIN_SATURATION,
    DARK_MIN_VALUE,
    MAX_OFF_PALETTE_RATIO,
    NEUTRAL_NAMES,
    STANDARD_MIN_SATURATION,
    STANDARD_MIN_VALUE,
    _chromatic_requested_family,
    _colour_family_members,
    _hue_distance,
    _is_dark_target,
    _matches_requested_neutral,
    _nearest_named_colour,
    _nearest_requested_family,
    _rgb_to_hsv,
    colour_descriptor,
    constrain_to_requested_palette,
)


SOURCE = Path(
    "/data/programmes/AEF3E9-636515/backgrounds/"
    "sd-3-5-medium/image-a.raw.png"
)

TARGET = Path(
    "/data/programmes/AEF3E9-636515/backgrounds/"
    "sd-3-5-medium/image-a.detail-preserving.png"
)

PRIMARY = "royal blue"
SECONDARY = "red"


def build_requested_targets():
    descriptors = [
        colour_descriptor(PRIMARY),
        colour_descriptor(SECONDARY),
    ]

    requested = []

    for descriptor in descriptors:
        name = descriptor["base_name"]
        rgb = descriptor["rgb"]

        if (
            not name
            or rgb is None
        ):
            continue

        requested.append(
            (
                name,
                rgb,
                descriptor,
            )
        )

    targets = []

    for (
        name,
        rgb,
        descriptor,
    ) in requested:
        if name in NEUTRAL_NAMES:
            continue

        hue, _, _ = _rgb_to_hsv(
            rgb
        )

        targets.append(
            {
                "name": name,
                "rgb": rgb,
                "hue": hue,
                "descriptor": descriptor,
                "family_members": (
                    _colour_family_members(
                        name
                    )
                ),
            }
        )

    return requested, targets


def inspect_palette(
    image: Image.Image,
):
    requested, requested_targets = (
        build_requested_targets()
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

    sample = (
        image.convert("RGB")
        .resize(
            (
                128,
                176,
            ),
            Image.Resampling.LANCZOS,
        )
    )

    rgb_array = np.asarray(
        sample,
        dtype=np.uint8,
    )

    hsv_array = cv2.cvtColor(
        rgb_array,
        cv2.COLOR_RGB2HSV,
    )

    rgb_pixels = rgb_array.reshape(
        -1,
        3,
    )

    hsv_pixels = hsv_array.reshape(
        -1,
        3,
    )

    total = len(
        hsv_pixels
    )

    accepted = Counter()
    rejected_nearest = Counter()
    rejection_reason = Counter()

    rejected_values = []
    rejected_saturations = []
    rejected_hues = []

    for (
        pixel_rgb,
        pixel_hsv,
    ) in zip(
        rgb_pixels,
        hsv_pixels,
    ):
        hue = int(
            pixel_hsv[0]
        )

        saturation = int(
            pixel_hsv[1]
        )

        value = int(
            pixel_hsv[2]
        )

        neutral_match = (
            _matches_requested_neutral(
                saturation,
                value,
                requested_neutrals,
            )
        )

        if (
            neutral_match
            is not None
        ):
            accepted[
                neutral_match
            ] += 1

            continue

        chromatic_match = (
            _chromatic_requested_family(
                hue,
                saturation,
                value,
                requested_targets,
            )
        )

        if (
            chromatic_match
            is not None
        ):
            accepted[
                chromatic_match
            ] += 1

            continue

        nearest = (
            _nearest_named_colour(
                pixel_rgb
            )
        )

        family_match = (
            _nearest_requested_family(
                nearest,
                pixel_rgb,
                requested_targets,
            )
        )

        if (
            family_match
            is not None
        ):
            accepted[
                family_match
            ] += 1

            continue

        rejected_nearest[
            nearest
        ] += 1

        rejected_values.append(
            value
        )

        rejected_saturations.append(
            saturation
        )

        rejected_hues.append(
            hue
        )

        target_failures = []

        for target in requested_targets:
            target_name = (
                target["name"]
            )

            target_rgb = (
                target["rgb"]
            )

            target_hue = (
                target["hue"]
            )

            dark_target = (
                _is_dark_target(
                    target_name,
                    target_rgb,
                )
            )

            minimum_saturation = (
                DARK_MIN_SATURATION
                if dark_target
                else STANDARD_MIN_SATURATION
            )

            minimum_value = (
                DARK_MIN_VALUE
                if dark_target
                else STANDARD_MIN_VALUE
            )

            hue_distance = (
                _hue_distance(
                    hue,
                    target_hue,
                )
            )

            failures = []

            if (
                saturation
                < minimum_saturation
            ):
                failures.append(
                    "saturation"
                )

            if (
                value
                < minimum_value
            ):
                failures.append(
                    "value"
                )

            if (
                hue_distance
                > 12
            ):
                failures.append(
                    "hue"
                )

            if not failures:
                failures.append(
                    "other"
                )

            target_failures.append(
                (
                    target_name,
                    failures,
                    hue_distance,
                )
            )

        #
        # Determine the most informative failure reason.
        #
        all_failure_sets = [
            set(
                failures
            )
            for (
                _,
                failures,
                _,
            )
            in target_failures
        ]

        if any(
            "value" in failures
            and "hue" not in failures
            for failures
            in all_failure_sets
        ):
            rejection_reason[
                "value_too_low"
            ] += 1

        elif any(
            "saturation" in failures
            and "hue" not in failures
            for failures
            in all_failure_sets
        ):
            rejection_reason[
                "saturation_too_low"
            ] += 1

        elif all(
            "hue" in failures
            for failures
            in all_failure_sets
        ):
            rejection_reason[
                "hue_mismatch"
            ] += 1

        else:
            rejection_reason[
                "mixed_or_other"
            ] += 1

    rejected_total = sum(
        rejected_nearest.values()
    )

    off_palette_ratio = (
        rejected_total
        / total
    )

    print()
    print(
        "========================================"
    )
    print(
        "PALETTE DIAGNOSTIC"
    )
    print(
        "========================================"
    )

    print(
        "TOTAL PIXELS:",
        total,
    )

    print(
        "ACCEPTED PIXELS:",
        total
        - rejected_total,
    )

    print(
        "REJECTED PIXELS:",
        rejected_total,
    )

    print(
        "OFF_PALETTE:",
        round(
            off_palette_ratio,
            6,
        ),
    )

    print(
        "MAX_ALLOWED:",
        MAX_OFF_PALETTE_RATIO,
    )

    print()
    print(
        "ACCEPTED FAMILIES:",
        dict(
            accepted
        ),
    )

    print()
    print(
        "REJECTED NEAREST COLOURS:",
        dict(
            rejected_nearest
        ),
    )

    print()
    print(
        "REJECTION REASONS:",
        dict(
            rejection_reason
        ),
    )

    if rejected_total:
        print()
        print(
            "REJECTED VALUE RANGE:",
            min(
                rejected_values
            ),
            "->",
            max(
                rejected_values
            ),
        )

        print(
            "REJECTED VALUE MEAN:",
            round(
                float(
                    np.mean(
                        rejected_values
                    )
                ),
                2,
            ),
        )

        print()
        print(
            "REJECTED SATURATION RANGE:",
            min(
                rejected_saturations
            ),
            "->",
            max(
                rejected_saturations
            ),
        )

        print(
            "REJECTED SATURATION MEAN:",
            round(
                float(
                    np.mean(
                        rejected_saturations
                    )
                ),
                2,
            ),
        )

        print()
        print(
            "REJECTED HUE RANGE:",
            min(
                rejected_hues
            ),
            "->",
            max(
                rejected_hues
            ),
        )

    print()
    print(
        "STANDARD_MIN_SATURATION:",
        STANDARD_MIN_SATURATION,
    )

    print(
        "STANDARD_MIN_VALUE:",
        STANDARD_MIN_VALUE,
    )

    print(
        "DARK_MIN_SATURATION:",
        DARK_MIN_SATURATION,
    )

    print(
        "DARK_MIN_VALUE:",
        DARK_MIN_VALUE,
    )

    print(
        "========================================"
    )


def main():
    with Image.open(
        SOURCE
    ) as image:
        recovered = (
            constrain_to_requested_palette(
                image.convert(
                    "RGB"
                ),
                PRIMARY,
                SECONDARY,
            )
        )

    recovered.save(
        TARGET,
        format="PNG",
    )

    print(
        "OUTPUT:",
        TARGET,
    )

    with Image.open(
        TARGET
    ) as image:
        inspect_palette(
            image.convert(
                "RGB"
            )
        )


if __name__ == "__main__":
    main()