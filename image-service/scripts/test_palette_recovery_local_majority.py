from pathlib import Path

from PIL import Image

from app.services.background_policy import (
    constrain_to_requested_palette,
    validate_palette,
)


SOURCE = Path(
    "/data/programmes/5592B0-741491/backgrounds/"
    "sd-3-5-medium/image-a.raw.png"
)

TARGET = Path(
    "/data/programmes/5592B0-741491/backgrounds/"
    "sd-3-5-medium/image-a.local-majority.png"
)

PRIMARY = "royal blue"
SECONDARY = "red"


def main():
    with Image.open(SOURCE) as image:
        recovered = constrain_to_requested_palette(
            image.convert("RGB"),
            PRIMARY,
            SECONDARY,
        )

    recovered.save(
        TARGET,
        format="PNG",
    )

    with Image.open(TARGET) as image:
        result = validate_palette(
            image.convert("RGB"),
            PRIMARY,
            SECONDARY,
        )

    print("OUTPUT:", TARGET)
    print("OFF_PALETTE:", result["off_palette_ratio"])
    print("PALETTE_MATCH:", result["palette_match_ratio"])
    print("PRIMARY:", result["primary_colour_ratio"])
    print("SECONDARY:", result["secondary_colour_ratio"])
    print(
        "CENTER_EDGE:",
        result["center_structural_edge_density"],
    )
    print(
        "OUTER_EDGE:",
        result["outer_structural_edge_density"],
    )
    print(
        "CENTER_TO_OUTER:",
        result["center_to_outer_edge_ratio"],
    )
    print(
        "CENTER_SAFE:",
        result["center_safe_region_passed"],
    )


if __name__ == "__main__":
    main()