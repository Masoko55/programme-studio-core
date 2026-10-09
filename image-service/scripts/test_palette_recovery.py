from pathlib import Path

from PIL import Image

from app.services.background_policy import (
    constrain_to_requested_palette,
    validate_palette,
)


source = Path(
    "/data/programmes/AEF3E9-636515/backgrounds/"
    "sd-3-5-medium/image-a.raw.png"
)

target = Path(
    "/data/programmes/AEF3E9-636515/backgrounds/"
    "sd-3-5-medium/image-a.detail-preserving.png"
)

with Image.open(source) as image:
    recovered = constrain_to_requested_palette(
        image,
        "royal blue",
        "red",
    )

    recovered.save(
        target,
        format="PNG",
    )

with Image.open(target) as image:
    result = validate_palette(
        image.convert("RGB"),
        "royal blue",
        "red",
    )

print("OUTPUT:", target)
print("OFF_PALETTE:", result["off_palette_ratio"])
print("PRIMARY:", result["primary_colour_ratio"])
print("SECONDARY:", result["secondary_colour_ratio"])
print("MATCH:", result["palette_match_ratio"])