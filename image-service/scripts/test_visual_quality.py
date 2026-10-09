from pathlib import Path

from PIL import Image

from app.services.visual_quality import (
    validate_visual_quality,
)


path = Path(
    "/data/programmes/AEF3E9-636515/backgrounds/"
    "sd-3-5-medium/image-a.raw.png"
)

with Image.open(path) as image:
    result = validate_visual_quality(
        image.convert("RGB")
    )

print(result)