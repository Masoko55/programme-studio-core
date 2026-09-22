import hashlib
from pathlib import Path

from PIL import Image

from app.config.settings import settings


def validate_final_png(
    image_path: str,
    expected_sha256: str | None = None,
) -> dict:
    path = Path(
        image_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"PNG not found: {image_path}"
        )

    data = path.read_bytes()

    if not data.startswith(
        b"\x89PNG\r\n\x1a\n"
    ):
        raise ValueError(
            "Artifact is not a valid PNG."
        )

    actual_sha256 = hashlib.sha256(
        data
    ).hexdigest()

    if (
        expected_sha256 is not None
        and actual_sha256
        != expected_sha256
    ):
        raise ValueError(
            "Artifact SHA-256 mismatch."
        )

    with Image.open(
        path
    ) as image:
        image.load()
        if image.format != "PNG":
            raise ValueError("Artifact is not PNG")
        width, height = (
            image.size
        )

        if (
            width
            != settings.final_image_width
        ):
            raise ValueError(
                f"Expected width "
                f"{settings.final_image_width}, "
                f"got {width}."
            )

        if (
            height
            != settings.final_image_height
        ):
            raise ValueError(
                f"Expected height "
                f"{settings.final_image_height}, "
                f"got {height}."
            )

        dpi = image.info.get(
            "dpi"
        )

        if dpi is None:
            raise ValueError(
                "PNG does not contain DPI metadata."
            )

        horizontal_dpi = round(
            dpi[0]
        )

        vertical_dpi = round(
            dpi[1]
        )

        if (
            horizontal_dpi
            != settings.final_image_dpi
            or vertical_dpi
            != settings.final_image_dpi
        ):
            raise ValueError(
                "PNG DPI metadata does not "
                f"match {settings.final_image_dpi}."
            )

    return {
        "path": str(path),
        "sha256": actual_sha256,
        "width": width,
        "height": height,
        "dpi": (
            settings.final_image_dpi
        ),
        "valid": True,
    }