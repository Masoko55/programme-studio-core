import json
import re
from pathlib import Path

from app.config.settings import settings


REFERENCE_PATTERN = re.compile(
    r"^[A-F0-9]{6}-[0-9]{6}$"
)


def validate_reference_number(
    reference_number: str,
) -> str:
    normalized_reference = (
        reference_number.strip().upper()
    )

    if not REFERENCE_PATTERN.fullmatch(
        normalized_reference
    ):
        raise ValueError(
            "Invalid reference number format."
        )

    return normalized_reference


def get_job_directory(
    reference_number: str,
) -> Path:
    safe_reference = (
        validate_reference_number(
            reference_number
        )
    )

    return (
        settings.programme_data_path
        / safe_reference
    )


def get_prompts_path(
    reference_number: str,
) -> Path:
    return (
        get_job_directory(
            reference_number
        )
        / "prompts.json"
    )


def load_prompts_document(
    reference_number: str,
) -> dict:
    prompts_path = get_prompts_path(
        reference_number
    )

    if not prompts_path.exists():
        raise FileNotFoundError(
            f"prompts.json not found for "
            f"{reference_number}."
        )

    with prompts_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def get_direction(
    prompts_document: dict,
    direction_id: str,
) -> dict:
    for item in prompts_document.get(
        "directions",
        [],
    ):
        direction = item.get(
            "direction",
            {}
        )

        if (
            direction.get(
                "direction_id"
            )
            == direction_id
        ):
            return direction

    raise ValueError(
        f"Direction {direction_id} "
        "was not found in prompts.json."
    )