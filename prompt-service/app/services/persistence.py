import json
import re
from datetime import datetime, timezone
from pathlib import Path

from app.config.settings import settings
from app.schemas.creative_direction import (
    CreativeDirectionOutput,
)
from app.schemas.prompts_document import (
    PromptsDocument,
)


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


def get_job_metadata_path(
    reference_number: str,
) -> Path:
    return (
        get_job_directory(reference_number)
        / "job.json"
    )


def initialize_job(
    reference_number: str,
    input_sha256: str,
    brief: dict,
) -> None:
    job_directory = get_job_directory(
        reference_number
    )

    job_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    persist_brief(
        reference_number=reference_number,
        brief=brief,
    )

    metadata = {
        "reference_number": (
            reference_number
        ),
        "input_sha256": input_sha256,
        "status": "processing",
        "current_stage": "created",
        "created_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "updated_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "error": None,
    }

    persist_job_metadata(
        reference_number,
        metadata,
    )


def persist_job_metadata(
    reference_number: str,
    metadata: dict,
) -> Path:
    job_directory = get_job_directory(
        reference_number
    )

    job_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        job_directory
        / "job.json"
    )

    metadata["updated_at"] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    return metadata_path


def load_job_metadata(
    reference_number: str,
) -> dict:
    metadata_path = (
        get_job_metadata_path(
            reference_number
        )
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Job metadata not found for "
            f"{reference_number}."
        )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def update_job_status(
    reference_number: str,
    status: str,
    current_stage: str,
    error: str | None = None,
) -> None:
    metadata = load_job_metadata(
        reference_number
    )

    metadata["status"] = status

    metadata["current_stage"] = (
        current_stage
    )

    metadata["error"] = error

    persist_job_metadata(
        reference_number,
        metadata,
    )


def persist_brief(
    reference_number: str,
    brief: dict,
) -> Path:
    job_directory = get_job_directory(
        reference_number
    )

    job_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    brief_path = (
        job_directory
        / "brief.json"
    )

    with brief_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            brief,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    return brief_path


def load_brief(
    reference_number: str,
) -> dict:
    brief_path = (
        get_job_directory(
            reference_number
        )
        / "brief.json"
    )

    if not brief_path.exists():
        raise FileNotFoundError(
            f"Brief not found for "
            f"{reference_number}."
        )

    with brief_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def get_direction_checkpoint_path(
    reference_number: str,
    direction_id: str,
) -> Path:
    normalized_id = (
        direction_id.strip().lower()
    )

    if normalized_id not in {
        "a",
        "b",
        "c",
    }:
        raise ValueError(
            "Invalid direction ID."
        )

    return (
        get_job_directory(
            reference_number
        )
        / f"direction-{normalized_id}.json"
    )


def persist_direction_checkpoint(
    reference_number: str,
    direction_id: str,
    direction: CreativeDirectionOutput,
) -> Path:
    checkpoint_path = (
        get_direction_checkpoint_path(
            reference_number,
            direction_id,
        )
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with checkpoint_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            direction.model_dump(
                mode="json"
            ),
            file,
            ensure_ascii=False,
            indent=2,
        )

    return checkpoint_path


def load_direction_checkpoint(
    reference_number: str,
    direction_id: str,
) -> CreativeDirectionOutput | None:
    checkpoint_path = (
        get_direction_checkpoint_path(
            reference_number,
            direction_id,
        )
    )

    if not checkpoint_path.exists():
        return None

    with checkpoint_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return (
        CreativeDirectionOutput
        .model_validate(data)
    )


def persist_prompts_document(
    document: PromptsDocument,
) -> Path:
    job_directory = get_job_directory(
        document.reference_number
    )

    job_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    prompts_path = (
        job_directory
        / "prompts.json"
    )

    with prompts_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            document.model_dump(
                mode="json"
            ),
            file,
            ensure_ascii=False,
            indent=2,
        )

    return prompts_path


def load_prompts_document(
    reference_number: str,
) -> PromptsDocument:
    prompts_path = (
        get_job_directory(
            reference_number
        )
        / "prompts.json"
    )

    if not prompts_path.exists():
        raise FileNotFoundError(
            f"Prompts document not found for "
            f"{reference_number}."
        )

    with prompts_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return PromptsDocument.model_validate(
        data
    )


def job_exists(
    reference_number: str,
) -> bool:
    return get_job_directory(
        reference_number
    ).exists()


def get_job_status(
    reference_number: str,
) -> str:
    job_directory = get_job_directory(
        reference_number
    )

    if not job_directory.exists():
        return "not_found"

    metadata_path = (
        job_directory
        / "job.json"
    )

    if metadata_path.exists():
        metadata = load_job_metadata(
            reference_number
        )

        return metadata.get(
            "status",
            "unknown",
        )

    prompts_path = (
        job_directory
        / "prompts.json"
    )

    if prompts_path.exists():
        return "complete"

    return "incomplete"