import json
import re
from pathlib import Path

from app.config.settings import settings
from app.schemas.image_job_state import (
    ImageJobState,
    ImageOutputRecord,
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


def get_image_job_state_path(
    reference_number: str,
) -> Path:
    return (
        get_job_directory(
            reference_number
        )
        / "image-job.json"
    )


def initialize_image_job(
    reference_number: str,
    execution_plan: dict,
) -> ImageJobState:
    outputs = []

    for step in execution_plan[
        "execution_order"
    ]:
        outputs.append(
            ImageOutputRecord(
                engine_id=step[
                    "engine_id"
                ],
                engine_label=step[
                    "engine_label"
                ],
                direction_id=step[
                    "direction"
                ],
                status="pending",
                output_path=None,
                sha256=None,
                error=None,
            )
        )

    state = ImageJobState(
        reference_number=(
            reference_number
        ),
        status="ready",
        current_stage="planned",
        total_outputs=len(outputs),
        completed_outputs=0,
        outputs=outputs,
        error=None,
    )

    persist_image_job_state(
        state
    )

    return state


def persist_image_job_state(
    state: ImageJobState,
) -> Path:
    job_directory = get_job_directory(
        state.reference_number
    )

    job_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = get_image_job_state_path(
        state.reference_number
    )

    from app.services.atomic import write_json
    write_json(path, state.model_dump(mode="json"))

    return path


def load_image_job_state(
    reference_number: str,
) -> ImageJobState:
    path = get_image_job_state_path(
        reference_number
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Image job state not found for "
            f"{reference_number}."
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return ImageJobState.model_validate(
        data
    )


def image_job_exists(
    reference_number: str,
) -> bool:
    return get_image_job_state_path(
        reference_number
    ).exists()


def update_output(
    state: ImageJobState,
    engine_id: str,
    direction_id: str,
    status: str,
    output_path: str | None = None,
    sha256: str | None = None,
    error: str | None = None,
) -> ImageJobState:
    for output in state.outputs:
        if (
            output.engine_id == engine_id
            and output.direction_id == direction_id
        ):
            output.status = status
            output.output_path = (
                output_path
            )
            output.sha256 = sha256
            output.error = error

            break

    state.completed_outputs = sum(
        1
        for output in state.outputs
        if output.status == "complete"
    )

    persist_image_job_state(
        state
    )

    return state