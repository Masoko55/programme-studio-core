import json
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path

from app.config.settings import (
    settings,
)


def get_workflow_state_path(
    reference_number: str,
) -> Path:
    return (
        settings.programme_data_path
        / reference_number
        / "image-workflow.json"
    )


def save_workflow_state(
    reference_number: str,
    state: dict,
) -> Path:
    path = get_workflow_state_path(
        reference_number
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    document = {
        "updated_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        **state,
    }

    temporary_path = (
        path.with_suffix(
            ".json.part"
        )
    )

    temporary_path.write_text(
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(
        path
    )

    return path


def load_workflow_state(
    reference_number: str,
) -> dict | None:
    path = get_workflow_state_path(
        reference_number
    )

    if not path.exists():
        return None

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )