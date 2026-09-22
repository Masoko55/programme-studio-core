from typing import (
    Any,
    TypedDict,
)


class ImageWorkflowState(
    TypedDict,
    total=False,
):
    reference_number: str

    status: str

    current_stage: str

    image_job: dict[str, Any]

    composition_results: list[
        dict[str, Any]
    ]

    manifest: dict[str, Any]

    publication: dict[str, Any]

    error: str | None