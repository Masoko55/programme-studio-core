from typing import TypedDict

from app.schemas.creative_direction import (
    CreativeDirectionOutput,
)
from app.schemas.prompts_document import (
    PromptsDocument,
)


class PromptWorkflowState(
    TypedDict,
    total=False,
):
    reference_number: str

    input_sha256: str

    brief: dict

    direction_a: (
        CreativeDirectionOutput
        | None
    )

    direction_b: (
        CreativeDirectionOutput
        | None
    )

    direction_c: (
        CreativeDirectionOutput
        | None
    )

    validation_error_a: str | None

    validation_error_b: str | None

    validation_error_c: str | None

    retry_count_a: int

    retry_count_b: int

    retry_count_c: int

    prompts_document: (
        PromptsDocument
        | None
    )