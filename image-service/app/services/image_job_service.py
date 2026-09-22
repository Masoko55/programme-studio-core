from app.services.execution_plan import (
    build_execution_plan,
)
from app.services.job_persistence import (
    image_job_exists,
    initialize_image_job,
    load_image_job_state,
)
from app.services.prompt_repository import (
    load_prompts_document,
)


def create_image_job(
    reference_number: str,
):
    if image_job_exists(
        reference_number
    ):
        return load_image_job_state(
            reference_number
        )

    prompts_document = (
        load_prompts_document(
            reference_number
        )
    )

    execution_plan = (
        build_execution_plan(
            prompts_document
        )
    )

    return initialize_image_job(
        reference_number=(
            reference_number
        ),
        execution_plan=(
            execution_plan
        ),
    )


def get_image_job(
    reference_number: str,
):
    return load_image_job_state(
        reference_number
    )