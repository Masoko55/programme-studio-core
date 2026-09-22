import logging
from app.config.settings import settings

from app.graph.workflow import (
    prompt_workflow,
)
from app.services.persistence import (
    initialize_job,
    job_exists,
    load_direction_checkpoint,
    persist_prompts_document,
    update_job_status,
)


logger = logging.getLogger(
    "uvicorn.error"
)


async def run_prompt_workflow(
    reference_number: str,
    input_sha256: str,
    brief: dict,
    resume: bool = False,
) -> dict:
    if not resume:
        initialize_job(
            reference_number=(
                reference_number
            ),
            input_sha256=(
                input_sha256
            ),
            brief=brief,
        )

    elif not job_exists(
        reference_number
    ):
        raise ValueError(
            "Cannot resume a job "
            "that does not exist."
        )

    direction_a = (
        load_direction_checkpoint(
            reference_number,
            "A",
        )
    )

    direction_b = (
        load_direction_checkpoint(
            reference_number,
            "B",
        )
    )

    direction_c = (
        load_direction_checkpoint(
            reference_number,
            "C",
        )
    )

    if direction_a is not None:
        logger.info(
            "Loaded Direction A checkpoint"
        )

    if direction_b is not None:
        logger.info(
            "Loaded Direction B checkpoint"
        )

    if direction_c is not None:
        logger.info(
            "Loaded Direction C checkpoint"
        )

    initial_state = {
        "reference_number": (
            reference_number
        ),
        "input_sha256": input_sha256,
        "brief": brief,
        "direction_a": direction_a,
        "direction_b": direction_b,
        "direction_c": direction_c,
        "retry_count_a": 0,
        "retry_count_b": 0,
        "retry_count_c": 0,
        "validation_error_a": None,
        "validation_error_b": None,
        "validation_error_c": None,
    }

    try:
        result = (
            await prompt_workflow.ainvoke(
                initial_state
            )
        )

        document = result[
            "prompts_document"
        ]

        prompts_path = (
            persist_prompts_document(
                document
            )
        )

        update_job_status(
            reference_number=(
                reference_number
            ),
            status="processing",
            current_stage="handoff_queued",
            error=None,
        )

        result["brief_path"] = str(settings.programme_data_path / reference_number / "brief.json")

        result["prompts_path"] = str(
            prompts_path
        )

        return result

    except Exception as error:
        update_job_status(
            reference_number=(
                reference_number
            ),
            status="failed",
            current_stage="failed",
            error=str(error),
        )

        raise
