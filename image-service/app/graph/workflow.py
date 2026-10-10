import logging

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from app.graph.state import (
    ImageWorkflowState,
)
from app.services.image_execution import (
    execute_image_job,
)
from app.services.image_job_service import (
    create_image_job,
)


logger = logging.getLogger(
    "uvicorn.error"
)


def prepare_job(
    state: ImageWorkflowState,
) -> dict:
    reference_number = state[
        "reference_number"
    ]

    logger.info(
        "Workflow preparing image job %s",
        reference_number,
    )

    image_job = create_image_job(
        reference_number
    )

    return {
        "status": "processing",
        "current_stage": "prepared",
        "image_job": (
            image_job.model_dump(
                mode="json"
            )
        ),
        "error": None,
    }


async def generate_images(
    state: ImageWorkflowState,
) -> dict:
    reference_number = state[
        "reference_number"
    ]

    logger.info(
        "Workflow generating images for %s",
        reference_number,
    )

    image_job = (
        await execute_image_job(
            reference_number
        )
    )

    return {
        "status": "processing",
        "current_stage": (
            "generation_complete"
        ),
        "image_job": (
            image_job.model_dump(
                mode="json"
            )
        ),
    }



builder = StateGraph(
    ImageWorkflowState
)

builder.add_node(
    "prepare_job",
    prepare_job,
)

builder.add_node(
    "generate_images",
    generate_images,
)


builder.add_edge(
    START,
    "prepare_job",
)

builder.add_edge(
    "prepare_job",
    "generate_images",
)

builder.add_edge(
    "generate_images",
    END,
)


image_workflow = builder.compile()
