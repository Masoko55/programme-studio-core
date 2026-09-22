import logging

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from app.graph.state import (
    ImageWorkflowState,
)
from app.services.composition_service import (
    compose_generated_background,
)
from app.services.image_execution import (
    execute_image_job,
)
from app.services.image_job_service import (
    create_image_job,
)
from app.services.manifest_service import (
    build_manifest,
    persist_manifest,
)
from app.services.publication_service import (
    publish_programme,
)


logger = logging.getLogger(
    "uvicorn.error"
)


async def prepare_job(
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


async def compose_images(
    state: ImageWorkflowState,
) -> dict:
    reference_number = state[
        "reference_number"
    ]

    image_job = state[
        "image_job"
    ]

    logger.info(
        "Workflow composing final PNGs "
        "for %s",
        reference_number,
    )

    composition_results = []

    for output in image_job[
        "outputs"
    ]:
        if output[
            "status"
        ] != "complete":
            raise RuntimeError(
                "Cannot compose "
                f"{output['engine_id']}/"
                f"{output['direction_id']} "
                "because image generation "
                "is not complete."
            )

        background_path = output.get(
            "output_path"
        )

        if not background_path:
            raise RuntimeError(
                "Generated background path "
                "is missing for "
                f"{output['engine_id']}/"
                f"{output['direction_id']}."
            )

        result = (
            compose_generated_background(
                reference_number=(
                    reference_number
                ),
                engine_id=(
                    output[
                        "engine_id"
                    ]
                ),
                direction_id=(
                    output[
                        "direction_id"
                    ]
                ),
                background_path=(
                    background_path
                ),
            )
        )

        composition_results.append(
            result
        )

        logger.info(
            "Composed %s/%s",
            output["engine_id"],
            output["direction_id"],
        )

    return {
        "status": "processing",
        "current_stage": (
            "composition_complete"
        ),
        "composition_results": (
            composition_results
        ),
    }


async def create_manifest(
    state: ImageWorkflowState,
) -> dict:
    reference_number = state[
        "reference_number"
    ]

    logger.info(
        "Workflow building manifest "
        "for %s",
        reference_number,
    )

    manifest = build_manifest(
        reference_number
    )

    persist_manifest(
        manifest
    )

    if not manifest.complete:
        raise RuntimeError(
            "Manifest is incomplete after "
            "composition. "
            f"{manifest.completed_artifact_count}/"
            f"{manifest.expected_artifact_count} "
            "artifacts passed validation."
        )

    return {
        "status": "processing",
        "current_stage": (
            "manifest_complete"
        ),
        "manifest": (
            manifest.model_dump(
                mode="json"
            )
        ),
    }


async def publish_artifacts(
    state: ImageWorkflowState,
) -> dict:
    reference_number = state[
        "reference_number"
    ]

    logger.info(
        "Workflow publishing %s",
        reference_number,
    )

    manifest = (
        await publish_programme(
            reference_number
        )
    )

    return {
        "status": "complete",
        "current_stage": "complete",
        "manifest": (
            manifest.model_dump(
                mode="json"
            )
        ),
        "publication": {
            "status": "published",
            "reference_number": (
                reference_number
            ),
        },
        "error": None,
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

builder.add_node(
    "compose_images",
    compose_images,
)

builder.add_node(
    "create_manifest",
    create_manifest,
)

builder.add_node(
    "publish_artifacts",
    publish_artifacts,
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
    "compose_images",
)

builder.add_edge(
    "compose_images",
    "create_manifest",
)

builder.add_edge(
    "create_manifest",
    "publish_artifacts",
)

builder.add_edge(
    "publish_artifacts",
    END,
)


image_workflow = builder.compile()