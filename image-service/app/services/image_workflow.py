import logging

from app.graph.workflow import (
    image_workflow,
)
from app.services.workflow_state import (
    save_workflow_state,
)


logger = logging.getLogger(
    "uvicorn.error"
)


async def run_image_workflow(
    reference_number: str,
) -> dict:
    state = {
        "reference_number": (
            reference_number
        ),
        "status": "processing",
        "current_stage": "starting",
        "composition_results": [],
        "error": None,
    }

    save_workflow_state(
        reference_number,
        state,
    )

    try:
        async for update in (
            image_workflow.astream(
                state,
                stream_mode="updates",
            )
        ):
            for (
                node_name,
                node_update,
            ) in update.items():
                if not isinstance(
                    node_update,
                    dict,
                ):
                    continue

                state.update(
                    node_update
                )

                state[
                    "last_completed_node"
                ] = node_name

                save_workflow_state(
                    reference_number,
                    state,
                )

                logger.info(
                    "Workflow checkpoint %s "
                    "for %s",
                    node_name,
                    reference_number,
                )

        state["status"] = "complete"
        state["current_stage"] = (
            "complete"
        )

        save_workflow_state(
            reference_number,
            state,
        )

        return state

    except Exception as error:
        logger.exception(
            "Image workflow failed "
            "for %s",
            reference_number,
        )

        state["status"] = "failed"
        state["current_stage"] = (
            "failed"
        )
        state["error"] = str(
            error
        )

        save_workflow_state(
            reference_number,
            state,
        )

        raise