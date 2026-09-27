import asyncio
import json
import logging
from pathlib import Path
import httpx
from app.config.settings import settings
from app.engines.registry import get_engine
from app.services.comfyui_client import ComfyUIClient, ComfyUIError
from app.services.execution_plan import build_execution_plan
from app.services.gpu_lease import GPULease
from app.services.job_persistence import load_image_job_state, persist_image_job_state, update_output
from app.services.prompt_repository import get_direction, load_prompts_document

logger = logging.getLogger("uvicorn.error")


def _candidate_was_rejected(reference_number: str, engine_id: str, direction_id: str) -> bool:
    """Only retry a completed generation rejected by local background QA.

    Transport, validation, and submission failures can be ambiguous.  They must
    remain resumable failures rather than being resubmitted automatically.
    """
    record_path = (
        settings.programme_data_path
        / reference_number
        / "backgrounds"
        / engine_id
        / f"image-{direction_id.lower()}.json"
    )
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return record.get("status") == "rejected" and bool(record.get("rejection_reason"))



async def execute_image_job(
    reference_number: str,
    max_outputs: int | None = None,
):
    """Generate background candidates only.

    The nine outputs produced here are background candidates. No programme
    title, programme rows, event details, headshot, or logo are composed at
    this stage.

    Once all candidates are generated the job enters awaiting_selection.
    """

    async with GPULease():
        state = load_image_job_state(reference_number)
        document = load_prompts_document(reference_number)

        expected = [
            (
                step["engine_id"],
                step["direction"],
            )
            for step in build_execution_plan(document)["execution_order"]
        ]

        actual = [
            (
                output.engine_id,
                output.direction_id,
            )
            for output in state.outputs
        ]

        if actual != expected:
            raise ValueError(
                "Historical engine plan differs from the current "
                "configuration. Preserve this job and create a new "
                "reference number."
            )

        state.status = "processing"
        state.current_stage = "generating_backgrounds"
        state.error = None

        persist_image_job_state(state)

        try:
            async with httpx.AsyncClient(timeout=60) as http:
                response = await http.get(
                    settings.ollama_base_url + "/api/ps"
                )

                response.raise_for_status()

                if response.json().get("models"):
                    raise RuntimeError(
                        "Ollama has resident models; unload them "
                        "before image generation."
                    )

            async with ComfyUIClient() as client:
                current_engine = None
                generated = 0

                try:
                    for output in state.outputs:
                        if current_engine != output.engine_id:
                            queue = (
                                await client.request(
                                    "GET",
                                    "/queue",
                                )
                            ).json()

                            entries = (
                                queue.get("queue_running", [])
                                + queue.get("queue_pending", [])
                            )

                            own_path = (
                                settings.programme_data_path
                                / reference_number
                                / "backgrounds"
                                / output.engine_id
                                / (
                                    "image-"
                                    f"{output.direction_id.lower()}"
                                    ".json"
                                )
                            )

                            own = (
                                json.loads(
                                    own_path.read_text(
                                        encoding="utf-8"
                                    )
                                )
                                if own_path.exists()
                                else {}
                            )

                            if entries:
                                if not own or any(
                                    entry[3].get(
                                        "programme_request_id"
                                    )
                                    != own.get("request_id")
                                    for entry in entries
                                ):
                                    raise RuntimeError(
                                        "ComfyUI has another active "
                                        "job; retry later."
                                    )
                            else:
                                await client.release_models()

                            current_engine = output.engine_id

                        state.current_stage = (
                            f"{output.engine_id}:"
                            f"{output.direction_id}"
                        )

                        persist_image_job_state(state)

                        direction = get_direction(
                            document,
                            output.direction_id,
                        )

                        try:
                            for retry_count in range(
                                settings.max_candidate_retries + 1
                            ):
                                try:
                                    result = (
                                        await get_engine(
                                            output.engine_id
                                        ).generate(
                                            reference_number=(
                                                reference_number
                                            ),
                                            direction_id=(
                                                output.direction_id
                                            ),
                                            positive_prompt=(
                                                direction[
                                                    "positive_prompt"
                                                ]
                                            ),
                                            negative_prompt=(
                                                direction.get(
                                                    "negative_prompt",
                                                    "",
                                                )
                                            ),
                                        )
                                    )

                                    break

                                except ComfyUIError as error:
                                    if (
                                        retry_count
                                        >= settings.max_candidate_retries
                                        or not _candidate_was_rejected(
                                            reference_number,
                                            output.engine_id,
                                            output.direction_id,
                                        )
                                    ):
                                        raise

                                    delay_seconds = 2 ** retry_count

                                    logger.warning(
                                        "event=candidate_retry "
                                        "reference=%s "
                                        "engine=%s "
                                        "direction=%s "
                                        "retry=%s/%s "
                                        "delay_seconds=%s "
                                        "reason=%s",
                                        reference_number,
                                        output.engine_id,
                                        output.direction_id,
                                        retry_count + 1,
                                        settings.max_candidate_retries,
                                        delay_seconds,
                                        error,
                                    )

                                    await asyncio.sleep(
                                        delay_seconds
                                    )

                            output.seed = result["seed"]
                            output.prompt_id = result["prompt_id"]
                            output.workflow_sha256 = (
                                result["workflow_sha256"]
                            )

                            update_output(
                                state,
                                output.engine_id,
                                output.direction_id,
                                "complete",
                                result["output_path"],
                                result["sha256"],
                            )

                            logger.info(
                                "event=candidate_complete "
                                "reference=%s "
                                "engine=%s "
                                "direction=%s "
                                "reused=%s",
                                reference_number,
                                output.engine_id,
                                output.direction_id,
                                result["reused"],
                            )

                            if not result["reused"]:
                                generated += 1

                            if (
                                max_outputs is not None
                                and generated >= max_outputs
                            ):
                                break

                        except Exception as error:
                            update_output(
                                state,
                                output.engine_id,
                                output.direction_id,
                                "failed",
                                error=str(error),
                            )

                            raise

                finally:
                    queue = (
                        await client.request(
                            "GET",
                            "/queue",
                        )
                    ).json()

                    if (
                        not queue.get("queue_running")
                        and not queue.get("queue_pending")
                    ):
                        await client.release_models()

            if state.completed_outputs == state.total_outputs:
                state.status = "awaiting_selection"
                state.current_stage = "backgrounds_complete"
            else:
                state.status = "paused"
                state.current_stage = "generation_checkpoint"

            persist_image_job_state(state)

            return state

        except Exception as error:
            state.status = "failed"
            state.current_stage = "generation_failed"
            state.error = str(error)

            persist_image_job_state(state)

            raise