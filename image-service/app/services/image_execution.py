"""Execute background-candidate generation.

Each engine/direction pair is an independent candidate.

A candidate rejected by deterministic QA is retried with a new seed and
stronger corrective prompting. If that candidate still fails after its retry
budget is exhausted, it is marked failed and the remaining candidates continue.

Infrastructure/transport failures remain hard failures because blindly
resubmitting ambiguous remote jobs could create duplicates.
"""

import asyncio
import json
import logging

import httpx

from app.config.settings import (
    settings,
)
from app.engines.registry import (
    get_engine,
)
from app.services.comfyui_client import (
    ComfyUIClient,
    ComfyUIError,
)
from app.services.execution_plan import (
    build_execution_plan,
)
from app.services.gpu_lease import (
    GPULease,
)
from app.services.job_persistence import (
    load_image_job_state,
    persist_image_job_state,
    update_output,
)
from app.services.prompt_repository import (
    get_direction,
    load_prompts_document,
)


logger = logging.getLogger(
    "uvicorn.error"
)


def _candidate_record(
    reference_number: str,
    engine_id: str,
    direction_id: str,
) -> dict:
    record_path = (
        settings.programme_data_path
        / reference_number
        / "backgrounds"
        / engine_id
        / (
            "image-"
            f"{direction_id.lower()}"
            ".json"
        )
    )

    try:
        return json.loads(
            record_path.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {}


def _candidate_was_rejected(
    reference_number: str,
    engine_id: str,
    direction_id: str,
) -> bool:
    record = _candidate_record(
        reference_number,
        engine_id,
        direction_id,
    )

    return (
        record.get(
            "status"
        )
        == "rejected"
        and bool(
            record.get(
                "rejection_reason"
            )
        )
    )


def _candidate_rejection_reason(
    reference_number: str,
    engine_id: str,
    direction_id: str,
) -> str:
    record = _candidate_record(
        reference_number,
        engine_id,
        direction_id,
    )

    return str(
        record.get(
            "rejection_reason"
        )
        or ""
    )


def _strengthen_prompts(
    positive_prompt: str,
    negative_prompt: str,
    rejection_reason: str,
) -> tuple[
    str,
    str,
]:
    """Correct the next retry based on deterministic QA feedback."""

    reason = (
        rejection_reason
        .lower()
    )

    positive_extra = []
    negative_extra = []

    if (
        "human figure"
        in reason
        or "person"
        in reason
        or "people"
        in reason
    ):
        positive_extra.append(
            (
                "STRICT RETRY REQUIREMENT: "
                "create pure abstract nonrepresentational "
                "surface design only. No central subject, "
                "no silhouette, no body-shaped form, no face, "
                "no human-like object, no clothing, no mannequin. "
                "Use only geometry, material texture, lines, "
                "light, gradients and ornament."
            )
        )

        negative_extra.append(
            (
                "person, people, human, human figure, "
                "man, woman, child, face, portrait, "
                "silhouette, body, head, hands, arms, "
                "legs, clothing, mannequin, character"
            )
        )

    if (
        "palette"
        in reason
        or "monochrome"
        in reason
        or "colour"
        in reason
        or "color"
        in reason
    ):
        positive_extra.append(
            (
                "STRICT RETRY REQUIREMENT: remain inside "
                "the requested colour palette only. "
                "For a black-and-white brief use ONLY "
                "achromatic black, white and neutral gray. "
                "Zero coloured tinting: no blue, cyan, green, "
                "yellow, beige, brown, gold, orange, red, pink, "
                "purple or other chromatic hues."
            )
        )

        negative_extra.append(
            (
                "colour cast, colored tint, coloured tint, "
                "blue, cyan, green, yellow, beige, brown, "
                "gold, orange, red, pink, purple"
            )
        )

    if (
        "ocr"
        in reason
        or "text"
        in reason
        or "readable"
        in reason
    ):
        positive_extra.append(
            (
                "STRICT RETRY REQUIREMENT: decorative visual "
                "background only. No symbols resembling letters, "
                "numbers or written language."
            )
        )

        negative_extra.append(
            (
                "text, letters, words, numbers, typography, "
                "writing, logo, watermark, signage"
            )
        )

    if not positive_extra:
        positive_extra.append(
            (
                "STRICT RETRY REQUIREMENT: correct the "
                "previous QA violation while preserving "
                "the requested abstract event background."
            )
        )

    strengthened_positive = (
        positive_prompt.rstrip(
            ". "
        )
        + ". "
        + " ".join(
            positive_extra
        )
    )

    strengthened_negative = (
        ", ".join(
            [
                value
                for value in (
                    negative_prompt.strip(),
                    *negative_extra,
                )
                if value
            ]
        )
    )

    return (
        strengthened_positive,
        strengthened_negative,
    )


async def execute_image_job(
    reference_number: str,
    max_outputs: int | None = None,
):
    """Generate all selectable background candidates.

    Successful existing candidates are reused.

    QA rejection:
        retry with stronger corrective prompting.

    Exhausted QA rejection:
        mark only that candidate failed and continue.

    Infrastructure/transport ambiguity:
        abort the job so an operator can safely resume it.
    """

    async with GPULease():
        state = load_image_job_state(
            reference_number
        )

        document = (
            load_prompts_document(
                reference_number
            )
        )

        expected = [
            (
                step[
                    "engine_id"
                ],
                step[
                    "direction"
                ],
            )
            for step
            in build_execution_plan(
                document
            )[
                "execution_order"
            ]
        ]

        actual = [
            (
                output.engine_id,
                output.direction_id,
            )
            for output
            in state.outputs
        ]

        if actual != expected:
            raise ValueError(
                "Historical engine plan differs from "
                "the current configuration. Preserve "
                "this job and create a new reference."
            )

        state.status = (
            "processing"
        )

        state.current_stage = (
            "generating_backgrounds"
        )

        state.error = None

        persist_image_job_state(
            state
        )

        try:
            # Ollama must release GPU memory before
            # ComfyUI image generation.
            async with httpx.AsyncClient(
                timeout=60
            ) as http:
                response = (
                    await http.get(
                        settings.ollama_base_url
                        + "/api/ps"
                    )
                )

                response.raise_for_status()

                if (
                    response.json()
                    .get(
                        "models"
                    )
                ):
                    raise RuntimeError(
                        "Ollama has resident models; "
                        "unload them before image generation."
                    )

            async with ComfyUIClient() as client:
                current_engine = None
                newly_generated = 0

                try:
                    for output in state.outputs:
                        if (
                            current_engine
                            != output.engine_id
                        ):
                            queue = (
                                await client.request(
                                    "GET",
                                    "/queue",
                                )
                            ).json()

                            entries = (
                                queue.get(
                                    "queue_running",
                                    [],
                                )
                                + queue.get(
                                    "queue_pending",
                                    [],
                                )
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
                                if (
                                    not own
                                    or any(
                                        (
                                            entry[3]
                                            .get(
                                                "programme_request_id"
                                            )
                                            != own.get(
                                                "request_id"
                                            )
                                        )
                                        for entry
                                        in entries
                                    )
                                ):
                                    raise RuntimeError(
                                        "ComfyUI has another active "
                                        "job; retry later."
                                    )

                            else:
                                await (
                                    client
                                    .release_models()
                                )

                            current_engine = (
                                output.engine_id
                            )

                        state.current_stage = (
                            f"{output.engine_id}:"
                            f"{output.direction_id}"
                        )

                        persist_image_job_state(
                            state
                        )

                        direction = (
                            get_direction(
                                document,
                                output.direction_id,
                            )
                        )

                        positive_prompt = (
                            direction[
                                "positive_prompt"
                            ]
                        )

                        negative_prompt = (
                            direction.get(
                                "negative_prompt",
                                "",
                            )
                        )

                        candidate_completed = False

                        for retry_count in range(
                            settings.max_candidate_retries
                            + 1
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
                                            positive_prompt
                                        ),
                                        negative_prompt=(
                                            negative_prompt
                                        ),
                                    )
                                )

                                output.seed = (
                                    result[
                                        "seed"
                                    ]
                                )

                                output.prompt_id = (
                                    result[
                                        "prompt_id"
                                    ]
                                )

                                output.workflow_sha256 = (
                                    result[
                                        "workflow_sha256"
                                    ]
                                )

                                update_output(
                                    state,
                                    output.engine_id,
                                    output.direction_id,
                                    "complete",
                                    result[
                                        "output_path"
                                    ],
                                    result[
                                        "sha256"
                                    ],
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
                                    result[
                                        "reused"
                                    ],
                                )

                                if not result[
                                    "reused"
                                ]:
                                    newly_generated += 1

                                candidate_completed = True

                                break

                            except ComfyUIError as error:
                                is_qa_rejection = (
                                    _candidate_was_rejected(
                                        reference_number,
                                        output.engine_id,
                                        output.direction_id,
                                    )
                                )

                                # Transport, workflow, queue and submission
                                # errors are intentionally NOT blindly retried.
                                if not is_qa_rejection:
                                    raise

                                rejection_reason = (
                                    _candidate_rejection_reason(
                                        reference_number,
                                        output.engine_id,
                                        output.direction_id,
                                    )
                                    or str(
                                        error
                                    )
                                )

                                if (
                                    retry_count
                                    >= settings.max_candidate_retries
                                ):
                                    update_output(
                                        state,
                                        output.engine_id,
                                        output.direction_id,
                                        "failed",
                                        error=(
                                            rejection_reason
                                        ),
                                    )

                                    logger.error(
                                        "event=candidate_exhausted "
                                        "reference=%s "
                                        "engine=%s "
                                        "direction=%s "
                                        "attempts=%s "
                                        "reason=%s",
                                        reference_number,
                                        output.engine_id,
                                        output.direction_id,
                                        (
                                            settings
                                            .max_candidate_retries
                                            + 1
                                        ),
                                        rejection_reason,
                                    )

                                    # Important:
                                    # do NOT abort the remaining candidates.
                                    break

                                (
                                    positive_prompt,
                                    negative_prompt,
                                ) = _strengthen_prompts(
                                    positive_prompt,
                                    negative_prompt,
                                    rejection_reason,
                                )

                                delay_seconds = (
                                    2
                                    ** retry_count
                                )

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
                                    retry_count
                                    + 1,
                                    (
                                        settings
                                        .max_candidate_retries
                                    ),
                                    delay_seconds,
                                    rejection_reason,
                                )

                                await asyncio.sleep(
                                    delay_seconds
                                )

                        if (
                            max_outputs
                            is not None
                            and newly_generated
                            >= max_outputs
                        ):
                            break

                        # Explicitly continue after an exhausted
                        # candidate rather than aborting the batch.
                        if not candidate_completed:
                            continue

                finally:
                    queue = (
                        await client.request(
                            "GET",
                            "/queue",
                        )
                    ).json()

                    if (
                        not queue.get(
                            "queue_running"
                        )
                        and not queue.get(
                            "queue_pending"
                        )
                    ):
                        await (
                            client
                            .release_models()
                        )

            completed = (
                state.completed_outputs
            )

            failed = sum(
                1
                for output
                in state.outputs
                if (
                    output.status
                    == "failed"
                )
            )

            pending = sum(
                1
                for output
                in state.outputs
                if (
                    output.status
                    == "pending"
                )
            )

            # At least one selectable background exists.
            if completed > 0:
                state.status = (
                    "awaiting_selection"
                )

                if (
                    completed
                    == state.total_outputs
                ):
                    state.current_stage = (
                        "backgrounds_complete"
                    )

                    state.error = None

                else:
                    state.current_stage = (
                        "backgrounds_complete_with_failures"
                    )

                    state.error = (
                        f"{completed}/"
                        f"{state.total_outputs} "
                        "background candidates completed; "
                        f"{failed} failed; "
                        f"{pending} remain pending."
                    )

            else:
                state.status = (
                    "failed"
                )

                state.current_stage = (
                    "generation_failed"
                )

                state.error = (
                    "No valid background candidates "
                    "were produced."
                )

            persist_image_job_state(
                state
            )

            return state

        except Exception as error:
            state.status = (
                "failed"
            )

            state.current_stage = (
                "generation_failed"
            )

            state.error = str(
                error
            )

            persist_image_job_state(
                state
            )

            raise