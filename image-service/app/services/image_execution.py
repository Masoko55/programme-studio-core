"""Execute background-candidate generation.

Every engine/direction pair is independent.

QA-rejected candidates are retried with:
- a new seed
- stronger correction instructions based on the actual rejection reason

If a candidate exhausts its retry budget, only that candidate fails.
The remaining candidates continue generating.

Transport, queue, workflow and ambiguous submission failures remain hard
failures because blindly resubmitting them could duplicate remote jobs.
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
    path = (
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
            path.read_text(
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


def _rejection_reason(
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
    reason: str,
) -> tuple[
    str,
    str,
]:
    reason_lower = (
        reason.lower()
    )

    positive_additions = []
    negative_additions = []

    if (
        "human"
        in reason_lower
        or "person"
        in reason_lower
        or "people"
        in reason_lower
    ):
        positive_additions.append(
            (
                "STRICT RETRY CORRECTION: "
                "pure abstract nonrepresentational "
                "background only. No human figure, "
                "no silhouette, no face, no body, "
                "no mannequin, no clothing, no "
                "human-shaped object and no central character."
            )
        )

        negative_additions.append(
            (
                "person, people, human, man, woman, "
                "child, face, portrait, silhouette, "
                "body, head, arms, hands, legs, "
                "clothing, mannequin, character"
            )
        )

    if (
        "monochrome"
        in reason_lower
        or "black-and-white"
        in reason_lower
        or "palette"
        in reason_lower
        or "colour"
        in reason_lower
        or "color"
        in reason_lower
    ):
        positive_additions.append(
            (
                "STRICT RETRY CORRECTION: obey the requested "
                "colour palette exactly. If the requested "
                "palette is black and white, generate ONLY "
                "achromatic black, white and neutral gray. "
                "No coloured tinting, warm cast, cool cast, "
                "sepia or chromatic lighting."
            )
        )

        negative_additions.append(
            (
                "blue, cyan, green, yellow, beige, brown, "
                "gold, orange, red, pink, purple, "
                "colour cast, colored tint, coloured tint"
            )
        )

    if (
        "ocr"
        in reason_lower
        or "text"
        in reason_lower
        or "readable"
        in reason_lower
    ):
        positive_additions.append(
            (
                "STRICT RETRY CORRECTION: no readable "
                "characters of any kind. Pure visual "
                "background only."
            )
        )

        negative_additions.append(
            (
                "text, typography, words, letters, "
                "numbers, writing, logo, watermark, signage"
            )
        )

    if not (
        positive_additions
    ):
        positive_additions.append(
            (
                "STRICT RETRY CORRECTION: correct the "
                "previous QA violation while preserving "
                "the requested abstract background."
            )
        )

    corrected_positive = (
        positive_prompt.rstrip(
            ". "
        )
        + ". "
        + " ".join(
            positive_additions
        )
    )

    corrected_negative = (
        ", ".join(
            value
            for value in (
                negative_prompt.strip(),
                *negative_additions,
            )
            if value
        )
    )

    return (
        corrected_positive,
        corrected_negative,
    )


async def execute_image_job(
    reference_number: str,
    max_outputs: int | None = None,
):
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
                "Historical engine plan differs from the "
                "current configuration. Preserve this job "
                "and create a new reference number."
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
                generated = 0

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

                        candidate_complete = False

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

                                if not (
                                    result[
                                        "reused"
                                    ]
                                ):
                                    generated += 1

                                candidate_complete = True

                                break

                            except ComfyUIError as error:
                                rejected = (
                                    _candidate_was_rejected(
                                        reference_number,
                                        output.engine_id,
                                        output.direction_id,
                                    )
                                )

                                # Do not blindly retry ambiguous
                                # infrastructure failures.
                                if not rejected:
                                    raise

                                reason = (
                                    _rejection_reason(
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
                                        error=reason,
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
                                        reason,
                                    )

                                    break

                                (
                                    positive_prompt,
                                    negative_prompt,
                                ) = _strengthen_prompts(
                                    positive_prompt,
                                    negative_prompt,
                                    reason,
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
                                    settings.max_candidate_retries,
                                    delay_seconds,
                                    reason,
                                )

                                await asyncio.sleep(
                                    delay_seconds
                                )

                        if (
                            max_outputs
                            is not None
                            and generated
                            >= max_outputs
                        ):
                            break

                        if not (
                            candidate_complete
                        ):
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