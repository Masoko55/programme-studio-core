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
    record = (
        _candidate_record(
            reference_number,
            engine_id,
            direction_id,
        )
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
    record = (
        _candidate_record(
            reference_number,
            engine_id,
            direction_id,
        )
    )

    return str(
        record.get(
            "rejection_reason"
        )
        or ""
    )


def _normalise_colour(
    value: str | None,
) -> str:
    return (
        str(
            value
            or ""
        )
        .strip()
        .lower()
    )


def _palette_instruction(
    primary_colour: str | None,
    secondary_colour: str | None,
    retry_number: int,
) -> str:
    primary = (
        _normalise_colour(
            primary_colour
        )
    )

    secondary = (
        _normalise_colour(
            secondary_colour
        )
    )

    if (
        primary
        and secondary
    ):
        if (
            retry_number
            >= 4
        ):
            return (
                "STRICT RETRY COLOUR CORRECTION: "
                f"use an unmistakable {primary} base covering most "
                "of the composition and clearly visible "
                f"{secondary} decorative accents across multiple "
                "regions of the image. "
                f"Both {primary} and {secondary} must be visibly "
                "present in the rendered pixels. "
                f"{primary} must remain dominant. "
                f"{secondary} must remain a supporting colour. "
                "Do not replace either requested colour with "
                "neutral grey or another hue."
            )

        return (
            "STRICT RETRY COLOUR CORRECTION: "
            f"the requested primary colour is {primary} and the "
            f"requested secondary colour is {secondary}. "
            f"Make {primary} clearly dominant and ensure "
            f"{secondary} is visibly present as repeated supporting "
            "accents. Preserve both requested colours."
        )

    if primary:
        return (
            "STRICT RETRY COLOUR CORRECTION: "
            f"the requested colour is {primary}. "
            f"Make {primary} unmistakably visible throughout "
            "the composition."
        )

    return (
        "STRICT RETRY COLOUR CORRECTION: "
        "obey the requested palette exactly and do not introduce "
        "unrequested chromatic colours."
    )


def _strengthen_prompts(
    positive_prompt: str,
    negative_prompt: str,
    reason: str,
    primary_colour: str | None,
    secondary_colour: str | None,
    retry_number: int,
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
        or "face"
        in reason_lower
        or "figure"
        in reason_lower
    ):
        if (
            retry_number
            >= 4
        ):
            positive_additions.append(
                (
                    "STRICT RETRY HUMAN-FREE CORRECTION: "
                    "remove every human-like visual structure. "
                    "Use only environmental, architectural, abstract, "
                    "geometric, ornamental or material forms. "
                    "Do not arrange shapes into a head, torso, limbs, "
                    "face, body, pose, person or character. "
                    "Avoid central figure-like compositions."
                )
            )

        else:
            positive_additions.append(
                (
                    "STRICT RETRY HUMAN-FREE CORRECTION: "
                    "background only with no person, human figure, "
                    "silhouette, face, body, mannequin, clothing, "
                    "character or human-shaped object."
                )
            )

        negative_additions.append(
            (
                "person, people, human, human figure, man, woman, "
                "child, face, portrait, silhouette, body, head, "
                "arms, hands, legs, clothing, mannequin, character, "
                "human-shaped form, humanoid"
            )
        )

    colour_failure = (
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
        or "primary ratio"
        in reason_lower
        or "secondary ratio"
        in reason_lower
    )

    if colour_failure:
        positive_additions.append(
            _palette_instruction(
                primary_colour,
                secondary_colour,
                retry_number,
            )
        )

        negative_additions.append(
            (
                "off-palette colours, colour drift, "
                "unrequested chromatic accents"
            )
        )

    if (
        "secondary"
        in reason_lower
        and secondary_colour
    ):
        secondary = (
            _normalise_colour(
                secondary_colour
            )
        )

        positive_additions.append(
            (
                "MANDATORY SECONDARY COLOUR CORRECTION: "
                f"the colour {secondary} must be clearly visible "
                "in several supporting accents rather than being "
                "absent, imperceptible or replaced by neutral tones."
            )
        )

    if (
        "primary"
        in reason_lower
        and primary_colour
    ):
        primary = (
            _normalise_colour(
                primary_colour
            )
        )

        positive_additions.append(
            (
                "MANDATORY PRIMARY COLOUR CORRECTION: "
                f"the colour {primary} must visibly dominate "
                "the composition."
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
                "STRICT RETRY TEXT-FREE CORRECTION: "
                "no readable characters, pseudo-letters, words, "
                "numbers, logos, labels or signs of any kind."
            )
        )

        negative_additions.append(
            (
                "text, typography, words, letters, numbers, "
                "writing, logo, watermark, signage, labels"
            )
        )

    if not (
        positive_additions
    ):
        positive_additions.append(
            (
                "STRICT RETRY CORRECTION: correct the previous "
                "QA violation while preserving the requested "
                "background design and colour requirements."
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
            for value
            in (
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
        state = (
            load_image_job_state(
                reference_number
            )
        )

        document = (
            load_prompts_document(
                reference_number
            )
        )

        brief = (
            document.get(
                "brief",
                {},
            )
        )

        primary_colour = (
            brief.get(
                "primary_colour"
            )
        )

        secondary_colour = (
            brief.get(
                "secondary_colour"
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
            in (
                build_execution_plan(
                    document
                )[
                    "execution_order"
                ]
            )
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
                    response
                    .json()
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
                                            entry[
                                                3
                                            ]
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
                                ) = (
                                    _strengthen_prompts(
                                        positive_prompt,
                                        negative_prompt,
                                        reason,
                                        primary_colour,
                                        secondary_colour,
                                        retry_count
                                        + 1,
                                    )
                                )

                                delay_seconds = min(
                                    2
                                    ** retry_count,
                                    settings
                                    .max_candidate_retry_delay_seconds,
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