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


def _is_palette_failure(
    reason: str,
) -> bool:
    reason_lower = (
        reason.lower()
    )

    terms = (
        "monochrome",
        "black-and-white",
        "palette",
        "colour",
        "color",
        "primary ratio",
        "secondary ratio",
        "off-palette",
    )

    return any(
        term in reason_lower
        for term in terms
    )


def _is_human_failure(
    reason: str,
) -> bool:
    reason_lower = (
        reason.lower()
    )

    terms = (
        "human",
        "person",
        "people",
        "face",
        "figure",
        "portrait",
    )

    return any(
        term in reason_lower
        for term in terms
    )


def _is_text_failure(
    reason: str,
) -> bool:
    reason_lower = (
        reason.lower()
    )

    terms = (
        "ocr",
        "text",
        "readable",
        "letter",
        "word",
    )

    return any(
        term in reason_lower
        for term in terms
    )


def _standard_palette_instruction(
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
                "STRICT PALETTE RECOVERY. "
                f"Use {primary} as the dominant visible colour across "
                "most large surfaces. "
                f"Use {secondary} as a clearly visible supporting "
                "colour in several separate regions. "
                "Do not introduce unrelated hues. "
                "Neutral black, white or grey may appear only where "
                "needed for contrast."
            )

        return (
            "STRICT COLOUR CORRECTION. "
            f"The primary colour is {primary}. "
            f"The secondary colour is {secondary}. "
            f"Make {primary} visibly dominant and make {secondary} "
            "clearly visible as repeated supporting accents."
        )

    if primary:
        return (
            "STRICT COLOUR CORRECTION. "
            f"Make {primary} unmistakably visible and dominant."
        )

    return (
        "STRICT COLOUR CORRECTION. "
        "Obey the requested palette exactly and remove "
        "unrequested chromatic colours."
    )


def _sd35_palette_instruction(
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
            >= 6
        ):
            return (
                "SD3.5 EMERGENCY PALETTE RECOVERY. "
                "Create a simple abstract background with very large "
                "clean colour regions. "
                f"Approximately 60 percent of the visible chromatic "
                f"area should read clearly as {primary}. "
                f"Approximately 20 to 30 percent should read clearly "
                f"as {secondary}. "
                "The remaining area may use neutral black, white or grey "
                "only for contrast. "
                f"Use broad unmistakable {primary} fields. "
                f"Use broad unmistakable {secondary} bands, shapes, "
                "lines or accents in multiple places. "
                "Do not use coloured reflections, rainbow effects, "
                "multicolour gradients, amber light, cyan light, blue "
                "light, red light, pink light or unrelated hues unless "
                "they are explicitly one of the requested colours. "
                "Prefer simple geometric composition over decorative "
                "complexity. "
                "Colour compliance is more important than style detail."
            )

        if (
            retry_number
            >= 3
        ):
            return (
                "SD3.5 STRICT TWO-COLOUR RECOVERY. "
                f"Make {primary} cover most of the major visible "
                "surfaces and remain unmistakably dominant. "
                f"Make {secondary} clearly visible in multiple large "
                "supporting regions, not tiny accents. "
                "Reduce decorative complexity. "
                "Reduce coloured lighting. "
                "Avoid hue shifts caused by glow or reflections. "
                "Use neutral black, white or grey only where needed. "
                "No unrelated chromatic colours."
            )

        return (
            "SD3.5 PALETTE CORRECTION. "
            f"Use a strong {primary} dominant base and clearly visible "
            f"{secondary} supporting accents. "
            f"Both {primary} and {secondary} must appear as actual "
            "rendered surface colours rather than only lighting effects. "
            "Do not replace either colour with a neighbouring hue."
        )

    if primary:
        return (
            "SD3.5 PALETTE CORRECTION. "
            f"Use {primary} as a broad, unmistakable rendered surface "
            "colour across the image."
        )

    return (
        "SD3.5 PALETTE CORRECTION. "
        "Use only the requested chromatic palette and neutral tones."
    )


def _human_instruction(
    retry_number: int,
) -> tuple[
    str,
    str,
]:
    if (
        retry_number
        >= 4
    ):
        positive = (
            "STRICT HUMAN-FREE RECOVERY. "
            "Use only abstract geometry, environmental forms, "
            "patterns, architecture, materials and ornamental shapes. "
            "Do not arrange shapes into heads, faces, torsos, limbs, "
            "bodies, silhouettes, poses or character-like structures. "
            "Avoid a central subject."
        )

    else:
        positive = (
            "STRICT HUMAN-FREE CORRECTION. "
            "Background only. No person, human figure, silhouette, "
            "face, body, mannequin, clothing, character or "
            "human-shaped object."
        )

    negative = (
        "person, people, human, human figure, man, woman, child, "
        "face, portrait, silhouette, body, head, arms, hands, legs, "
        "clothing, mannequin, character, hero, superhero, humanoid, "
        "human-shaped form"
    )

    return (
        positive,
        negative,
    )


def _text_instruction() -> tuple[
    str,
    str,
]:
    return (
        (
            "STRICT TEXT-FREE CORRECTION. "
            "Do not render readable characters, pseudo-letters, words, "
            "numbers, logos, labels, signs, typography or writing."
        ),
        (
            "text, typography, words, letters, numbers, writing, "
            "logo, watermark, signage, labels, pseudo-text"
        ),
    )


def _sd35_minimal_recovery_prompt(
    primary_colour: str | None,
    secondary_colour: str | None,
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
        return (
            "Abstract A4 portrait decorative event background only. "
            f"Large simple {primary} colour fields dominating the "
            "composition. "
            f"Large clearly visible {secondary} supporting geometric "
            "bands, lines, gradients and shapes across several areas. "
            f"{primary} must remain the dominant chromatic colour. "
            f"{secondary} must remain clearly visible and measurable. "
            "Use black, white and grey only as neutral contrast. "
            "Strong tonal contrast. "
            "Simple geometric composition. "
            "Minimal decorative complexity. "
            "No unrelated colours. "
            "No coloured environmental lighting. "
            "No rainbow gradient. "
            "No people. "
            "No faces. "
            "No characters. "
            "No silhouettes. "
            "No text. "
            "No letters. "
            "No numbers. "
            "No logos. "
            "No watermark. "
            "Leave clean quiet regions for later programme overlays."
        )

    if primary:
        return (
            "Abstract A4 portrait decorative event background only. "
            f"Large simple {primary} colour fields dominating the "
            "composition. "
            "Use black, white and grey only as neutral contrast. "
            "Strong tonal contrast. "
            "Simple geometric composition. "
            "No unrelated chromatic colours. "
            "No people, faces, characters, text, letters, numbers, "
            "logos or watermark."
        )

    return (
        "Abstract A4 portrait decorative event background only. "
        "Simple geometric composition using only the requested palette. "
        "Strong tonal contrast. "
        "No people, faces, characters, text, letters, numbers, "
        "logos or watermark."
    )


def _strengthen_prompts(
    base_positive_prompt: str,
    base_negative_prompt: str,
    reason: str,
    primary_colour: str | None,
    secondary_colour: str | None,
    retry_number: int,
    engine_id: str,
) -> tuple[
    str,
    str,
]:
    positive_additions = []
    negative_additions = []

    palette_failure = (
        _is_palette_failure(
            reason
        )
    )

    human_failure = (
        _is_human_failure(
            reason
        )
    )

    text_failure = (
        _is_text_failure(
            reason
        )
    )

    if (
        engine_id
        == settings.engine_3_id
        and palette_failure
        and retry_number
        >= 6
    ):
        positive_prompt = (
            _sd35_minimal_recovery_prompt(
                primary_colour,
                secondary_colour,
            )
        )

    else:
        positive_prompt = (
            base_positive_prompt
            .rstrip(
                ". "
            )
        )

        if palette_failure:
            if (
                engine_id
                == settings.engine_3_id
            ):
                positive_additions.append(
                    _sd35_palette_instruction(
                        primary_colour,
                        secondary_colour,
                        retry_number,
                    )
                )

            else:
                positive_additions.append(
                    _standard_palette_instruction(
                        primary_colour,
                        secondary_colour,
                        retry_number,
                    )
                )

        if human_failure:
            (
                positive_human,
                negative_human,
            ) = (
                _human_instruction(
                    retry_number
                )
            )

            positive_additions.append(
                positive_human
            )

            negative_additions.append(
                negative_human
            )

        if text_failure:
            (
                positive_text,
                negative_text,
            ) = (
                _text_instruction()
            )

            positive_additions.append(
                positive_text
            )

            negative_additions.append(
                negative_text
            )

        if (
            "secondary"
            in reason.lower()
            and secondary_colour
        ):
            secondary = (
                _normalise_colour(
                    secondary_colour
                )
            )

            positive_additions.append(
                (
                    "MANDATORY SECONDARY COLOUR REQUIREMENT. "
                    f"Render {secondary} as actual visible surface colour "
                    "in several substantial regions. "
                    "Do not make it a tiny highlight. "
                    "Do not hide it inside lighting or reflections."
                )
            )

        if (
            "primary"
            in reason.lower()
            and primary_colour
        ):
            primary = (
                _normalise_colour(
                    primary_colour
                )
            )

            positive_additions.append(
                (
                    "MANDATORY PRIMARY COLOUR REQUIREMENT. "
                    f"Render {primary} as actual visible surface colour "
                    "across large portions of the composition."
                )
            )

        if not positive_additions:
            positive_additions.append(
                (
                    "STRICT RETRY CORRECTION. "
                    "Correct the previous QA violation while keeping "
                    "the requested background-only composition."
                )
            )

        positive_prompt = (
            positive_prompt
            + ". "
            + " ".join(
                positive_additions
            )
        )

    negative_prompt = (
        base_negative_prompt
        .strip()
    )

    if palette_failure:
        negative_additions.append(
            (
                "off-palette colours, colour drift, unrelated hues, "
                "multicolour lighting, rainbow gradient, coloured "
                "reflections, unrequested chromatic accents"
            )
        )

    if (
        engine_id
        == settings.engine_3_id
        and palette_failure
        and retry_number
        >= 3
    ):
        negative_additions.append(
            (
                "complex multicolour materials, coloured atmospheric "
                "lighting, cinematic colour grading, rainbow glow, "
                "iridescence, unrelated neon hues"
            )
        )

    corrected_negative = (
        ", ".join(
            value
            for value
            in (
                negative_prompt,
                *negative_additions,
            )
            if value
        )
    )

    return (
        positive_prompt,
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

                        base_positive_prompt = (
                            direction[
                                "positive_prompt"
                            ]
                        )

                        base_negative_prompt = (
                            direction.get(
                                "negative_prompt",
                                "",
                            )
                        )

                        positive_prompt = (
                            base_positive_prompt
                        )

                        negative_prompt = (
                            base_negative_prompt
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

                                next_retry = (
                                    retry_count
                                    + 1
                                )

                                (
                                    positive_prompt,
                                    negative_prompt,
                                ) = (
                                    _strengthen_prompts(
                                        base_positive_prompt,
                                        base_negative_prompt,
                                        reason,
                                        primary_colour,
                                        secondary_colour,
                                        next_retry,
                                        output.engine_id,
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
                                    next_retry,
                                    settings.max_candidate_retries,
                                    delay_seconds,
                                    reason,
                                )

                                if (
                                    output.engine_id
                                    == settings.engine_3_id
                                    and _is_palette_failure(
                                        reason
                                    )
                                ):
                                    logger.info(
                                        "event=sd35_palette_recovery "
                                        "reference=%s "
                                        "direction=%s "
                                        "retry=%s "
                                        "mode=%s",
                                        reference_number,
                                        output.direction_id,
                                        next_retry,
                                        (
                                            "minimal"
                                            if next_retry >= 6
                                            else (
                                                "strict"
                                                if next_retry >= 3
                                                else "normal"
                                            )
                                        ),
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