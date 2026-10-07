from __future__ import annotations

import asyncio
import json
import logging

import httpx

from app.config.settings import settings

from app.engines.registry import (
    get_engine,
)

from app.services.comfyui_client import (
    ComfyUIClient,
    ComfyUIError,
    SubmissionUncertain,
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
            f"image-"
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

    return bool(
        record.get(
            "rejection_reason"
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
        or record.get(
            "error"
        )
        or ""
    )


def _normalise_colour(
    value: str | None,
) -> str:
    return (
        str(
            value or ""
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
        "unrequested white",
        "unrequested colour",
        "unrequested color",
        "colours that were not requested",
        "colors that were not requested",
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
        "body",
        "silhouette",
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
        "readable text",
        "contains text",
        "letter",
        "word",
        "number",
        "logo",
    )

    return any(
        term in reason_lower
        for term in terms
    )


def _is_quality_failure(
    reason: str,
) -> bool:
    reason_lower = (
        reason.lower()
    )

    terms = (
        "visual quality",
        "scanline",
        "scan line",
        "raster",
        "banding",
        "moire",
        "glitch",
        "corrupt",
        "degenerate",
        "blank",
        "flat colour",
        "flat color",
        "flat or blank",
        "insufficient visual structure",
        "central title/programme region",
        "too visually dense",
    )

    return any(
        term in reason_lower
        for term in terms
    )


def _is_runtime_failure(
    reason: str,
) -> bool:
    reason_lower = (
        reason.lower()
    )

    terms = (
        "comfyui execution failed",
        "execution_error",
        "execution interrupted",
        "execution_interrupted",
        "generation timed out",
        "cuda",
        "out of memory",
        "allocation",
        "expected exactly one persisted",
        "comfyui rejected the workflow",
    )

    return any(
        term in reason_lower
        for term in terms
    )


def _retry_strength_instruction(
    effective_retry: int,
) -> str:
    if effective_retry >= 16:
        return (
            "MAXIMUM COMPLIANCE RETRY. "
            "Many previous generations failed deterministic QA. "
            "Use the simplest coherent professional event background "
            "that satisfies every requirement literally. "
            "Remove unnecessary materials, reflections, complicated "
            "lighting, texture noise and ambiguous decorative effects."
        )

    if effective_retry >= 13:
        return (
            "EXTREME RECOVERY RETRY. "
            "Previous first-wave and recovery candidates still failed QA. "
            "Simplify the composition aggressively. Use large coherent "
            "forms, restrained detail, clean surfaces and literal "
            "constraint compliance."
        )

    if effective_retry >= 10:
        return (
            "SECOND-WAVE STRICT RECOVERY. "
            "The first complete generation wave exhausted all attempts. "
            "Reduce creative complexity substantially and obey every "
            "palette, human-free, text-free and visual-quality requirement "
            "literally."
        )

    if effective_retry >= 7:
        return (
            "VERY STRICT RETRY. "
            "Previous candidates repeatedly failed QA. "
            "Use cleaner larger forms, fewer visual effects, simpler "
            "materials and stronger literal compliance."
        )

    if effective_retry >= 4:
        return (
            "STRICT RETRY. "
            "Strengthen compliance beyond the previous attempt. "
            "Reduce unnecessary complexity and make every required "
            "constraint obvious in the rendered image."
        )

    if effective_retry >= 2:
        return (
            "ENHANCED RETRY. "
            "The previous candidate failed QA. "
            "Apply the required correction substantially more strongly."
        )

    return (
        "RETRY CORRECTION. "
        "The previous candidate failed deterministic QA. "
        "Correct the identified violation explicitly."
    )


def _initial_engine_prompts(
    engine_id: str,
    base_positive_prompt: str,
    base_negative_prompt: str,
    primary_colour: str | None,
    secondary_colour: str | None,
) -> tuple[
    str,
    str,
]:
    """
    Engine-specific attempt-zero prompt adaptation.
    """

    positive = (
        base_positive_prompt
    )

    negative = (
        base_negative_prompt
    )

    #
    # FLUX already performs well. Do not unnecessarily
    # disturb its initial creative prompt.
    #
    if (
        engine_id
        != settings.engine_2_id
    ):
        return (
            positive,
            negative,
        )

    #
    # SDXL has been repeatedly drifting toward white,
    # silver, cream and grey. Start strict immediately.
    #
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
        positive += (
            ". SDXL STRICT TWO-COLOUR GRAPHIC MODE. "
            f"Render using only the {primary} and {secondary} "
            "colour families. "
            f"Use {primary} across most large visible surfaces. "
            f"Use {secondary} across several substantial supporting "
            "regions. Use matte graphic illustration, clean geometric "
            "forms and flat or gently tonal surfaces. "
            "Create depth exclusively using darker or lighter tonal "
            "variants inside those two requested colour families. "
            "Use colour-area contrast rather than neutral highlights. "
            "No third colour family."
        )

    elif primary:
        positive += (
            ". SDXL STRICT SINGLE-PALETTE GRAPHIC MODE. "
            f"Use only tonal variations belonging to {primary}. "
            "Use clean matte graphic surfaces and geometric forms."
        )

    else:
        positive += (
            ". SDXL STRICT GRAPHIC MODE. "
            "Use only the explicitly requested colour palette."
        )

    requested = {
        value
        for value
        in (
            primary,
            secondary,
        )
        if value
    }

    forbidden_neutrals = [
        colour
        for colour
        in (
            "white",
            "off-white",
            "ivory",
            "cream",
            "beige",
            "silver",
            "grey",
            "gray",
            "charcoal",
            "black",
        )
        if colour
        not in requested
    ]

    if forbidden_neutrals:
        negative += (
            ", "
            + ", ".join(
                forbidden_neutrals
            )
            + ", neutral highlight, neutral shadow, "
            + "metallic highlight, metallic reflection, "
            + "colourless reflection, studio lighting, "
            + "white rim light, white glow, "
            + "silver surface, grey surface, "
            + "cream surface, beige surface"
        )

    return (
        positive,
        negative,
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
        if retry_number >= 7:
            return (
                "EXTREME PALETTE ENFORCEMENT. "
                f"Use only {primary} and {secondary}. "
                f"Make {primary} occupy most major surfaces. "
                f"Make {secondary} occupy clearly visible supporting "
                "regions in several areas. "
                "Every neutral or chromatic colour family that was not "
                "requested is forbidden. "
                "All shadows and highlights must remain tonal variations "
                "inside the requested colour families."
            )

        if retry_number >= 4:
            return (
                "STRICT PALETTE RECOVERY. "
                f"Use {primary} as the dominant visible colour across "
                "most large surfaces. "
                f"Use {secondary} as a substantial supporting colour. "
                "Do not introduce any third colour family. "
                "Do not use unrequested neutral highlights or shadows."
            )

        return (
            "STRICT COLOUR CORRECTION. "
            f"The primary colour is {primary}. "
            f"The secondary colour is {secondary}. "
            f"Make {primary} visibly dominant and {secondary} "
            "clearly visible in repeated supporting regions. "
            "Every additional colour family is forbidden."
        )

    if primary:
        return (
            "STRICT COLOUR CORRECTION. "
            f"Use {primary} as the only requested visible "
            "colour family."
        )

    return (
        "STRICT COLOUR CORRECTION. "
        "Obey the requested palette exactly."
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
        if retry_number >= 6:
            return (
                "SD3.5 EMERGENCY PALETTE RECOVERY. "
                "Create a simple coherent event background with large "
                "clean colour regions. "
                f"Approximately 65 percent should read as {primary}. "
                f"Approximately 25 to 35 percent should read as {secondary}. "
                f"Use only {primary} and {secondary}. "
                "No third colour family. No unrequested neutral fills. "
                "Avoid colour-changing glow, reflections and iridescence."
            )

        if retry_number >= 3:
            return (
                "SD3.5 STRICT TWO-COLOUR RECOVERY. "
                f"Make {primary} cover most major surfaces. "
                f"Make {secondary} clearly visible across multiple "
                "substantial supporting regions. "
                f"Use only {primary} and {secondary}."
            )

        return (
            "SD3.5 PALETTE CORRECTION. "
            f"Use a strong {primary} dominant base and clearly visible "
            f"{secondary} supporting accents."
        )

    if primary:
        return (
            "SD3.5 PALETTE CORRECTION. "
            f"Use the {primary} colour family."
        )

    return (
        "SD3.5 PALETTE CORRECTION. "
        "Use only the explicitly requested palette."
    )


def _human_instruction(
    retry_number: int,
) -> tuple[
    str,
    str,
]:
    if retry_number >= 4:
        positive = (
            "STRICT HUMAN-FREE RECOVERY. "
            "Use only abstract geometry, environmental forms, patterns, "
            "architecture, materials and ornamental shapes. "
            "Do not arrange shapes into heads, faces, torsos, limbs, "
            "bodies, silhouettes or character-like structures."
        )

    else:
        positive = (
            "STRICT HUMAN-FREE CORRECTION. "
            "Background only with abstract or environmental structure."
        )

    negative = (
        "person, people, human, human figure, man, woman, child, "
        "face, portrait, silhouette, body, head, arms, hands, legs, "
        "clothing, mannequin, character, hero, superhero, humanoid"
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
            "Use visual artwork only with no readable symbols."
        ),
        (
            "text, typography, words, letters, numbers, writing, "
            "logo, watermark, signage, labels, pseudo-text"
        ),
    )


def _quality_instruction(
    effective_retry: int,
) -> tuple[
    str,
    str,
]:
    if effective_retry >= 10:
        positive = (
            "SECOND-WAVE IMAGE QUALITY RECOVERY. "
            "Produce a clean coherent rendered background with stable "
            "continuous visual forms. Use large intentional shapes, "
            "clean surfaces and controlled detail."
        )

    elif effective_retry >= 5:
        positive = (
            "STRICT IMAGE QUALITY RECOVERY. "
            "Use coherent shapes and clean surfaces. "
            "Avoid repetitive raster textures and degenerate output."
        )

    else:
        positive = (
            "IMAGE QUALITY CORRECTION. "
            "Render a coherent designed event background."
        )

    negative = (
        "scanlines, scan lines, horizontal scanlines, vertical scanlines, "
        "raster banding, horizontal banding, vertical banding, repetitive "
        "full-frame stripes, digital interference, moire, glitch, "
        "corrupted texture, broken render, blank image, empty flat field"
    )

    return (
        positive,
        negative,
    )


def _strengthen_prompts(
    base_positive_prompt: str,
    base_negative_prompt: str,
    reason: str,
    primary_colour: str | None,
    secondary_colour: str | None,
    retry_number: int,
    engine_id: str,
    recovery_wave: int,
) -> tuple[
    str,
    str,
]:
    (
        positive_prompt,
        negative_prompt,
    ) = (
        _initial_engine_prompts(
            engine_id,
            base_positive_prompt,
            base_negative_prompt,
            primary_colour,
            secondary_colour,
        )
    )

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

    quality_failure = (
        _is_quality_failure(
            reason
        )
    )

    runtime_failure = (
        _is_runtime_failure(
            reason
        )
    )

    effective_retry = (
        retry_number
        + (
            recovery_wave
            * (
                settings
                .max_candidate_retries
                + 1
            )
        )
    )

    #
    # Every retry becomes stronger.
    #
    positive_additions.append(
        _retry_strength_instruction(
            effective_retry
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
                    effective_retry,
                )
            )

        else:
            positive_additions.append(
                _standard_palette_instruction(
                    primary_colour,
                    secondary_colour,
                    effective_retry,
                )
            )

    #
    # SDXL needs a substantially different strategy after
    # repeated palette drift.
    #
    if (
        engine_id
        == settings.engine_2_id
        and palette_failure
    ):
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
            effective_retry >= 6
            and primary
            and secondary
        ):
            positive_prompt = (
                "Abstract A4 portrait decorative event background. "
                f"Use only {primary} and {secondary}. "
                f"Large matte {primary} surfaces dominate the image. "
                f"Large matte {secondary} geometric accents appear "
                "around outer edges, side borders and lower corners. "
                "Simple graphic city geometry and abstract line patterns. "
                "Large calm central region. "
                "Flat graphic illustration. "
                "Clean vector-like shapes. "
                "No realistic materials. "
                "No realistic illumination. "
                "Depth comes only from tonal changes inside the "
                "requested colour families."
            )

        elif effective_retry >= 3:
            positive_additions.append(
                (
                    "SDXL STRICT FLAT-COLOUR RECOVERY. "
                    "Remove realistic materials, reflective surfaces, "
                    "metallic appearance, cinematic illumination, "
                    "photographic shading and colourless highlights. "
                    "Use large matte colour regions with clean graphic "
                    "geometry."
                )
            )

        else:
            positive_additions.append(
                (
                    "SDXL PALETTE RECOVERY. "
                    "Strengthen the requested colours and remove "
                    "neutral highlights, neutral shading and metallic tones."
                )
            )

        negative_additions.append(
            (
                "white highlight, white surface, off-white, ivory, "
                "cream surface, beige surface, silver surface, "
                "grey surface, gray surface, neutral surface, "
                "neutral highlight, neutral shadow, metallic surface, "
                "chrome, steel, photographic lighting, studio lighting, "
                "white rim light, white glow, colourless reflection"
            )
        )

    if human_failure:
        (
            positive_human,
            negative_human,
        ) = (
            _human_instruction(
                effective_retry
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

    if quality_failure:
        (
            positive_quality,
            negative_quality,
        ) = (
            _quality_instruction(
                effective_retry
            )
        )

        positive_additions.append(
            positive_quality
        )

        negative_additions.append(
            negative_quality
        )

    if runtime_failure:
        positive_additions.append(
            (
                "RUNTIME RECOVERY ATTEMPT. "
                "Use a simpler coherent composition with fewer "
                "intricate micro-details and larger clean regions."
            )
        )

        if (
            engine_id
            == settings.engine_3_id
        ):
            positive_additions.append(
                (
                    "SD3.5 RUNTIME RECOVERY. "
                    "Prefer broad geometric forms, restrained layering "
                    "and uncomplicated surfaces."
                )
            )

    reason_lower = (
        reason.lower()
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
                "MANDATORY SECONDARY COLOUR REQUIREMENT. "
                f"Render {secondary} as actual visible surface colour "
                "in several substantial separate regions."
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
                "MANDATORY PRIMARY COLOUR REQUIREMENT. "
                f"Render {primary} across large visible surfaces."
            )
        )

    if recovery_wave > 0:
        positive_additions.append(
            (
                f"RECOVERY WAVE {recovery_wave}. "
                "A complete previous nine-attempt generation wave "
                "failed. Simplify the image and satisfy every "
                "validation requirement literally."
            )
        )

    positive_prompt = (
        positive_prompt.rstrip(
            ". "
        )
        + ". "
        + " ".join(
            positive_additions
        )
    )

    negative_prompt = (
        ", ".join(
            value
            for value
            in (
                negative_prompt.strip(
                    ", "
                ),
                *negative_additions,
            )
            if value
        )
    )

    return (
        positive_prompt,
        negative_prompt,
    )


async def _prepare_engine(
    client: ComfyUIClient,
    reference_number: str,
    output,
    current_engine: str | None,
) -> str:
    if (
        current_engine
        == output.engine_id
    ):
        return (
            current_engine
        )

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
            f"image-"
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
                if len(entry) > 3
            )
        ):
            raise RuntimeError(
                "ComfyUI has another active job; "
                "retry later."
            )

    else:
        await (
            client.release_models()
        )

    return (
        output.engine_id
    )


async def _attempt_output_wave(
    *,
    client: ComfyUIClient,
    state,
    document: dict,
    output,
    reference_number: str,
    primary_colour: str | None,
    secondary_colour: str | None,
    recovery_wave: int,
) -> tuple[
    bool,
    int,
]:
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

    (
        positive_prompt,
        negative_prompt,
    ) = (
        _initial_engine_prompts(
            output.engine_id,
            base_positive_prompt,
            base_negative_prompt,
            primary_colour,
            secondary_colour,
        )
    )

    #
    # Recovery wave starts already strengthened.
    #
    if recovery_wave > 0:
        last_reason = (
            _rejection_reason(
                reference_number,
                output.engine_id,
                output.direction_id,
            )
            or output.error
            or (
                "Previous attempt wave exhausted."
            )
        )

        (
            positive_prompt,
            negative_prompt,
        ) = (
            _strengthen_prompts(
                base_positive_prompt,
                base_negative_prompt,
                last_reason,
                primary_colour,
                secondary_colour,
                1,
                output.engine_id,
                recovery_wave,
            )
        )

    generated = 0

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
                "reused=%s "
                "recovery_wave=%s",
                reference_number,
                output.engine_id,
                output.direction_id,
                result[
                    "reused"
                ],
                recovery_wave,
            )

            if not (
                result[
                    "reused"
                ]
            ):
                generated = 1

            return (
                True,
                generated,
            )

        except SubmissionUncertain:
            raise

        except ComfyUIError as error:
            reason = (
                _rejection_reason(
                    reference_number,
                    output.engine_id,
                    output.direction_id,
                )
                or str(error)
            )

            #
            # Validation rejection AND remote execution failure
            # are candidate-level failures and therefore retryable.
            #
            retryable = (
                _candidate_was_rejected(
                    reference_number,
                    output.engine_id,
                    output.direction_id,
                )
                or _is_runtime_failure(
                    reason
                )
            )

            if not retryable:
                raise

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
                        reason
                    ),
                )

                logger.error(
                    "event=candidate_exhausted "
                    "reference=%s "
                    "engine=%s "
                    "direction=%s "
                    "attempts=%s "
                    "recovery_wave=%s "
                    "reason=%s",
                    reference_number,
                    output.engine_id,
                    output.direction_id,
                    (
                        settings
                        .max_candidate_retries
                        + 1
                    ),
                    recovery_wave,
                    reason,
                )

                return (
                    False,
                    0,
                )

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
                    recovery_wave,
                )
            )

            delay_seconds = min(
                2 ** retry_count,
                settings
                .max_candidate_retry_delay_seconds,
            )

            effective_retry = (
                next_retry
                + (
                    recovery_wave
                    * (
                        settings
                        .max_candidate_retries
                        + 1
                    )
                )
            )

            logger.warning(
                "event=candidate_retry "
                "reference=%s "
                "engine=%s "
                "direction=%s "
                "retry=%s/%s "
                "effective_retry=%s "
                "recovery_wave=%s "
                "delay_seconds=%s "
                "reason=%s",
                reference_number,
                output.engine_id,
                output.direction_id,
                next_retry,
                settings.max_candidate_retries,
                effective_retry,
                recovery_wave,
                delay_seconds,
                reason,
            )

            logger.info(
                "event=prompt_strengthened "
                "reference=%s "
                "engine=%s "
                "direction=%s "
                "effective_retry=%s "
                "palette_failure=%s "
                "human_failure=%s "
                "text_failure=%s "
                "quality_failure=%s "
                "runtime_failure=%s",
                reference_number,
                output.engine_id,
                output.direction_id,
                effective_retry,
                _is_palette_failure(
                    reason
                ),
                _is_human_failure(
                    reason
                ),
                _is_text_failure(
                    reason
                ),
                _is_quality_failure(
                    reason
                ),
                _is_runtime_failure(
                    reason
                ),
            )

            if (
                output.engine_id
                == settings.engine_2_id
                and (
                    _is_palette_failure(
                        reason
                    )
                    or _is_quality_failure(
                        reason
                    )
                )
            ):
                logger.info(
                    "event=sdxl_recovery "
                    "reference=%s "
                    "direction=%s "
                    "effective_retry=%s "
                    "recovery_wave=%s",
                    reference_number,
                    output.direction_id,
                    effective_retry,
                    recovery_wave,
                )

            if (
                output.engine_id
                == settings.engine_3_id
            ):
                logger.info(
                    "event=sd35_recovery "
                    "reference=%s "
                    "direction=%s "
                    "effective_retry=%s "
                    "recovery_wave=%s "
                    "runtime_failure=%s "
                    "palette_failure=%s",
                    reference_number,
                    output.direction_id,
                    effective_retry,
                    recovery_wave,
                    _is_runtime_failure(
                        reason
                    ),
                    _is_palette_failure(
                        reason
                    ),
                )

            await asyncio.sleep(
                delay_seconds
            )

    return (
        False,
        0,
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
            #
            # Ollama must not hold GPU memory while image
            # generation is running.
            #
            async with httpx.AsyncClient(
                timeout=60
            ) as http:
                response = (
                    await http.get(
                        settings
                        .ollama_base_url
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

            async with (
                ComfyUIClient()
                as client
            ):
                generated = 0
                current_engine = None

                stopped_for_max_outputs = (
                    False
                )

                first_wave_failures = []

                try:
                    #
                    # =====================================
                    # FIRST WAVE
                    # =====================================
                    #
                    # 1 initial attempt + 8 retries
                    # = 9 attempts for every candidate.
                    #
                    for output in state.outputs:
                        if (
                            output.status
                            == "complete"
                        ):
                            continue

                        current_engine = (
                            await _prepare_engine(
                                client,
                                reference_number,
                                output,
                                current_engine,
                            )
                        )

                        state.current_stage = (
                            f"{output.engine_id}:"
                            f"{output.direction_id}"
                        )

                        persist_image_job_state(
                            state
                        )

                        (
                            complete,
                            newly_generated,
                        ) = (
                            await _attempt_output_wave(
                                client=client,
                                state=state,
                                document=document,
                                output=output,
                                reference_number=(
                                    reference_number
                                ),
                                primary_colour=(
                                    primary_colour
                                ),
                                secondary_colour=(
                                    secondary_colour
                                ),
                                recovery_wave=0,
                            )
                        )

                        generated += (
                            newly_generated
                        )

                        if not complete:
                            first_wave_failures.append(
                                output
                            )

                        if (
                            max_outputs
                            is not None
                            and generated
                            >= max_outputs
                        ):
                            stopped_for_max_outputs = (
                                True
                            )

                            break

                    #
                    # =====================================
                    # RECOVERY WAVES
                    # =====================================
                    #
                    # Failed candidates are retried only after
                    # every engine/direction has finished its
                    # first nine-attempt wave.
                    #
                    if (
                        not stopped_for_max_outputs
                        and first_wave_failures
                        and (
                            settings
                            .failed_candidate_recovery_waves
                            > 0
                        )
                    ):
                        logger.warning(
                            "event=failed_candidate_recovery_wave_start "
                            "reference=%s "
                            "failed_candidates=%s "
                            "attempts_per_candidate=%s "
                            "configured_waves=%s",
                            reference_number,
                            len(
                                first_wave_failures
                            ),
                            (
                                settings
                                .max_candidate_retries
                                + 1
                            ),
                            settings
                            .failed_candidate_recovery_waves,
                        )

                        state.current_stage = (
                            "retrying_exhausted_candidates"
                        )

                        persist_image_job_state(
                            state
                        )

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
                                client.release_models()
                            )

                        current_engine = None

                        recovery_failures = (
                            first_wave_failures
                        )

                        for recovery_wave in range(
                            1,
                            (
                                settings
                                .failed_candidate_recovery_waves
                                + 1
                            ),
                        ):
                            if (
                                not recovery_failures
                            ):
                                break

                            next_failures = []

                            logger.warning(
                                "event=failed_candidate_recovery_round "
                                "reference=%s "
                                "recovery_wave=%s "
                                "candidates=%s",
                                reference_number,
                                recovery_wave,
                                len(
                                    recovery_failures
                                ),
                            )

                            for output in (
                                recovery_failures
                            ):
                                update_output(
                                    state,
                                    output.engine_id,
                                    output.direction_id,
                                    "pending",
                                    error=None,
                                )

                                current_engine = (
                                    await _prepare_engine(
                                        client,
                                        reference_number,
                                        output,
                                        current_engine,
                                    )
                                )

                                state.current_stage = (
                                    f"recovery-"
                                    f"{recovery_wave}:"
                                    f"{output.engine_id}:"
                                    f"{output.direction_id}"
                                )

                                persist_image_job_state(
                                    state
                                )

                                (
                                    complete,
                                    newly_generated,
                                ) = (
                                    await _attempt_output_wave(
                                        client=client,
                                        state=state,
                                        document=document,
                                        output=output,
                                        reference_number=(
                                            reference_number
                                        ),
                                        primary_colour=(
                                            primary_colour
                                        ),
                                        secondary_colour=(
                                            secondary_colour
                                        ),
                                        recovery_wave=(
                                            recovery_wave
                                        ),
                                    )
                                )

                                generated += (
                                    newly_generated
                                )

                                if not complete:
                                    next_failures.append(
                                        output
                                    )

                                if (
                                    max_outputs
                                    is not None
                                    and generated
                                    >= max_outputs
                                ):
                                    stopped_for_max_outputs = (
                                        True
                                    )

                                    break

                            recovery_failures = (
                                next_failures
                            )

                            if (
                                stopped_for_max_outputs
                            ):
                                break

                        logger.warning(
                            "event=failed_candidate_recovery_wave_complete "
                            "reference=%s "
                            "remaining_failed=%s",
                            reference_number,
                            len(
                                recovery_failures
                            ),
                        )

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
                            client.release_models()
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

            if (
                stopped_for_max_outputs
                and pending > 0
            ):
                state.status = (
                    "processing"
                )

                state.current_stage = (
                    "generation_paused"
                )

                state.error = (
                    f"Generation paused after "
                    f"{generated} new outputs; "
                    f"{completed}/"
                    f"{state.total_outputs} "
                    "candidates complete."
                )

            elif completed > 0:
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

            state.error = (
                str(error)
            )

            persist_image_job_state(
                state
            )

            raise