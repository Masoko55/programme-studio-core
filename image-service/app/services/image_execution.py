from __future__ import annotations

import asyncio
import json
import logging

import httpx

from app.config.settings import settings
from app.engines.registry import get_engine
from app.services.comfyui_client import (
    ComfyUIClient,
    ComfyUIError,
)
from app.services.execution_plan import (
    build_execution_plan,
)
from app.services.gpu_lease import GPULease
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
        / f"image-{direction_id.lower()}.json"
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
        record.get("status")
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


def _normalise_colour(
    value: str | None,
) -> str:
    return str(
        value or ""
    ).strip().lower()


def _is_palette_failure(
    reason: str,
) -> bool:
    reason_lower = reason.lower()

    terms = (
        "monochrome",
        "black-and-white",
        "palette",
        "colour",
        "color",
        "primary ratio",
        "secondary ratio",
        "off-palette",
        "unrequested",
    )

    return any(
        term in reason_lower
        for term in terms
    )


def _is_human_failure(
    reason: str,
) -> bool:
    reason_lower = reason.lower()

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
    reason_lower = reason.lower()

    terms = (
        "ocr",
        "text",
        "readable",
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
    reason_lower = reason.lower()

    terms = (
        "visual quality",
        "scanline",
        "scan line",
        "raster",
        "banding",
        "moire",
        "glitch",
        "corrupt",
        "flat or blank",
        "insufficient visual structure",
        "directional dominance",
        "full-span",
    )

    return any(
        term in reason_lower
        for term in terms
    )


def _retry_strength_instruction(
    effective_retry: int,
    recovery_wave: int,
) -> str:
    if effective_retry >= 15:
        return (
            "MAXIMUM RETRY ENFORCEMENT. "
            "Previous generations repeatedly failed deterministic QA. "
            "Use the simplest coherent composition that satisfies every "
            "constraint literally. Remove unnecessary effects, complex "
            "materials, decorative noise and ambiguous visual forms. "
            "Prioritise valid requested colours, clean rendering, "
            "background-only structure and reliable composition over "
            "creative complexity."
        )

    if effective_retry >= 10:
        return (
            "SECOND-WAVE STRICT RECOVERY. "
            "The first complete generation wave failed QA. "
            "Simplify the design substantially. Obey all palette, "
            "human-free, text-free and image-quality requirements "
            "literally. Avoid unnecessary lighting effects, material "
            "effects, reflections and complex visual noise."
        )

    if effective_retry >= 7:
        return (
            "VERY STRICT RETRY. "
            "Previous candidates continued to fail QA. "
            "Use cleaner larger forms, fewer effects, simpler materials "
            "and more literal compliance with every validation rule."
        )

    if effective_retry >= 4:
        return (
            "STRICT RETRY. "
            "Strengthen compliance beyond the previous attempt. "
            "Reduce unnecessary complexity and make every required "
            "visual constraint unmistakable in the rendered pixels."
        )

    if effective_retry >= 2:
        return (
            "ENHANCED RETRY. "
            "The previous candidate failed QA. "
            "Apply the correction more strongly and make the requested "
            "constraints visually explicit."
        )

    return (
        "RETRY CORRECTION. "
        "The previous candidate failed deterministic QA. "
        "Correct the detected violation explicitly."
    )


def _standard_palette_instruction(
    primary_colour: str | None,
    secondary_colour: str | None,
    retry_number: int,
) -> str:
    primary = _normalise_colour(
        primary_colour
    )

    secondary = _normalise_colour(
        secondary_colour
    )

    if primary and secondary:
        if retry_number >= 7:
            return (
                "EXTREME PALETTE ENFORCEMENT. "
                f"Use only {primary} and {secondary}. "
                f"Make {primary} occupy most major surfaces. "
                f"Make {secondary} occupy clearly visible supporting "
                "regions in several areas. "
                "Every visible neutral or chromatic family not explicitly "
                "requested is forbidden. Shadows, highlights, depth and "
                "contrast must remain tonal variations of the requested "
                "colour families."
            )

        if retry_number >= 4:
            return (
                "STRICT PALETTE RECOVERY. "
                f"Use {primary} as the dominant visible colour across "
                "most large surfaces. "
                f"Use {secondary} as a clearly visible supporting "
                "colour in several separate regions. "
                "Do not introduce any other colour family, including "
                "unrequested black, white, grey, silver, green, cyan, "
                "purple, pink, orange, yellow, brown or unrelated hues. "
                "Shading must remain inside the requested colour families."
            )

        return (
            "STRICT COLOUR CORRECTION. "
            f"The primary colour is {primary}. "
            f"The secondary colour is {secondary}. "
            f"Make {primary} visibly dominant and make {secondary} "
            "clearly visible as repeated supporting accents. "
            "Every other visible colour family is forbidden."
        )

    if primary:
        return (
            "STRICT COLOUR CORRECTION. "
            f"Make {primary} unmistakably visible and dominant. "
            "Do not introduce any other visible colour family."
        )

    return (
        "STRICT COLOUR CORRECTION. "
        "Obey the requested palette exactly and remove every "
        "unrequested colour family."
    )


def _sdxl_recovery_instruction(
    primary_colour: str | None,
    secondary_colour: str | None,
    effective_retry: int,
) -> tuple[str, str]:
    primary = _normalise_colour(
        primary_colour
    )

    secondary = _normalise_colour(
        secondary_colour
    )

    palette_text = (
        f"Use only {primary} and {secondary}."
        if primary and secondary
        else (
            f"Use only {primary}."
            if primary
            else (
                "Use only the explicitly "
                "requested palette."
            )
        )
    )

    if effective_retry >= 10:
        positive = (
            "SDXL SECOND-WAVE FLAT-COLOUR RECOVERY. "
            f"{palette_text} "
            "Render a clean graphic event background using matte flat "
            "colour surfaces and controlled tonal variants inside the "
            "requested colour families only. Use simple architectural, "
            "geometric or abstract forms. Use colour-area contrast rather "
            "than white highlights, grey shading, metallic reflections, "
            "photorealistic illumination or neutral shadows. "
            "No realistic lighting. No specular highlights. "
            "No colourless reflections. No neutral depth shading. "
            "The result must remain a coherent designed background."
        )
    elif effective_retry >= 5:
        positive = (
            "SDXL STRICT GRAPHIC PALETTE RECOVERY. "
            f"{palette_text} "
            "Prefer clean flat graphic surfaces. Remove realistic "
            "illumination, photographic shadows, bright neutral highlights, "
            "metallic materials, reflective surfaces and colourless glow. "
            "Create depth only through requested-colour tonal variation."
        )
    else:
        positive = (
            "SDXL PALETTE RECOVERY. "
            f"{palette_text} "
            "Keep highlights and shadows inside the requested colour "
            "families. Avoid neutral lighting and reflective colour drift."
        )

    negative = (
        "white highlight, grey highlight, gray highlight, silver highlight, "
        "cream highlight, beige highlight, neutral shadow, grey shadow, "
        "gray shadow, silver material, metallic reflection, neutral "
        "reflection, photographic lighting, studio lighting, white glow, "
        "colourless glow, photorealistic shading, neutral gradient"
    )

    return (
        positive,
        negative,
    )


def _sd35_palette_instruction(
    primary_colour: str | None,
    secondary_colour: str | None,
    retry_number: int,
) -> str:
    primary = _normalise_colour(
        primary_colour
    )

    secondary = _normalise_colour(
        secondary_colour
    )

    if primary and secondary:
        if retry_number >= 6:
            return (
                "SD3.5 EMERGENCY PALETTE RECOVERY. "
                "Create a simple abstract background with very large "
                "clean colour regions. "
                f"Approximately 65 percent of the visible chromatic area "
                f"should read clearly as {primary}. "
                f"Approximately 25 to 35 percent should read clearly "
                f"as {secondary}. "
                f"Use only {primary} and {secondary}. "
                "Do not add white, black, grey, silver or any third "
                "chromatic hue unless explicitly requested. "
                "Avoid colour-changing glow, reflections, iridescence "
                "and multicolour gradients. "
                "Colour compliance is more important than complexity."
            )

        if retry_number >= 3:
            return (
                "SD3.5 STRICT TWO-COLOUR RECOVERY. "
                f"Make {primary} cover most major visible surfaces. "
                f"Make {secondary} clearly visible in multiple substantial "
                "supporting regions. "
                f"Use only {primary} and {secondary}. "
                "No third colour family, no neutral fills unless explicitly "
                "requested and no coloured lighting drift."
            )

        return (
            "SD3.5 PALETTE CORRECTION. "
            f"Use a strong {primary} dominant base and clearly visible "
            f"{secondary} supporting accents. "
            f"Both {primary} and {secondary} must appear as actual "
            "rendered surface colours. "
            "Do not replace either colour with a neighbouring hue "
            "and do not introduce any additional colour family."
        )

    if primary:
        return (
            "SD3.5 PALETTE CORRECTION. "
            f"Use {primary} as the only visible colour family."
        )

    return (
        "SD3.5 PALETTE CORRECTION. "
        "Use only the explicitly requested palette."
    )


def _human_instruction(
    retry_number: int,
) -> tuple[str, str]:
    if retry_number >= 4:
        positive = (
            "STRICT HUMAN-FREE RECOVERY. "
            "Use only abstract geometry, environmental forms, patterns, "
            "architecture, materials and ornamental shapes. "
            "Do not arrange shapes into heads, faces, torsos, limbs, "
            "bodies, silhouettes, poses or character-like structures. "
            "Avoid a central figure-like subject."
        )
    else:
        positive = (
            "STRICT HUMAN-FREE CORRECTION. "
            "Background only. No person, human figure, silhouette, face, "
            "body, mannequin, clothing, character or human-shaped object."
        )

    negative = (
        "person, people, human, human figure, man, woman, child, face, "
        "portrait, silhouette, body, head, arms, hands, legs, clothing, "
        "mannequin, character, hero, superhero, humanoid, human-shaped form"
    )

    return (
        positive,
        negative,
    )


def _text_instruction() -> tuple[str, str]:
    return (
        (
            "STRICT TEXT-FREE CORRECTION. "
            "Do not render readable characters, pseudo-letters, words, "
            "numbers, logos, labels, signs, typography or writing."
        ),
        (
            "text, typography, words, letters, numbers, writing, logo, "
            "watermark, signage, labels, pseudo-text"
        ),
    )


def _quality_instruction(
    effective_retry: int,
) -> tuple[str, str]:
    if effective_retry >= 10:
        positive = (
            "SECOND-WAVE IMAGE QUALITY RECOVERY. "
            "Produce a clean coherent rendered background with stable "
            "continuous forms and intentional visual structure. "
            "Use large clean shapes and smooth coherent surfaces. "
            "Do not create raster-like repetition, repeated scan lines, "
            "full-frame striping, digital corruption or blank flat output."
        )
    elif effective_retry >= 5:
        positive = (
            "STRICT IMAGE QUALITY RECOVERY. "
            "Use coherent shapes and clean continuous surfaces. "
            "Avoid excessive repetitive lines, raster textures, banding "
            "and degenerate flat fields."
        )
    else:
        positive = (
            "IMAGE QUALITY CORRECTION. "
            "Render a coherent designed background rather than scanline, "
            "banded, corrupted, blank or degenerate output."
        )

    negative = (
        "scanlines, scan lines, horizontal scanlines, vertical scanlines, "
        "raster banding, horizontal banding, vertical banding, repetitive "
        "full-frame stripes, moire, glitch, corrupted texture, broken "
        "render, digital interference, blank image, empty flat field"
    )

    return (
        positive,
        negative,
    )


def _sd35_minimal_recovery_prompt(
    primary_colour: str | None,
    secondary_colour: str | None,
) -> str:
    primary = _normalise_colour(
        primary_colour
    )

    secondary = _normalise_colour(
        secondary_colour
    )

    if primary and secondary:
        return (
            "Abstract A4 portrait decorative event background only. "
            f"Large coherent {primary} colour fields dominate the "
            "composition. "
            f"Large clearly visible {secondary} supporting geometric "
            "forms appear across several areas. "
            f"Use only {primary} and {secondary}. "
            "No third colour family and no unrequested neutral fills. "
            "Strong contrast must come only from tonal variation inside "
            "the requested colours. "
            "Use coherent geometric shapes, not repetitive raster lines. "
            "No scanlines. No banding. No glitch patterns. "
            "No people. No faces. No characters. No silhouettes. "
            "No text. No letters. No numbers. No logos. No watermark. "
            "Leave clean quiet regions for later overlays."
        )

    if primary:
        return (
            "Abstract A4 portrait decorative event background only. "
            f"Use only the requested {primary} colour family with tonal "
            "variation inside that family. "
            "Use coherent geometric shapes and surfaces. "
            "No scanlines, banding, glitch textures, people, faces, "
            "characters, text, letters, numbers, logos or watermark."
        )

    return (
        "Abstract A4 portrait decorative event background only. "
        "Simple coherent geometric composition using only the explicitly "
        "requested palette. No scanlines, banding or glitch textures. "
        "No people, faces, characters, text, letters, numbers, logos "
        "or watermark."
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
) -> tuple[str, str]:
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

    effective_retry = (
        retry_number
        + (
            recovery_wave
            * (
                settings.max_candidate_retries
                + 1
            )
        )
    )

    positive_additions.append(
        _retry_strength_instruction(
            effective_retry,
            recovery_wave,
        )
    )

    if (
        engine_id
        == settings.engine_3_id
        and palette_failure
        and effective_retry >= 6
        and not quality_failure
    ):
        positive_prompt = (
            _sd35_minimal_recovery_prompt(
                primary_colour,
                secondary_colour,
            )
        )

        positive_prompt += (
            " "
            + _retry_strength_instruction(
                effective_retry,
                recovery_wave,
            )
        )
    else:
        positive_prompt = (
            base_positive_prompt.rstrip(
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

        if (
            engine_id
            == settings.engine_2_id
            and (
                palette_failure
                or quality_failure
            )
        ):
            (
                sdxl_positive,
                sdxl_negative,
            ) = _sdxl_recovery_instruction(
                primary_colour,
                secondary_colour,
                effective_retry,
            )

            positive_additions.append(
                sdxl_positive
            )

            negative_additions.append(
                sdxl_negative
            )

        if human_failure:
            (
                positive_human,
                negative_human,
            ) = _human_instruction(
                effective_retry
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
            ) = _text_instruction()

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
            ) = _quality_instruction(
                effective_retry
            )

            positive_additions.append(
                positive_quality
            )

            negative_additions.append(
                negative_quality
            )

        reason_lower = reason.lower()

        if (
            "secondary" in reason_lower
            and secondary_colour
        ):
            secondary = _normalise_colour(
                secondary_colour
            )

            positive_additions.append(
                (
                    "MANDATORY SECONDARY COLOUR REQUIREMENT. "
                    f"Render {secondary} as actual visible surface colour "
                    "in several substantial regions. Do not make it a tiny "
                    "highlight and do not hide it inside lighting or "
                    "reflections."
                )
            )

        if (
            "primary" in reason_lower
            and primary_colour
        ):
            primary = _normalise_colour(
                primary_colour
            )

            positive_additions.append(
                (
                    "MANDATORY PRIMARY COLOUR REQUIREMENT. "
                    f"Render {primary} as actual visible surface colour "
                    "across large portions of the composition."
                )
            )

        if recovery_wave > 0:
            positive_additions.append(
                (
                    f"RECOVERY WAVE {recovery_wave}. "
                    "A complete previous generation wave exhausted all "
                    "allowed attempts. Simplify the image further and obey "
                    "all deterministic validation requirements literally."
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
        base_negative_prompt.strip()
    )

    if palette_failure:
        negative_additions.append(
            (
                "off-palette colours, colour drift, unrelated hues, "
                "third colour, extra colour, multicolour lighting, "
                "rainbow gradient, coloured reflections, unrequested "
                "chromatic accents, unrequested black, unrequested white, "
                "unrequested grey, unrequested silver, unrequested cream, "
                "unrequested beige, unrequested charcoal"
            )
        )

    if (
        engine_id
        == settings.engine_3_id
        and palette_failure
        and effective_retry >= 3
    ):
        negative_additions.append(
            (
                "complex multicolour materials, coloured atmospheric "
                "lighting, cinematic colour grading, rainbow glow, "
                "iridescence, unrelated neon hues"
            )
        )

    if quality_failure:
        negative_additions.append(
            (
                "scanlines, raster banding, repetitive horizontal lines, "
                "repetitive vertical lines, digital interference, moire, "
                "glitch, corrupted image, broken texture, blank image, "
                "flat empty field"
            )
        )

    corrected_negative = ", ".join(
        value
        for value in (
            negative_prompt,
            *negative_additions,
        )
        if value
    )

    return (
        positive_prompt,
        corrected_negative,
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
        return current_engine

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
                for entry in entries
            )
        ):
            raise RuntimeError(
                "ComfyUI has another active job; "
                "retry later."
            )
    else:
        await client.release_models()

    return output.engine_id


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
) -> tuple[bool, int]:
    direction = get_direction(
        document,
        output.direction_id,
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
        ) = _strengthen_prompts(
            base_positive_prompt,
            base_negative_prompt,
            last_reason,
            primary_colour,
            secondary_colour,
            1,
            output.engine_id,
            recovery_wave,
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

            output.seed = result[
                "seed"
            ]

            output.prompt_id = result[
                "prompt_id"
            ]

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
                result["output_path"],
                result["sha256"],
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
                result["reused"],
                recovery_wave,
            )

            if not result["reused"]:
                generated = 1

            return (
                True,
                generated,
            )

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
                or str(error)
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
                    "recovery_wave=%s "
                    "reason=%s",
                    reference_number,
                    output.engine_id,
                    output.direction_id,
                    (
                        settings.max_candidate_retries
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
            ) = _strengthen_prompts(
                base_positive_prompt,
                base_negative_prompt,
                reason,
                primary_colour,
                secondary_colour,
                next_retry,
                output.engine_id,
                recovery_wave,
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
                        settings.max_candidate_retries
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
                "quality_failure=%s",
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
                    "recovery_wave=%s "
                    "mode=%s",
                    reference_number,
                    output.direction_id,
                    next_retry,
                    recovery_wave,
                    (
                        "minimal"
                        if effective_retry >= 6
                        else (
                            "strict"
                            if effective_retry >= 3
                            else "normal"
                        )
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
        state = load_image_job_state(
            reference_number
        )

        document = (
            load_prompts_document(
                reference_number
            )
        )

        brief = document.get(
            "brief",
            {},
        )

        primary_colour = brief.get(
            "primary_colour"
        )

        secondary_colour = brief.get(
            "secondary_colour"
        )

        expected = [
            (
                step["engine_id"],
                step["direction"],
            )
            for step in (
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
            for output in state.outputs
        ]

        if actual != expected:
            raise ValueError(
                "Historical engine plan differs from the "
                "current configuration. Preserve this job "
                "and create a new reference number."
            )

        state.status = "processing"
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
                response = await http.get(
                    settings.ollama_base_url
                    + "/api/ps"
                )

                response.raise_for_status()

                if (
                    response.json().get(
                        "models"
                    )
                ):
                    raise RuntimeError(
                        "Ollama has resident models; "
                        "unload them before image generation."
                    )

            async with ComfyUIClient() as client:
                generated = 0
                current_engine = None
                stopped_for_max_outputs = False

                try:
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

                        if (
                            max_outputs
                            is not None
                            and generated
                            >= max_outputs
                        ):
                            stopped_for_max_outputs = True
                            break


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
                        await client.release_models()

            completed = (
                state.completed_outputs
            )

            failed = sum(
                1
                for output in state.outputs
                if output.status
                == "failed"
            )

            pending = sum(
                1
                for output in state.outputs
                if output.status
                == "pending"
            )

            if (
                stopped_for_max_outputs
                and pending > 0
            ):
                state.status = "processing"
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
                state.status = "failed"
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
            state.status = "failed"
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