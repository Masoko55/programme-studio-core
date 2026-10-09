"""ComfyUI HTTP adapter and deterministic candidate validation."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import re
import secrets
import time
import uuid

from pathlib import Path
from typing import Any

import httpx
import pytesseract
import cv2
import numpy as np

from PIL import Image

from app.config.settings import settings

from app.services.atomic import (
    write_bytes,
    write_json,
)

from app.services.background_policy import (
    BACKGROUND_ONLY_NEGATIVE,
    build_engine_prompt,
    constrain_to_requested_palette,
    detect_human_signals,
    palette_negative_contract,
    validate_palette,
    validate_visual_quality,
)

from app.services.job_persistence import (
    get_job_directory,
)

from app.services.prompt_repository import (
    load_prompts_document,
)
from app.services.sampling_profiles import SamplingProfile, select_sampling_profile, summarize_failures
from app.services.prompt_compiler import failure_category as classify_failure


logger = logging.getLogger(
    "uvicorn.error"
)

# A large third-colour area needs a new generation, not automatic recolouring.
# ============================================================
# Deterministic palette-only recovery limits
# ============================================================
#
# Final palette validation remains unchanged.
#
# These values control only whether a structurally valid candidate is eligible
# for deterministic palette normalization BEFORE the normal strict palette
# validator is executed again.
#
# They are intentionally engine-specific.
#
# SDXL already had a broader recovery range because that engine frequently
# generates otherwise-correct artwork with substantial neutral/metallic drift.
#
# SD3.5 uses a much narrower limit. Controlled programme-studio probes showed
# structurally healthy SD3.5 outputs at approximately 0.09 off-palette, while
# catastrophic historical drift was much larger. We therefore allow only mild
# SD3.5 palette-only correction.
#

MAX_SDXL_RECOVERABLE_OFF_PALETTE_RATIO = 0.35
MAX_SD35_RECOVERABLE_OFF_PALETTE_RATIO = 0.15


def _palette_recovery_limit(
    engine_id: str | None,
) -> float:
    """Return the maximum raw palette drift eligible for correction."""

    if (
        engine_id
        == settings.engine_3_id
    ):
        return (
            MAX_SD35_RECOVERABLE_OFF_PALETTE_RATIO
        )

    #
    # Preserve the historical helper behaviour when tests/callers omit an
    # engine ID. SDXL was the original consumer of palette-only recovery.
    #
    if (
        engine_id is None
        or engine_id
        == settings.engine_2_id
    ):
        return (
            MAX_SDXL_RECOVERABLE_OFF_PALETTE_RATIO
        )

    #
    # FLUX and unknown engines are not eligible.
    #
    return 0.0


def _recoverable_palette_error(
    reason: str,
    engine_id: str | None = None,
) -> bool:
    """Return True only for bounded off-palette validation failures.

    This function does NOT relax final palette validation.

    It only decides whether deterministic recolouring may be attempted.
    The recoloured image must subsequently pass validate_background() with
    normal palette enforcement enabled.
    """

    match = re.search(
        r"off-palette ratio ([0-9.]+)",
        reason,
    )

    if not match:
        return False

    ratio = float(
        match.group(
            1
        )
    )

    return (
        ratio
        <= _palette_recovery_limit(
            engine_id
        )
    )


def _edge_geometry_similarity(source: Path, target: Path) -> float:
    """Compare edge locations after local contrast adjustment, not brightness."""
    edge_maps = []
    for path in (source, target):
        with Image.open(path) as image:
            grey = np.asarray(image.convert("L").resize((256, 352)), dtype=np.uint8)
        grey = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(grey)
        magnitude = np.hypot(
            cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3),
        )
        edge_maps.append(magnitude > np.quantile(magnitude, 0.85))
    first, second = edge_maps
    count = int(first.sum()) + int(second.sum())
    return 2 * int((first & second).sum()) / count if count else 0.0


def _palette_only_recovery(
    source: Path,
    target: Path,
    reference_number: str,
    primary_colour: str | None,
    secondary_colour: str | None,
) -> tuple[dict, dict]:
    """Recolour only a raw candidate that passes every non-palette gate."""
    structural = validate_background(source, reference_number=reference_number, enforce_palette=False)
    with Image.open(source) as image:
        recoloured = constrain_to_requested_palette(image, primary_colour, secondary_colour)
        recoloured.save(target, format="PNG")
    try:
        final = validate_background(target, reference_number=reference_number)
        if _edge_geometry_similarity(source, target) < 0.60:
            raise ValueError("Palette normalization changed structural edge locations too much.")
        return structural, final
    except Exception:
        target.unlink(missing_ok=True)
        raise


def _valid_background_dimensions() -> set[tuple[int, int]]:
    """Return every native generation size accepted by the compositor."""
    return {
        (
            settings.generation_width,
            settings.generation_height,
        ),
        (
            settings.sdxl_generation_width,
            settings.sdxl_generation_height,
        ),
        (
            settings.sd35_generation_width,
            settings.sd35_generation_height,
        ),
    }


def _attempt_budget_exhausted(
    record: dict | None,
) -> bool:
    """Prevent a restarted job from exceeding its persisted retry budget."""
    if not record:
        return False

    try:
        attempt_count = int(record.get("attempt_count", 0))
    except (TypeError, ValueError):
        return False

    return attempt_count >= settings.max_candidate_retries + 1


def _engine_positive_prompt(
    engine_id: str,
    positive_prompt: str,
    primary_colour: str | None,
    secondary_colour: str | None,
) -> str:
    """Add safety guidance without burying SD3.5's compact scene brief."""
    if engine_id == settings.engine_3_id:
        # SD3.5's T5 field is intentionally capped in build_workflow. Its
        # engine-specific subject and palette must therefore come first. The
        # general background wrapper is useful for other engines but consumed
        # the whole SD3.5 encoder budget before the actual creative direction.
        return (
            positive_prompt.rstrip(". ")
            + ". Background artwork only; no people, characters, readable "
            "text, logos, watermarks, scanlines or raster artefacts."
        )

    return build_engine_prompt(
        positive_prompt,
        primary_colour,
        secondary_colour,
    )


def _transport_prompts(
    engine_id: str,
    positive_prompt: str,
    negative_prompt: str,
    primary_colour: str | None,
    secondary_colour: str | None,
    compiled_prompt=None,
) -> tuple[str, str]:
    """Keep native compiled semantics unchanged at the ComfyUI boundary."""
    if compiled_prompt is not None:
        return compiled_prompt.positive, compiled_prompt.negative
    return (
        _engine_positive_prompt(engine_id, positive_prompt, primary_colour, secondary_colour),
        ", ".join(
            value for value in (
                BACKGROUND_ONLY_NEGATIVE,
                negative_prompt,
                palette_negative_contract(primary_colour, secondary_colour),
            ) if value
        ),
    )


class ComfyUIError(
    RuntimeError
):
    """ComfyUI execution or validation failure."""


class SubmissionUncertain(
    ComfyUIError
):
    """The remote server may already have accepted the request."""


def workflow_name(
    engine_id: str,
) -> str:
    names = {
        settings.engine_1_id: "flux2",
        settings.engine_2_id: "sdxl",
        settings.engine_3_id: "sd35-medium",
    }

    if engine_id not in names:
        raise ValueError(
            f"Unknown engine: {engine_id}"
        )

    return names[
        engine_id
    ]


def _extract_comfyui_execution_error(
    history: dict | None,
) -> str:
    """
    Extract the useful exception emitted by ComfyUI.

    ComfyUI normally places node execution failures inside
    status.messages as execution_error records.
    """

    if not history:
        return (
            "ComfyUI execution failed without "
            "remote history details."
        )

    status = (
        history.get(
            "status",
            {},
        )
        or {}
    )

    messages = (
        status.get(
            "messages",
            [],
        )
        or []
    )

    details = []

    for item in messages:
        try:
            if (
                not isinstance(
                    item,
                    (list, tuple),
                )
                or len(item) < 2
            ):
                continue

            message_type = str(
                item[0]
            )

            payload = (
                item[1]
                if isinstance(
                    item[1],
                    dict,
                )
                else {}
            )

            if (
                message_type
                not in {
                    "execution_error",
                    "execution_interrupted",
                }
            ):
                continue

            node_id = (
                payload.get(
                    "node_id"
                )
            )

            node_type = (
                payload.get(
                    "node_type"
                )
            )

            exception_type = (
                payload.get(
                    "exception_type"
                )
            )

            exception_message = (
                payload.get(
                    "exception_message"
                )
            )

            traceback_lines = (
                payload.get(
                    "traceback"
                )
                or []
            )

            parts = []

            if node_id:
                parts.append(
                    f"node={node_id}"
                )

            if node_type:
                parts.append(
                    f"type={node_type}"
                )

            if exception_type:
                parts.append(
                    f"exception={exception_type}"
                )

            if exception_message:
                parts.append(
                    f"message={exception_message}"
                )

            if traceback_lines:
                tail = (
                    str(
                        traceback_lines[-1]
                    )
                    .strip()
                )

                if tail:
                    parts.append(
                        f"trace={tail}"
                    )

            if parts:
                details.append(
                    "; ".join(
                        parts
                    )
                )

        except Exception:
            continue

    if details:
        return (
            "ComfyUI execution failed: "
            + " | ".join(
                details
            )
        )

    return (
        "ComfyUI execution failed; remote history "
        "did not contain a structured execution_error message."
    )


def _normalise_ocr_token(
    value: str,
) -> str:
    """Return alphabetic OCR content suitable for text validation."""

    return re.sub(
        r"[^A-Za-z]",
        "",
        str(
            value
            or ""
        ),
    )


def _ocr_confidence(
    value: Any,
) -> float:
    try:
        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return -1.0


def _similar_ocr_tokens(
    first: str,
    second: str,
) -> bool:
    """Return True when two meaningful OCR tokens substantially agree.

    Short OCR fragments are intentionally rejected before any equality or
    similarity checks because diffusion textures frequently generate tiny
    letter-like patterns such as:

        ti
        ii
        gl
        ht

    Only tokens with at least four alphabetic characters can participate in
    OCR confirmation.
    """

    first = (
        str(
            first
            or ""
        )
        .strip()
        .lower()
    )

    second = (
        str(
            second
            or ""
        )
        .strip()
        .lower()
    )

    #
    # The minimum-length guard MUST happen before exact equality.
    #
    # Without this:
    #
    #     _similar_ocr_tokens("ti", "ti")
    #
    # incorrectly returns True.
    #
    if (
        len(
            first
        )
        < 4
        or len(
            second
        )
        < 4
    ):
        return False

    if (
        first
        == second
    ):
        return True

    #
    # OCR may add or remove a character around the edge of a crop.
    #
    if (
        first in second
        or second in first
    ):
        return True

    #
    # Allow one OCR substitution/insertion/deletion for otherwise similar
    # tokens.
    #
    # Example:
    #
    #     title
    #     titie
    #
    if (
        abs(
            len(
                first
            )
            - len(
                second
            )
        )
        > 1
    ):
        return False

    mismatches = sum(
        1
        for left, right
        in zip(
            first,
            second,
        )
        if left
        != right
    )

    mismatches += abs(
        len(
            first
        )
        - len(
            second
        )
    )

    return (
        mismatches
        <= 1
    )


def _confirm_ocr_candidate(
    image: Image.Image,
    *,
    token: str,
    left: int,
    top: int,
    width: int,
    height: int,
) -> bool:
    """Confirm a first-pass OCR token using an enlarged local crop.

    Diffusion artwork frequently produces architectural lines, windows and
    decorative geometry that Tesseract interprets as short words.

    A genuine text region should survive a second OCR pass after cropping and
    enlargement. Random skyline geometry normally does not.
    """

    image_width, image_height = (
        image.size
    )

    #
    # Give the OCR engine context around the candidate.
    #
    # Padding scales with the candidate box but is always large enough to
    # include nearby glyphs.
    #
    pad_x = max(
        12,
        int(
            width
            * 1.5
        ),
    )

    pad_y = max(
        12,
        int(
            height
            * 1.0
        ),
    )

    x1 = max(
        0,
        left
        - pad_x,
    )

    y1 = max(
        0,
        top
        - pad_y,
    )

    x2 = min(
        image_width,
        left
        + width
        + pad_x,
    )

    y2 = min(
        image_height,
        top
        + height
        + pad_y,
    )

    crop = (
        image
        .crop(
            (
                x1,
                y1,
                x2,
                y2,
            )
        )
        .convert(
            "L"
        )
    )

    #
    # Enlargement makes real glyph edges easier to distinguish while noisy
    # image texture usually remains inconsistent.
    #
    crop = crop.resize(
        (
            max(
                1,
                crop.width
                * 3,
            ),
            max(
                1,
                crop.height
                * 3,
            ),
        ),
        Image.Resampling.LANCZOS,
    )

    #
    # Test both:
    #
    # PSM 7 -> one expected text line
    # PSM 6 -> compact text block
    #
    for psm in (
        7,
        6,
    ):
        data = (
            pytesseract.image_to_data(
                crop,
                config=(
                    f"--psm {psm}"
                ),
                output_type=(
                    pytesseract
                    .Output
                    .DICT
                ),
            )
        )

        for (
            confirmed_text,
            confirmed_confidence,
        ) in zip(
            data[
                "text"
            ],
            data[
                "conf"
            ],
        ):
            normalized = (
                _normalise_ocr_token(
                    confirmed_text
                )
            )

            score = (
                _ocr_confidence(
                    confirmed_confidence
                )
            )

            if (
                len(
                    normalized
                )
                < 4
                or score
                < 45
            ):
                continue

            if (
                _similar_ocr_tokens(
                    token,
                    normalized,
                )
            ):
                return True

    return False


def detected_text_tokens(
    image: Image.Image,
) -> list[str]:
    """Detect meaningful readable text while suppressing texture hallucination.

    First-pass OCR remains deliberately sensitive so that potential text is
    noticed.

    A moderate-confidence isolated token must then survive an independent OCR
    pass over an enlarged crop before the background is rejected.

    Very strong OCR detections are accepted immediately.
    """

    grey = (
        image.convert(
            "L"
        )
    )

    data = (
        pytesseract.image_to_data(
            grey,
            config="--psm 11",
            output_type=(
                pytesseract
                .Output
                .DICT
            ),
        )
    )

    confirmed_tokens: list[
        str
    ] = []

    for index, (
        raw_token,
        raw_confidence,
    ) in enumerate(
        zip(
            data[
                "text"
            ],
            data[
                "conf"
            ],
        )
    ):
        normalized = (
            _normalise_ocr_token(
                raw_token
            )
        )

        score = (
            _ocr_confidence(
                raw_confidence
            )
        )

        #
        # Ignore very short fragments entirely.
        #
        # These are extremely common around window frames, web geometry and
        # comic-book architectural details.
        #
        if (
            len(
                normalized
            )
            < 4
        ):
            continue

        #
        # Preserve the existing sensitive first-pass floor.
        #
        if (
            score
            < 45
        ):
            continue

        #
        # Strong longer detections are unlikely to be accidental.
        #
        # Example:
        #
        #     "BIRTHDAY" confidence 92
        #
        if (
            score
            >= 85
            and len(
                normalized
            )
            >= 5
        ):
            confirmed_tokens.append(
                normalized
            )

            continue

        left = int(
            data[
                "left"
            ][
                index
            ]
        )

        top = int(
            data[
                "top"
            ][
                index
            ]
        )

        width = int(
            data[
                "width"
            ][
                index
            ]
        )

        height = int(
            data[
                "height"
            ][
                index
            ]
        )

        #
        # Tiny boxes are particularly prone to false detections, but rather
        # than simply ignoring them, require independent confirmation.
        #
        if (
            _confirm_ocr_candidate(
                image,
                token=normalized,
                left=left,
                top=top,
                width=width,
                height=height,
            )
        ):
            confirmed_tokens.append(
                normalized
            )

    return confirmed_tokens


def build_workflow(
    engine_id: str,
    positive_prompt: str,
    negative_prompt: str,
    seed: int,
    output_prefix: str,
    compiled_prompt=None,
    sampling_profile: SamplingProfile | None = None,
) -> dict:
    template_path = (
        settings.comfyui_workflow_path
        / (
            f"{workflow_name(engine_id)}.json"
        )
    )

    template = json.loads(
        template_path.read_text(
            encoding="utf-8"
        )
    )

    values = (
        settings.model_dump()
    )

    def compact_prompt(value: str, limit: int) -> str:
        compact = " ".join(str(value or "").split())
        return compact if len(compact) <= limit else compact[:limit].rsplit(" ", 1)[0]

    # SD3.5 has three encoders with different jobs. CLIP-L receives a short
    # visual label, CLIP-G receives the composition contract, and T5 receives
    # a compact creative description. Negative conditioning is intentionally
    # empty for SD3.5: long negative prompts make this model collapse into
    # flat fields and raster artifacts.
    clip_l_positive = (
        compiled_prompt.clip_l if compiled_prompt and compiled_prompt.clip_l
        else compact_prompt(positive_prompt, 160)
    )
    clip_g_positive = (
        compiled_prompt.clip_g if compiled_prompt and compiled_prompt.clip_g
        else "portrait event background, edge-weighted composition, open title "
             "and programme zones, large matte graphic forms, requested palette"
    )
    sd35_positive = (
        compiled_prompt.t5 if compiled_prompt and compiled_prompt.t5
        else compact_prompt(positive_prompt, 520)
    )
    sd35_negative = compiled_prompt.negative if compiled_prompt else ""

    if engine_id == settings.engine_2_id:
        width = settings.sdxl_generation_width
        height = settings.sdxl_generation_height
    elif engine_id == settings.engine_3_id:
        width = settings.sd35_generation_width
        height = settings.sd35_generation_height
    else:
        width = settings.generation_width
        height = settings.generation_height

    profile = sampling_profile or select_sampling_profile(engine_id, 1, summarize_failures(None))
    values.update(
        positive_prompt=(
            positive_prompt
        ),
        negative_prompt=(
            negative_prompt
        ),
        clip_l_positive_prompt=clip_l_positive,
        clip_g_positive_prompt=clip_g_positive,
        t5_positive_prompt=sd35_positive,
        clip_l_negative_prompt=sd35_negative,
        clip_g_negative_prompt=sd35_negative,
        t5_negative_prompt=sd35_negative,
        sd35_shift=profile.shift if profile and profile.shift is not None else 3.0,
        steps=profile.steps if profile else 28,
        cfg=profile.cfg if profile else 6.0,
        sampler_name=profile.sampler_name if profile else "dpmpp_2m",
        scheduler=profile.scheduler if profile else "karras",
        seed=seed,
        width=width,
        height=height,
        output_prefix=(
            output_prefix
        ),
    )

    def inject(
        value: Any,
    ) -> Any:
        if (
            isinstance(
                value,
                str,
            )
            and value.startswith(
                "${"
            )
            and value.endswith(
                "}"
            )
        ):
            return values[
                value[2:-1]
            ]

        if isinstance(
            value,
            dict,
        ):
            return {
                key: inject(
                    item
                )
                for (
                    key,
                    item,
                )
                in value.items()
            }

        if isinstance(
            value,
            list,
        ):
            return [
                inject(
                    item
                )
                for item in value
            ]

        return value

    workflow = inject(template)
    if engine_id == settings.engine_3_id and profile and profile.shift is None:
        # The checkpoint's native sampling configuration is the official
        # SD3.5 path. Only recovery profiles explicitly override its shift.
        workflow.pop("55")
        workflow["3"]["inputs"]["model"] = ["4", 0]
    return workflow


def validate_workflow(
    workflow: dict,
    object_info: dict,
) -> None:
    for (
        node_id,
        node,
    ) in workflow.items():
        kind = (
            node[
                "class_type"
            ]
        )

        if kind not in object_info:
            raise ComfyUIError(
                "ComfyUI node "
                f"{kind} is unavailable "
                f"(node {node_id})."
            )

        schema = (
            object_info[
                kind
            ][
                "input"
            ]
        )

        inputs = (
            node[
                "inputs"
            ]
        )

        missing = (
            set(
                schema.get(
                    "required",
                    {},
                )
            )
            - inputs.keys()
        )

        if missing:
            raise ComfyUIError(
                f"Node {kind} missing "
                "required inputs: "
                f"{sorted(missing)}"
            )

        fields = {
            **schema.get(
                "required",
                {},
            ),
            **schema.get(
                "optional",
                {},
            ),
        }

        for (
            key,
            specification,
        ) in fields.items():
            if (
                key not in inputs
                or isinstance(
                    inputs[key],
                    list,
                )
            ):
                continue

            options = (
                specification[0]
                if isinstance(
                    specification[0],
                    list,
                )
                else (
                    specification[1].get(
                        "options"
                    )
                    if (
                        len(specification) > 1
                        and isinstance(
                            specification[1],
                            dict,
                        )
                    )
                    else None
                )
            )

            if (
                options is not None
                and inputs[key]
                not in options
            ):
                raise ComfyUIError(
                    "Configured "
                    f"{kind}.{key}="
                    f"{inputs[key]!r} "
                    "is unavailable on ComfyUI."
                )


def validate_background(
    path: Path,
    *,
    reference_number: str | None = None,
    expected_sha256: str | None = None,
    enforce_palette: bool = True,
) -> dict:
    data = (
        path.read_bytes()
    )

    if not data.startswith(
        b"\x89PNG\r\n\x1a\n"
    ):
        raise ValueError(
            "Generated background is not PNG."
        )

    digest = (
        hashlib.sha256(
            data
        ).hexdigest()
    )

    if (
        expected_sha256 is not None
        and digest
        != expected_sha256
    ):
        raise ValueError(
            "Generated background SHA-256 mismatch."
        )

    with Image.open(
        io.BytesIO(
            data
        )
    ) as image:
        image.verify()

    with Image.open(
        io.BytesIO(
            data
        )
    ) as image:
        image.load()

        image = image.convert(
            "RGB"
        )

        width, height = (
            image.size
        )

        tokens = (
            detected_text_tokens(
                image
            )
        )

        human_signals = (
            detect_human_signals(
                image
            )
        )

        quality_result = (
            validate_visual_quality(
                image
            )
        )

        palette_result = {
            "palette_checked": False,
        }

        if reference_number and enforce_palette:
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

            palette_result = (
                validate_palette(
                    image,
                    brief.get(
                        "primary_colour"
                    ),
                    brief.get(
                        "secondary_colour"
                    ),
                )
            )

    if (width, height) not in _valid_background_dimensions():
        raise ValueError(
            "Unexpected background dimensions: "
            f"{width}x{height}."
        )

    if tokens:
        raise ValueError(
            "Generated background contains OCR text: "
            + ", ".join(
                tokens[:8]
            )
        )

    if (
        human_signals[
            "detected"
        ]
    ):
        raise ValueError(
            "Generated background appears to contain "
            "a human figure or face."
        )

    return {
        "output_path": (
            str(path)
        ),
        "sha256": (
            digest
        ),
        "width": (
            width
        ),
        "height": (
            height
        ),
        "person_detection_count": (
            len(
                human_signals[
                    "people"
                ]
            )
        ),
        "face_detection_count": (
            len(
                human_signals[
                    "faces"
                ]
            )
        ),
        "profile_detection_count": (
            len(
                human_signals[
                    "profiles"
                ]
            )
        ),
        "upper_body_detection_count": (
            len(
                human_signals[
                    "upper_bodies"
                ]
            )
        ),
        **quality_result,
        **palette_result,
    }


class ComfyUIClient:
    def __init__(
        self,
    ):
        self.http = (
            httpx.AsyncClient(
                base_url=(
                    settings
                    .comfyui_base_url
                    .rstrip(
                        "/"
                    )
                ),
                timeout=(
                    httpx.Timeout(
                        60,
                        connect=(
                            settings
                            .comfyui_connect_timeout_seconds
                        ),
                    )
                ),
            )
        )

    async def __aenter__(
        self,
    ):
        return self

    async def __aexit__(
        self,
        *args,
    ):
        await (
            self.http
            .aclose()
        )

    async def request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> httpx.Response:
        try:
            response = (
                await self.http.request(
                    method,
                    path,
                    **kwargs,
                )
            )

            response.raise_for_status()

            return response

        except httpx.HTTPError as error:
            code = getattr(
                getattr(
                    error,
                    "response",
                    None,
                ),
                "status_code",
                None,
            )

            raise ComfyUIError(
                "ComfyUI "
                f"{method} {path} failed "
                f"(HTTP {code or 'unavailable'})."
            ) from error

    async def health(
        self,
    ) -> dict:
        stats = (
            await self.request(
                "GET",
                "/system_stats",
            )
        ).json()

        object_info = (
            await self.request(
                "GET",
                "/object_info",
            )
        ).json()

        for engine_id in (
            settings.engine_1_id,
            settings.engine_2_id,
            settings.engine_3_id,
        ):
            validate_workflow(
                build_workflow(
                    engine_id,
                    "readiness",
                    "",
                    0,
                    "readiness",
                ),
                object_info,
            )

        return {
            "status": "available",
            "version": (
                stats.get(
                    "system",
                    {},
                )
                .get(
                    "comfyui_version"
                )
            ),
            "engines": [
                settings.engine_1_id,
                settings.engine_2_id,
                settings.engine_3_id,
            ],
        }

    async def release_models(
        self,
    ) -> None:
        queue = (
            await self.request(
                "GET",
                "/queue",
            )
        ).json()

        if (
            queue.get(
                "queue_running"
            )
            or queue.get(
                "queue_pending"
            )
        ):
            raise ComfyUIError(
                "ComfyUI has active jobs; "
                "cannot release models safely."
            )

        await self.request(
            "POST",
            "/free",
            json={
                "unload_models": True,
                "free_memory": True,
            },
        )

        deadline = (
            time.monotonic()
            + 60
        )

        while True:
            stats = (
                await self.request(
                    "GET",
                    "/system_stats",
                )
            ).json()

            devices = (
                stats.get(
                    "devices",
                    [],
                )
            )

            if (
                devices
                and all(
                    (
                        device.get(
                            "torch_vram_total",
                            0,
                        )
                        <= (
                            64
                            * 1024
                            * 1024
                        )
                        and (
                            device.get(
                                "vram_total",
                                0,
                            )
                            - device.get(
                                "vram_free",
                                0,
                            )
                        )
                        < (
                            1024
                            * 1024
                            * 1024
                        )
                    )
                    for device
                    in devices
                )
            ):
                logger.info(
                    "event=comfyui_models_released"
                )

                return

            if (
                time.monotonic()
                >= deadline
            ):
                raise ComfyUIError(
                    "ComfyUI model unload was not "
                    "confirmed within 60 seconds."
                )

            await asyncio.sleep(
                1
            )

    async def recover_submission(
        self,
        request_id: str,
    ) -> str | None:
        queue = (
            await self.request(
                "GET",
                "/queue",
            )
        ).json()

        history = (
            await self.request(
                "GET",
                "/history",
                params={
                    "max_items": 1000,
                },
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

        entries += [
            entry["prompt"]
            for entry
            in history.values()
            if "prompt" in entry
        ]

        matches = {
            item[1]
            for item
            in entries
            if (
                len(item) > 3
                and item[3].get(
                    "programme_request_id"
                )
                == request_id
            )
        }

        if len(matches) > 1:
            raise SubmissionUncertain(
                "Multiple remote jobs match "
                "the submission."
            )

        return next(
            iter(matches),
            None,
        )

    async def generate_image(
        self,
        reference_number: str,
        engine_id: str,
        direction_id: str,
        positive_prompt: str,
        negative_prompt: str,
        compiled_prompt=None,
        spec_sha256: str | None = None,
        direction_role: str | None = None,
        retry_stage: str | None = None,
        failure_category: str | None = None,
        sampling_profile: SamplingProfile | None = None,
        seed_override: int | None = None,
    ) -> dict:
        workflow_name(
            engine_id
        )

        if (
            direction_id
            not in {
                "A",
                "B",
                "C",
            }
        ):
            raise ValueError(
                "Direction must be A, B or C."
            )

        directory = (
            get_job_directory(
                reference_number
            )
            / "backgrounds"
            / engine_id
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        record_path = (
            directory
            / (
                "image-"
                f"{direction_id.lower()}"
                ".json"
            )
        )

        image_path = (
            directory
            / (
                "image-"
                f"{direction_id.lower()}"
                ".png"
            )
        )

        previous = (
            json.loads(
                record_path.read_text(
                    encoding="utf-8"
                )
            )
            if record_path.exists()
            else None
        )

        if (
            previous and spec_sha256 and previous.get("spec_sha256")
            and previous["spec_sha256"] != spec_sha256
        ):
            raise ComfyUIError(
                "Candidate specification changed across attempts; "
                "preserve this job and use a new reference."
            )

        if (
            previous
            and previous.get("status") != "complete"
            and _attempt_budget_exhausted(previous)
        ):
            raise ComfyUIError(
                "Candidate attempt budget is exhausted; no additional "
                "ComfyUI submission will be made."
            )

        #
        # Reuse completed candidate only if it still validates.
        #
        if (
            previous
            and previous.get(
                "status"
            )
            == "complete"
        ):
            try:
                verified = (
                    validate_background(
                        image_path,
                        reference_number=(
                            reference_number
                        ),
                        expected_sha256=(
                            previous[
                                "sha256"
                            ]
                        ),
                    )
                )

                return {
                    **previous,
                    **verified,
                    "reused": True,
                }

            except (
                OSError,
                ValueError,
            ):
                logger.warning(
                    "event=background_invalid "
                    "reference=%s "
                    "engine=%s "
                    "direction=%s",
                    reference_number,
                    engine_id,
                    direction_id,
                )

        retrying_rejected_candidate = bool(
            previous
            and previous.get(
                "rejection_reason"
            )
            and previous.get(
                "status"
            )
            == "rejected"
        )

        retrying_failed_execution = bool(
            previous
            and previous.get(
                "status"
            )
            in {
                "failed",
                "runtime_failed",
            }
        )

        retrying_candidate = (
            retrying_rejected_candidate
            or retrying_failed_execution
        )

        #
        # Every rejected/runtime-failed attempt gets a fresh seed.
        #
        if seed_override is not None:
            if not 0 <= seed_override < 2**63:
                raise ValueError("Probe seed must fit the supported signed 63-bit range")
            seed = seed_override
        elif retrying_candidate:
            seed = (
                secrets.randbits(
                    63
                )
            )

        elif previous:
            seed = (
                previous[
                    "seed"
                ]
            )

        else:
            seed = (
                secrets.randbits(
                    63
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

        actual_positive_prompt, actual_negative_prompt = _transport_prompts(
            engine_id, positive_prompt, negative_prompt,
            primary_colour, secondary_colour, compiled_prompt,
        )

        selected_profile = sampling_profile or select_sampling_profile(
            engine_id,
            int(previous.get("attempt_count", 0)) + 1 if retrying_candidate else 1,
            summarize_failures(previous),
        )

        workflow = (
            build_workflow(
                engine_id,
                actual_positive_prompt,
                actual_negative_prompt,
                seed,
                (
                    "programme-studio/"
                    f"{reference_number}/"
                    f"{engine_id}/"
                    f"{direction_id.lower()}"
                ),
                compiled_prompt=compiled_prompt,
                sampling_profile=selected_profile,
            )
        )

        fingerprint = (
            hashlib.sha256(
                json.dumps(
                    workflow,
                    sort_keys=True,
                ).encode(
                    "utf-8"
                )
            ).hexdigest()
        )

        if (
            previous
            and not retrying_candidate
            and previous.get(
                "workflow_sha256"
            )
            and previous[
                "workflow_sha256"
            ]
            != fingerprint
        ):
            raise ComfyUIError(
                "Frozen workflow differs from stored "
                "candidate; use a new reference."
            )

        rejected_attempts = (
            list(
                previous.get(
                    "rejected_attempts",
                    [],
                )
            )
            if previous
            else []
        )

        failed_attempts = (
            list(
                previous.get(
                    "failed_attempts",
                    [],
                )
            )
            if previous
            else []
        )

        if (
            retrying_rejected_candidate
            and previous
        ):
            rejected_attempts.append(
                {
                    "attempt_count": (
                        previous.get(
                            "attempt_count",
                            1,
                        )
                    ),
                    "seed": (
                        previous.get(
                            "seed"
                        )
                    ),
                    "prompt_id": (
                        previous.get(
                            "prompt_id"
                        )
                    ),
                    "sha256": (
                        previous.get(
                            "sha256"
                        )
                    ),
                    "rejection_reason": (
                        previous.get(
                            "rejection_reason"
                        )
                    ),
                    "failure_category": previous.get("failure_category"),
                    "retry_stage": previous.get("retry_stage"),
                    "spec_sha256": previous.get("spec_sha256"),
                    "direction_role": previous.get("direction_role"),
                    "compiled_prompt": previous.get("compiled_prompt"),
                    "sampling_profile": previous.get("sampling_profile"),
                    "outcome_category": previous.get("outcome_category"),
                }
            )

        if (
            retrying_failed_execution
            and previous
        ):
            failed_attempts.append(
                {
                    "attempt_count": (
                        previous.get(
                            "attempt_count",
                            1,
                        )
                    ),
                    "seed": (
                        previous.get(
                            "seed"
                        )
                    ),
                    "prompt_id": (
                        previous.get(
                            "prompt_id"
                        )
                    ),
                    "error": (
                        previous.get(
                            "error"
                        )
                    ),
                    "remote_status": (
                        previous.get(
                            "remote_status"
                        )
                    ),
                    "failure_category": previous.get("failure_category"),
                    "retry_stage": previous.get("retry_stage"),
                    "spec_sha256": previous.get("spec_sha256"),
                    "direction_role": previous.get("direction_role"),
                    "compiled_prompt": previous.get("compiled_prompt"),
                    "sampling_profile": previous.get("sampling_profile"),
                    "outcome_category": previous.get("outcome_category"),
                }
            )

        record = {
            "reference_number": (
                reference_number
            ),
            "engine_id": (
                engine_id
            ),
            "direction_id": (
                direction_id
            ),
            "seed": (
                seed
            ),
            "workflow_sha256": (
                fingerprint
            ),
            "request_id": (
                str(
                    uuid.uuid4()
                )
            ),
            "status": (
                "prepared"
            ),
            "attempt_count": (
                (
                    previous.get(
                        "attempt_count",
                        1,
                    )
                    + 1
                )
                if (
                    retrying_candidate
                    and previous
                )
                else 1
            ),
            "positive_prompt": (
                actual_positive_prompt
            ),
            "negative_prompt": (
                actual_negative_prompt
                if (
                    engine_id
                    != settings.engine_1_id
                )
                else None
            ),
            "adaptations": (
                [
                    (
                        "Negative prompt omitted because "
                        "the configured FLUX workflow does "
                        "not expose negative conditioning."
                    )
                ]
                if (
                    engine_id
                    == settings.engine_1_id
                )
                else []
            ),
            "workflow": (
                workflow
            ),
            "spec_sha256": spec_sha256,
            "direction_role": direction_role,
            "retry_stage": compiled_prompt.retry_stage if compiled_prompt else retry_stage,
            "failure_category": compiled_prompt.failure_category if compiled_prompt else failure_category,
            "compiled_prompt": compiled_prompt.record() if compiled_prompt else {
                "positive": actual_positive_prompt,
                "negative": actual_negative_prompt,
                "clip_l": None,
                "clip_g": None,
                "t5": None,
            },
            "sampling_profile": selected_profile.record() if selected_profile else None,
        }

        if rejected_attempts:
            record[
                "rejected_attempts"
            ] = (
                rejected_attempts
            )

        if failed_attempts:
            record[
                "failed_attempts"
            ] = (
                failed_attempts
            )

        if (
            previous
            and not retrying_candidate
        ):
            record = previous

        prompt_id = (
            record.get(
                "prompt_id"
            )
        )

        if (
            not prompt_id
            and previous
            and not retrying_candidate
        ):
            prompt_id = (
                await self.recover_submission(
                    record[
                        "request_id"
                    ]
                )
            )

            if (
                prompt_id is None
            ):
                raise SubmissionUncertain(
                    "Submission outcome is unknown; "
                    "inspect ComfyUI history."
                )

        if not prompt_id:
            object_info = (
                await self.request(
                    "GET",
                    "/object_info",
                )
            ).json()

            validate_workflow(
                workflow,
                object_info,
            )

            queue = (
                await self.request(
                    "GET",
                    "/queue",
                )
            ).json()

            if (
                queue.get(
                    "queue_running"
                )
                or queue.get(
                    "queue_pending"
                )
            ):
                raise ComfyUIError(
                    "ComfyUI is busy; retry after "
                    "its current queue finishes."
                )

            write_json(
                record_path,
                record,
            )

            result = (
                await self.request(
                    "POST",
                    "/prompt",
                    json={
                        "prompt": (
                            workflow
                        ),
                        "client_id": (
                            record[
                                "request_id"
                            ]
                        ),
                        "extra_data": {
                            "programme_request_id": (
                                record[
                                    "request_id"
                                ]
                            )
                        },
                    },
                )
            ).json()

            if (
                result.get(
                    "node_errors"
                )
                or not result.get(
                    "prompt_id"
                )
            ):
                node_errors = (
                    result.get(
                        "node_errors"
                    )
                    or {}
                )

                message = (
                    "ComfyUI rejected the workflow."
                )

                if node_errors:
                    message += (
                        " "
                        + json.dumps(
                            node_errors,
                            ensure_ascii=False,
                        )[:1500]
                    )

                record.update(
                    status=(
                        "runtime_failed"
                    ),
                    error=(
                        message
                    ),
                    rejection_reason=(
                        message
                    ),
                )

                write_json(
                    record_path,
                    record,
                )

                raise ComfyUIError(
                    message
                )

            prompt_id = (
                result[
                    "prompt_id"
                ]
            )

        record.update(
            prompt_id=(
                prompt_id
            ),
            status=(
                "submitted"
            ),
        )

        write_json(
            record_path,
            record,
        )

        logger.info(
            "event=comfyui_submitted "
            "reference=%s "
            "engine=%s "
            "direction=%s "
            "attempt=%s "
            "seed=%s "
            "prompt_id=%s",
            reference_number,
            engine_id,
            direction_id,
            record.get(
                "attempt_count"
            ),
            seed,
            prompt_id,
        )

        deadline = (
            time.monotonic()
            + settings
            .comfyui_generation_timeout_seconds
        )

        history = None

        while True:
            history = (
                await self.request(
                    "GET",
                    (
                        f"/history/"
                        f"{prompt_id}"
                    ),
                )
            ).json().get(
                prompt_id
            )

            if history:
                status = (
                    history.get(
                        "status",
                        {},
                    )
                    or {}
                )

                if (
                    status.get(
                        "status_str"
                    )
                    == "error"
                ):
                    error_message = (
                        _extract_comfyui_execution_error(
                            history
                        )
                    )

                    record.update(
                        status=(
                            "runtime_failed"
                        ),
                        error=(
                            error_message
                        ),
                        rejection_reason=(
                            error_message
                        ),
                        remote_status=(
                            status
                        ),
                    )

                    write_json(
                        record_path,
                        record,
                    )

                    logger.warning(
                        "event=comfyui_execution_failed "
                        "reference=%s "
                        "engine=%s "
                        "direction=%s "
                        "attempt=%s "
                        "seed=%s "
                        "error=%s",
                        reference_number,
                        engine_id,
                        direction_id,
                        record.get(
                            "attempt_count"
                        ),
                        seed,
                        error_message,
                    )

                    raise ComfyUIError(
                        error_message
                    )

                if (
                    status.get(
                        "completed"
                    )
                ):
                    break

            if (
                time.monotonic()
                >= deadline
            ):
                message = (
                    "ComfyUI generation timed out."
                )

                record.update(
                    status=(
                        "runtime_failed"
                    ),
                    error=(
                        message
                    ),
                    rejection_reason=(
                        message
                    ),
                )

                write_json(
                    record_path,
                    record,
                )

                raise ComfyUIError(
                    message
                )

            await asyncio.sleep(
                settings
                .comfyui_poll_interval_seconds
            )

        save_nodes = [
            node_id
            for (
                node_id,
                item,
            ) in workflow.items()
            if (
                item[
                    "class_type"
                ]
                == "SaveImage"
            )
        ]

        images = [
            image
            for node_id
            in save_nodes
            for image
            in (
                history.get(
                    "outputs",
                    {},
                )
                .get(
                    node_id,
                    {},
                )
                .get(
                    "images",
                    [],
                )
            )
        ]

        if (
            len(images) != 1
            or images[0].get(
                "type"
            )
            != "output"
        ):
            message = (
                "Expected exactly one persisted "
                "ComfyUI output image."
            )

            record.update(
                status=(
                    "runtime_failed"
                ),
                error=(
                    message
                ),
                rejection_reason=(
                    message
                ),
            )

            write_json(
                record_path,
                record,
            )

            raise ComfyUIError(
                message
            )

        descriptor = (
            images[0]
        )

        response = (
            await self.request(
                "GET",
                "/view",
                params={
                    "filename": (
                        descriptor[
                            "filename"
                        ]
                    ),
                    "subfolder": (
                        descriptor.get(
                            "subfolder",
                            "",
                        )
                    ),
                    "type": "output",
                },
            )
        )

        candidate_path = (
            image_path.with_suffix(
                ".download.part"
            )
        )

        write_bytes(
            candidate_path,
            response.content,
        )

        raw_validation = None
        normalized_path = image_path.with_suffix(".normalized.part.png")
        palette_normalized = False
        raw_palette_error = None

        try:
            try:
                raw_validation = validate_background(
                    candidate_path,
                    reference_number=reference_number,
                )
                validation = raw_validation

            except ValueError as error:
                raw_palette_error = (
                    str(
                        error
                    )
                )

                # ====================================================
                # Engine-specific deterministic palette-only recovery
                # ====================================================
                #
                # This path is eligible ONLY when:
                #
                # - the validator specifically reported bounded
                #   off-palette drift;
                # - a requested primary colour exists;
                # - the engine is SDXL or SD3.5;
                # - SDXL has reached its existing minimum attempt
                #   threshold;
                # - SD3.5 is within its tighter 0.15 recovery ceiling.
                #
                # _palette_only_recovery() first executes complete
                # non-palette validation on the raw candidate.
                #
                # Therefore raster-corrupt, human-containing,
                # text-containing, flat, noisy or otherwise structurally
                # invalid candidates cannot be rescued merely by recolouring.
                #
                # The recoloured result then goes through the normal FULL
                # validate_background() path again, so final palette
                # acceptance remains strict.
                # ====================================================

                palette_recovery_engine = bool(
                    engine_id
                    in {
                        settings.engine_2_id,
                        settings.engine_3_id,
                    }
                )

                sdxl_recovery_ready = bool(
                    engine_id
                    == settings.engine_2_id
                    and record[
                        "attempt_count"
                    ]
                    >= 3
                )

                sd35_recovery_ready = bool(
                    engine_id
                    == settings.engine_3_id
                )

                palette_recovery_ready = bool(
                    palette_recovery_engine
                    and (
                        sdxl_recovery_ready
                        or sd35_recovery_ready
                    )
                    and _recoverable_palette_error(
                        raw_palette_error,
                        engine_id,
                    )
                    and primary_colour
                )

                if (
                    palette_recovery_ready
                ):
                    try:
                        (
                            raw_validation,
                            validation,
                        ) = (
                            _palette_only_recovery(
                                candidate_path,
                                normalized_path,
                                reference_number,
                                primary_colour,
                                secondary_colour,
                            )
                        )

                        palette_normalized = (
                            True
                        )

                        logger.info(
                            "event=palette_normalization_succeeded "
                            "reference=%s "
                            "engine=%s "
                            "direction=%s "
                            "attempt=%s "
                            "raw_error=%s",
                            reference_number,
                            engine_id,
                            direction_id,
                            record.get(
                                "attempt_count"
                            ),
                            raw_palette_error,
                        )

                    except ValueError as recovery_error:
                        logger.info(
                            "event=palette_normalization_rejected "
                            "reference=%s "
                            "engine=%s "
                            "direction=%s "
                            "attempt=%s "
                            "raw_error=%s "
                            "recovery_error=%s",
                            reference_number,
                            engine_id,
                            direction_id,
                            record.get(
                                "attempt_count"
                            ),
                            raw_palette_error,
                            str(
                                recovery_error
                            ),
                        )

                if not (
                    palette_normalized
                ):
                    record.update(
                        status=(
                            "rejected"
                        ),
                        rejection_reason=(
                            str(
                                error
                            )
                        ),
                        outcome_category=(
                            classify_failure(
                                str(
                                    error
                                )
                            )
                        ),
                        error=None,
                        remote_image=(
                            descriptor
                        ),
                    )

                    write_json(
                        record_path,
                        record,
                    )

                    raise ComfyUIError(
                        str(
                            error
                        )
                    ) from error

            if (
                previous
                and not retrying_candidate
                and previous.get(
                    "sha256"
                )
                and validation[
                    "sha256"
                ]
                != previous[
                    "sha256"
                ]
            ):
                raise ComfyUIError(
                    "Remote output changed since "
                    "the original generation."
                )

            if palette_normalized:
                raw_image_path = image_path.with_suffix(".raw.png")
                candidate_path.replace(raw_image_path)
                normalized_path.replace(candidate_path)
            candidate_path.replace(
                image_path
            )

        finally:
            candidate_path.unlink(
                missing_ok=True
            )
            normalized_path.unlink(missing_ok=True)

        record.update(
            validation,
            raw_validation=raw_validation,
            final_validation=validation,
            palette_normalized=palette_normalized,
            raw_palette_error=raw_palette_error if palette_normalized else None,
            outcome_category="PASS",
            raw_image_path=str(image_path.with_suffix(".raw.png")) if palette_normalized else None,
            status=(
                "complete"
            ),
            error=None,
            rejection_reason=None,
            output_path=(
                str(image_path)
            ),
            remote_image=(
                descriptor
            ),
        )

        write_json(
            record_path,
            record,
        )

        logger.info(
            "event=background_verified "
            "reference=%s "
            "engine=%s "
            "direction=%s "
            "sha256=%s "
            "palette_mode=%s "
            "palette_ratio=%s",
            reference_number,
            engine_id,
            direction_id,
            record[
                "sha256"
            ],
            record.get(
                "palette_mode"
            ),
            record.get(
                "palette_match_ratio"
            ),
        )

        return {
            **record,
            "reused": False,
        }


async def get_runtime_health() -> dict:
    async with ComfyUIClient() as client:
        return (
            await client.health()
        )


async def generate_remote_image(
    reference_number: str,
    engine_id: str,
    direction_id: str,
    positive_prompt: str,
    negative_prompt: str,
    compiled_prompt=None,
    spec_sha256: str | None = None,
    direction_role: str | None = None,
    retry_stage: str | None = None,
    failure_category: str | None = None,
    sampling_profile: SamplingProfile | None = None,
) -> dict:
    async with ComfyUIClient() as client:
        return (
            await client.generate_image(
                reference_number,
                engine_id,
                direction_id,
                positive_prompt,
                negative_prompt,
                compiled_prompt=compiled_prompt,
                spec_sha256=spec_sha256,
                direction_role=direction_role,
                retry_stage=retry_stage,
                failure_category=failure_category,
                sampling_profile=sampling_profile,
            )
        )