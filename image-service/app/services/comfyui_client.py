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

from PIL import Image

from app.config.settings import settings

from app.services.atomic import (
    write_bytes,
    write_json,
)

from app.services.background_policy import (
    BACKGROUND_ONLY_NEGATIVE,
    build_engine_prompt,
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


logger = logging.getLogger(
    "uvicorn.error"
)


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


def detected_text_tokens(
    image: Image.Image,
) -> list[str]:
    data = (
        pytesseract.image_to_data(
            image.convert(
                "L"
            ),
            config="--psm 11",
            output_type=(
                pytesseract.Output.DICT
            ),
        )
    )

    tokens = []

    for (
        token,
        confidence,
    ) in zip(
        data["text"],
        data["conf"],
    ):
        normalized = re.sub(
            r"[^A-Za-z]",
            "",
            token,
        )

        try:
            score = float(
                confidence
            )

        except (
            TypeError,
            ValueError,
        ):
            score = -1

        if (
            len(normalized) >= 4
            and score >= 45
        ):
            tokens.append(
                normalized
            )

    return tokens


def build_workflow(
    engine_id: str,
    positive_prompt: str,
    negative_prompt: str,
    seed: int,
    output_prefix: str,
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
    clip_l_positive = compact_prompt(positive_prompt, 160)
    clip_g_positive = (
        "portrait event background, edge-weighted composition, open title "
        "and programme zones, large matte graphic forms, requested palette"
    )
    sd35_positive = compact_prompt(positive_prompt, 520)

    if engine_id == settings.engine_2_id:
        width = settings.sdxl_generation_width
        height = settings.sdxl_generation_height
    elif engine_id == settings.engine_3_id:
        width = settings.sd35_generation_width
        height = settings.sd35_generation_height
    else:
        width = settings.generation_width
        height = settings.generation_height

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
        clip_l_negative_prompt="",
        clip_g_negative_prompt="",
        t5_negative_prompt="",
        sd35_shift=3.0,
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

    return inject(
        template
    )


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
        if retrying_candidate:
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

        actual_positive_prompt = _engine_positive_prompt(
            engine_id,
            positive_prompt,
            primary_colour,
            secondary_colour,
        )

        actual_negative_prompt = (
            ", ".join(
                value
                for value
                in (
                    BACKGROUND_ONLY_NEGATIVE,
                    negative_prompt,
                    palette_negative_contract(
                        primary_colour,
                        secondary_colour,
                    ),
                )
                if value
            )
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

        try:
            try:
                # Preserve what the model generated. A candidate must meet the
                # quality, composition and requested-palette rules directly;
                # a retry is preferable to painting over native artwork.
                raw_validation = validate_background(
                    candidate_path,
                    reference_number=reference_number,
                )
                validation = raw_validation

            except ValueError as error:
                record.update(
                    status=(
                        "rejected"
                    ),
                    rejection_reason=(
                        str(error)
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
                    str(error)
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

            candidate_path.replace(
                image_path
            )

        finally:
            candidate_path.unlink(
                missing_ok=True
            )

        record.update(
            validation,
            raw_validation=raw_validation,
            final_validation=validation,
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
) -> dict:
    async with ComfyUIClient() as client:
        return (
            await client.generate_image(
                reference_number,
                engine_id,
                direction_id,
                positive_prompt,
                negative_prompt,
            )
        )
