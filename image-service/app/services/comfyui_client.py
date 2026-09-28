"""ComfyUI HTTP adapter and background-candidate QA."""

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

import cv2
import httpx
import numpy as np
import pytesseract

from PIL import Image

from app.config.settings import settings
from app.services.atomic import (
    write_bytes,
    write_json,
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


BACKGROUND_ONLY_SUFFIX = (
    "A4 portrait decorative background only. "
    "Pure abstract nonrepresentational artwork. "
    "No central subject. "
    "No focal character. "
    "No human silhouette. "
    "No body-shaped form. "
    "No person-like object. "
    "No face-like focal object. "
    "No portrait composition. "
    "Absolutely no people, human figures, faces, portraits, "
    "characters, mannequins, silhouettes, heads, bodies, "
    "hands, arms, legs, or clothing. "
    "Absolutely no words, letters, typography, logos, "
    "signatures, watermarks, labels, numbers, or readable characters. "
    "Do not create a poster, invitation, certificate, menu, card, "
    "document, signage, or framed information panel. "
    "Use abstract materials, geometric ornament, gradients, "
    "textures, lines, shapes, lighting, and pattern only. "
    "Leave the reserved title and programme areas visually quiet."
)


BACKGROUND_ONLY_NEGATIVE = (
    "(person:2.0), "
    "(people:2.0), "
    "(human:2.0), "
    "(human figure:2.0), "
    "(man:2.0), "
    "(woman:2.0), "
    "(child:2.0), "
    "(face:2.0), "
    "(portrait:2.0), "
    "(body:2.0), "
    "(silhouette:2.0), "
    "(character:2.0), "
    "(head:1.8), "
    "(hands:1.8), "
    "(arms:1.8), "
    "(legs:1.8), "
    "(clothing:1.8), "
    "(text:2.0), "
    "(words:2.0), "
    "(letters:2.0), "
    "(typography:2.0), "
    "(writing:2.0), "
    "(calligraphy:1.8), "
    "(signature:1.8), "
    "(logo:1.8), "
    "(watermark:1.8), "
    "(poster:2.0), "
    "(card:2.0), "
    "(document:2.0), "
    "(menu:2.0), "
    "(certificate:2.0), "
    "(signage:2.0), "
    "numbers, labels, invitation"
)


ABSTRACT_BACKGROUND_PREFIX = (
    "Abstract nonrepresentational event background, "
    "decorative surface design only, "
    "no subject, no character, no figure, "
    "no photographic scene, no narrative scene, "
    "no foreground object, generous negative space, "
)


SDXL_CONTENT_CUES = re.compile(
    (
        r"\b(?:programme|program|ceremony|event|award|title|"
        r"information|hierarchy|layout|page|invitation|cover)\b"
    ),
    re.IGNORECASE,
)


FORBIDDEN_BACKGROUND_PROMPT_TERMS = (
    "typography",
    "lettering",
    "watermark",
    "readable text",
    "written text",
    "text",
    "words",
    "writing",
    "person",
    "people",
    "human",
    "face",
    "figure",
    "portrait",
)


NAMED_COLOURS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (220, 45, 45),
    "blue": (45, 95, 220),
    "green": (45, 150, 80),
    "pink": (235, 115, 170),
    "purple": (125, 75, 185),
    "violet": (120, 80, 185),
    "orange": (230, 130, 45),
    "yellow": (230, 195, 45),
    "gold": (195, 150, 50),
    "golden": (195, 150, 50),
    "silver": (175, 180, 190),
    "grey": (125, 125, 125),
    "gray": (125, 125, 125),
    "navy": (25, 45, 100),
    "teal": (35, 135, 135),
    "cyan": (55, 170, 190),
    "brown": (120, 80, 50),
    "beige": (210, 190, 150),
    "cream": (240, 225, 190),
    "maroon": (115, 30, 50),
    "burgundy": (120, 35, 60),
}


class ComfyUIError(
    RuntimeError
):
    """Unavailable dependency, invalid workflow, or failed remote execution."""


class SubmissionUncertain(
    ComfyUIError
):
    """The server may have accepted a request; never blindly resubmit it."""


def workflow_name(
    engine_id: str,
) -> str:
    names = {
        settings.engine_1_id: (
            "flux2"
        ),
        settings.engine_2_id: (
            "sdxl"
        ),
        settings.engine_3_id: (
            "sd35-medium"
        ),
    }

    if engine_id not in names:
        raise ValueError(
            f"Unknown engine: {engine_id}"
        )

    return names[
        engine_id
    ]


def sanitize_background_prompt(
    positive_prompt: str,
) -> str:
    clauses = re.split(
        r"[,;.!?]+",
        positive_prompt,
    )

    kept = [
        clause.strip()
        for clause in clauses
        if clause.strip()
        and not any(
            re.search(
                rf"\b{re.escape(term)}\b",
                clause,
                re.IGNORECASE,
            )
            for term in (
                FORBIDDEN_BACKGROUND_PROMPT_TERMS
            )
        )
    ]

    sanitized = (
        ", ".join(
            kept
        )
    )

    return (
        sanitized
        or (
            "abstract nonrepresentational background "
            "with generous open space"
        )
    )


def build_background_only_prompt(
    positive_prompt: str,
) -> str:
    sanitized = (
        sanitize_background_prompt(
            positive_prompt
        )
    )

    return (
        ABSTRACT_BACKGROUND_PREFIX
        + sanitized.rstrip(
            ". "
        )
        + ". "
        + BACKGROUND_ONLY_SUFFIX
    )


def build_engine_prompt(
    engine_id: str,
    positive_prompt: str,
) -> str:
    """Give every engine an abstract-only prompt.

    FLUX does not have a negative-conditioning input in the configured
    workflow, so its positive prompt must carry the full safety constraint.
    """

    sanitized = (
        sanitize_background_prompt(
            positive_prompt
        )
    )

    visual_sentences = [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            sanitized,
        )
        if (
            sentence.strip()
            and not SDXL_CONTENT_CUES.search(
                sentence
            )
        )
    ]

    visual_detail = (
        " ".join(
            visual_sentences
        )
    )

    if not visual_detail:
        visual_detail = (
            "refined abstract material texture"
        )

    return (
        ABSTRACT_BACKGROUND_PREFIX
        + visual_detail.rstrip(
            ". "
        )
        + ". "
        + BACKGROUND_ONLY_SUFFIX
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
            len(
                normalized
            )
            >= 4
            and score >= 45
        ):
            tokens.append(
                normalized
            )

    return tokens


def detect_people(
    image: Image.Image,
) -> list[
    tuple[
        int,
        int,
        int,
        int,
    ]
]:
    """Detect obvious full/upper-body human figures using OpenCV HOG.

    This is intentionally conservative. It is used as a second safety net
    after the diffusion prompt, not as the only person-avoidance mechanism.
    """

    rgb = np.array(
        image.convert(
            "RGB"
        )
    )

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )

    maximum_dimension = max(
        bgr.shape[
            :2
        ]
    )

    if maximum_dimension > 1280:
        scale = (
            1280
            / maximum_dimension
        )

        bgr = cv2.resize(
            bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=(
                cv2.INTER_AREA
            ),
        )

    hog = cv2.HOGDescriptor()

    hog.setSVMDetector(
        cv2.HOGDescriptor_getDefaultPeopleDetector()
    )

    rectangles, weights = (
        hog.detectMultiScale(
            bgr,
            winStride=(
                8,
                8,
            ),
            padding=(
                16,
                16,
            ),
            scale=1.05,
        )
    )

    detected = []

    for (
        rectangle,
        weight,
    ) in zip(
        rectangles,
        weights,
    ):
        score = float(
            weight
        )

        if score < 0.45:
            continue

        x, y, width, height = [
            int(value)
            for value in rectangle
        ]

        detected.append(
            (
                x,
                y,
                width,
                height,
            )
        )

    return detected


def parse_colour(
    value: str | None,
) -> tuple[
    int,
    int,
    int,
] | None:
    if not value:
        return None

    normalized = (
        value
        .strip()
        .lower()
    )

    if normalized in (
        NAMED_COLOURS
    ):
        return (
            NAMED_COLOURS[
                normalized
            ]
        )

    hexadecimal = (
        re.fullmatch(
            r"#?([0-9a-f]{6})",
            normalized,
        )
    )

    if hexadecimal:
        raw = (
            hexadecimal.group(
                1
            )
        )

        return (
            int(
                raw[0:2],
                16,
            ),
            int(
                raw[2:4],
                16,
            ),
            int(
                raw[4:6],
                16,
            ),
        )

    for (
        name,
        rgb,
    ) in NAMED_COLOURS.items():
        if re.search(
            rf"\b{re.escape(name)}\b",
            normalized,
        ):
            return rgb

    return None


def colour_distance(
    first: tuple[
        int,
        int,
        int,
    ],
    second: tuple[
        int,
        int,
        int,
    ],
) -> float:
    return float(
        np.linalg.norm(
            np.array(
                first,
                dtype=np.float32,
            )
            - np.array(
                second,
                dtype=np.float32,
            )
        )
    )


def validate_palette(
    image: Image.Image,
    primary_colour: str | None,
    secondary_colour: str | None,
) -> dict:
    """Reject candidates that materially leave the requested palette.

    Black/white is treated more strictly because it is explicitly
    monochromatic. Other palettes allow neutral values and variations
    of the requested colours.
    """

    primary = parse_colour(
        primary_colour
    )

    secondary = parse_colour(
        secondary_colour
    )

    palette = [
        colour
        for colour in (
            primary,
            secondary,
        )
        if colour is not None
    ]

    if not palette:
        return {
            "palette_checked": (
                False
            ),
        }

    sample = (
        image
        .convert(
            "RGB"
        )
        .resize(
            (
                96,
                132,
            ),
            Image.Resampling.LANCZOS,
        )
    )

    pixels = np.array(
        sample,
        dtype=np.int16,
    ).reshape(
        -1,
        3,
    )

    normalized_names = {
        (
            primary_colour
            or ""
        )
        .strip()
        .lower(),
        (
            secondary_colour
            or ""
        )
        .strip()
        .lower(),
    }

    monochrome_requested = (
        "black"
        in normalized_names
        and "white"
        in normalized_names
    )

    if monochrome_requested:
        chroma = (
            pixels.max(
                axis=1
            )
            - pixels.min(
                axis=1
            )
        )

        monochrome_ratio = float(
            np.mean(
                chroma <= 28
            )
        )

        if monochrome_ratio < 0.90:
            raise ValueError(
                "Generated background left the requested "
                "black-and-white palette "
                f"(monochrome ratio {monochrome_ratio:.2f})."
            )

        return {
            "palette_checked": (
                True
            ),
            "palette_mode": (
                "black-white"
            ),
            "palette_match_ratio": (
                monochrome_ratio
            ),
        }

    neutrals = [
        (
            0,
            0,
            0,
        ),
        (
            255,
            255,
            255,
        ),
        (
            128,
            128,
            128,
        ),
    ]

    allowed = (
        palette
        + neutrals
    )

    matched = 0

    for pixel in pixels:
        current = tuple(
            int(value)
            for value in pixel
        )

        minimum_distance = min(
            colour_distance(
                current,
                candidate,
            )
            for candidate in allowed
        )

        if minimum_distance <= 95:
            matched += 1

    ratio = (
        matched
        / len(
            pixels
        )
    )

    if ratio < 0.72:
        raise ValueError(
            "Generated background does not sufficiently "
            "match the requested primary/secondary palette "
            f"(match ratio {ratio:.2f})."
        )

    return {
        "palette_checked": True,
        "palette_mode": "theme",
        "palette_match_ratio": ratio,
    }


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
            f"{workflow_name(engine_id)}"
            ".json"
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

    values.update(
        positive_prompt=(
            positive_prompt
        ),
        negative_prompt=(
            negative_prompt
        ),
        seed=seed,
        width=(
            settings.generation_width
        ),
        height=(
            settings.generation_height
        ),
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
                value[
                    2:-1
                ]
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
                ) in value.items()
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

        if kind not in (
            object_info
        ):
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

        all_fields = {
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
        ) in all_fields.items():
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
                        len(
                            specification
                        )
                        > 1
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
                and inputs[
                    key
                ]
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
        expected_sha256
        is not None
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

        people = (
            detect_people(
                image
            )
        )

        palette_result = {
            "palette_checked": (
                False
            ),
        }

        if reference_number:
            document = (
                load_prompts_document(
                    reference_number
                )
            )

            brief = (
                document.get(
                    "brief",
                    {}
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

    if (
        width,
        height,
    ) != (
        settings.generation_width,
        settings.generation_height,
    ):
        raise ValueError(
            "Unexpected background dimensions: "
            f"{width}x{height}."
        )

    if tokens:
        raise ValueError(
            "Generated background contains "
            "OCR text: "
            + ", ".join(
                tokens[:8]
            )
        )

    if people:
        raise ValueError(
            "Generated background appears to contain "
            "a human figure."
        )

    return {
        "output_path": (
            str(path)
        ),
        "sha256": digest,
        "width": width,
        "height": height,
        "person_detection_count": (
            len(
                people
            )
        ),
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

        info = (
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
                info,
            )

        return {
            "status": "available",
            "version": (
                stats.get(
                    "system",
                    {}
                ).get(
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
                "unload_models": (
                    True
                ),
                "free_memory": (
                    True
                ),
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
                    for device in devices
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
                    "max_items": (
                        1000
                    )
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
            entry[
                "prompt"
            ]
            for entry in (
                history.values()
            )
            if "prompt" in entry
        ]

        matches = {
            item[1]
            for item in entries
            if (
                len(item)
                > 3
                and item[3].get(
                    "programme_request_id"
                )
                == request_id
            )
        }

        if len(
            matches
        ) > 1:
            raise SubmissionUncertain(
                "Multiple remote jobs match "
                "the submission."
            )

        return next(
            iter(
                matches
            ),
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

        if direction_id not in {
            "A",
            "B",
            "C",
        }:
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
        )

        seed = (
            secrets.randbits(
                63
            )
            if retrying_rejected_candidate
            else (
                previous[
                    "seed"
                ]
                if previous
                else secrets.randbits(
                    63
                )
            )
        )

        actual_positive_prompt = (
            build_engine_prompt(
                engine_id,
                positive_prompt,
            )
        )

        actual_negative_prompt = (
            (
                f"{BACKGROUND_ONLY_NEGATIVE}, "
                f"{negative_prompt}"
            )
            .strip(
                ", "
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
                ).encode()
            ).hexdigest()
        )

        if (
            previous
            and not retrying_rejected_candidate
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

        if retrying_rejected_candidate:
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
            "seed": seed,
            "workflow_sha256": (
                fingerprint
            ),
            "request_id": (
                str(
                    uuid.uuid4()
                )
            ),
            "status": "prepared",
            "attempt_count": 1,
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
                        "Negative prompt omitted: "
                        "FLUX.2 Klein BasicGuider "
                        "has no negative input."
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

        if (
            previous
            and not retrying_rejected_candidate
        ):
            record = (
                previous
            )

        elif (
            retrying_rejected_candidate
        ):
            record.update(
                attempt_count=(
                    previous.get(
                        "attempt_count",
                        1,
                    )
                    + 1
                ),
                rejected_attempts=(
                    rejected_attempts
                ),
            )

        prompt_id = (
            record.get(
                "prompt_id"
            )
        )

        if (
            not prompt_id
            and previous
            and not retrying_rejected_candidate
        ):
            prompt_id = (
                await self.recover_submission(
                    record[
                        "request_id"
                    ]
                )
            )

            if prompt_id is None:
                raise SubmissionUncertain(
                    "Submission outcome is unknown; "
                    "inspect ComfyUI history."
                )

        if not prompt_id:
            info = (
                await self.request(
                    "GET",
                    "/object_info",
                )
            ).json()

            validate_workflow(
                workflow,
                info,
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
                raise ComfyUIError(
                    "ComfyUI rejected the workflow."
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
            status="submitted",
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
            "prompt_id=%s",
            reference_number,
            engine_id,
            direction_id,
            prompt_id,
        )

        deadline = (
            time.monotonic()
            + (
                settings
                .comfyui_generation_timeout_seconds
            )
        )

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
                )

                if (
                    status.get(
                        "status_str"
                    )
                    == "error"
                ):
                    record.update(
                        status="failed",
                        error=(
                            "ComfyUI execution failed; "
                            "see remote history."
                        ),
                        remote_status=(
                            status
                        ),
                    )

                    write_json(
                        record_path,
                        record,
                    )

                    raise ComfyUIError(
                        record[
                            "error"
                        ]
                    )

                if status.get(
                    "completed"
                ):
                    break

            if (
                time.monotonic()
                >= deadline
            ):
                raise ComfyUIError(
                    "ComfyUI generation timed out."
                )

            await asyncio.sleep(
                settings
                .comfyui_poll_interval_seconds
            )

        save_nodes = [
            node
            for (
                node,
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
            for node in save_nodes
            for image in (
                history.get(
                    "outputs",
                    {},
                )
                .get(
                    node,
                    {},
                )
                .get(
                    "images",
                    [],
                )
            )
        ]

        if (
            len(
                images
            )
            != 1
            or images[0].get(
                "type"
            )
            != "output"
        ):
            raise ComfyUIError(
                "Expected exactly one persisted "
                "ComfyUI output image."
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
                    "type": (
                        "output"
                    ),
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

        try:
            try:
                validation = (
                    validate_background(
                        candidate_path,
                        reference_number=(
                            reference_number
                        ),
                    )
                )

            except ValueError as error:
                record.update(
                    status="rejected",
                    rejection_reason=(
                        str(error)
                    ),
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
                and not retrying_rejected_candidate
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
            status="complete",
            output_path=(
                str(
                    image_path
                )
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
            "sha256=%s",
            reference_number,
            engine_id,
            direction_id,
            record[
                "sha256"
            ],
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