"""ComfyUI HTTP adapter. No remote filesystem paths cross this boundary."""
import asyncio
import hashlib
import io
import json
import logging
import secrets
import time
import uuid
import re
from pathlib import Path
from typing import Any

import httpx
from PIL import Image
import pytesseract

from app.config.settings import settings
from app.services.atomic import write_bytes, write_json
from app.services.job_persistence import get_job_directory

logger = logging.getLogger("uvicorn.error")


BACKGROUND_ONLY_SUFFIX = (
    "Decorative background only. Absolutely no words, letters, typography, "
    "logos, signatures, watermarks, labels, or readable characters. "
    "Use unframed abstract ornament only: never a poster, card, document, "
    "menu, certificate, signage, or central panel. Leave the reserved content "
    "zones visually quiet."
)

BACKGROUND_ONLY_NEGATIVE = (
    "(text:2.0), (words:2.0), (letters:2.0), (typography:2.0), "
    "(writing:2.0), (calligraphy:1.8), (signature:1.8), (logo:1.8), "
    "(watermark:1.8), (poster:2.0), (card:2.0), (document:2.0), "
    "(menu:2.0), (certificate:2.0), (signage:2.0), numbers, labels, invitation"
)

SDXL_ABSTRACT_PROMPT = (
    "Abstract black and gold material study, polished obsidian, brushed metal, "
    "soft nonrepresentational light, asymmetric edge ornament and generous open "
    "negative space. No focal object, no frame, no placard, no page layout, and "
    "no information hierarchy."
)

SDXL_CONTENT_CUES = re.compile(
    r"\b(?:programme|program|ceremony|event|award|title|information|"
    r"hierarchy|layout|page|invitation|cover)\b",
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
)

class ComfyUIError(RuntimeError):
    """Unavailable dependency, invalid workflow, or failed remote execution."""

class SubmissionUncertain(ComfyUIError):
    """The server may have accepted a request; never blindly resubmit it."""


def workflow_name(engine_id: str) -> str:
    names = {settings.engine_1_id: "flux2", settings.engine_2_id: "sdxl", settings.engine_3_id: "sd35-medium"}
    if engine_id not in names:
        raise ValueError(f"Unknown engine: {engine_id}")
    return names[engine_id]


def build_background_only_prompt(
    positive_prompt: str,
) -> str:
    lowered = positive_prompt.lower()

    for term in FORBIDDEN_BACKGROUND_PROMPT_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            raise ComfyUIError(
                "Creative direction requests "
                f"{term}; regenerate the direction before "
                "image generation."
            )

    return f"{positive_prompt.rstrip('. ')}. {BACKGROUND_ONLY_SUFFIX}"


def build_engine_prompt(engine_id: str, positive_prompt: str) -> str:
    """Remove editorial cues before text-prone diffusion engines see them."""
    if engine_id not in {settings.engine_2_id, settings.engine_3_id}:
        return build_background_only_prompt(positive_prompt)

    visual_sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", positive_prompt)
        if sentence.strip() and not SDXL_CONTENT_CUES.search(sentence)
    ]
    visual_detail = " ".join(visual_sentences)
    if not visual_detail:
        visual_detail = "black and gold abstract material texture"
    return f"{SDXL_ABSTRACT_PROMPT} {visual_detail}. {BACKGROUND_ONLY_SUFFIX}"


def detected_text_tokens(image: Image.Image) -> list[str]:
    """Return confident OCR tokens so background-only candidates fail closed."""
    data = pytesseract.image_to_data(
        image.convert("L"),
        config="--psm 11",
        output_type=pytesseract.Output.DICT,
    )
    tokens = []
    for token, confidence in zip(data["text"], data["conf"]):
        normalized = re.sub(r"[^A-Za-z]", "", token)
        if len(normalized) >= 4 and float(confidence) >= 45:
            tokens.append(normalized)
    return tokens


def build_workflow(engine_id: str, positive_prompt: str, negative_prompt: str,
                   seed: int, output_prefix: str) -> dict:
    template = json.loads((settings.comfyui_workflow_path / f"{workflow_name(engine_id)}.json").read_text())
    values = settings.model_dump()
    values.update(positive_prompt=positive_prompt, negative_prompt=negative_prompt,
                  seed=seed, width=settings.generation_width, height=settings.generation_height,
                  output_prefix=output_prefix)
    def inject(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            return values[value[2:-1]]
        if isinstance(value, dict):
            return {key: inject(item) for key, item in value.items()}
        if isinstance(value, list):
            return [inject(item) for item in value]
        return value
    return inject(template)


def validate_workflow(workflow: dict, object_info: dict) -> None:
    for node_id, node in workflow.items():
        kind = node["class_type"]
        if kind not in object_info:
            raise ComfyUIError(f"ComfyUI node {kind} is unavailable (node {node_id})")
        schema = object_info[kind]["input"]
        inputs = node["inputs"]
        missing = set(schema.get("required", {})) - inputs.keys()
        if missing:
            raise ComfyUIError(f"Node {kind} missing required inputs: {sorted(missing)}")
        for key, spec in {**schema.get("required", {}), **schema.get("optional", {})}.items():
            if key not in inputs or isinstance(inputs[key], list):
                continue
            options = spec[0] if isinstance(spec[0], list) else (spec[1].get("options") if len(spec) > 1 and isinstance(spec[1], dict) else None)
            if options is not None and inputs[key] not in options:
                raise ComfyUIError(f"Configured {kind}.{key}={inputs[key]!r} is not available on ComfyUI")


def validate_background(path: Path, expected_sha256: str | None = None) -> dict:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Generated background is not PNG")
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Generated background SHA-256 mismatch")
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        width, height = image.size
        tokens = detected_text_tokens(image)
    if (width, height) != (settings.generation_width, settings.generation_height):
        raise ValueError(f"Unexpected background dimensions: {width}x{height}")
    if tokens:
        raise ValueError(f"Generated background contains OCR text: {', '.join(tokens[:8])}")
    return {"output_path": str(path), "sha256": digest, "width": width, "height": height}


class ComfyUIClient:
    def __init__(self):
        self.http = httpx.AsyncClient(
            base_url=settings.comfyui_base_url.rstrip("/"),
            timeout=httpx.Timeout(60, connect=settings.comfyui_connect_timeout_seconds),
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.http.aclose()

    async def request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            response = await self.http.request(method, path, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPError as error:
            code = getattr(getattr(error, "response", None), "status_code", None)
            raise ComfyUIError(f"ComfyUI {method} {path} failed (HTTP {code or 'unavailable'})") from error

    async def health(self) -> dict:
        stats = (await self.request("GET", "/system_stats")).json()
        info = (await self.request("GET", "/object_info")).json()
        for engine_id in [settings.engine_1_id, settings.engine_2_id, settings.engine_3_id]:
            validate_workflow(build_workflow(engine_id, "readiness", "", 0, "readiness"), info)
        return {"status": "available", "version": stats.get("system", {}).get("comfyui_version"),
                "engines": [settings.engine_1_id, settings.engine_2_id, settings.engine_3_id]}

    async def release_models(self) -> None:
        queue = (await self.request("GET", "/queue")).json()
        if queue.get("queue_running") or queue.get("queue_pending"):
            raise ComfyUIError("ComfyUI has active jobs; cannot release models safely")
        await self.request("POST", "/free", json={"unload_models": True, "free_memory": True})
        # /free acknowledges a flag; wait until the worker has released torch allocations.
        deadline = time.monotonic() + 60
        while True:
            stats = (await self.request("GET", "/system_stats")).json()
            devices = stats.get("devices", [])
            if devices and all(d.get("torch_vram_total", 0) <= 64 * 1024 * 1024 and d.get("vram_total", 0) - d.get("vram_free", 0) < 1024 * 1024 * 1024 for d in devices):
                logger.info("event=comfyui_models_released")
                return
            if time.monotonic() >= deadline:
                raise ComfyUIError("ComfyUI model unload was not confirmed within 60 seconds")
            await asyncio.sleep(1)

    async def recover_submission(self, request_id: str) -> str | None:
        queue = (await self.request("GET", "/queue")).json()
        history = (await self.request("GET", "/history", params={"max_items": 1000})).json()
        entries = queue.get("queue_running", []) + queue.get("queue_pending", [])
        entries += [entry["prompt"] for entry in history.values() if "prompt" in entry]
        matches = {item[1] for item in entries if len(item) > 3 and item[3].get("programme_request_id") == request_id}
        if len(matches) > 1:
            raise SubmissionUncertain("Multiple remote jobs match the submission; manual reconciliation required")
        return next(iter(matches), None)

    async def generate_image(self, reference_number: str, engine_id: str, direction_id: str,
                             positive_prompt: str, negative_prompt: str) -> dict:
        workflow_name(engine_id)
        if direction_id not in {"A", "B", "C"}:
            raise ValueError("Direction must be A, B or C")
        directory = get_job_directory(reference_number) / "backgrounds" / engine_id
        record_path = directory / f"image-{direction_id.lower()}.json"
        image_path = directory / f"image-{direction_id.lower()}.png"
        previous = json.loads(record_path.read_text()) if record_path.exists() else None
        if previous and previous.get("status") == "complete":
            try:
                verified = validate_background(image_path, previous["sha256"])
                return {**previous, **verified, "reused": True}
            except (OSError, ValueError):
                logger.warning(
                    "event=background_invalid reference=%s engine=%s direction=%s",
                    reference_number,
                    engine_id,
                    direction_id,
                )
        retrying_rejected_candidate = bool(
            previous and previous.get("rejection_reason")
        )
        seed = secrets.randbits(63) if retrying_rejected_candidate else (previous["seed"] if previous else secrets.randbits(63))
        actual_positive_prompt = build_engine_prompt(engine_id, positive_prompt)
        actual_negative_prompt = (
            f"{BACKGROUND_ONLY_NEGATIVE}, {negative_prompt}".strip(", ")
        )
        workflow = build_workflow(engine_id, actual_positive_prompt, actual_negative_prompt, seed,
                                  f"programme-studio/{reference_number}/{engine_id}/{direction_id.lower()}")
        fingerprint = hashlib.sha256(json.dumps(workflow, sort_keys=True).encode()).hexdigest()
        if previous and not retrying_rejected_candidate and previous["workflow_sha256"] != fingerprint:
            raise ComfyUIError("Frozen workflow differs from stored candidate; use a new reference")
        rejected_attempts = list(previous.get("rejected_attempts", [])) if previous else []
        if retrying_rejected_candidate:
            rejected_attempts.append({
                "attempt_count": previous.get("attempt_count", 1),
                "seed": previous.get("seed"),
                "prompt_id": previous.get("prompt_id"),
                "sha256": previous.get("sha256"),
                "rejection_reason": previous.get("rejection_reason"),
            })
        record = {"reference_number": reference_number, "engine_id": engine_id,
            "direction_id": direction_id, "seed": seed, "workflow_sha256": fingerprint,
            "request_id": str(uuid.uuid4()), "status": "prepared", "attempt_count": 1,
            "positive_prompt": actual_positive_prompt,
            "negative_prompt": actual_negative_prompt if engine_id != settings.engine_1_id else None,
            "adaptations": ["negative_prompt omitted: FLUX.2 Klein BasicGuider has no negative input"] if engine_id == settings.engine_1_id else [],
            "workflow": workflow}
        if previous and not retrying_rejected_candidate:
            record = previous
        elif retrying_rejected_candidate:
            record.update(
                attempt_count=previous.get("attempt_count", 1) + 1,
                rejected_attempts=rejected_attempts,
            )
        prompt_id = record.get("prompt_id")
        if not prompt_id and previous and not retrying_rejected_candidate:
            prompt_id = await self.recover_submission(record["request_id"])
            if prompt_id is None:
                raise SubmissionUncertain("Submission outcome is unknown; inspect ComfyUI history before retrying. No duplicate was submitted.")
        if not prompt_id:
            info = (await self.request("GET", "/object_info")).json()
            validate_workflow(workflow, info)
            queue = (await self.request("GET", "/queue")).json()
            if queue.get("queue_running") or queue.get("queue_pending"):
                raise ComfyUIError("ComfyUI is busy; retry after its current queue finishes")
            write_json(record_path, record)
            result = (await self.request("POST", "/prompt", json={"prompt": workflow,
                "client_id": record["request_id"], "extra_data": {"programme_request_id": record["request_id"]}})).json()
            if result.get("node_errors") or not result.get("prompt_id"):
                raise ComfyUIError("ComfyUI rejected the workflow; inspect the configured nodes/models")
            prompt_id = result["prompt_id"]
        record.update(prompt_id=prompt_id, status="submitted")
        write_json(record_path, record)
        logger.info("event=comfyui_submitted reference=%s engine=%s direction=%s prompt_id=%s", reference_number, engine_id, direction_id, prompt_id)
        deadline = time.monotonic() + settings.comfyui_generation_timeout_seconds
        while True:
            history = (await self.request("GET", f"/history/{prompt_id}")).json().get(prompt_id)
            if history:
                status = history.get("status", {})
                if status.get("status_str") == "error":
                    record.update(status="failed", error="ComfyUI execution failed; see remote history", remote_status=status)
                    write_json(record_path, record)
                    raise ComfyUIError(record["error"])
                if status.get("completed"):
                    break
            if time.monotonic() >= deadline:
                raise ComfyUIError(f"ComfyUI generation timed out; resume will reconnect to {prompt_id}")
            await asyncio.sleep(settings.comfyui_poll_interval_seconds)
        save_nodes = [node for node, item in workflow.items() if item["class_type"] == "SaveImage"]
        images = [image for node in save_nodes for image in history.get("outputs", {}).get(node, {}).get("images", [])]
        if len(images) != 1 or images[0].get("type") != "output":
            raise ComfyUIError("Expected exactly one persisted ComfyUI output image")
        descriptor = images[0]
        response = await self.request("GET", "/view", params={"filename": descriptor["filename"],
            "subfolder": descriptor.get("subfolder", ""), "type": "output"})
        # Validate before replacing any existing candidate.
        candidate_path = image_path.with_suffix(".download.part")
        write_bytes(candidate_path, response.content)
        try:
            try:
                result = validate_background(candidate_path)
            except ValueError as error:
                record.update(
                    status="rejected",
                    rejection_reason=str(error),
                    remote_image=descriptor,
                )
                write_json(record_path, record)
                raise ComfyUIError(str(error)) from error
            if (previous and not retrying_rejected_candidate and previous.get("sha256")
                    and result["sha256"] != previous["sha256"]):
                raise ComfyUIError("Remote output changed since the original generation")
            candidate_path.replace(image_path)
        finally:
            candidate_path.unlink(missing_ok=True)
        record.update(result, status="complete", output_path=str(image_path), remote_image=descriptor)
        write_json(record_path, record)
        logger.info("event=background_verified reference=%s engine=%s direction=%s sha256=%s", reference_number, engine_id, direction_id, record["sha256"])
        return {**record, "reused": False}


async def get_runtime_health() -> dict:
    async with ComfyUIClient() as client:
        return await client.health()

async def generate_remote_image(reference_number: str, engine_id: str, direction_id: str,
                                positive_prompt: str, negative_prompt: str) -> dict:
    async with ComfyUIClient() as client:
        return await client.generate_image(reference_number, engine_id, direction_id, positive_prompt, negative_prompt)
