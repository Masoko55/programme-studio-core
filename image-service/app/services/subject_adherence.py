"""Optional image-text subject check through the existing Ollama service.

The checker is enabled by SUBJECT_VISION_MODEL. An unavailable model produces
an explicit unchecked diagnostic; it never masquerades as a passing score.
"""
from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path

import httpx
from PIL import Image

from app.config.settings import settings


def select_vision_endpoint(model_name: str) -> str:
    """Use the endpoint verified for the installed MiniCPM vision model."""
    return "/api/chat" if model_name.strip().casefold().startswith("minicpm-v4.5") else "/api/generate"


def _model_answer(payload: dict, endpoint: str) -> dict:
    """Decode the Ollama envelope and one complete JSON object in its text."""
    message = payload.get("message")
    if endpoint == "/api/chat" and not isinstance(message, dict):
        raise ValueError("Missing Ollama chat message")
    content = message.get("content") if endpoint == "/api/chat" else payload.get("response")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Empty or missing model response")
    content = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", content.strip(), flags=re.I)
    try:
        answer = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        if start < 0:
            raise
        answer, _ = json.JSONDecoder().raw_decode(content[start:])
    if isinstance(answer, str):
        answer = json.loads(answer)
    if not isinstance(answer, dict):
        raise ValueError("Model response must be an object")
    return answer


def assess_subject_adherence(path: Path, visual_design: dict, *, model: str | None = None) -> dict:
    subject = str(visual_design.get("primary_visual_subject") or "").strip()
    anchors = [str(value) for value in visual_design.get("visual_anchors", []) if str(value).strip()]
    chosen_model = settings.subject_vision_model if model is None else model
    threshold = settings.subject_adherence_threshold
    base = {"checked": False, "score": None, "threshold": threshold,
            "subject": subject, "anchors": anchors, "model": chosen_model,
            "decision": "unavailable", "reason": ""}
    if not chosen_model or not subject:
        return {**base, "reason": "Vision model or requested visual subject is not configured"}

    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((768, 768), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
    prompt = (
        "Assess whether this generated background visibly depicts the requested visual subject. "
        "Judge the environment and its distinctive anchors, not the palette alone. "
        "Intentional abstract or sculptural subjects are valid when requested. "
        "Do not infer a subject from colour or decorative lines alone. "
        "Return JSON with score between 0 and 1, observed_evidence as a list of visible objects/forms, "
        "and reason. Requested subject: " + subject + ". Anchors: " + "; ".join(anchors[:5]) + "."
    )
    endpoint = select_vision_endpoint(chosen_model)
    debug = {"endpoint": endpoint, "model": chosen_model}
    try:
        encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")
        request = {"model": chosen_model, "stream": False, "format": "json",
                   "think": False, "keep_alive": 0}
        if endpoint == "/api/chat":
            request["messages"] = [{"role": "user", "content": prompt, "images": [encoded_image]}]
        else:
            request.update(prompt=prompt, images=[encoded_image])
        response = httpx.post(
            settings.ollama_base_url.rstrip("/") + endpoint,
            json=request, timeout=90,
        )
        debug["http_status"] = response.status_code
        debug["response_content_type"] = response.headers.get("content-type", "")[:80]
        response.raise_for_status()
        payload = response.json()
        debug["response_shape"] = sorted(str(key) for key in payload)[:20] if isinstance(payload, dict) else type(payload).__name__
        if not isinstance(payload, dict):
            raise ValueError("Ollama response must be an object")
        message = payload.get("message") if endpoint == "/api/chat" else None
        raw_text = message.get("content") if isinstance(message, dict) else payload.get("response")
        thinking = message.get("thinking") if isinstance(message, dict) else payload.get("thinking")
        debug["thinking_present"] = bool(thinking)
        debug["thinking_preview_length"] = len(thinking) if isinstance(thinking, str) else 0
        debug["response_length"] = len(raw_text) if isinstance(raw_text, str) else 0
        debug["final_source"] = ("message.content" if endpoint == "/api/chat" else "response") if isinstance(raw_text, str) and raw_text.strip() else None
        if isinstance(raw_text, str):
            debug["raw_text_preview"] = raw_text[:400]
        answer = _model_answer(payload, endpoint)
        score = float(answer["score"])
        if not 0 <= score <= 1:
            raise ValueError("Vision score must be between zero and one")
        evidence = answer.get("observed_evidence", answer.get("matched_anchors", []))
        if not isinstance(evidence, list):
            raise ValueError("Vision response lacks observed evidence")
    except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        debug["parse_error"] = f"{type(error).__name__}: {str(error)[:160]}"
        return {**base, "reason": f"Vision check unavailable: {type(error).__name__}",
                "vision_debug": debug}
    decision = "pass" if score >= threshold else "SUBJECT_DRIFT"
    return {**base, "checked": True, "score": score, "decision": decision,
            "reason": str(answer.get("reason") or ""), "observed_evidence": evidence,
            "model": chosen_model, "vision_debug": debug}
