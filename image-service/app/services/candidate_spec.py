"""Frozen creative contract for one engine and direction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from app.services.prompt_repository import get_direction


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    reference_number: str
    engine_id: str
    direction_id: str
    direction_role: str
    event_type: str
    theme: str
    creative_description: str
    background_inspiration: str
    theme_reference_treatment: str
    original_positive_prompt: str
    original_negative_prompt: str
    primary_colour: str
    secondary_colour: str
    # A canonical string prevents mutation through a nested dict on a frozen
    # dataclass. Callers receive a fresh dict from the property below.
    layout_guidance_json: str

    @property
    def layout_guidance(self) -> dict:
        return json.loads(self.layout_guidance_json)

    @property
    def spec_sha256(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_candidate_spec(document: dict, engine_id: str, direction_id: str) -> CandidateSpec:
    brief = document.get("brief")
    if not isinstance(brief, dict):
        raise ValueError("prompts.json has no structured brief")
    direction = get_direction(document, direction_id)
    layout = direction.get("layout_guidance") or {}
    if not isinstance(layout, dict):
        raise ValueError("Direction layout guidance must be an object")

    def field(source: dict, key: str) -> str:
        return str(source.get(key) or "").strip()

    spec = CandidateSpec(
        reference_number=field(document, "reference_number"),
        engine_id=engine_id,
        direction_id=direction_id,
        direction_role=field(direction, "role"),
        event_type=field(brief, "event_type"),
        theme=field(brief, "theme"),
        creative_description=field(brief, "creative_description"),
        background_inspiration=field(brief, "background_inspiration"),
        theme_reference_treatment=field(brief, "theme_reference_treatment"),
        original_positive_prompt=field(direction, "positive_prompt"),
        original_negative_prompt=field(direction, "negative_prompt"),
        primary_colour=field(brief, "primary_colour"),
        secondary_colour=field(brief, "secondary_colour"),
        layout_guidance_json=json.dumps(layout, sort_keys=True, separators=(",", ":")),
    )
    if not spec.reference_number or not spec.original_positive_prompt:
        raise ValueError("Candidate specification lacks reference or direction prompt")
    return spec
