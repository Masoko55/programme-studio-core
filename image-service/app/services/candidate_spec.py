"""Frozen creative contract for one engine and direction."""

from __future__ import annotations

import hashlib
import json
import re
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
    visual_design_json: str

    @property
    def visual_design(self) -> dict:
        return json.loads(self.visual_design_json)

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

    def values(source: dict, key: str) -> list[str]:
        value = source.get(key) or []
        if isinstance(value, str):
            value = [value]
        return [str(item).strip() for item in value if str(item).strip()] if isinstance(value, list) else []

    palette = brief.get("palette")
    if not isinstance(palette, list):
        palette = [
            {"colour": field(brief, key), "role": role}
            for key, role in (("primary_colour", "primary"), ("secondary_colour", "secondary"))
            if field(brief, key)
        ]
    palette = [item for item in palette if isinstance(item, dict) and item.get("colour")]
    visual_sentence = next((sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", field(brief, "creative_description"))
                            if re.match(r"\s*(?:create|draw|depict|illustrate|render|feature)\b", sentence, re.I)), "")
    anchors = values(brief, "visual_anchors") or [value for value in (
        field(brief, "background_inspiration"),
        visual_sentence if not field(brief, "background_inspiration") else "",
        field(brief, "theme_reference_treatment") if not field(brief, "background_inspiration") and not visual_sentence else "",
        field(brief, "theme") if not field(brief, "background_inspiration")
            and not visual_sentence and not field(brief, "theme_reference_treatment") else "",
    ) if value]
    visual_design = {
        "event_context": field(brief, "event_context") or field(brief, "event_type"),
        "audience_context": field(brief, "audience_context"),
        "mood": field(brief, "mood"),
        "visual_style": field(brief, "visual_style"),
        "requested_palette": palette,
        "palette_relationship": field(brief, "palette_relationship"),
        "visual_anchors": anchors,
        "symbolic_motifs": values(brief, "symbolic_motifs"),
        "environmental_motifs": values(brief, "environmental_motifs"),
        "material_language": field(brief, "material_language"),
        "texture_language": field(brief, "texture_language"),
        "lighting_language": field(brief, "lighting_language"),
        "composition_strategy": field(direction, "composition_strategy") or field(direction, "positive_prompt"),
        "focal_strategy": field(direction, "focal_strategy"),
        "depth_strategy": field(direction, "depth_strategy"),
        "border_or_perimeter_strategy": field(direction, "border_or_perimeter_strategy"),
        "negative_constraints": values(brief, "negative_constraints"),
        "title_safe_region_strategy": field(direction, "title_safe_region_strategy"),
        "programme_safe_region_strategy": field(direction, "programme_safe_region_strategy"),
        "abstraction_level": field(brief, "abstraction_level"),
        "realism_level": field(brief, "realism_level"),
        "energy_level": field(brief, "energy_level"),
        "symmetry_preference": field(brief, "symmetry_preference"),
        "density_preference": field(brief, "density_preference"),
    }

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
        visual_design_json=json.dumps(visual_design, sort_keys=True, separators=(",", ":")),
    )
    if not spec.reference_number or not spec.original_positive_prompt:
        raise ValueError("Candidate specification lacks reference or direction prompt")
    return spec
