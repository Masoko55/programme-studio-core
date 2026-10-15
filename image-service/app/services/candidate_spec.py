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
    required_strategies = (
        "composition_strategy", "focal_strategy", "depth_strategy",
        "border_or_perimeter_strategy", "title_safe_region_strategy",
        "programme_safe_region_strategy",
    )
    missing = [name for name in required_strategies
               if len(str(direction.get(name) or "").strip()) < 8]
    if missing:
        raise ValueError(
            f"Direction {direction_id} lacks required structural strategies: {', '.join(missing)}"
        )
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
    context = brief.get("grill_me_context")
    semantics = brief.get("visual_semantics")
    if not isinstance(semantics, dict) and isinstance(context, dict):
        semantics = context.get("visual_semantics")
    if not isinstance(semantics, dict):
        semantics = {}
    motif_words = values(brief, "symbolic_motifs") + values(brief, "environmental_motifs")
    # A raw description can contain planning context or a named reference.
    # Preserve descriptive prose unless it repeats the event label or contains
    # a backstory sentence; the semantic stage owns interpretation of those.
    raw_description = field(brief, "creative_description")
    event_label = field(brief, "event_type")
    fallback_description = raw_description if (
        raw_description
        and (not event_label or event_label.casefold() not in raw_description.casefold())
        and not re.search(r"\b(?:loves?|guest of honour|favorite character|favourite character)\b", raw_description, re.I)
    ) else ""
    atmosphere = values(semantics, "atmosphere_descriptors") or [value for value in (
        field(brief, "mood"), visual_sentence or fallback_description
    ) if value]
    audience = values(semantics, "audience_descriptors") or [value for value in (
        field(brief, "audience_context"), field(brief, "age_group")
    ) if value]
    visual_semantics = {
        "atmosphere_descriptors": atmosphere,
        "audience_descriptors": audience,
        "motif_descriptors": values(semantics, "motif_descriptors") or motif_words,
        "motif_structure_description": field(semantics, "motif_structure_description") or ", ".join(motif_words),
        "lighting_descriptors": values(semantics, "lighting_descriptors") or values(brief, "lighting_language"),
        "incompatible_treatments": values(semantics, "incompatible_treatments") or values(brief, "negative_constraints"),
        "scene_family": field(semantics, "scene_family") or "general",
        "composition_affinities": values(semantics, "composition_affinities"),
    }
    visual_design = {
        "primary_visual_subject": field(brief, "background_inspiration") or visual_sentence or field(brief, "theme_reference_treatment") or field(brief, "theme"),
        "event_context": field(brief, "event_context") or field(brief, "event_type"),
        "render_exclusion_text": [field(brief, key) for key in (
            "title_preference", "event_date", "venue"
        ) if field(brief, key)],
        "audience_context": field(brief, "audience_context"),
        "age_group": field(brief, "age_group"),
        "creative_description": field(brief, "creative_description"),
        "visual_semantics": visual_semantics,
        "theme_reference_treatment": field(brief, "theme_reference_treatment"),
        "mood": field(brief, "mood"),
        "visual_style": field(brief, "visual_style"),
        "requested_palette": palette,
        "palette_mode": field(brief, "palette_mode") or "dominant",
        "palette_relationship": field(brief, "palette_relationship"),
        "visual_anchors": anchors,
        "symbolic_motifs": values(brief, "symbolic_motifs"),
        "environmental_motifs": values(brief, "environmental_motifs"),
        "material_language": field(brief, "material_language"),
        "texture_language": field(brief, "texture_language"),
        "lighting_language": field(brief, "lighting_language"),
        "composition_strategy": field(direction, "composition_strategy"),
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
