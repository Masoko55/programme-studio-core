"""Frozen creative contract for one engine and direction."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass

from app.services.prompt_repository import get_direction
from app.services.semantic_contract import (
    VERSION, contract_hash, normalize_contract, semantic_completeness, semantic_consistency,
)
from app.services.visual_hierarchy import derive_visual_hierarchy


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
    semantic_contract_json: str = ""

    @property
    def visual_design(self) -> dict:
        return json.loads(self.visual_design_json)

    @property
    def layout_guidance(self) -> dict:
        return json.loads(self.layout_guidance_json)

    @property
    def semantic_contract(self) -> dict:
        return json.loads(self.semantic_contract_json)

    @property
    def semantic_contract_sha256(self) -> str:
        return contract_hash(self.semantic_contract)

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
        "material_descriptors": values(semantics, "material_descriptors") or values(brief, "material_language"),
        "texture_descriptors": values(semantics, "texture_descriptors") or values(brief, "texture_language"),
        "incompatible_treatments": values(semantics, "incompatible_treatments") or values(brief, "negative_constraints"),
        "scene_family": field(semantics, "scene_family") or "general",
        "composition_affinities": values(semantics, "composition_affinities"),
    }
    subject = field(brief, "background_inspiration") or visual_sentence or field(brief, "theme_reference_treatment") or field(brief, "theme")
    supplied_contract = brief.get("semantic_contract")
    if supplied_contract is not None:
        contract = normalize_contract(supplied_contract)
        expected_hash = field(brief, "semantic_contract_sha256")
        if not expected_hash or contract_hash(supplied_contract) != expected_hash or contract_hash(contract) != expected_hash:
            raise ValueError("Semantic contract hash mismatch between prompt and image services")
        if (field(brief, "background_inspiration") and contract["primary_visual_subject"] != field(brief, "background_inspiration")):
            raise ValueError("Semantic subject changed after contract creation")
        if not field(brief, "background_inspiration") and field(brief, "theme") and contract["primary_visual_subject"] != field(brief, "theme"):
            raise ValueError("Semantic subject changed after contract creation")
        if contract["palette"] != palette:
            raise ValueError("Semantic palette changed after contract creation")
        for key in ("palette_mode", "palette_relationship", "theme_reference_treatment",
                    "abstraction_level", "realism_level", "energy_level",
                    "symmetry_preference", "density_preference"):
            if field(brief, key) and contract[key] != field(brief, key):
                raise ValueError(f"Semantic {key} changed after contract creation")
        if values(brief, "visual_anchors") and contract["visual_anchors"] != values(brief, "visual_anchors"):
            raise ValueError("Semantic anchors changed after contract creation")
        for key, value in semantics.items():
            if key in contract and value and contract[key] != value:
                raise ValueError(f"Semantic {key} changed after contract creation")
    else:
        contract = normalize_contract({
            "semantic_contract_version": VERSION,
            "primary_visual_subject": subject, "visual_anchors": anchors,
            **visual_semantics, "palette": palette,
            "palette_mode": field(brief, "palette_mode") or "dominant",
            "palette_relationship": field(brief, "palette_relationship"),
            "abstraction_level": field(brief, "abstraction_level"),
            "realism_level": field(brief, "realism_level"),
            "energy_level": field(brief, "energy_level"),
            "symmetry_preference": field(brief, "symmetry_preference"),
            "density_preference": field(brief, "density_preference"),
            "theme_reference_treatment": field(brief, "theme_reference_treatment"),
            "semantic_provenance": {
                "primary_visual_subject": "USER_EXPLICIT" if field(brief, "background_inspiration") else "SAFE_FALLBACK",
                "atmosphere_descriptors": ("SEMANTIC_INFERENCE" if values(semantics, "atmosphere_descriptors")
                                           else "USER_EXPLICIT" if field(brief, "mood") or fallback_description
                                           else "SAFE_FALLBACK"),
                "palette": "USER_EXPLICIT" if palette else "SAFE_FALLBACK",
            },
        })
    consistency = semantic_consistency(contract)
    if not consistency["valid"]:
        error = ValueError(f"Semantic contract requires clarification: {consistency['conflicts']}")
        error.semantic_consistency = consistency
        raise error
    completeness = semantic_completeness(contract)
    visual_design = {
        "primary_visual_subject": contract["primary_visual_subject"],
        "event_context": field(brief, "event_context") or field(brief, "event_type"),
        "render_exclusion_text": [field(brief, key) for key in (
            "title_preference", "event_date", "venue"
        ) if field(brief, key)],
        "audience_context": field(brief, "audience_context"),
        "age_group": field(brief, "age_group"),
        "creative_description": field(brief, "creative_description"),
        "visual_semantics": {key: contract[key] for key in visual_semantics},
        "semantic_contract_version": VERSION,
        "semantic_contract_sha256": contract_hash(contract),
        "semantic_consistency": consistency,
        "semantic_completeness": completeness,
        "theme_reference_treatment": contract["theme_reference_treatment"],
        "mood": field(brief, "mood"),
        "visual_style": field(brief, "visual_style"),
        "requested_palette": contract["palette"],
        "palette_mode": contract["palette_mode"],
        "palette_relationship": contract["palette_relationship"],
        "visual_anchors": contract["visual_anchors"],
        "symbolic_motifs": values(brief, "symbolic_motifs"),
        "environmental_motifs": values(brief, "environmental_motifs"),
        "material_language": ", ".join(contract["material_descriptors"]) or field(brief, "material_language"),
        "texture_language": ", ".join(contract["texture_descriptors"]) or field(brief, "texture_language"),
        "lighting_language": ", ".join(contract["lighting_descriptors"]) or field(brief, "lighting_language"),
        "composition_strategy": field(direction, "composition_strategy"),
        "focal_strategy": field(direction, "focal_strategy"),
        "depth_strategy": field(direction, "depth_strategy"),
        "border_or_perimeter_strategy": field(direction, "border_or_perimeter_strategy"),
        "negative_constraints": values(brief, "negative_constraints"),
        "title_safe_region_strategy": field(direction, "title_safe_region_strategy"),
        "programme_safe_region_strategy": field(direction, "programme_safe_region_strategy"),
        "abstraction_level": contract["abstraction_level"],
        "realism_level": contract["realism_level"],
        "energy_level": contract["energy_level"],
        "symmetry_preference": contract["symmetry_preference"],
        "density_preference": contract["density_preference"],
        "visual_hierarchy": derive_visual_hierarchy(contract, direction),
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
        semantic_contract_json=json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
    )
    if not spec.reference_number or not spec.original_positive_prompt:
        raise ValueError("Candidate specification lacks reference or direction prompt")
    return spec
