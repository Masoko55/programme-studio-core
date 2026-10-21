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


def _safe_spatial_treatment(raw: str, affinities: list[str]) -> str:
    """Translate explicitly stated camera/motion cues without depicting a reference."""
    text = raw.casefold()
    parts = []
    if re.search(r"\bfirst[- ]person\b", text):
        parts.append("immersive first-person viewpoint")
    if re.search(r"\b(?:view from above|aerial view|overhead view)\b", text):
        parts.append("aerial viewpoint")
    if re.search(r"\b(?:swing\w*|fly\w*|mov\w*|travel\w*|travers\w*|glid\w*|rush\w*)\b", text):
        parts.append("forward traversal with strong receding perspective")
    if re.search(r"\b(?:swing\w*|arc\w*|sweep\w*)\b", text):
        parts.append("arcing spatial motion")
    environment = re.search(r"\b(?:through|between)\s+(?:the\s+)?([a-z]+(?:\s+[a-z]+){0,2})", text)
    if environment:
        words = re.split(r"\b(?:in|from|with|while|as|first|person|view)\b", environment.group(1), maxsplit=1)[0].strip()
        if words:
            parts.append(f"movement through {words}")
    parts.extend(phrase for phrase in affinities if re.search(
        r"\b(?:viewpoint|perspective|traversal|movement|motion|depth)\b", phrase, re.I))
    return "; ".join(dict.fromkeys(parts))


def _motif_literalization(contract: dict, subject: str, has_reference) -> tuple[str, str]:
    """Separate a motif's visual pattern from an explicitly requested subject."""
    motifs = " ".join((*contract["motif_descriptors"], contract["motif_structure_description"]))
    if not motifs.strip():
        return "literal_allowed", ""
    source = contract["semantic_provenance"].get("motif_descriptors", "SAFE_FALLBACK")
    pattern = bool(re.search(r"\b(?:motif|pattern|texture|geometry|geometric|abstract|"
                             r"topology|lattice|print|shapes?|inspired|like)\b", motifs, re.I))
    policy = "motif_only" if source not in {"USER_EXPLICIT", "GRILL_ME_CLARIFIED"} or pattern else "literal_allowed"
    explicit_subject = contract["semantic_provenance"].get("primary_visual_subject") in {
        "USER_EXPLICIT", "GRILL_ME_CLARIFIED"}
    subject_is_pattern = bool(re.search(r"\b(?:motif|pattern|texture|geometry|abstract|"
                                        r"lattice|print|shapes?|inspired|like)\b", subject, re.I))
    authorized = subject if explicit_subject and not subject_is_pattern and not has_reference(subject) else ""
    return policy, authorized


def _characters_explicitly_prohibited(constraints: list[str]) -> bool:
    """Recognize explicit bans on depicted characters or character-like figures."""
    subject = r"(?:characters?|fictional figures?|humanoid figures?|costumed figures?|superhero figures?)"
    qualifier = r"(?:(?:any|all|depicted|visible)\s+){0,2}"
    paired_people = r"(?:(?:people|persons|humans)\s+(?:or|and)\s+)?"
    leading = rf"\b(?:no|without|exclude|avoid)\s+{qualifier}{paired_people}{subject}\b"
    imperative = rf"\bdo not\s+(?:depict|show|include|render|feature)\s+{qualifier}{subject}\b"
    trailing = rf"\b{subject}\b\W+(?:are\W+)?(?:prohibited|forbidden|excluded|not allowed)\b"
    return any(re.search(rf"(?:{leading}|{imperative}|{trailing})", item, re.I)
               for item in constraints)


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
    generation_constraints = (values(context, "generation_constraints")
                              if isinstance(context, dict) else [])
    generation_constraints.extend(values(brief, "negative_constraints"))
    characters_prohibited = _characters_explicitly_prohibited(generation_constraints)
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
        try:
            contract = normalize_contract(supplied_contract)
        except (TypeError, ValueError, KeyError):
            supplied_contract = None
        # The clarified brief wins over stale or missing provenance metadata.
        # Keep the contract for diagnostics, but rebuild it from current brief
        # fields below when it disagrees with what the user confirmed.
        if supplied_contract is not None and ((subject and contract["primary_visual_subject"] != subject)
                or contract["palette"] != palette
                or any(value and contract.get(key) != value
                       for key, value in visual_semantics.items()
                       if key in contract)):
            supplied_contract = None
    if supplied_contract is None:
        provenance = (context.get("semantic_provenance") or {}) if isinstance(context, dict) else {}
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
            "hard_dimensions": brief.get("hard_dimensions", []),
            "semantic_provenance": {
                "primary_visual_subject": "USER_EXPLICIT" if field(brief, "background_inspiration") else "SAFE_FALLBACK",
                "atmosphere_descriptors": ("SEMANTIC_INFERENCE" if values(semantics, "atmosphere_descriptors")
                                           else "USER_EXPLICIT" if field(brief, "mood") or fallback_description
                                           else "SAFE_FALLBACK"),
                "palette": "USER_EXPLICIT" if palette else "SAFE_FALLBACK",
                "palette_relationship": ("SAFE_FALLBACK" if field(brief, "palette_relationship_source") == "default"
                                         else "USER_EXPLICIT" if field(brief, "palette_relationship") else "SAFE_FALLBACK"),
                "visual_anchors": "USER_EXPLICIT" if values(brief, "visual_anchors") or field(brief, "background_inspiration") else "SAFE_FALLBACK",
                "theme_reference_treatment": "USER_EXPLICIT" if field(brief, "theme_reference_treatment") else "SAFE_FALLBACK",
                "motif_descriptors": provenance.get("motif_descriptors") or (
                    "USER_EXPLICIT" if motif_words else "SEMANTIC_INFERENCE" if values(semantics, "motif_descriptors") else "SAFE_FALLBACK"),
                "motif_structure_description": provenance.get("motif_structure_description") or (
                    "SEMANTIC_INFERENCE" if field(semantics, "motif_structure_description") else "USER_EXPLICIT" if motif_words else "SAFE_FALLBACK"),
                "audience_descriptors": provenance.get("audience_descriptors") or (
                    "USER_EXPLICIT" if field(brief, "audience_context") or field(brief, "age_group") else "SEMANTIC_INFERENCE" if audience else "SAFE_FALLBACK"),
            },
        })
    consistency = semantic_consistency(contract)
    # Semantic contradictions are useful diagnostics, not a second schema gate.
    completeness = semantic_completeness(contract)
    reference_name = (field(context, "reference_name") if isinstance(context, dict)
                      and context.get("reference_detected") else "")
    reference_parts = [part for part in re.split(r"[\W_]+", reference_name) if part]
    reference_pattern = (r"(?<!\w)" + r"[\W_]*".join(map(re.escape, reference_parts)) + r"(?!\w)"
                         if reference_parts else "")

    def has_reference(text: str) -> bool:
        return bool(reference_pattern and re.search(reference_pattern, text, re.I))

    semantic_motifs = [*contract["motif_descriptors"]]
    structure = contract["motif_structure_description"]
    if structure:
        semantic_motifs.append(structure)
    semantic_motifs = [item for item in dict.fromkeys(semantic_motifs) if not has_reference(item)]
    if reference_name:
        visual_semantics["motif_descriptors"] = [item for item in contract["motif_descriptors"]
                                                 if not has_reference(item)]
        visual_semantics["motif_structure_description"] = (
            structure if not has_reference(structure) else ", ".join(semantic_motifs))
    safe_scene = contract["scene_family"] if contract["scene_family"] != "general" else ""
    render_subject = contract["primary_visual_subject"]
    if has_reference(render_subject):
        if not semantic_motifs:
            raise ValueError("Detected reference lacks a renderable semantic motif")
        render_subject = ", ".join(semantic_motifs)
    render_anchors = [item for item in contract["visual_anchors"] if not has_reference(item)]
    if reference_name and not render_anchors:
        render_anchors = semantic_motifs or ([safe_scene] if safe_scene else [])
    render_treatment = contract["theme_reference_treatment"]
    if has_reference(render_treatment):
        render_treatment = _safe_spatial_treatment(
            render_treatment, [item for item in contract["composition_affinities"] if not has_reference(item)])
        if not render_treatment:
            render_treatment = structure if structure and not has_reference(structure) else ", ".join(semantic_motifs)
    else:
        render_treatment = re.split(
            r"\b(?:do not|don't|never|avoid|without|no people|no characters|no text)\b",
            render_treatment, maxsplit=1, flags=re.I,
        )[0].rstrip(" ,.;")
    motif_policy, authorized_literal_subject = _motif_literalization(contract, subject, has_reference)
    visual_design = {
        "primary_visual_subject": render_subject,
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
        "semantic_provenance": contract["semantic_provenance"],
        "hard_dimensions": contract.get("hard_dimensions", []),
        "theme_reference_treatment": render_treatment,
        "motif_literalization_policy": motif_policy,
        "authorized_literal_subject": authorized_literal_subject,
        "mood": field(brief, "mood"),
        "visual_style": field(brief, "visual_style"),
        "requested_palette": contract["palette"],
        "palette_mode": contract["palette_mode"],
        "palette_relationship": contract["palette_relationship"],
        "palette_relationship_source": (field(brief, "palette_relationship_source")
                                        or contract.get("semantic_provenance", {}).get("palette_relationship")),
        "visual_anchors": render_anchors,
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
        "characters_prohibited": characters_prohibited,
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
