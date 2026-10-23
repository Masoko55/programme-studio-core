"""Build the portable semantic payload before creative directions are rendered."""

from __future__ import annotations

import hashlib
import json
import re


VERSION = "1.0"


def resolve_palette_relationship(brief: dict) -> tuple[str, str]:
    """Preserve colour relationships stated in the user's visual brief."""
    colours = [str(brief.get(key) or "").strip() for key in ("primary_colour", "secondary_colour")]
    relationship = str(brief.get("palette_relationship") or "").strip()
    source = str(brief.get("palette_relationship_source") or "").casefold()
    if relationship and source in {"user", "user_explicit", "grill_me_clarified"}:
        return relationship, source
    clauses = []
    for key in ("creative_description", "background_inspiration", "theme_reference_treatment"):
        for clause in re.split(r"[.!?;\n]+", str(brief.get(key) or "")):
            if _has_colour_relationship(clause, colours):
                clauses.append(clause.strip())
    if clauses:
        return "; ".join(dict.fromkeys(clauses)), "user_explicit"
    return ("only these colours" if brief.get("palette_mode") == "strict" else
            "requested colours dominate with natural tonal variation"), "default"


def _has_colour_relationship(clause: str, colours: list[str]) -> bool:
    names = "|".join(re.escape(colour) for colour in colours if colour)
    if not names or not re.search(rf"\b(?:{names})\b", clause, re.I):
        return False
    colour = rf"\b(?:{names})\b"
    return bool(re.search(
        rf"{colour}\s+(?:(?:must|should|is|be|remain|the|visually|as|only)\s+)*"
        r"(?:dominant|dominate|main|support\w*|accent\w*)\b|"
        rf"\b(?:mostly|mainly|predominantly)\s+{colour}|"
        rf"\b(?:small|subtle|supporting)\s+{colour}\s+(?:details|accents|highlights)\b|"
        r"\b(?:balanced|equal|equally|only|monochrom\w*)\b", clause, re.I))


def contract_hash(contract: dict) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def build_semantic_contract(brief: dict) -> dict:
    import re

    def redaction_shell(value: str) -> bool:
        text = re.sub(r"\s+", " ", value.casefold()).strip(" ,.;-_")
        if re.search(r"\b(?:with|featuring|including|using)\b", text):
            return False
        if re.fullmatch(r"(?:(?:a|an|the)\s+)?(?:inspired\s+)?(?:theme|style|background|design)\s+for\s+.+", text):
            return True
        words = re.findall(r"[a-z]+", text)
        return bool(words) and set(words) <= {
            "a", "an", "the", "use", "apply", "inspired", "by", "theme", "style",
            "background", "design", "visual", "trait", "traits", "colour", "colours",
            "color", "colors", "and",
        }

    semantics = brief.get("visual_semantics") or {}
    context = brief.get("grill_me_context") or {}
    reference_detected = bool(context.get("reference_detected"))
    inspiration = str(brief.get("background_inspiration") or "").strip()
    if reference_detected and redaction_shell(inspiration):
        inspiration = ""
    anchors = brief.get("visual_anchors") or []
    if reference_detected:
        anchors = [anchor for anchor in anchors
                   if not isinstance(anchor, str) or not redaction_shell(anchor)]
    inferred_sources = context.get("semantic_provenance") or {}
    palette = brief.get("palette")
    if not isinstance(palette, list):
        palette = [{"colour": str(brief[key]).strip(), "role": role}
                   for key, role in (("primary_colour", "primary"), ("secondary_colour", "secondary"))
                   if brief.get(key)]
    fields = {
        "primary_visual_subject": inspiration or brief.get("theme") or "",
        "visual_anchors": anchors or ([inspiration] if inspiration else []),
        "atmosphere_descriptors": semantics.get("atmosphere_descriptors") or [],
        "audience_descriptors": semantics.get("audience_descriptors") or [],
        "motif_descriptors": semantics.get("motif_descriptors") or [],
        "motif_structure_description": semantics.get("motif_structure_description") or "",
        "lighting_descriptors": semantics.get("lighting_descriptors") or [],
        "material_descriptors": semantics.get("material_descriptors") or [],
        "texture_descriptors": semantics.get("texture_descriptors") or [],
        "incompatible_treatments": semantics.get("incompatible_treatments") or [],
        "scene_family": semantics.get("scene_family") or "general",
        "composition_affinities": semantics.get("composition_affinities") or [],
        "palette": palette,
        "palette_mode": brief.get("palette_mode") or "dominant",
        "palette_relationship": brief.get("palette_relationship") or "",
        "abstraction_level": brief.get("abstraction_level") or "",
        "realism_level": brief.get("realism_level") or "",
        "energy_level": brief.get("energy_level") or "",
        "symmetry_preference": brief.get("symmetry_preference") or "",
        "density_preference": brief.get("density_preference") or "",
        "theme_reference_treatment": brief.get("theme_reference_treatment") or "",
    }
    sources = {}
    for key, value in fields.items():
        if not value:
            sources[key] = "SAFE_FALLBACK"
        elif key == "palette_relationship" and brief.get("palette_relationship_source") == "default":
            sources[key] = "SAFE_FALLBACK"
        elif key == "palette_relationship" and str(brief.get("palette_relationship_source") or "").casefold() in {"user", "user_explicit", "grill_me_clarified"}:
            sources[key] = ("GRILL_ME_CLARIFIED" if str(brief["palette_relationship_source"]).casefold() == "grill_me_clarified" else "USER_EXPLICIT")
        elif key == "primary_visual_subject" and reference_detected and not inspiration and brief.get("theme"):
            sources[key] = "USER_EXPLICIT"
        elif key in inferred_sources:
            sources[key] = inferred_sources[key]
        elif key in {"primary_visual_subject", "visual_anchors", "palette", "palette_mode", "palette_relationship", "theme_reference_treatment"}:
            sources[key] = "USER_EXPLICIT" if brief.get("background_inspiration" if key in {"primary_visual_subject", "visual_anchors"} else key) else "SAFE_FALLBACK"
        elif key in semantics:
            sources[key] = "SEMANTIC_INFERENCE"
        else:
            sources[key] = "USER_EXPLICIT"
    confidence = {key: value for key, value in (context.get("semantic_confidence") or {}).items()
                  if key in fields and sources[key] != "USER_EXPLICIT" and isinstance(value, (float, int))
                  and not isinstance(value, bool) and 0 <= value <= 1}
    return {"semantic_contract_version": VERSION, **fields,
            "hard_dimensions": [field for field in brief.get("hard_dimensions", [])
                                if field in {"subject_score", "motif_score", "audience_fit_score", "atmosphere_score"}],
            "semantic_provenance": sources, "semantic_confidence": confidence}
