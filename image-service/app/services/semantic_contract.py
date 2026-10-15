"""Versioned, direction-independent semantic payload used by every renderer."""

from __future__ import annotations

import hashlib
import json
import re


VERSION = "1.0"
SEMANTIC_FIELDS = (
    "primary_visual_subject", "visual_anchors", "atmosphere_descriptors",
    "audience_descriptors", "motif_descriptors", "motif_structure_description",
    "lighting_descriptors", "material_descriptors", "texture_descriptors",
    "incompatible_treatments", "scene_family", "composition_affinities",
    "palette", "palette_mode", "palette_relationship", "abstraction_level",
    "realism_level", "energy_level", "symmetry_preference", "density_preference",
    "theme_reference_treatment",
)
LIST_FIELDS = frozenset(("visual_anchors", "atmosphere_descriptors", "audience_descriptors",
    "motif_descriptors", "lighting_descriptors", "material_descriptors",
    "texture_descriptors", "incompatible_treatments", "composition_affinities"))
PROVENANCE = frozenset(("USER_EXPLICIT", "GRILL_ME_CLARIFIED", "SEMANTIC_INFERENCE",
                        "CREATIVE_DIRECTION", "SAFE_FALLBACK"))


def contract_hash(contract: dict) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def normalize_contract(raw: dict) -> dict:
    if not isinstance(raw, dict) or raw.get("semantic_contract_version") != VERSION:
        raise ValueError("Unsupported or missing semantic contract version")
    contract = {"semantic_contract_version": VERSION}
    for key in SEMANTIC_FIELDS:
        value = raw.get(key)
        if key == "palette":
            contract[key] = [dict(item) for item in value if isinstance(item, dict) and item.get("colour")] if isinstance(value, list) else []
        elif key in LIST_FIELDS:
            if isinstance(value, str):
                value = [value]
            contract[key] = [str(item).strip() for item in value if str(item).strip()] if isinstance(value, list) else []
        else:
            contract[key] = str(value or "").strip()
    contract["semantic_provenance"] = {
        key: source for key, source in (raw.get("semantic_provenance") or {}).items()
        if key in SEMANTIC_FIELDS and source in PROVENANCE
    }
    contract["semantic_confidence"] = {
        key: float(score) for key, score in (raw.get("semantic_confidence") or {}).items()
        if key in SEMANTIC_FIELDS and contract["semantic_provenance"].get(key) != "USER_EXPLICIT"
        and isinstance(score, (float, int)) and not isinstance(score, bool) and 0 <= score <= 1
    }
    return contract


def semantic_consistency(contract: dict) -> dict:
    """Check dimension conflicts without looking at named event or theme labels."""
    atmosphere = " ".join(contract.get("atmosphere_descriptors", []) + contract.get("lighting_descriptors", [])).casefold()
    density = str(contract.get("density_preference") or "").casefold()
    realism = str(contract.get("realism_level") or "").casefold()
    abstraction = str(contract.get("abstraction_level") or "").casefold()
    subject = str(contract.get("primary_visual_subject") or "").casefold()
    excluded = " ".join(contract.get("incompatible_treatments", [])).casefold()
    palette = [str(item.get("colour") or "").casefold() for item in contract.get("palette", [])]
    conflicts = []

    def conflict(fields: list[str], reason: str) -> None:
        conflicts.append({"fields": fields, "reason": reason})

    if re.search(r"\b(calm|peaceful|tranquil)\b", atmosphere) and re.search(r"\b(chaotic|frantic)\b", atmosphere):
        conflict(["atmosphere_descriptors"], "Opposing calm and chaotic atmosphere cues")
    if (re.search(r"\bminimal\b", atmosphere + " " + density)
            and re.search(r"\b(extremely dense|maximal|cluttered)\b", atmosphere + " " + density)):
        conflict(["atmosphere_descriptors", "density_preference"], "Minimal and dense treatments conflict")
    if re.search(r"\bphotorealistic\b", realism + " " + atmosphere) and re.search(r"\b(flat vector|vector only)\b", abstraction + " " + atmosphere):
        conflict(["realism_level", "abstraction_level"], "Photorealistic and flat-vector-only treatments conflict")
    if re.search(r"\b(visible|depicted|portrait of)\s+(human|person|character)\b", subject) and re.search(r"\b(no|without)\s+(human|people|person|character)s?\b", excluded):
        conflict(["primary_visual_subject", "incompatible_treatments"], "Visible character conflicts with exclusion")
    if contract.get("palette_mode") in {"strict", "achromatic"} and {"black", "white"}.issubset(palette) and any(colour not in {"black", "white", "grey", "gray"} for colour in palette):
        conflict(["palette", "palette_mode"], "Strict black-and-white palette also requests chromatic accents")
    if re.search(r"\b(dark|restrained|somber|solemn)\b", atmosphere) and re.search(r"\b(neon celebratory|neon party)\b", atmosphere):
        conflict(["atmosphere_descriptors"], "Restrained and neon-celebratory treatments conflict")
    return {"valid": not conflicts, "conflicts": conflicts,
            "requires_clarification": bool(conflicts) and any(
                contract.get("semantic_provenance", {}).get(field) == "USER_EXPLICIT"
                for item in conflicts for field in item["fields"])}


def semantic_completeness(contract: dict) -> dict:
    required = ("primary_visual_subject", "palette")
    optional = ("atmosphere_descriptors", "audience_descriptors", "motif_descriptors",
                "motif_structure_description", "material_descriptors", "composition_affinities")
    status = {key: ("PRESENT" if contract.get(key) else "REQUIRED_BUT_MISSING") for key in required}
    status.update({key: ("PRESENT" if contract.get(key) else "OPTIONAL_NOT_PROVIDED") for key in optional})
    return {"fields": status, "score": sum(status[key] == "PRESENT" for key in required) / len(required),
            "ready": all(status[key] == "PRESENT" for key in required)}
