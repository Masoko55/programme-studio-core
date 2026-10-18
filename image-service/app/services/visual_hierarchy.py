"""Derive generic visual emphasis from an immutable brief and direction."""

from __future__ import annotations

import re


def density_level(value: str) -> str:
    words = set(re.findall(r"[a-z]+", str(value).casefold()))
    if words & {"sparse", "minimal", "minimalist", "simple", "restrained"}:
        return "sparse"
    if words & {"dense", "crowded", "packed", "intricate"}:
        return "dense"
    if words & {"rich", "lush", "layered", "detailed"}:
        return "rich"
    return "moderate"


def derive_visual_hierarchy(contract: dict, direction: dict) -> dict:
    """Return renderer guidance without inferring a new subject or scene object."""
    level = density_level(contract.get("density_preference", ""))
    motif = str(contract.get("motif_structure_description") or "").strip()
    motif_words = contract.get("motif_descriptors") or []
    treatment = str(contract.get("theme_reference_treatment") or "").casefold()
    emphasized = any(str(word).casefold() in treatment for word in motif_words if len(str(word)) > 3)
    prominence = "primary" if emphasized else "supporting" if motif or motif_words or treatment else "subtle"
    safe_regions = [str(direction.get(key) or "").strip() for key in
                    ("title_safe_region_strategy", "programme_safe_region_strategy")]
    return {
        "visual_priority": ["primary subject", "required motif", "palette relationship",
                            "atmosphere", "composition structure"],
        "semantic_emphasis": {"subject": "PRIMARY", "motif": prominence.upper(),
                              "palette": "PRIMARY", "environment": "SECONDARY",
                              "decorative_detail": "SUPPORTING"},
        "detail_density": level,
        "motif_prominence": prominence,
        "subject_coverage_preference": "selective concentration with visible breathing room",
        "quiet_space_requirement": "integrated low-detail visual regions",
        "safe_region_strategies": safe_regions,
        "palette_dominance_strength": "exclusive" if contract.get("palette_mode") == "strict" else "major visual masses",
    }
