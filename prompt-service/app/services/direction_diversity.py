"""Small, brief-compatible structural plan for successive directions."""

import re

from app.schemas.creative_direction import CreativeDirectionOutput
from app.services.grill_me_semantic import grounded_scene_family
from app.services.validation import direction_signature


_FIELDS = (
    "focal_horizontal_band", "focal_vertical_band", "viewpoint",
    "composition_family", "major_flow", "mass_distribution",
    "depth_plan", "perimeter_plan", "quiet_region_pair",
)


def locked_structural_fields(brief: dict) -> dict[str, str]:
    """Map supported explicit composition language to signature dimensions."""
    context = brief.get("grill_me_context") or {}
    semantics = brief.get("visual_semantics") or context.get("visual_semantics") or {}
    provenance = brief.get("semantic_provenance") or context.get("semantic_provenance") or {}
    if provenance.get("composition_affinities") not in {"USER_EXPLICIT", "GRILL_ME_CLARIFIED"}:
        return {}
    affinities = semantics.get("composition_affinities") or []
    if isinstance(affinities, str):
        affinities = [affinities]
    text = " ".join(str(item) for item in affinities).casefold()
    rules = {
        "viewpoint": ((r"first[- ]person", "first-person"), (r"aerial|overhead|bird.?s.eye", "aerial"),
                      (r"low[- ]angle|looking up", "low-angle"), (r"frontal|front[- ]on", "frontal"),
                      (r"oblique", "oblique"), (r"panoramic|wide[- ]angle", "panoramic")),
        "depth_plan": ((r"reced|strong perspective|forward traversal|near[- ]to[- ]far", "receding"),
                       (r"shallow|flat depth", "shallow"), (r"layered depth", "layered")),
        "major_flow": ((r"arc(?:ing)? spatial motion|curved flow|sweep(?:ing)? (?:motion|movement)", "curved"),
                       (r"diagonal flow", "diagonal"), (r"radial flow", "radial"),
                       (r"horizontal flow", "horizontal"), (r"vertical flow", "vertical")),
        "composition_family": ((r"\bsymmetr(?:ic|ical|y)\b", "symmetric"),),
    }
    return {field: next(value for pattern, value in options if re.search(pattern, text))
            for field, options in rules.items() if any(re.search(pattern, text) for pattern, _ in options)}


def _scene_kind(brief: dict) -> str:
    family = grounded_scene_family(brief)
    return family if family in {"organic", "architectural-interior", "abstract"} else "general"


def _compatible_choices(brief: dict) -> dict[str, tuple[str, ...]]:
    kind = _scene_kind(brief)
    composition = {
        "organic": ("asymmetric-cluster", "diagonal-depth", "panoramic-layered", "radial"),
        "abstract": ("central-focal", "diagonal-depth", "asymmetric-cluster", "panoramic-layered", "radial"),
        "architectural-interior": ("central-focal", "diagonal-depth", "panoramic-layered", "asymmetric-cluster"),
        "general": ("asymmetric-cluster", "central-focal", "diagonal-depth", "panoramic-layered"),
    }[kind]
    semantics = brief.get("visual_semantics") or (brief.get("grill_me_context") or {}).get("visual_semantics") or {}
    affinities = semantics.get("composition_affinities") or []
    if isinstance(affinities, list):
        preferred = [value for value in affinities if value in composition]
        composition = tuple(dict.fromkeys((*preferred, *composition)))
    viewpoint = {
        "organic": ("low-angle", "panoramic", "close-crop", "oblique"),
        "abstract": ("close-crop", "low-angle", "panoramic", "oblique"),
        "architectural-interior": ("frontal", "oblique", "panoramic", "low-angle"),
        "general": ("frontal", "oblique", "panoramic", "low-angle"),
    }[kind]
    choices = {
        "focal_horizontal_band": ("left", "right", "centre"),
        "focal_vertical_band": ("upper", "middle", "lower"),
        "viewpoint": viewpoint,
        "composition_family": composition,
        "major_flow": ("diagonal", "horizontal", "curved", "vertical", "radial"),
        "mass_distribution": ("left-heavy", "right-heavy", "distributed", "central", "asymmetric"),
        "depth_plan": ("receding", "layered", "shallow"),
        "perimeter_plan": ("open", "framed"),
        "quiet_region_pair": ("upper/lower", "left/right", "right/lower", "lower/upper"),
    }
    for field, value in locked_structural_fields(brief).items():
        choices[field] = (value,)
    return choices


def build_diversity_constraints(
    direction_id: str,
    previous_directions: list[CreativeDirectionOutput],
    brief: dict,
    *,
    retry_count: int = 0,
    failed_diagnostic: dict | None = None,
) -> dict:
    """Expose occupied choices and viable alternatives before generation."""
    signatures = [
        {"direction_id": item.direction_id, **direction_signature(item)}
        for item in previous_directions
    ]
    choices = _compatible_choices(brief)
    locked = locked_structural_fields(brief)
    occupied = {
        field: sorted({signature[field] for signature in signatures
                       if signature[field] != "unspecified"})
        for field in _FIELDS
    }
    available = {
        field: [choice for choice in options if choice not in occupied[field]]
        for field, options in choices.items() if field not in locked
    }
    rejected = (failed_diagnostic or {}).get("rejected_signature") or {}
    selected = {}
    for field, options in choices.items():
        if field in locked:
            selected[field] = locked[field]
            continue
        candidates = available.get(field) or list(options)
        if retry_count and len(candidates) > 1:
            candidates = [value for value in candidates if value != rejected.get(field)] or candidates
        if candidates:
            selected[field] = candidates[retry_count % len(candidates)]
    return {
        "direction_id": direction_id,
        "scene_kind": _scene_kind(brief),
        "previous_signatures": signatures,
        "occupied": occupied,
        "available_alternatives": available,
        "locked_fields": locked,
        "unlocked_fields": [field for field in _FIELDS if field not in locked],
        "suggested_combination": selected,
        "required_material_changes_per_previous": 2,
        "retry_count": retry_count,
        "failed_diagnostic": failed_diagnostic,
        "instruction": (
            "Choose a brief-compatible combination that changes at least two material "
            "structural fields relative to EACH accepted direction. Treat suggestions as "
            "options, not mandatory scene content; preserve the requested subject."
        ),
    }
