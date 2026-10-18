"""Small, brief-compatible structural plan for successive directions."""

import re

from app.schemas.creative_direction import CreativeDirectionOutput
from app.services.grill_me_semantic import grounded_scene_family
from app.services.validation import STRUCTURAL_SIGNATURE_CHOICES, direction_signature


_FIELDS = (
    "focal_horizontal_band", "focal_vertical_band", "viewpoint",
    "composition_family", "major_flow", "mass_distribution",
    "depth_plan", "perimeter_plan", "quiet_region_pair",
)


_LOCK_TARGET = {
    "viewpoint": "composition_strategy",
    "depth_plan": "depth_strategy",
    "major_flow": "composition_strategy",
    "composition_family": "composition_strategy",
}
_LOCK_PHRASE = {
    "viewpoint": lambda value: f"viewpoint: {value}",
    "depth_plan": lambda value: f"depth plan: {value}",
    "major_flow": lambda value: f"major flow: {value}",
    "composition_family": lambda value: f"composition family: {value}",
}


def materialize_locked_structural_fields(
    direction: CreativeDirectionOutput, locked_fields: dict[str, str]
) -> CreativeDirectionOutput:
    """Write only explicit structural locks into the model's existing strategies."""
    if not locked_fields:
        return direction
    changes = {}
    for field in ("viewpoint", "depth_plan", "major_flow", "composition_family"):
        value = locked_fields.get(field)
        if value is None:
            continue
        if value not in {label for _, label in STRUCTURAL_SIGNATURE_CHOICES[field]}:
            raise ValueError(f"Unsupported structural lock: {field}={value}")
        target = _LOCK_TARGET[field]
        strategy_fields = (
            ("depth_strategy",) if field == "depth_plan" else
            ("composition_strategy",) if field == "composition_family" else
            ("composition_strategy", "focal_strategy", "depth_strategy", "border_or_perimeter_strategy")
        )
        for strategy_field in strategy_fields:
            prose = changes.get(strategy_field, getattr(direction, strategy_field))
            for pattern, label in STRUCTURAL_SIGNATURE_CHOICES[field]:
                if label != value:
                    if field == "depth_plan" and label == "layered":
                        continue  # Layering can coexist with receding or shallow depth.
                    suffix = {
                        "viewpoint": r"(?:\s+(?:viewpoint|view|angle|perspective))?",
                        "depth_plan": r"(?:\s+(?:depth|perspective))?",
                        "major_flow": r"(?:\s+(?:visual|spatial))?\s+(?:flow|motion|movement|direction)",
                        "composition_family": r"(?:\s+(?:composition|arrangement|layout))?",
                    }[field]
                    if field == "viewpoint" and label in {"panoramic", "close-crop"}:
                        suffix = r"\s+(?:viewpoint|view|angle|perspective)"
                    marker = field.replace("_", " ")
                    prose = re.sub(
                        rf"(?:\b{marker}:\s*)?\b(?:{pattern})[a-z-]*\b{suffix}",
                        "", prose, flags=re.I,
                    )
            prose = re.sub(r"\s+", " ", prose).strip(" ,;.")
            if len(prose) < 8:
                prose = _LOCK_PHRASE[field](value)
            changes[strategy_field] = prose
        prose = changes.get(target, getattr(direction, target))
        changes[target] = f"{_LOCK_PHRASE[field](value)}; {prose}".strip(" ;")
    return direction.model_copy(update=changes)


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
