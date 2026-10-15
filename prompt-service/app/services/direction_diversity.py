"""Small, brief-compatible structural plan for successive directions."""

from app.schemas.creative_direction import CreativeDirectionOutput
from app.services.validation import direction_signature


_FIELDS = (
    "focal_horizontal_band", "focal_vertical_band", "viewpoint",
    "composition_family", "major_flow", "mass_distribution",
    "depth_plan", "perimeter_plan", "quiet_region_pair",
)


def _scene_kind(brief: dict) -> str:
    context = " ".join(str(brief.get(key) or "") for key in (
        "event_type", "theme", "background_inspiration", "creative_description"
    )).casefold()
    if any(word in context for word in ("garden", "botanic", "flower", "foliage")):
        return "organic"
    if any(word in context for word in ("ballroom", "black-tie", "black tie", "formal")):
        return "architectural-interior"
    if any(word in context for word in ("abstract", "sculptur", "geometric")):
        return "abstract"
    return "general"


def _compatible_choices(brief: dict) -> dict[str, tuple[str, ...]]:
    kind = _scene_kind(brief)
    composition = {
        "organic": ("asymmetric-cluster", "diagonal-depth", "panoramic-layered", "radial"),
        "abstract": ("central-monument", "diagonal-depth", "asymmetric-cluster", "panoramic-layered", "radial"),
        "architectural-interior": ("central-monument", "diagonal-depth", "panoramic-layered", "asymmetric-cluster"),
        "general": ("asymmetric-cluster", "central-monument", "diagonal-depth", "panoramic-layered"),
    }[kind]
    viewpoint = {
        "organic": ("low-angle", "panoramic", "close-crop", "oblique"),
        "abstract": ("close-crop", "low-angle", "panoramic", "oblique"),
        "architectural-interior": ("frontal", "oblique", "panoramic", "low-angle"),
        "general": ("frontal", "oblique", "panoramic", "low-angle"),
    }[kind]
    return {
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
    occupied = {
        field: sorted({signature[field] for signature in signatures
                       if signature[field] != "unspecified"})
        for field in _FIELDS
    }
    available = {
        field: [choice for choice in options if choice not in occupied[field]]
        for field, options in choices.items()
    }
    selected = {field: values[0] for field, values in available.items() if values}
    return {
        "direction_id": direction_id,
        "scene_kind": _scene_kind(brief),
        "previous_signatures": signatures,
        "occupied": occupied,
        "available_alternatives": available,
        "suggested_combination": selected,
        "required_material_changes_per_previous": 3,
        "retry_count": retry_count,
        "failed_diagnostic": failed_diagnostic,
        "instruction": (
            "Choose a brief-compatible combination that changes at least three material "
            "structural fields relative to EACH accepted direction. Treat suggestions as "
            "options, not mandatory scene content; preserve the requested subject."
        ),
    }
