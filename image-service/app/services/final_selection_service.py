"""Compose one client-selected candidate with optional supplied assets."""
import hashlib
from pathlib import Path

from app.schemas.final_selection import FinalSelectionRequest
from app.services.atomic import write_json
from app.services.composer import compose_programme
from app.services.composition_service import get_final_output_path
from app.services.job_persistence import load_image_job_state
from app.services.prompt_repository import get_direction, get_job_directory, load_prompts_document


def _asset_hash(path: str | None) -> str | None:
    if path is None:
        return None
    asset = Path(path)
    if not asset.is_file():
        raise ValueError(f"Supplied asset does not exist: {path}")
    if asset.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Supplied asset must be a JPEG or PNG")
    return hashlib.sha256(asset.read_bytes()).hexdigest()


def _selection_layout(layout: dict, request: FinalSelectionRequest) -> dict:
    """Keep frozen title/agenda geometry and add asset zones only when chosen."""
    result = {
        "title_zone": layout["title_zone"],
        "programme_zone": layout["programme_zone"],
    }
    if request.headshot_path:
        result["headshot_zone"] = {"x": 0.10, "y": 0.24, "width": 0.22, "height": 0.18}
    if request.logo_path:
        result["logo_zone"] = {"x": 0.68, "y": 0.24, "width": 0.22, "height": 0.18}
    return result


def select_final(reference_number: str, request: FinalSelectionRequest) -> dict:
    """Persist a client decision and a deterministic selected A4 PNG."""
    state = load_image_job_state(reference_number)
    output = next(
        (
            item for item in state.outputs
            if item.engine_id == request.engine_id and item.direction_id == request.direction_id
        ),
        None,
    )
    if output is None or output.status != "complete" or not output.output_path:
        raise ValueError("Selected candidate is not a completed image output")

    document = load_prompts_document(reference_number)
    direction = get_direction(document, request.direction_id)
    layout = direction.get("layout_contract") or direction.get("layout_guidance")
    if not layout:
        raise ValueError("Selected direction has no layout contract")

    asset_hashes = {
        "headshot": _asset_hash(request.headshot_path),
        "logo": _asset_hash(request.logo_path),
    }
    selected_brief = {
        **document["brief"],
        "headshot_path": request.headshot_path,
        "logo_path": request.logo_path,
    }
    selected_layout = _selection_layout(layout, request)
    selection_path = (
        get_job_directory(reference_number) / "selected" /
        f"{request.engine_id}-{request.direction_id.lower()}.png"
    )
    result = compose_programme(
        reference_number=reference_number,
        engine_id=request.engine_id,
        direction_id=request.direction_id,
        background_path=output.output_path,
        brief=selected_brief,
        layout_guidance=selected_layout,
        output_path=selection_path,
    )
    record = {
        "reference_number": reference_number.upper(),
        "status": "selected",
        "selection": request.model_dump(),
        "asset_sha256": asset_hashes,
        "source_candidate": {
            "engine_id": output.engine_id,
            "direction_id": output.direction_id,
            "background_path": output.output_path,
            "background_sha256": output.sha256,
            "candidate_preview_path": str(
                get_final_output_path(reference_number, output.engine_id, output.direction_id)
            ),
        },
        "selected_output": result.model_dump(mode="json"),
    }
    write_json(get_job_directory(reference_number) / "selected-final.json", record)
    return record
