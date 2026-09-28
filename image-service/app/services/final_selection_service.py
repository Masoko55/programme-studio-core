"""Create the single final programme from the client-selected background."""

import hashlib
import json
from pathlib import Path

from app.schemas.final_selection import (
    FinalSelectionRequest,
)
from app.services.atomic import write_json
from app.services.composer import compose_programme
from app.services.job_persistence import (
    load_image_job_state,
)
from app.services.prompt_repository import (
    get_direction,
    get_job_directory,
    load_prompts_document,
)


def _asset_hash(
    path: str | None,
) -> str | None:
    if path is None:
        return None

    asset = Path(path)

    if not asset.is_file():
        raise ValueError(
            f"Supplied asset does not exist: {path}"
        )

    if asset.suffix.lower() not in {
        ".jpg",
        ".jpeg",
        ".png",
    }:
        raise ValueError(
            "Supplied asset must be a JPEG or PNG."
        )

    return hashlib.sha256(
        asset.read_bytes()
    ).hexdigest()

def _selection_layout(
    layout: dict,
    request: FinalSelectionRequest,
) -> dict:
    result = {
        "title_zone": (
            layout[
                "title_zone"
            ]
        ),
        "programme_zone": (
            layout[
                "programme_zone"
            ]
        ),
    }

    headshot_side = (
        request.headshot_placement
        or "left"
    )

    logo_side = (
        request.logo_placement
        or "right"
    )

    if (
        request.headshot_path
        and request.logo_path
        and headshot_side
        == logo_side
    ):
        raise ValueError(
            "Headshot and logo placements must "
            "use different sides."
        )

    if request.headshot_path:
        result[
            "headshot_zone"
        ] = {
            "x": (
                0.12
                if headshot_side
                == "left"
                else 0.70
            ),
            "y": 0.25,
            "width": 0.18,
            "height": 0.15,
            "shape": (
                request.headshot_shape
                or "rounded"
            ),
        }

    if request.logo_path:
        result[
            "logo_zone"
        ] = {
            "x": (
                0.12
                if logo_side
                == "left"
                else 0.68
            ),
            "y": 0.25,
            "width": 0.22,
            "height": 0.15,
        }

    return result



def select_final(
    reference_number: str,
    request: FinalSelectionRequest,
) -> dict:
    """Turn exactly one selected background into the final programme."""

    state = load_image_job_state(
        reference_number
    )

    if state.status not in {
        "awaiting_selection",
        "generated",  # compatibility with older jobs
    }:
        raise ValueError(
            "Background generation must finish "
            "before selecting a final image."
        )

    output = next(
        (
            item
            for item in state.outputs
            if (
                item.engine_id
                == request.engine_id
                and item.direction_id
                == request.direction_id
            )
        ),
        None,
    )

    if (
        output is None
        or output.status != "complete"
        or not output.output_path
    ):
        raise ValueError(
            "Selected candidate is not a completed "
            "background output."
        )

    document = load_prompts_document(
        reference_number
    )

    direction = get_direction(
        document,
        request.direction_id,
    )

    layout = (
        direction.get("layout_contract")
        or direction.get("layout_guidance")
    )

    if not layout:
        raise ValueError(
            "Selected direction has no layout contract."
        )

    job_directory = get_job_directory(
        reference_number
    )

    pending_path = (
        job_directory
        / "pending-final.json"
    )

    pending = (
        json.loads(
            pending_path.read_text(
                encoding="utf-8"
            )
        )
        if pending_path.exists()
        else {}
    )

    details = (
        request.programme_details.model_dump()
        if request.programme_details
        else pending
    )

    if not details:
        raise ValueError(
            "No programme details are available "
            "for final composition."
        )

    programme = details.get(
        "programme",
        []
    )

    if len(programme) > 15:
        raise ValueError(
            "A programme may contain at most 15 rows."
        )

    headshot_path = (
        request.headshot_path
        if request.headshot_path is not None
        else pending.get("headshot_path")
    )

    logo_path = (
        request.logo_path
        if request.logo_path is not None
        else pending.get("logo_path")
    )

    consent_confirmed = (
        request.rights_and_consent_confirmed
        or pending.get(
            "rights_and_consent_confirmed",
            False,
        )
    )

    if (
        headshot_path or logo_path
    ) and not consent_confirmed:
        raise ValueError(
            "Confirmed rights and consent are "
            "required for selected assets."
        )

    asset_hashes = {
        "headshot": _asset_hash(
            headshot_path
        ),
        "logo": _asset_hash(
            logo_path
        ),
    }

    selected_brief = {
        **details,
        "headshot_path": headshot_path,
        "logo_path": logo_path,
    }

    selection_request = (
        request.model_copy(
            update={
                "headshot_path": (
                    headshot_path
                ),
                "logo_path": (
                    logo_path
                ),
                "headshot_shape": (
                    request.headshot_shape
                    or pending.get(
                        "headshot_shape"
                    )
                ),
                "headshot_placement": (
                    request.headshot_placement
                    or pending.get(
                        "headshot_placement"
                    )
                ),
                "logo_placement": (
                    request.logo_placement
                    or pending.get(
                        "logo_placement"
                    )
                ),
            }
        )
    )

    selected_layout = _selection_layout(
        layout,
        selection_request,
    )

    # There is only ONE final programme.
    #
    # The nine generated PNGs remain background candidates under
    # /backgrounds. Only this selected candidate is composed with
    # title, date, venue and programme information.
    final_path = (
        job_directory
        / "final"
        / "programme.png"
    )

    result = compose_programme(
        reference_number=reference_number,
        engine_id=request.engine_id,
        direction_id=request.direction_id,
        background_path=output.output_path,
        brief=selected_brief,
        layout_guidance=selected_layout,
        output_path=final_path,
    )

    record = {
        "reference_number": (
            reference_number.upper()
        ),
        "status": "selected",
        "selection": {
            "engine_id": request.engine_id,
            "direction_id": (
                request.direction_id
            ),
        },
        "asset_sha256": asset_hashes,
        "source_candidate": {
            "engine_id": output.engine_id,
            "direction_id": (
                output.direction_id
            ),
            "background_path": (
                output.output_path
            ),
            "background_sha256": (
                output.sha256
            ),
        },
        "selected_output": (
            result.model_dump(
                mode="json"
            )
        ),
    }

    write_json(
        job_directory
        / "selected-final.json",
        record,
    )

    state.status = "selected"
    state.current_stage = (
        "final_programme_composed"
    )

    from app.services.job_persistence import (
        persist_image_job_state,
    )

    persist_image_job_state(
        state
    )

    return record