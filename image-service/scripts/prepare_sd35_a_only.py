import sys

from app.services.job_persistence import (
    load_image_job_state,
    persist_image_job_state,
)


if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: prepare_sd35_a_only.py REFERENCE_NUMBER"
    )


REFERENCE = sys.argv[1].strip().upper()

state = load_image_job_state(
    REFERENCE
)


for output in state.outputs:
    is_target = (
        output.engine_id == "sd-3-5-medium"
        and output.direction_id == "A"
    )

    if is_target:
        output.status = "pending"
        output.output_path = None
        output.sha256 = None
        output.error = None
        output.seed = None
        output.prompt_id = None
        output.workflow_sha256 = None

    else:
        output.status = "failed"
        output.output_path = None
        output.sha256 = None
        output.error = (
            "Skipped for isolated SD3.5 "
            "direction A production test."
        )
        output.seed = None
        output.prompt_id = None
        output.workflow_sha256 = None


state.status = "ready"
state.current_stage = "planned"
state.completed_outputs = 0
state.error = None


persist_image_job_state(
    state
)


print(
    "Prepared isolated production test:"
)

print(
    "REFERENCE:",
    REFERENCE,
)

print(
    "TARGET:",
    "sd-3-5-medium direction A",
)