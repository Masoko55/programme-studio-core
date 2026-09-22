import logging
from datetime import datetime, timezone

from langgraph.graph import END, START, StateGraph

from app.config.settings import settings
from app.graph.state import PromptWorkflowState
from app.schemas.prompts_document import (
    DirectionRecord,
    ModelProvenance,
    PromptsDocument,
)
from app.services.creative_direction import (
    generate_creative_direction,
)
from app.services.persistence import (
    persist_direction_checkpoint,
    update_job_status,
)
from app.services.validation import (
    validate_creative_direction,
)


logger = logging.getLogger("uvicorn.error")

MAX_RETRIES = 2


async def generate_direction_a(
    state: PromptWorkflowState,
) -> dict:
    existing = state.get("direction_a")

    if existing is not None:
        logger.info(
            "Reusing Direction A checkpoint"
        )
        return {}

    try:
        direction = await generate_creative_direction(
            direction_id="A",
            brief=state["brief"],
            correction_error=state.get(
                "validation_error_a"
            ),
        )

        return {
            "direction_a": direction,
            "validation_error_a": None,
        }

    except ValueError as error:
        return {
            "direction_a": None,
            "validation_error_a": str(error),
            "retry_count_a": (
                state.get("retry_count_a", 0) + 1
            ),
        }


def validate_direction_a(
    state: PromptWorkflowState,
) -> dict:
    if state.get("direction_a") is None:
        return {}

    try:
        validate_creative_direction(
            state["direction_a"],
            state["brief"],
        )

        return {
            "validation_error_a": None,
        }

    except ValueError as error:
        return {
            "validation_error_a": str(error),
            "retry_count_a": (
                state.get("retry_count_a", 0) + 1
            ),
            "direction_a": None,
        }


def checkpoint_direction_a(
    state: PromptWorkflowState,
) -> dict:
    persist_direction_checkpoint(
        reference_number=state["reference_number"],
        direction_id="A",
        direction=state["direction_a"],
    )

    update_job_status(
        reference_number=state["reference_number"],
        status="processing",
        current_stage="direction_a_complete",
    )

    logger.info(
        "Saved Direction A checkpoint"
    )

    return {}


def route_after_validation_a(
    state: PromptWorkflowState,
) -> str:
    error = state.get("validation_error_a")

    if error is None and state.get("direction_a") is not None:
        return "checkpoint_direction_a"

    retry_count = state.get(
        "retry_count_a",
        0,
    )

    if retry_count <= MAX_RETRIES:
        return "direction_a"

    raise ValueError(
        "Direction A failed validation "
        f"after {MAX_RETRIES} retries: {error}"
    )


async def generate_direction_b(
    state: PromptWorkflowState,
) -> dict:
    existing = state.get("direction_b")

    if existing is not None:
        logger.info(
            "Reusing Direction B checkpoint"
        )
        return {}

    try:
        direction = await generate_creative_direction(
            direction_id="B",
            brief=state["brief"],
            correction_error=state.get(
                "validation_error_b"
            ),
        )

        return {
            "direction_b": direction,
            "validation_error_b": None,
        }

    except ValueError as error:
        return {
            "direction_b": None,
            "validation_error_b": str(error),
            "retry_count_b": (
                state.get("retry_count_b", 0) + 1
            ),
        }


def validate_direction_b(
    state: PromptWorkflowState,
) -> dict:
    if state.get("direction_b") is None:
        return {}

    try:
        validate_creative_direction(
            state["direction_b"],
            state["brief"],
        )

        return {
            "validation_error_b": None,
        }

    except ValueError as error:
        return {
            "validation_error_b": str(error),
            "retry_count_b": (
                state.get("retry_count_b", 0) + 1
            ),
            "direction_b": None,
        }


def checkpoint_direction_b(
    state: PromptWorkflowState,
) -> dict:
    persist_direction_checkpoint(
        reference_number=state["reference_number"],
        direction_id="B",
        direction=state["direction_b"],
    )

    update_job_status(
        reference_number=state["reference_number"],
        status="processing",
        current_stage="direction_b_complete",
    )

    logger.info(
        "Saved Direction B checkpoint"
    )

    return {}


def route_after_validation_b(
    state: PromptWorkflowState,
) -> str:
    error = state.get("validation_error_b")

    if error is None and state.get("direction_b") is not None:
        return "checkpoint_direction_b"

    retry_count = state.get(
        "retry_count_b",
        0,
    )

    if retry_count <= MAX_RETRIES:
        return "direction_b"

    raise ValueError(
        "Direction B failed validation "
        f"after {MAX_RETRIES} retries: {error}"
    )


async def generate_direction_c(
    state: PromptWorkflowState,
) -> dict:
    existing = state.get("direction_c")

    if existing is not None:
        logger.info(
            "Reusing Direction C checkpoint"
        )
        return {}

    try:
        direction = await generate_creative_direction(
            direction_id="C",
            brief=state["brief"],
            correction_error=state.get(
                "validation_error_c"
            ),
        )

        return {
            "direction_c": direction,
            "validation_error_c": None,
        }

    except ValueError as error:
        return {
            "direction_c": None,
            "validation_error_c": str(error),
            "retry_count_c": (
                state.get("retry_count_c", 0) + 1
            ),
        }


def validate_direction_c(
    state: PromptWorkflowState,
) -> dict:
    if state.get("direction_c") is None:
        return {}

    try:
        validate_creative_direction(
            state["direction_c"],
            state["brief"],
        )

        return {
            "validation_error_c": None,
        }

    except ValueError as error:
        return {
            "validation_error_c": str(error),
            "retry_count_c": (
                state.get("retry_count_c", 0) + 1
            ),
            "direction_c": None,
        }


def checkpoint_direction_c(
    state: PromptWorkflowState,
) -> dict:
    persist_direction_checkpoint(
        reference_number=state["reference_number"],
        direction_id="C",
        direction=state["direction_c"],
    )

    update_job_status(
        reference_number=state["reference_number"],
        status="processing",
        current_stage="direction_c_complete",
    )

    logger.info(
        "Saved Direction C checkpoint"
    )

    return {}


def route_after_validation_c(
    state: PromptWorkflowState,
) -> str:
    error = state.get("validation_error_c")

    if error is None and state.get("direction_c") is not None:
        return "checkpoint_direction_c"

    retry_count = state.get(
        "retry_count_c",
        0,
    )

    if retry_count <= MAX_RETRIES:
        return "direction_c"

    raise ValueError(
        "Direction C failed validation "
        f"after {MAX_RETRIES} retries: {error}"
    )


def build_prompts_document(
    state: PromptWorkflowState,
) -> dict:
    document = PromptsDocument(
        schema_version="1.0",
        reference_number=state["reference_number"],
        input_sha256=state["input_sha256"],
        created_at=datetime.now(timezone.utc),
        brief=state["brief"],
        directions=[
            DirectionRecord(
                provenance=ModelProvenance(
                    provider="ollama",
                    model=settings.direction_a_model,
                ),
                direction=state["direction_a"],
            ),
            DirectionRecord(
                provenance=ModelProvenance(
                    provider="ollama",
                    model=settings.direction_b_model,
                ),
                direction=state["direction_b"],
            ),
            DirectionRecord(
                provenance=ModelProvenance(
                    provider="ollama",
                    model=settings.direction_c_model,
                ),
                direction=state["direction_c"],
            ),
        ],
    )

    return {
        "prompts_document": document,
    }


builder = StateGraph(
    PromptWorkflowState
)

builder.add_node(
    "direction_a",
    generate_direction_a,
)
builder.add_node(
    "validate_direction_a",
    validate_direction_a,
)
builder.add_node(
    "checkpoint_direction_a",
    checkpoint_direction_a,
)

builder.add_node(
    "direction_b",
    generate_direction_b,
)
builder.add_node(
    "validate_direction_b",
    validate_direction_b,
)
builder.add_node(
    "checkpoint_direction_b",
    checkpoint_direction_b,
)

builder.add_node(
    "direction_c",
    generate_direction_c,
)
builder.add_node(
    "validate_direction_c",
    validate_direction_c,
)
builder.add_node(
    "checkpoint_direction_c",
    checkpoint_direction_c,
)

builder.add_node(
    "build_prompts_document",
    build_prompts_document,
)

builder.add_edge(
    START,
    "direction_a",
)

builder.add_edge(
    "direction_a",
    "validate_direction_a",
)

builder.add_conditional_edges(
    "validate_direction_a",
    route_after_validation_a,
    {
        "direction_a": "direction_a",
        "checkpoint_direction_a": "checkpoint_direction_a",
    },
)

builder.add_edge(
    "checkpoint_direction_a",
    "direction_b",
)

builder.add_edge(
    "direction_b",
    "validate_direction_b",
)

builder.add_conditional_edges(
    "validate_direction_b",
    route_after_validation_b,
    {
        "direction_b": "direction_b",
        "checkpoint_direction_b": "checkpoint_direction_b",
    },
)

builder.add_edge(
    "checkpoint_direction_b",
    "direction_c",
)

builder.add_edge(
    "direction_c",
    "validate_direction_c",
)

builder.add_conditional_edges(
    "validate_direction_c",
    route_after_validation_c,
    {
        "direction_c": "direction_c",
        "checkpoint_direction_c": "checkpoint_direction_c",
    },
)

builder.add_edge(
    "checkpoint_direction_c",
    "build_prompts_document",
)

builder.add_edge(
    "build_prompts_document",
    END,
)

prompt_workflow = builder.compile()