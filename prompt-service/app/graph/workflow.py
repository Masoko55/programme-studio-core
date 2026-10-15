import logging
import json

from datetime import (
    datetime,
    timezone,
)

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from app.config.settings import (
    settings,
)

from app.graph.state import (
    PromptWorkflowState,
)

from app.schemas.prompts_document import (
    DirectionRecord,
    ModelProvenance,
    PromptsDocument,
)

from app.services.creative_direction import (
    direction_structural_summary,
    generate_creative_direction,
)
from app.services.direction_diversity import build_diversity_constraints

from app.services.persistence import (
    persist_direction_checkpoint,
    update_job_status,
)

from app.services.validation import (
    validate_creative_direction,
    validate_direction_diversity,
)


logger = logging.getLogger(
    "uvicorn.error"
)


MAX_RETRIES = 2


def _retry_count(
    state: PromptWorkflowState,
    direction_id: str,
) -> int:
    return int(
        state.get(
            (
                "retry_count_"
                + direction_id.lower()
            ),
            0,
        )
        or 0
    )


def _validation_error(
    state: PromptWorkflowState,
    direction_id: str,
):
    return (
        state.get(
            (
                "validation_error_"
                + direction_id.lower()
            )
        )
    )


def _log_generation_failure(
    direction_id: str,
    retry_count: int,
    error: Exception,
) -> None:
    label = ("initial" if retry_count == 1 else
             f"retry={retry_count - 1}/{MAX_RETRIES}")
    logger.warning(
        "Creative direction generation failed: "
        "direction=%s attempt=%s/%s %s error=%s",
        direction_id,
        retry_count,
        MAX_RETRIES + 1,
        label,
        str(
            error
        ),
    )


def _log_validation_failure(
    direction_id: str,
    retry_count: int,
    error: Exception,
) -> None:
    label = ("initial" if retry_count == 1 else
             f"retry={retry_count - 1}/{MAX_RETRIES}")
    logger.warning(
        "Creative direction validation failed: "
        "direction=%s attempt=%s/%s %s error=%s",
        direction_id,
        retry_count,
        MAX_RETRIES + 1,
        label,
        str(
            error
        ),
    )


def _diversity_failure(error: ValueError, direction_id: str) -> dict | None:
    diagnostic = getattr(error, "diversity_diagnostics", None)
    if isinstance(diagnostic, dict):
        logger.warning("direction_diversity=%s", json.dumps(
            {"direction_id": direction_id, **diagnostic}, sort_keys=True
        ))
        return diagnostic
    return None


async def generate_direction_a(
    state: PromptWorkflowState,
) -> dict:
    existing = (
        state.get(
            "direction_a"
        )
    )

    if existing is not None:
        logger.info(
            "Reusing Direction A checkpoint"
        )

        return {}

    current_retry = (
        _retry_count(
            state,
            "A",
        )
    )

    try:
        direction = (
            await generate_creative_direction(
                direction_id="A",
                brief=state[
                    "brief"
                ],
                correction_error=(
                    state.get(
                        "validation_error_a"
                    )
                ),
            )
        )

        return {
            "direction_a": (
                direction
            ),
            "validation_error_a": None,
        }

    except ValueError as error:
        next_retry = (
            current_retry
            + 1
        )

        _log_generation_failure(
            "A",
            next_retry,
            error,
        )

        return {
            "direction_a": None,
            "validation_error_a": (
                str(
                    error
                )
            ),
            "retry_count_a": (
                next_retry
            ),
        }


def validate_direction_a(
    state: PromptWorkflowState,
) -> dict:
    if (
        state.get(
            "direction_a"
        )
        is None
    ):
        return {}

    try:
        validate_creative_direction(
            state[
                "direction_a"
            ],
            state[
                "brief"
            ],
        )

        logger.info(
            "Direction A validation passed"
        )

        return {
            "validation_error_a": None,
        }

    except ValueError as error:
        next_retry = (
            _retry_count(
                state,
                "A",
            )
            + 1
        )

        _log_validation_failure(
            "A",
            next_retry,
            error,
        )

        return {
            "validation_error_a": (
                str(
                    error
                )
            ),
            "retry_count_a": (
                next_retry
            ),
            "direction_a": None,
        }


def checkpoint_direction_a(
    state: PromptWorkflowState,
) -> dict:
    persist_direction_checkpoint(
        reference_number=(
            state[
                "reference_number"
            ]
        ),
        direction_id="A",
        direction=(
            state[
                "direction_a"
            ]
        ),
    )

    update_job_status(
        reference_number=(
            state[
                "reference_number"
            ]
        ),
        status="processing",
        current_stage=(
            "direction_a_complete"
        ),
    )

    logger.info(
        "Saved Direction A checkpoint"
    )

    return {}


def route_after_validation_a(
    state: PromptWorkflowState,
) -> str:
    error = (
        state.get(
            "validation_error_a"
        )
    )

    if (
        error is None
        and state.get(
            "direction_a"
        )
        is not None
    ):
        return (
            "checkpoint_direction_a"
        )

    retry_count = (
        state.get(
            "retry_count_a",
            0,
        )
    )

    if (
        retry_count
        <= MAX_RETRIES
    ):
        return (
            "direction_a"
        )

    logger.error(
        "Direction A permanently failed: "
        "retry_count=%s error=%s",
        retry_count,
        error,
    )

    raise ValueError(
        "Direction A failed validation "
        f"after {MAX_RETRIES} retries: "
        f"{error}"
    )


async def generate_direction_b(
    state: PromptWorkflowState,
) -> dict:
    existing = (
        state.get(
            "direction_b"
        )
    )

    if existing is not None:
        logger.info(
            "Reusing Direction B checkpoint"
        )

        return {}

    current_retry = (
        _retry_count(
            state,
            "B",
        )
    )

    try:
        direction = (
            await generate_creative_direction(
                direction_id="B",
                brief=state[
                    "brief"
                ],
                previous_directions=[direction_structural_summary(state["direction_a"], state["brief"])] if state.get("direction_a") else [],
                diversity_constraints=build_diversity_constraints(
                    "B", [state["direction_a"]] if state.get("direction_a") else [], state["brief"],
                    retry_count=current_retry,
                    failed_diagnostic=state.get("diversity_diagnostics_b"),
                ),
                correction_error=(
                    state.get(
                        "validation_error_b"
                    )
                ),
            )
        )

        return {
            "direction_b": (
                direction
            ),
            "validation_error_b": None,
        }

    except ValueError as error:
        next_retry = (
            current_retry
            + 1
        )

        _log_generation_failure(
            "B",
            next_retry,
            error,
        )

        return {
            "direction_b": None,
            "validation_error_b": (
                str(
                    error
                )
            ),
            "retry_count_b": (
                next_retry
            ),
        }


def validate_direction_b(
    state: PromptWorkflowState,
) -> dict:
    if (
        state.get(
            "direction_b"
        )
        is None
    ):
        return {}

    try:
        validate_creative_direction(
            state[
                "direction_b"
            ],
            state[
                "brief"
            ],
        )
        if state.get("direction_a"):
            validate_direction_diversity(state["direction_b"], [state["direction_a"]])

        logger.info(
            "Direction B validation passed"
        )

        return {
            "validation_error_b": None,
            "diversity_diagnostics_b": None,
        }

    except ValueError as error:
        diversity_diagnostic = _diversity_failure(error, "B")
        next_retry = (
            _retry_count(
                state,
                "B",
            )
            + 1
        )

        _log_validation_failure(
            "B",
            next_retry,
            error,
        )

        return {
            "validation_error_b": (
                str(
                    error
                )
            ),
            "retry_count_b": (
                next_retry
            ),
            "diversity_diagnostics_b": diversity_diagnostic,
            "direction_b": None,
        }


def checkpoint_direction_b(
    state: PromptWorkflowState,
) -> dict:
    persist_direction_checkpoint(
        reference_number=(
            state[
                "reference_number"
            ]
        ),
        direction_id="B",
        direction=(
            state[
                "direction_b"
            ]
        ),
    )

    update_job_status(
        reference_number=(
            state[
                "reference_number"
            ]
        ),
        status="processing",
        current_stage=(
            "direction_b_complete"
        ),
    )

    logger.info(
        "Saved Direction B checkpoint"
    )

    return {}


def route_after_validation_b(
    state: PromptWorkflowState,
) -> str:
    error = (
        state.get(
            "validation_error_b"
        )
    )

    if (
        error is None
        and state.get(
            "direction_b"
        )
        is not None
    ):
        return (
            "checkpoint_direction_b"
        )

    retry_count = (
        state.get(
            "retry_count_b",
            0,
        )
    )

    if (
        retry_count
        <= MAX_RETRIES
    ):
        return (
            "direction_b"
        )

    logger.error(
        "Direction B permanently failed: "
        "retry_count=%s error=%s",
        retry_count,
        error,
    )

    raise ValueError(
        "Direction B failed validation "
        f"after {MAX_RETRIES} retries: "
        f"{error}"
    )


async def generate_direction_c(
    state: PromptWorkflowState,
) -> dict:
    existing = (
        state.get(
            "direction_c"
        )
    )

    if existing is not None:
        logger.info(
            "Reusing Direction C checkpoint"
        )

        return {}

    current_retry = (
        _retry_count(
            state,
            "C",
        )
    )

    try:
        direction = (
            await generate_creative_direction(
                direction_id="C",
                brief=state[
                    "brief"
                ],
                previous_directions=[
                    direction_structural_summary(item, state["brief"])
                    for item in (state.get("direction_a"), state.get("direction_b")) if item
                ],
                diversity_constraints=build_diversity_constraints(
                    "C", [item for item in (state.get("direction_a"), state.get("direction_b")) if item],
                    state["brief"], retry_count=current_retry,
                    failed_diagnostic=state.get("diversity_diagnostics_c"),
                ),
                correction_error=(
                    state.get(
                        "validation_error_c"
                    )
                ),
            )
        )

        return {
            "direction_c": (
                direction
            ),
            "validation_error_c": None,
        }

    except ValueError as error:
        next_retry = (
            current_retry
            + 1
        )

        _log_generation_failure(
            "C",
            next_retry,
            error,
        )

        return {
            "direction_c": None,
            "validation_error_c": (
                str(
                    error
                )
            ),
            "retry_count_c": (
                next_retry
            ),
        }


def validate_direction_c(
    state: PromptWorkflowState,
) -> dict:
    if (
        state.get(
            "direction_c"
        )
        is None
    ):
        return {}

    try:
        validate_creative_direction(
            state[
                "direction_c"
            ],
            state[
                "brief"
            ],
        )
        validate_direction_diversity(
            state["direction_c"],
            [item for item in (state.get("direction_a"), state.get("direction_b")) if item],
        )

        logger.info(
            "Direction C validation passed"
        )

        return {
            "validation_error_c": None,
            "diversity_diagnostics_c": None,
        }

    except ValueError as error:
        diversity_diagnostic = _diversity_failure(error, "C")
        next_retry = (
            _retry_count(
                state,
                "C",
            )
            + 1
        )

        _log_validation_failure(
            "C",
            next_retry,
            error,
        )

        return {
            "validation_error_c": (
                str(
                    error
                )
            ),
            "retry_count_c": (
                next_retry
            ),
            "diversity_diagnostics_c": diversity_diagnostic,
            "direction_c": None,
        }


def checkpoint_direction_c(
    state: PromptWorkflowState,
) -> dict:
    persist_direction_checkpoint(
        reference_number=(
            state[
                "reference_number"
            ]
        ),
        direction_id="C",
        direction=(
            state[
                "direction_c"
            ]
        ),
    )

    update_job_status(
        reference_number=(
            state[
                "reference_number"
            ]
        ),
        status="processing",
        current_stage=(
            "direction_c_complete"
        ),
    )

    logger.info(
        "Saved Direction C checkpoint"
    )

    return {}


def route_after_validation_c(
    state: PromptWorkflowState,
) -> str:
    error = (
        state.get(
            "validation_error_c"
        )
    )

    if (
        error is None
        and state.get(
            "direction_c"
        )
        is not None
    ):
        return (
            "checkpoint_direction_c"
        )

    retry_count = (
        state.get(
            "retry_count_c",
            0,
        )
    )

    if (
        retry_count
        <= MAX_RETRIES
    ):
        return (
            "direction_c"
        )

    logger.error(
        "Direction C permanently failed: "
        "retry_count=%s error=%s",
        retry_count,
        error,
    )

    raise ValueError(
        "Direction C failed validation "
        f"after {MAX_RETRIES} retries: "
        f"{error}"
    )


def build_prompts_document(
    state: PromptWorkflowState,
) -> dict:
    document = (
        PromptsDocument(
            schema_version="1.0",
            reference_number=(
                state[
                    "reference_number"
                ]
            ),
            input_sha256=(
                state[
                    "input_sha256"
                ]
            ),
            created_at=(
                datetime.now(
                    timezone.utc
                )
            ),
            brief=(
                state[
                    "brief"
                ]
            ),
            directions=[
                DirectionRecord(
                    provenance=(
                        ModelProvenance(
                            provider="ollama",
                            model=(
                                settings
                                .direction_a_model
                            ),
                        )
                    ),
                    direction=(
                        state[
                            "direction_a"
                        ]
                    ),
                ),
                DirectionRecord(
                    provenance=(
                        ModelProvenance(
                            provider="ollama",
                            model=(
                                settings
                                .direction_b_model
                            ),
                        )
                    ),
                    direction=(
                        state[
                            "direction_b"
                        ]
                    ),
                ),
                DirectionRecord(
                    provenance=(
                        ModelProvenance(
                            provider="ollama",
                            model=(
                                settings
                                .direction_c_model
                            ),
                        )
                    ),
                    direction=(
                        state[
                            "direction_c"
                        ]
                    ),
                ),
            ],
        )
    )

    return {
        "prompts_document": (
            document
        ),
    }


builder = (
    StateGraph(
        PromptWorkflowState
    )
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
        "direction_a": (
            "direction_a"
        ),
        "checkpoint_direction_a": (
            "checkpoint_direction_a"
        ),
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
        "direction_b": (
            "direction_b"
        ),
        "checkpoint_direction_b": (
            "checkpoint_direction_b"
        ),
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
        "direction_c": (
            "direction_c"
        ),
        "checkpoint_direction_c": (
            "checkpoint_direction_c"
        ),
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


prompt_workflow = (
    builder.compile()
)
