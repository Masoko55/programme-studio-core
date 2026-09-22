from app.engines.registry import (
    get_engine_descriptors,
)


def validate_prompts_document(
    prompts_document: dict,
) -> None:
    directions = prompts_document.get(
        "directions"
    )

    if not isinstance(
        directions,
        list,
    ):
        raise ValueError(
            "prompts.json does not contain "
            "a valid directions array."
        )

    if len(directions) != 3:
        raise ValueError(
            "prompts.json must contain exactly "
            "three creative directions."
        )

    found_ids = set()

    for item in directions:
        direction = item.get(
            "direction",
            {}
        )

        direction_id = direction.get(
            "direction_id"
        )

        if direction_id not in {
            "A",
            "B",
            "C",
        }:
            raise ValueError(
                "Each creative direction must "
                "have ID A, B, or C."
            )

        found_ids.add(
            direction_id
        )

    if found_ids != {
        "A",
        "B",
        "C",
    }:
        raise ValueError(
            "prompts.json must contain "
            "directions A, B, and C."
        )


def build_execution_plan(
    prompts_document: dict,
) -> dict:
    validate_prompts_document(
        prompts_document
    )

    reference_number = (
        prompts_document[
            "reference_number"
        ]
    )

    engines = get_engine_descriptors()

    execution_order = []

    for engine in engines:
        for direction_id in [
            "A",
            "B",
            "C",
        ]:
            execution_order.append(
                {
                    "engine_id": (
                        engine.engine_id
                    ),
                    "engine_label": (
                        engine.label
                    ),
                    "direction": (
                        direction_id
                    ),
                }
            )

    return {
        "reference_number": (
            reference_number
        ),
        "total_outputs": 9,
        "execution_order": (
            execution_order
        ),
    }