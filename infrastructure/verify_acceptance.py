#!/usr/bin/env python3

"""Verify one real Programme Studio run.

Usage:

    python3 infrastructure/verify_acceptance.py FB0B64-965302

The supplied reference must have:
1. completed Grill-Me,
2. generated all nine background candidates,
3. selected one candidate,
4. composed one final programme,
5. produced a final manifest.
"""

import json
import sys
from pathlib import Path
from urllib.request import urlopen


PROMPT_URL = (
    "http://127.0.0.1:8001"
)

IMAGE_URL = (
    "http://127.0.0.1:8002"
)


def get_json(
    url: str,
) -> dict:
    with urlopen(url) as response:
        return json.loads(
            response.read()
        )


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(
            message
        )


def verify(
    reference_number: str,
) -> None:
    prompt_job = get_json(
        f"{PROMPT_URL}/v1/prompt-jobs/"
        f"{reference_number}"
    )

    image_job = get_json(
        f"{IMAGE_URL}/v1/image-jobs/"
        f"{reference_number}"
    )

    require(
        image_job[
            "completed_outputs"
        ] == 9,
        "Expected nine generated "
        "background candidates.",
    )

    require(
        len(
            image_job[
                "outputs"
            ]
        ) == 9,
        "Expected nine candidate records.",
    )

    for candidate in image_job[
        "outputs"
    ]:
        require(
            candidate[
                "status"
            ] == "complete",
            (
                "Background candidate "
                f"{candidate['engine_id']}/"
                f"{candidate['direction_id']} "
                "is incomplete."
            ),
        )

        path = Path(
            candidate[
                "output_path"
            ]
        )

        require(
            path.exists(),
            (
                "Background candidate file "
                f"is missing: {path}"
            ),
        )

    job_directory = (
        Path("/data/programmes")
        / reference_number.upper()
    )

    selected_path = (
        job_directory
        / "selected-final.json"
    )

    require(
        selected_path.exists(),
        "No background has been selected.",
    )

    selected = json.loads(
        selected_path.read_text(
            encoding="utf-8"
        )
    )

    final_path = Path(
        selected[
            "selected_output"
        ][
            "output_path"
        ]
    )

    require(
        final_path.exists(),
        (
            "Selected final programme "
            "PNG does not exist."
        ),
    )

    require(
        final_path.name
        == "programme.png",
        (
            "Expected the final output "
            "to be programme.png."
        ),
    )

    print(
        "PASS"
    )

    print(
        "Reference:",
        reference_number.upper(),
    )

    print(
        "Background candidates:",
        9,
    )

    print(
        "Selected engine:",
        selected[
            "source_candidate"
        ][
            "engine_id"
        ],
    )

    print(
        "Selected direction:",
        selected[
            "source_candidate"
        ][
            "direction_id"
        ],
    )

    print(
        "Final programme:",
        final_path,
    )


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Usage: verify_acceptance.py "
            "<REFERENCE_NUMBER>",
            file=sys.stderr,
        )

        return 2

    verify(
        sys.argv[1]
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )

    except Exception as error:
        print(
            f"FAIL: {error}",
            file=sys.stderr,
        )

        raise SystemExit(1)