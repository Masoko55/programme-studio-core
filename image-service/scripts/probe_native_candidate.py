"""Generate one native-engine candidate under a fresh reference.

This probe intentionally generates exactly ONE image. It can either use the
production profile selector or an explicitly named sampling profile.

Examples:

    python scripts/probe_native_candidate.py \
        --source-reference 5592AE-741489 \
        --engine sd-3-5-medium \
        --direction A

    python scripts/probe_native_candidate.py \
        --source-reference 5592AE-741489 \
        --engine sd-3-5-medium \
        --direction A \
        --sampling-profile explicit_shift_3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid

from pathlib import Path


sys.path.insert(
    0,
    str(
        Path(
            __file__
        ).resolve().parents[1]
    ),
)


from app.config.settings import settings
from app.services.atomic import write_json
from app.services.candidate_spec import (
    build_candidate_spec,
)
from app.services.comfyui_client import (
    ComfyUIClient,
    ComfyUIError,
)
from app.services.prompt_compiler import (
    compile_candidate_prompt,
)
from app.services.prompt_repository import (
    load_prompts_document,
)
from app.services.sampling_profiles import (
    profile_by_name,
    probe_profiles,
)


def _new_reference() -> str:
    return (
        uuid.uuid4()
        .hex[:6]
        .upper()
        + "-"
        + str(
            uuid.uuid4().int
            % 1_000_000
        ).zfill(
            6
        )
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--source-reference",
        required=True,
    )

    parser.add_argument(
        "--engine",
        required=True,
        choices=(
            settings.engine_2_id,
            settings.engine_3_id,
        ),
    )

    parser.add_argument(
        "--direction",
        required=True,
        choices=(
            "A",
            "B",
            "C",
        ),
    )

    parser.add_argument(
        "--sampling-profile",
        default=None,
        help=(
            "Optional explicit sampling profile. "
            "If omitted, production profile selection "
            "is used."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional fixed 63-bit seed for controlled "
            "profile comparison."
        ),
    )

    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help=(
            "Print available profiles for the selected "
            "engine and exit."
        ),
    )

    return parser.parse_args()


async def main() -> None:
    args = _parse_args()

    if (
        args.list_profiles
    ):
        for profile in probe_profiles(
            args.engine
        ):
            print(
                json.dumps(
                    profile.record(),
                    ensure_ascii=False,
                )
            )

        return

    profile = None

    if (
        args.sampling_profile
    ):
        profile = profile_by_name(
            args.engine,
            args.sampling_profile,
        )

    document = (
        load_prompts_document(
            args.source_reference
        )
    )

    reference = (
        _new_reference()
    )

    directory = (
        settings.programme_data_path
        / reference
    )

    directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    document[
        "reference_number"
    ] = reference

    write_json(
        directory
        / "prompts.json",
        document,
    )

    write_json(
        directory
        / "brief.json",
        document[
            "brief"
        ],
    )

    write_json(
        directory
        / "probe-source.json",
        {
            "source_reference": (
                args.source_reference
            ),
            "engine": (
                args.engine
            ),
            "direction": (
                args.direction
            ),
            "sampling_profile": (
                profile.record()
                if profile
                else None
            ),
            "seed_override": (
                args.seed
            ),
        },
    )

    print(
        "NEW_REFERENCE="
        + reference,
        flush=True,
    )

    spec = (
        build_candidate_spec(
            document,
            args.engine,
            args.direction,
        )
    )

    compiled = (
        compile_candidate_prompt(
            spec,
            1,
        )
    )

    print(
        "SPEC_SHA256="
        + spec.spec_sha256,
        flush=True,
    )

    print(
        "RETRY_STAGE="
        + compiled.retry_stage,
        flush=True,
    )

    if profile:
        print(
            "SAMPLING_PROFILE="
            + json.dumps(
                profile.record(),
                sort_keys=True,
            ),
            flush=True,
        )

    try:
        async with (
            ComfyUIClient()
            as client
        ):
            await client.health()

            result = (
                await client.generate_image(
                    reference,
                    args.engine,
                    args.direction,
                    compiled.positive,
                    compiled.negative,
                    compiled_prompt=(
                        compiled
                    ),
                    spec_sha256=(
                        spec.spec_sha256
                    ),
                    direction_role=(
                        spec.direction_role
                    ),
                    retry_stage=(
                        compiled.retry_stage
                    ),
                    failure_category=(
                        compiled
                        .failure_category
                    ),
                    sampling_profile=(
                        profile
                    ),
                    seed_override=(
                        args.seed
                    ),
                )
            )

    except ComfyUIError as error:
        record_path = (
            directory
            / "backgrounds"
            / args.engine
            / (
                "image-"
                + args.direction.lower()
                + ".json"
            )
        )

        stored = {}

        if (
            record_path.exists()
        ):
            stored = json.loads(
                record_path.read_text(
                    encoding="utf-8"
                )
            )

        print(
            json.dumps(
                {
                    "reference": (
                        reference
                    ),
                    "engine": (
                        args.engine
                    ),
                    "direction": (
                        args.direction
                    ),
                    "status": (
                        stored.get(
                            "status",
                            "rejected",
                        )
                    ),
                    "error": (
                        str(
                            error
                        )
                    ),
                    "sampling_profile": (
                        stored.get(
                            "sampling_profile"
                        )
                    ),
                    "retry_stage": (
                        stored.get(
                            "retry_stage"
                        )
                    ),
                    "failure_category": (
                        stored.get(
                            "failure_category"
                        )
                    ),
                    "outcome_category": (
                        stored.get(
                            "outcome_category"
                        )
                    ),
                    "spec_sha256": (
                        stored.get(
                            "spec_sha256"
                        )
                    ),
                    "seed": (
                        stored.get(
                            "seed"
                        )
                    ),
                    "rejection_reason": (
                        stored.get(
                            "rejection_reason"
                        )
                    ),
                },
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )

        raise

    print(
        json.dumps(
            {
                "reference": (
                    reference
                ),
                "engine": (
                    args.engine
                ),
                "direction": (
                    args.direction
                ),
                "status": (
                    result[
                        "status"
                    ]
                ),
                "sampling_profile": (
                    result.get(
                        "sampling_profile"
                    )
                ),
                "retry_stage": (
                    result.get(
                        "retry_stage"
                    )
                ),
                "spec_sha256": (
                    result.get(
                        "spec_sha256"
                    )
                ),
                "seed": (
                    result.get(
                        "seed"
                    )
                ),
                "output_path": (
                    result.get(
                        "output_path"
                    )
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(
        main()
    )