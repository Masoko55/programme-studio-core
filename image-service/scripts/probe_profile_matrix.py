"""Print commands for a controlled native-engine sampling-profile matrix.

This script does NOT launch generations itself. It prints one probe command
per profile so the engineer can run and inspect each candidate deliberately.

Example:

    python scripts/probe_profile_matrix.py \
        --source-reference 5592AE-741489 \
        --engine sd-3-5-medium \
        --direction A
"""

from __future__ import annotations

import argparse
import shlex
import sys

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
from app.services.sampling_profiles import (
    probe_profiles,
)


def main() -> None:
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
        default="A",
        choices=(
            "A",
            "B",
            "C",
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261009,
        help=(
            "Use the same seed for every profile so the "
            "sampling comparison changes model parameters "
            "rather than random noise."
        ),
    )

    args = parser.parse_args()

    profiles = probe_profiles(
        args.engine
    )

    print(
        "# Controlled profile matrix"
    )

    print(
        "# engine="
        + args.engine
    )

    print(
        "# direction="
        + args.direction
    )

    print(
        "# seed="
        + str(
            args.seed
        )
    )

    print()

    for profile in profiles:
        command = [
            "python",
            "/app/scripts/"
            "probe_native_candidate.py",
            "--source-reference",
            args.source_reference,
            "--engine",
            args.engine,
            "--direction",
            args.direction,
            "--sampling-profile",
            profile.name,
            "--seed",
            str(
                args.seed
            ),
        ]

        print(
            "# "
            + profile.name
        )

        print(
            " ".join(
                shlex.quote(
                    item
                )
                for item
                in command
            )
        )

        print()


if __name__ == "__main__":
    main()