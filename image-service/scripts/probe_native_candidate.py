"""Generate one engine candidate under a fresh reference for diagnosis.

This probe intentionally generates exactly ONE image.

It can:

- use the production sampling-profile selector;
- use an explicitly named sampling profile;
- use a fixed seed for controlled comparisons;
- preserve rejected generated PNGs for visual inspection;
- capture raw non-palette QA before the normal palette gate rejects an image.

Important:

This script does NOT weaken production validation.

The normal ComfyUIClient generation path still performs its usual validation.
During this probe only, validate_background() is wrapped so that:

    1. the raw candidate is validated with enforce_palette=False;
    2. those raw structural metrics are captured;
    3. normal validation then runs unchanged;
    4. if normal validation rejects the candidate, the generated PNG is copied
       to image-<direction>.rejected.png before the normal temporary file is
       deleted.

Production code never enables this behavior.

Examples:

    # From the programme-studio repository root, use the image-service venv.
    ./image-service/.venv/bin/python image-service/scripts/probe_native_candidate.py \
        --source-reference 5592AE-741489 \
        --engine sd-3-5-medium \
        --direction A

    ./image-service/.venv/bin/python image-service/scripts/probe_native_candidate.py \
        --source-reference 5592AE-741489 \
        --engine sd-3-5-medium \
        --direction A \
        --sampling-profile alternate_sampler \
        --seed 20261009

    ./image-service/.venv/bin/python image-service/scripts/probe_native_candidate.py \
        --source-reference 5592AE-741489 \
        --engine sd-3-5-medium \
        --direction A \
        --list-profiles
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import shutil
import sys
import uuid

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


# ============================================================
# Make image-service/app importable when this script is mounted
# into the image container at /app/scripts.
# ============================================================

sys.path.insert(
    0,
    str(
        Path(
            __file__
        )
        .resolve()
        .parents[1]
    ),
)


from app.config.settings import settings
from app.services.atomic import write_json
from app.services.candidate_spec import (
    build_candidate_spec,
)

#
# Import the module itself as well as its classes.
#
# The module reference is required because the probe temporarily replaces
# comfyui_client.validate_background for this process only.
#
import app.services.comfyui_client as comfyui_module

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


# ============================================================
# Reference helpers
# ============================================================


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


# ============================================================
# Argument parsing
# ============================================================


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
            settings.engine_1_id,
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
            "If omitted, the production profile selector "
            "is used."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional fixed signed 63-bit seed for "
            "controlled profile comparison."
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


# ============================================================
# Probe validation capture
# ============================================================


class ProbeValidationCapture:
    """Mutable diagnostic state for one single-image probe."""

    def __init__(
        self,
        rejected_image_path: Path,
    ) -> None:
        self.rejected_image_path = (
            rejected_image_path
        )

        self.raw_validation: (
            dict[str, Any]
            | None
        ) = None

        self.raw_validation_error: (
            str
            | None
        ) = None

        self.full_validation_error: (
            str
            | None
        ) = None

        self.rejected_image_preserved = (
            False
        )

    def record_raw_validation(
        self,
        value: dict[str, Any],
    ) -> None:
        self.raw_validation = (
            dict(
                value
            )
        )

    def record_raw_error(
        self,
        error: Exception,
    ) -> None:
        self.raw_validation_error = (
            str(
                error
            )
        )

    def preserve_rejected_image(
        self,
        source_path: Path,
        error: Exception,
    ) -> None:
        self.full_validation_error = (
            str(
                error
            )
        )

        self.rejected_image_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copyfile(
            source_path,
            self.rejected_image_path,
        )

        self.rejected_image_preserved = (
            True
        )


@contextmanager
def _capture_probe_validation(
    capture: ProbeValidationCapture,
) -> Iterator[
    ProbeValidationCapture
]:
    """Capture raw QA and rejected pixels without altering acceptance rules.

    The wrapper is installed only inside this probe process.

    Production code continues to call the original validate_background()
    implementation normally.

    For an ordinary full-validation call we first evaluate the same candidate
    with palette enforcement disabled. This gives us structural, OCR, human,
    dimensional and visual-quality diagnostics before palette enforcement can
    terminate validation.

    We then call the original validator again with its original arguments.

    If that original validation raises ValueError, the raw generated PNG is
    copied to the probe-specific *.rejected.png location before the production
    generation code removes its temporary candidate.
    """

    original_validate = (
        comfyui_module
        .validate_background
    )

    def probe_validate_background(
        path,
        *args,
        **kwargs,
    ):
        source_path = (
            Path(
                path
            )
        )

        #
        # When production code explicitly asks for non-palette validation,
        # do not wrap that call again.
        #
        # This is particularly important for deterministic palette recovery,
        # where _palette_only_recovery() intentionally uses:
        #
        #     enforce_palette=False
        #
        if (
            kwargs.get(
                "enforce_palette",
                True,
            )
            is False
        ):
            return original_validate(
                path,
                *args,
                **kwargs,
            )

        #
        # Probe-only raw QA pass.
        #
        # Use exactly the same arguments as the real validation call except
        # palette enforcement is disabled.
        #
        raw_kwargs = (
            dict(
                kwargs
            )
        )

        raw_kwargs[
            "enforce_palette"
        ] = False

        try:
            raw_result = (
                original_validate(
                    path,
                    *args,
                    **raw_kwargs,
                )
            )

            capture.record_raw_validation(
                raw_result
            )

        except ValueError as raw_error:
            #
            # A structural/OCR/human/etc. rejection can happen even before
            # palette checking.
            #
            # Record it, but still execute the original full validation below
            # so that the actual application rejection reason remains exactly
            # what production would have produced.
            #
            capture.record_raw_error(
                raw_error
            )

        #
        # Actual application validation.
        #
        # Nothing about its thresholds or behavior is changed.
        #
        try:
            return original_validate(
                path,
                *args,
                **kwargs,
            )

        except ValueError as validation_error:
            #
            # Preserve the candidate before ComfyUIClient's finally block
            # removes its .download.part temporary image.
            #
            capture.preserve_rejected_image(
                source_path,
                validation_error,
            )

            raise

    comfyui_module.validate_background = (
        probe_validate_background
    )

    try:
        yield capture

    finally:
        #
        # Always restore the real validator, even when generation fails.
        #
        comfyui_module.validate_background = (
            original_validate
        )


# ============================================================
# Persist probe diagnostics
# ============================================================


def _record_path(
    reference: str,
    engine: str,
    direction: str,
) -> Path:
    return (
        settings.programme_data_path
        / reference
        / "backgrounds"
        / engine
        / (
            "image-"
            + direction.lower()
            + ".json"
        )
    )


def _load_record(
    reference: str,
    engine: str,
    direction: str,
) -> dict:
    path = _record_path(
        reference,
        engine,
        direction,
    )

    if not path.exists():
        return {}

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {}


def _persist_probe_diagnostics(
    *,
    reference: str,
    engine: str,
    direction: str,
    capture: ProbeValidationCapture,
) -> dict:
    """Add probe-only diagnostics to the existing candidate record."""

    path = _record_path(
        reference,
        engine,
        direction,
    )

    record = _load_record(
        reference,
        engine,
        direction,
    )

    record[
        "probe_mode"
    ] = True

    record[
        "probe_raw_validation"
    ] = (
        capture.raw_validation
    )

    record[
        "probe_raw_validation_error"
    ] = (
        capture.raw_validation_error
    )

    record[
        "probe_full_validation_error"
    ] = (
        capture.full_validation_error
    )

    record[
        "rejected_image_path"
    ] = (
        str(
            capture.rejected_image_path
        )
        if (
            capture
            .rejected_image_preserved
        )
        else None
    )

    #
    # Expose calculable raw QA fields at top level as well.
    #
    # This keeps inspect_candidate_prompts.py useful without changing the
    # production candidate schema.
    #
    # Do not overwrite output_path with the temporary .download.part path.
    # Do not overwrite palette_checked with False from the raw QA pass.
    #
    if (
        capture.raw_validation
        is not None
    ):
        for (
            key,
            value,
        ) in (
            capture
            .raw_validation
            .items()
        ):
            if (
                key
                in {
                    "output_path",
                    "palette_checked",
                }
            ):
                continue

            record[
                key
            ] = value

    if (
        capture
        .rejected_image_preserved
    ):
        record[
            "rejected_image_preserved"
        ] = True

    else:
        record[
            "rejected_image_preserved"
        ] = False

    write_json(
        path,
        record,
    )

    return record


# ============================================================
# Output helpers
# ============================================================


def _diagnostic_summary(
    *,
    reference: str,
    engine: str,
    direction: str,
    record: dict,
    error: str | None = None,
) -> dict:
    raw = (
        record.get(
            "probe_raw_validation"
        )
        or {}
    )

    profile = (
        record.get(
            "sampling_profile"
        )
        or {}
    )

    return {
        "reference": (
            reference
        ),
        "engine": (
            engine
        ),
        "direction": (
            direction
        ),
        "status": (
            record.get(
                "status"
            )
        ),
        "error": (
            error
        ),
        "sampling_profile": (
            profile
        ),
        "retry_stage": (
            record.get(
                "retry_stage"
            )
        ),
        "failure_category": (
            record.get(
                "failure_category"
            )
        ),
        "outcome_category": (
            record.get(
                "outcome_category"
            )
        ),
        "spec_sha256": (
            record.get(
                "spec_sha256"
            )
        ),
        "seed": (
            record.get(
                "seed"
            )
        ),
        "rejection_reason": (
            record.get(
                "rejection_reason"
            )
        ),
        "rejected_image_path": (
            record.get(
                "rejected_image_path"
            )
        ),
        "rejected_image_preserved": (
            record.get(
                "rejected_image_preserved"
            )
        ),
        "probe_raw_validation_error": (
            record.get(
                "probe_raw_validation_error"
            )
        ),
        "off_palette_ratio": (
            record.get(
                "off_palette_ratio"
            )
        ),
        "fine_horizontal_pair_ratio": (
            raw.get(
                "fine_horizontal_pair_ratio"
            )
            if (
                "fine_horizontal_pair_ratio"
                in raw
            )
            else record.get(
                "fine_horizontal_pair_ratio"
            )
        ),
        "fine_vertical_pair_ratio": (
            raw.get(
                "fine_vertical_pair_ratio"
            )
            if (
                "fine_vertical_pair_ratio"
                in raw
            )
            else record.get(
                "fine_vertical_pair_ratio"
            )
        ),
        "structural_edge_density": (
            raw.get(
                "structural_edge_density"
            )
            if (
                "structural_edge_density"
                in raw
            )
            else record.get(
                "structural_edge_density"
            )
        ),
        "outer_edge_density": (
            raw.get(
                "outer_edge_density"
            )
            if (
                "outer_edge_density"
                in raw
            )
            else record.get(
                "outer_edge_density"
            )
        ),
        "center_edge_density": (
            raw.get(
                "center_edge_density"
            )
            if (
                "center_edge_density"
                in raw
            )
            else record.get(
                "center_edge_density"
            )
        ),
        "coherent_edge_ratio": (
            raw.get(
                "coherent_edge_ratio"
            )
            if (
                "coherent_edge_ratio"
                in raw
            )
            else record.get(
                "coherent_edge_ratio"
            )
        ),
        "person_detection_count": (
            raw.get(
                "person_detection_count"
            )
            if (
                "person_detection_count"
                in raw
            )
            else record.get(
                "person_detection_count"
            )
        ),
        "face_detection_count": (
            raw.get(
                "face_detection_count"
            )
            if (
                "face_detection_count"
                in raw
            )
            else record.get(
                "face_detection_count"
            )
        ),
    }


# ============================================================
# Main
# ============================================================


async def main() -> None:
    args = _parse_args()

    # ========================================================
    # List profiles only
    # ========================================================

    if (
        args.list_profiles
    ):
        if args.engine == settings.engine_1_id:
            print("FLUX uses its configured workflow without a sampling-profile override.")
            return
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

    # ========================================================
    # Resolve optional explicit sampling profile
    # ========================================================

    profile = None

    if (
        args.sampling_profile
    ):
        profile = (
            profile_by_name(
                args.engine,
                args.sampling_profile,
            )
        )

    # ========================================================
    # Load the source semantic brief
    # ========================================================

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

    #
    # Every probe gets a fresh reference.
    #
    # The semantic contents remain copied from the source job.
    #
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
            "preserve_rejected_candidate": (
                True
            ),
        },
    )

    print(
        "NEW_REFERENCE="
        + reference,
        flush=True,
    )

    # ========================================================
    # Build immutable semantic contract
    # ========================================================

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

    if (
        profile
        is not None
    ):
        print(
            "SAMPLING_PROFILE="
            + json.dumps(
                profile.record(),
                sort_keys=True,
            ),
            flush=True,
        )

    # ========================================================
    # Probe-specific rejected-image destination
    # ========================================================

    rejected_image_path = (
        directory
        / "backgrounds"
        / args.engine
        / (
            "image-"
            + args.direction.lower()
            + ".rejected.png"
        )
    )

    capture = (
        ProbeValidationCapture(
            rejected_image_path
        )
    )

    # ========================================================
    # Generate exactly one candidate
    # ========================================================

    try:
        with _capture_probe_validation(
            capture
        ):
            async with (
                ComfyUIClient()
                as client
            ):
                await (
                    client.health()
                )

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
                            compiled.failure_category
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
        #
        # generate_image() has already persisted the normal rejection record.
        #
        # Add probe-only raw QA and rejected-image information now.
        #
        stored = (
            _persist_probe_diagnostics(
                reference=reference,
                engine=args.engine,
                direction=args.direction,
                capture=capture,
            )
        )

        print(
            json.dumps(
                _diagnostic_summary(
                    reference=reference,
                    engine=args.engine,
                    direction=args.direction,
                    record=stored,
                    error=str(
                        error
                    ),
                ),
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )

        rejected_path = stored.get("rejected_image_path")
        if rejected_path and Path(rejected_path).is_file():
            print(f"VIEW_COMMAND=xdg-open {shlex.quote(str(rejected_path))}", flush=True)

        #
        # Retain the non-zero exit status.
        #
        # A diagnostic rejection must still be a real rejection.
        #
        raise SystemExit(1) from None

    # ========================================================
    # Accepted candidate
    # ========================================================

    stored = (
        _persist_probe_diagnostics(
            reference=reference,
            engine=args.engine,
            direction=args.direction,
            capture=capture,
        )
    )

    print(
        json.dumps(
            {
                **_diagnostic_summary(
                    reference=reference,
                    engine=args.engine,
                    direction=args.direction,
                    record=stored,
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
    output_path = result.get("output_path")
    if output_path and Path(output_path).is_file():
        print(f"VIEW_COMMAND=xdg-open {shlex.quote(str(output_path))}", flush=True)


if __name__ == "__main__":
    asyncio.run(
        main()
    )
