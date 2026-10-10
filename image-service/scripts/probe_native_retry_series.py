"""Probe up to nine attempts for one native engine and one direction.

Run this locally on the machine with access to ComfyUI. Each direction gets a
new reference and an immutable copy of the source brief. Rejected PNGs are
kept with attempt numbers so every generated image can be inspected.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import settings
from app.services.atomic import write_json
from app.services.candidate_spec import build_candidate_spec
from app.services.comfyui_client import ComfyUIClient, ComfyUIError
from app.services.prompt_compiler import compile_candidate_prompt, failure_category, retry_stage
from app.services.image_execution import _initial_engine_prompts, _strengthen_prompts
from app.services.prompt_repository import load_prompts_document
from app.services.sampling_profiles import select_sampling_profile, summarize_failures

from probe_native_candidate import (
    ProbeValidationCapture,
    _capture_probe_validation,
    _diagnostic_summary,
    _load_record,
    _new_reference,
    _persist_probe_diagnostics,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-reference", required=True)
    parser.add_argument("--engine", required=True, choices=(settings.engine_1_id, settings.engine_2_id, settings.engine_3_id))
    parser.add_argument("--direction", required=True, choices=("A", "B", "C"))
    parser.add_argument("--attempts", type=int, default=9, choices=range(1, 10))
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional starting seed; each retry increments it by one.")
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    if args.attempts > settings.max_candidate_retries + 1:
        raise ValueError("Requested attempts exceed the configured candidate budget")
    if args.seed is not None and not 0 <= args.seed <= 2**63 - args.attempts:
        raise ValueError("Seed series must fit the supported signed 63-bit range")

    document = load_prompts_document(args.source_reference)
    reference = _new_reference()
    directory = settings.programme_data_path / reference
    directory.mkdir(parents=True, exist_ok=False)
    document["reference_number"] = reference
    write_json(directory / "prompts.json", document)
    write_json(directory / "brief.json", document["brief"])
    write_json(directory / "probe-source.json", {
        "source_reference": args.source_reference,
        "engine": args.engine,
        "direction": args.direction,
        "attempts_requested": args.attempts,
        "starting_seed": args.seed,
    })

    spec = build_candidate_spec(document, args.engine, args.direction)
    print(f"NEW_REFERENCE={reference}", flush=True)
    print(f"SPEC_SHA256={spec.spec_sha256}", flush=True)

    previous_reason = ""
    async with ComfyUIClient() as client:
        await client.health()
        for attempt in range(1, args.attempts + 1):
            previous = _load_record(reference, args.engine, args.direction)
            if args.engine == settings.engine_1_id:
                if attempt == 1:
                    positive, negative = _initial_engine_prompts(
                        args.engine, spec.original_positive_prompt, spec.original_negative_prompt,
                        spec.primary_colour, spec.secondary_colour,
                    )
                else:
                    positive, negative = _strengthen_prompts(
                        spec.original_positive_prompt, spec.original_negative_prompt,
                        previous_reason, spec.primary_colour, spec.secondary_colour,
                        attempt - 1, args.engine,
                    )
                compiled = None
                category = failure_category(previous_reason) if attempt > 1 else "INITIAL"
                stage = retry_stage(attempt)
            else:
                compiled = compile_candidate_prompt(spec, attempt, previous_reason)
                positive, negative = compiled.positive, compiled.negative
                category, stage = compiled.failure_category, compiled.retry_stage
            profile = select_sampling_profile(
                args.engine, attempt, summarize_failures(previous)
            )
            capture = ProbeValidationCapture(
                directory / "backgrounds" / args.engine /
                f"image-{args.direction.lower()}.attempt-{attempt:02d}.rejected.png"
            )
            print(f"ATTEMPT={attempt}/{args.attempts}", flush=True)
            print(f"RETRY_STAGE={stage}", flush=True)
            if profile is not None:
                print("SAMPLING_PROFILE=" + json.dumps(profile.record(), sort_keys=True), flush=True)

            error_message = None
            result = None
            try:
                with _capture_probe_validation(capture):
                    result = await client.generate_image(
                        reference, args.engine, args.direction,
                        positive, negative,
                        compiled_prompt=compiled,
                        spec_sha256=spec.spec_sha256,
                        direction_role=spec.direction_role,
                        retry_stage=stage,
                        failure_category=category,
                        sampling_profile=profile,
                        seed_override=(args.seed + attempt - 1 if args.seed is not None else None),
                    )
            except (ComfyUIError, ValueError) as error:
                error_message = str(error)

            record = _persist_probe_diagnostics(
                reference=reference, engine=args.engine,
                direction=args.direction, capture=capture,
            )
            summary = _diagnostic_summary(
                reference=reference, engine=args.engine,
                direction=args.direction, record=record, error=error_message,
            )
            summary["attempt"] = attempt
            conditioning = record.get("template_conditioning") or {}
            summary.update({
                "TEMPLATE_CONDITIONING_ENABLED": conditioning.get("enabled", False),
                "TEMPLATE_PATH": conditioning.get("template_path"),
                "CONDITIONING_SHA256": conditioning.get("conditioning_sha256"),
                "BASE_CONDITIONING_SHA256": conditioning.get("base_conditioning_sha256"),
                "ATTEMPT_CONDITIONING_SHA256": conditioning.get("attempt_conditioning_sha256"),
                "PROTECTED_MASK_PATH": conditioning.get("protected_mask_path"),
                "GENERATION_MASK_PATH": conditioning.get("generation_mask_path"),
                "MASKED_DENOISING_ENABLED": conditioning.get("masked_denoising_enabled", False),
                "REGIONAL_CONDITIONING_ENABLED": conditioning.get("regional_conditioning_enabled", False),
                "DENOISE": conditioning.get("denoise"),
                "TEMPLATE_ADHERENCE_PASS": record.get("template_adherence_passed"),
                "TEMPLATE_STRUCTURE_IOU": record.get("template_structure_iou"),
                "DECORATION_IN_PROTECTED_RATIO": record.get("decoration_in_protected_ratio"),
                "TITLE_ZONE_EDGE_DENSITY": record.get("title_zone_edge_density"),
                "PROGRAMME_ZONE_EDGE_DENSITY": record.get("programme_zone_edge_density"),
                "OUTER_EDGE_DENSITY": record.get("outer_edge_density"),
                "CENTER_EDGE_DENSITY": record.get("center_edge_density"),
                "DECORATIVE_REGION_EDGE_DENSITY": record.get("decorative_region_edge_density"),
                "PROTECTED_REGION_EDGE_DENSITY": record.get("protected_region_edge_density"),
                "DECORATIVE_TO_PROTECTED_EDGE_RATIO": record.get("decorative_to_protected_edge_ratio"),
                "DECORATIVE_COVERAGE_RATIO": record.get("decorative_coverage_ratio"),
            })
            print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

            view_path = (
                record.get("rejected_image_path")
                if error_message else (result or {}).get("output_path")
            )
            if view_path and Path(view_path).is_file():
                print(f"VIEW_COMMAND=xdg-open {shlex.quote(str(view_path))}", flush=True)

            if error_message is None:
                print("SERIES_STATUS=complete", flush=True)
                return
            if record.get("status") not in {"rejected", "failed", "runtime_failed"}:
                print("SERIES_STATUS=stopped_non_retryable", flush=True)
                raise SystemExit(1)
            previous_reason = record.get("rejection_reason") or error_message

    print("SERIES_STATUS=attempt_budget_exhausted", flush=True)
    raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
