"""Compare text-only and core-node template img2img under one fresh reference.

Run from the repository root with image-service/.venv/bin/python. This script
never changes the production candidate or retry records.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import settings
from app.services.candidate_spec import build_candidate_spec
from app.services.comfyui_client import (
    ComfyUIClient, ComfyUIError, _transport_prompts, _workflow_dimensions,
    build_workflow, validate_background, validate_workflow,
)
from app.services.prompt_compiler import compile_candidate_prompt
from app.services.prompt_repository import get_job_directory, load_prompts_document
from app.services.sampling_profiles import select_sampling_profile, summarize_failures
from app.services.template_conditioning import compile_template_conditioning, render_template


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-reference", required=True)
    parser.add_argument("--engine", required=True, choices=("sdxl-1-0", "sd-3-5-medium", "flux-2"))
    parser.add_argument("--direction", required=True, choices=("A", "B", "C"))
    parser.add_argument("--seed", required=True, type=int)
    denoise = parser.add_mutually_exclusive_group()
    denoise.add_argument("--denoise", type=float)
    denoise.add_argument("--denoise-sweep", help="Comma-separated values, e.g. 0.40,0.50,0.60,0.70")
    parser.add_argument("--failure-category", default="INITIAL", choices=(
        "INITIAL", "DENSE_SAFE_REGION", "WEAK_OUTER_STRUCTURE", "PALETTE_OFF",
        "PRIMARY_MISSING", "SECONDARY_MISSING", "RASTER", "FLAT", "HUMAN",
    ))
    args = parser.parse_args()
    if not 0 <= args.seed < 2**63:
        parser.error("Seed must fit the supported signed 63-bit range")
    return args


def _denoise_values(args) -> list[float]:
    if args.denoise_sweep:
        values = [float(value.strip()) for value in args.denoise_sweep.split(",")]
    else:
        values = [args.denoise]
    if any(value is not None and not 0 < value < 1 for value in values):
        raise ValueError("All denoise values must be between zero and one")
    if len(values) > 8 or not values:
        raise ValueError("Denoise sweep must contain between one and eight values")
    return values


async def _generate(client: ComfyUIClient, workflow: dict, output: Path, object_info: dict) -> None:
    validate_workflow(workflow, object_info)
    queue = (await client.request("GET", "/queue")).json()
    if queue.get("queue_running") or queue.get("queue_pending"):
        raise ComfyUIError("ComfyUI is busy; run the probe after its queue is empty")
    result = (await client.request("POST", "/prompt", json={
        "prompt": workflow, "client_id": str(uuid.uuid4()),
    })).json()
    if result.get("node_errors") or not result.get("prompt_id"):
        raise ComfyUIError(f"ComfyUI rejected probe workflow: {result.get('node_errors')}")
    prompt_id = result["prompt_id"]
    deadline = time.monotonic() + settings.comfyui_generation_timeout_seconds
    while time.monotonic() < deadline:
        history = (await client.request("GET", f"/history/{prompt_id}")).json().get(prompt_id)
        if history:
            if history.get("status", {}).get("status_str") == "error":
                raise ComfyUIError(f"Probe execution failed: {history.get('status')}")
            images = [image for node_id, node in workflow.items()
                      if node["class_type"] == "SaveImage"
                      for image in history.get("outputs", {}).get(node_id, {}).get("images", [])]
            if len(images) != 1:
                raise ComfyUIError("Probe expected exactly one ComfyUI image")
            descriptor = images[0]
            response = await client.request("GET", "/view", params={
                "filename": descriptor["filename"],
                "subfolder": descriptor.get("subfolder", ""), "type": "output",
            })
            output.write_bytes(response.content)
            return
        await asyncio.sleep(settings.comfyui_poll_interval_seconds)
    raise ComfyUIError("Probe generation timed out; inspect ComfyUI history")


async def _run(args) -> Path:
    source = load_prompts_document(args.source_reference)
    reference = f"{uuid.uuid4().hex[:6].upper()}-{uuid.uuid4().int % 1_000_000:06d}"
    source["reference_number"] = reference
    directory = get_job_directory(reference)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "prompts.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    candidate = build_candidate_spec(source, args.engine, args.direction)
    compiled = None if args.engine == settings.engine_1_id else compile_candidate_prompt(candidate, 1)
    positive, negative = _transport_prompts(
        args.engine, candidate.original_positive_prompt, candidate.original_negative_prompt,
        candidate.primary_colour, candidate.secondary_colour, compiled,
    )
    profile = select_sampling_profile(args.engine, 1, summarize_failures(None))
    probe = directory / "conditioning-probe"
    probe.mkdir()
    values = _denoise_values(args)
    base = compile_template_conditioning(candidate, denoise=values[0])
    active = compile_template_conditioning(
        candidate, denoise=values[0], failure_category=args.failure_category,
    )
    template = render_template(active, _workflow_dimensions(args.engine))
    (probe / "template.png").write_bytes(template.read_bytes())
    results = {"reference_number": reference, "source_reference": args.source_reference,
               "engine": args.engine, "direction": args.direction, "seed": args.seed,
               "candidate_sha256": candidate.spec_sha256, "base_conditioning_sha256": base.conditioning_sha256,
               "retry_conditioning_sha256": active.conditioning_sha256,
               "template_conditioning": active.record(),
               "modes": {}}
    async with ComfyUIClient() as client:
        object_info = (await client.request("GET", "/object_info")).json()
        required = {"LoadImage", "VAEEncode"}
        if args.engine == settings.engine_1_id:
            required.add("SplitSigmasDenoise")
        missing = required - object_info.keys()
        if missing:
            raise ComfyUIError(f"ComfyUI lacks required core img2img nodes: {sorted(missing)}")
        uploaded = (await client.request("POST", "/upload/image", files={
            "image": (f"{reference}-template.png", template.read_bytes(), "image/png"),
        }, data={"type": "input", "overwrite": "false"})).json()
        image_name = uploaded.get("name")
        if not image_name or uploaded.get("subfolder"):
            raise ComfyUIError("ComfyUI did not accept the template into its input root")
        # LoadImage's filename choices are generated from the current input
        # directory, so refresh the schema after uploading this new file.
        object_info = (await client.request("GET", "/object_info")).json()
        modes = [("text-only", None)] + [
            (f"template-img2img-{value:.2f}" if len(values) > 1 else "template-img2img", value)
            for value in values
        ]
        for name, denoise in modes:
            output = probe / f"{name}.png"
            workflow = build_workflow(
                args.engine, positive, negative, args.seed,
                f"programme-studio/{reference}/probe/{name}",
                compiled_prompt=compiled, sampling_profile=profile,
                template_image=image_name if denoise is not None else None,
                denoise=denoise,
            )
            entry = {"denoise": denoise, "output": str(output), "visual_review": "pending",
                     "conditioning_sha256": (compile_template_conditioning(
                         candidate, denoise=denoise, failure_category=args.failure_category,
                     ).conditioning_sha256 if denoise is not None else None)}
            try:
                await _generate(client, workflow, output, object_info)
                try:
                    entry["raw_validation"] = validate_background(
                        output, reference_number=reference, enforce_palette=False,
                    )
                except ValueError as error:
                    entry["raw_validation_error"] = str(error)
                entry["validation"] = validate_background(output, reference_number=reference)
                entry["validator_passed"] = True
            except (ComfyUIError, ValueError) as error:
                entry["validator_passed"] = False
                entry["error"] = str(error)
                if not output.exists():
                    results["modes"][name] = entry
                    break
            results["modes"][name] = entry
            (probe / "metrics.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    (probe / "metrics.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return probe


if __name__ == "__main__":
    print(asyncio.run(_run(_arguments())))
