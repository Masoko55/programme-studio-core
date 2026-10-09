"""Generate one native-engine candidate under a fresh reference for diagnosis."""

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import settings
from app.services.atomic import write_json
from app.services.candidate_spec import build_candidate_spec
from app.services.comfyui_client import ComfyUIClient
from app.services.prompt_compiler import compile_candidate_prompt
from app.services.prompt_repository import load_prompts_document


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-reference", required=True)
    parser.add_argument("--engine", required=True,
                        choices=(settings.engine_2_id, settings.engine_3_id))
    parser.add_argument("--direction", required=True, choices=("A", "B", "C"))
    args = parser.parse_args()

    document = load_prompts_document(args.source_reference)
    reference = uuid.uuid4().hex[:6].upper() + "-" + str(uuid.uuid4().int % 1000000).zfill(6)
    directory = settings.programme_data_path / reference
    directory.mkdir(exist_ok=False)
    document["reference_number"] = reference
    write_json(directory / "prompts.json", document)
    write_json(directory / "brief.json", document["brief"])
    write_json(directory / "probe-source.json", {
        "source_reference": args.source_reference,
        "engine": args.engine,
        "direction": args.direction,
    })
    print("NEW_REFERENCE=" + reference, flush=True)

    spec = build_candidate_spec(document, args.engine, args.direction)
    compiled = compile_candidate_prompt(spec, 1)
    async with ComfyUIClient() as client:
        await client.health()
        result = await client.generate_image(
            reference, args.engine, args.direction,
            compiled.positive, compiled.negative,
            compiled_prompt=compiled,
            spec_sha256=spec.spec_sha256,
            direction_role=spec.direction_role,
            retry_stage=compiled.retry_stage,
            failure_category=compiled.failure_category,
        )
    print(json.dumps({
        "reference": reference,
        "engine": args.engine,
        "direction": args.direction,
        "status": result["status"],
        "sampling_profile": result["sampling_profile"],
        "output_path": result["output_path"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
