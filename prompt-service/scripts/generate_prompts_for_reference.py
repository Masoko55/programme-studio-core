import argparse
import asyncio
import json

from app.config.settings import settings
from app.services.job_service import calculate_sha256
from app.services.prompt_workflow import run_prompt_workflow


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_number")
    args = parser.parse_args()

    reference_number = args.reference_number.upper()

    brief_path = (
        settings.programme_data_path
        / reference_number
        / "brief.json"
    )

    if not brief_path.exists():
        raise FileNotFoundError(
            f"brief.json not found for {reference_number}: "
            f"{brief_path}"
        )

    brief = json.loads(
        brief_path.read_text(
            encoding="utf-8"
        )
    )

    input_sha256 = calculate_sha256(
        brief
    )

    result = await run_prompt_workflow(
        reference_number=reference_number,
        input_sha256=input_sha256,
        brief=brief,
        resume=False,
    )

    print(
        "REFERENCE:",
        reference_number,
    )

    print(
        "BRIEF:",
        result["brief_path"],
    )

    print(
        "PROMPTS:",
        result["prompts_path"],
    )


if __name__ == "__main__":
    asyncio.run(
        main()
    )