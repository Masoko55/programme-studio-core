"""Check an existing image with the production subject-adherence path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.subject_adherence import assess_subject_adherence


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--anchor", action="append", default=[])
    parser.add_argument("--model", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    result = assess_subject_adherence(
        args.image,
        {"primary_visual_subject": args.subject, "visual_anchors": args.anchor},
        model=args.model,
    )
    output = {key: result.get(key) for key in (
        "checked", "score", "decision", "reason", "model", "vision_debug"
    )}
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return output


if __name__ == "__main__":
    main()
