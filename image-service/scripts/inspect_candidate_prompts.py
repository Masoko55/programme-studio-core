"""Print the stored semantic contract and conditioning for every candidate attempt."""

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/data/programmes"))
    args = parser.parse_args()
    root = args.data_root / args.reference / "backgrounds"
    for engine in ("flux-2", "sdxl-1-0", "sd-3-5-medium"):
        for direction in "abc":
            path = root / engine / f"image-{direction}.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            attempts = record.get("rejected_attempts", []) + record.get("failed_attempts", [])
            attempts.append(record)
            for item in sorted(attempts, key=lambda value: value.get("attempt_count", 0)):
                prompt = item.get("compiled_prompt") or {}
                profile = item.get("sampling_profile") or {}
                print(json.dumps({
                    "engine": engine,
                    "direction": direction.upper(),
                    "direction_role": item.get("direction_role"),
                    "attempt": item.get("attempt_count"),
                    "status": item.get("status") or "rejected",
                    "rejection_reason": item.get("rejection_reason"),
                    "off_palette_ratio": item.get("off_palette_ratio") or (
                        float(match.group(1)) if (
                            match := re.search(r"off-palette ratio ([0-9.]+)",
                                               item.get("rejection_reason") or "")
                        ) else None
                    ),
                    "raw_palette_error": item.get("raw_palette_error"),
                    "palette_normalized": item.get("palette_normalized"),
                    "fine_horizontal_pair_ratio": item.get("fine_horizontal_pair_ratio"),
                    "fine_vertical_pair_ratio": item.get("fine_vertical_pair_ratio"),
                    "sampling_profile": profile,
                    "sampler": profile.get("sampler_name"),
                    "scheduler": profile.get("scheduler"),
                    "steps": profile.get("steps"),
                    "cfg": profile.get("cfg"),
                    "shift": profile.get("shift"),
                    "seed": item.get("seed"),
                    "failure_category": item.get("failure_category"),
                    "outcome_category": item.get("outcome_category"),
                    "retry_stage": item.get("retry_stage"),
                    "spec_sha256": item.get("spec_sha256"),
                    "positive": prompt.get("positive"),
                    "clip_l": prompt.get("clip_l"),
                    "clip_g": prompt.get("clip_g"),
                    "t5": prompt.get("t5"),
                }, ensure_ascii=False))


if __name__ == "__main__":
    main()
