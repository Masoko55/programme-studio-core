#!/usr/bin/env python3
"""Verify durable image-workflow recovery after a deliberate service stop.

Run this only with a newly accepted prompt-job reference.  Start it while the
image workflow is active, stop the image container after at least one output is
complete, restart it, then invoke this script's resume phase.  It proves that
stored candidates are reused and no repository artifact IDs are duplicated.
"""
import argparse
import json
import sys
from urllib.request import Request, urlopen


IMAGE_URL = "http://127.0.0.1:8002"
REPOSITORY_URL = "http://127.0.0.1:8003"


def get_json(url: str) -> dict:
    with urlopen(url) as response:
        return json.loads(response.read())


def post_workflow(reference: str) -> dict:
    request = Request(
        f"{IMAGE_URL}/v1/image-jobs/{reference}/workflow",
        method="POST",
        data=b"",
    )
    with urlopen(request, timeout=7200) as response:
        return json.loads(response.read())


def artifact_map(reference: str) -> dict[str, tuple[str, str]]:
    manifest = get_json(f"{REPOSITORY_URL}/v1/references/{reference}/manifest")
    return {
        item["artifact_key"]: (item["artifact_id"], item["output_sha256"])
        for item in manifest.get("artifacts", [])
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("--before", help="Write current repository artifact map here")
    parser.add_argument("--resume", action="store_true", help="Resume the persisted image workflow")
    parser.add_argument("--after", help="Compare the completed artifact map with this saved map")
    args = parser.parse_args()

    if args.before:
        with open(args.before, "w", encoding="utf-8") as output:
            json.dump(artifact_map(args.reference), output, indent=2, sort_keys=True)
        print("CHECKPOINT: repository artifact map saved")

    if args.resume:
        state = post_workflow(args.reference)
        if state.get("status") != "complete":
            raise RuntimeError(f"resume did not complete: {state.get('status')}")
        print("PASS: persisted image workflow resumed to completion")

    if args.after:
        with open(args.after, encoding="utf-8") as source:
            before = json.load(source)
        after = artifact_map(args.reference)
        changed = {
            key for key, value in before.items()
            if key in after and after[key] != tuple(value)
        }
        if changed:
            raise RuntimeError(
                "existing artifact IDs or checksums changed after resume: "
                + ", ".join(sorted(changed))
            )
        if len(after) != 9:
            raise RuntimeError(f"expected nine artifacts after resume, got {len(after)}")
        print("PASS: resume produced exactly nine repository artifacts without changing prior artifacts")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
