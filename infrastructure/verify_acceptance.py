#!/usr/bin/env python3
"""Verify Programme Studio's three required end-to-end sample briefs.

Run after the Podman services are ready.  The script checks prompt completion,
image workflow completion, the required brief assets/rows, and repository
downloads with SHA-256 verification.
"""
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import urlopen


PROMPT_URL = "http://127.0.0.1:8001"
IMAGE_URL = "http://127.0.0.1:8002"
REPOSITORY_URL = "http://127.0.0.1:8003"

SAMPLES = {
    "funeral with headshot": {
        "reference": "6CA211-996230",
        "minimum_rows": 1,
        "headshot": True,
        "logo": False,
    },
    "celebration with logo": {
        "reference": "4E8EEE-671617",
        "minimum_rows": 1,
        "headshot": False,
        "logo": True,
    },
    "15-row programme with both assets": {
        "reference": "1B4566-784525",
        "minimum_rows": 15,
        "headshot": True,
        "logo": True,
    },
}


def get_json(url: str) -> dict:
    with urlopen(url) as response:
        return json.loads(response.read())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_sample(label: str, expectation: dict) -> None:
    reference = expectation["reference"]
    prompt_job = get_json(f"{PROMPT_URL}/v1/prompt-jobs/{reference}")
    workflow = get_json(f"{IMAGE_URL}/v1/image-jobs/{reference}/workflow")
    manifest = get_json(f"{REPOSITORY_URL}/v1/references/{reference}/manifest")
    brief = prompt_job["brief"]

    require(prompt_job["status"] == "complete", f"{label}: prompt job is not complete")
    require(workflow["status"] == "complete", f"{label}: image workflow is not complete")
    require(manifest["complete"] is True, f"{label}: manifest is not complete")
    require(len(brief["programme"]) >= expectation["minimum_rows"], f"{label}: insufficient programme rows")
    require(bool(brief.get("headshot_path")) is expectation["headshot"], f"{label}: headshot requirement failed")
    require(bool(brief.get("logo_path")) is expectation["logo"], f"{label}: logo requirement failed")

    artifacts = manifest["artifacts"]
    require(len(artifacts) == 9, f"{label}: expected 9 artifacts")
    expected_keys = {
        f"{engine}/{direction}"
        for engine in ("flux-2", "sdxl-1-0", "sd-3-5-medium")
        for direction in ("a", "b", "c")
    }
    require({item["artifact_key"] for item in artifacts} == expected_keys, f"{label}: artifact matrix is incomplete")

    for artifact in artifacts:
        artifact_id = artifact["artifact_id"]
        with urlopen(f"{REPOSITORY_URL}/v1/artifacts/{artifact_id}") as response:
            content = response.read()
        actual_sha256 = hashlib.sha256(content).hexdigest()
        require(actual_sha256 == artifact["output_sha256"], f"{label}: checksum mismatch for {artifact['artifact_key']}")

    print(f"PASS  {label}: {reference}; 9 repository artifacts verified")


def verify_service_readiness() -> None:
    for name, url in (
        ("Prompt Service", f"{PROMPT_URL}/ready"),
        ("Image Service", f"{IMAGE_URL}/ready"),
        ("Repository Service", f"{REPOSITORY_URL}/ready"),
    ):
        response = get_json(url)
        require(response.get("status") == "ready", f"{name}: not ready")
        print(f"PASS  {name}: ready")


def main() -> int:
    verify_service_readiness()
    for label, expectation in SAMPLES.items():
        verify_sample(label, expectation)
    print("COMPLETE: all three required end-to-end samples passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
