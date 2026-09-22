#!/usr/bin/env python3
"""Capture the GPU-host runtime inventory used by Programme Studio."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


OLLAMA_URL = "http://192.168.68.115:11434/api/tags"
COMFY_URL = "http://192.168.68.115:8188/object_info"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "infrastructure" / "evidence" / "runtime-inventory.json"


def fetch_json(url: str) -> dict:
    with urlopen(url, timeout=30) as response:
        return json.loads(response.read())


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ollama = fetch_json(OLLAMA_URL)
    object_info = fetch_json(COMFY_URL)
    workflows = {
        path.stem: {
            "path": str(path.relative_to(ROOT)),
            "sha256": file_sha256(path),
        }
        for path in sorted((ROOT / "image-service" / "workflows").glob("*.json"))
    }

    inventory = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "ollama_url": OLLAMA_URL,
        "comfyui_url": COMFY_URL,
        "ollama_models": [
            {
                "name": model.get("name"),
                "model": model.get("model"),
                "digest": model.get("digest"),
                "details": model.get("details"),
            }
            for model in ollama.get("models", [])
        ],
        "comfyui_node_types": sorted(object_info.keys()),
        "workflow_files": workflows,
        "note": (
            "Ollama digests are supplied by the running host. ComfyUI model-file "
            "digests require filesystem access on the GPU host and are not exposed "
            "by its HTTP API."
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Ollama models: {len(inventory['ollama_models'])}")
    print(f"ComfyUI node types: {len(inventory['comfyui_node_types'])}")
    print("Workflow hashes:")
    for name, workflow in workflows.items():
        print(f"  {name}: {workflow['sha256']}")


if __name__ == "__main__":
    main()
