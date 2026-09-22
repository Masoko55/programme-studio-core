#!/usr/bin/env python3
"""Capture the GPU-host runtime inventory used by Programme Studio."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


OLLAMA_URL = "http://192.168.68.115:11434/api/tags"
COMFY_URL = "http://192.168.68.115:8188/object_info"
COMFY_STATS_URL = "http://192.168.68.115:8188/system_stats"
REQUIRED_PROMPT_MODELS = (
    "qwen2.5:14b",
    "gemma3:27b",
    "mistral-small3.1:24b",
)
# The agreed host requirement is expressed as 24 GB (manufacturer decimal GB),
# not 24 GiB.  ComfyUI reports bytes, so compare it with 24,000,000,000.
MINIMUM_VRAM_BYTES = 24_000_000_000
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
    comfy_stats = fetch_json(COMFY_STATS_URL)
    installed_models = {
        model.get("name")
        for model in ollama.get("models", [])
        if model.get("name")
    }
    devices = comfy_stats.get("devices", [])
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
        "comfyui_system_stats_url": COMFY_STATS_URL,
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
        "gpu_devices": devices,
        "contract_validation": {
            "required_prompt_models": list(REQUIRED_PROMPT_MODELS),
            "missing_prompt_models": sorted(
                set(REQUIRED_PROMPT_MODELS) - installed_models
            ),
            "minimum_vram_bytes": MINIMUM_VRAM_BYTES,
            "has_24gb_gpu": any(
                device.get("vram_total", 0) >= MINIMUM_VRAM_BYTES
                or device.get("torch_vram_total", 0) >= MINIMUM_VRAM_BYTES
                for device in devices
            ),
        },
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
    validation = inventory["contract_validation"]
    print(f"24 GB GPU available: {validation['has_24gb_gpu']}")
    missing = validation["missing_prompt_models"]
    print("Missing required prompt models: " + (", ".join(missing) or "none"))
    print("Workflow hashes:")
    for name, workflow in workflows.items():
        print(f"  {name}: {workflow['sha256']}")


if __name__ == "__main__":
    main()
