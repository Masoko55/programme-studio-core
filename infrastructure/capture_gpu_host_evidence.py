#!/usr/bin/env python3
"""Run on the GPU host to capture capacity and ComfyUI model-file evidence."""
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "infrastructure" / "evidence" / "gpu-host-evidence.json"
MODEL_ROOT = Path(os.environ.get("COMFYUI_MODELS_DIR", "/opt/ComfyUI/models"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    if not shutil.which(command[0]):
        return None
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def main() -> None:
    weights = []
    if MODEL_ROOT.is_dir():
        for path in sorted(MODEL_ROOT.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".safetensors", ".ckpt", ".pt", ".pth", ".gguf"}:
                weights.append({
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                })
    payload = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "hostname": command_output(["hostname"]),
        "nvidia_smi": command_output([
            "nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"
        ]),
        "comfyui_models_dir": str(MODEL_ROOT),
        "weight_files": weights,
        "licence_approval": "Record commercial-use approval for each listed model before delivery.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Model files hashed: {len(weights)}")


if __name__ == "__main__":
    main()
