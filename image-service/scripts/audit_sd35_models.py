"""Audit the SD 3.5 Medium files on the machine that runs ComfyUI.

This script reads model files and optionally compares their SHA-256 digests
with Stability AI's Hugging Face LFS metadata. It does not generate images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.error
import urllib.request
from pathlib import Path


MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
MODEL_FILES = {
    "sd3.5_medium.safetensors": (
        ("checkpoints/sd3.5_medium.safetensors",),
        "sd3.5_medium.safetensors",
    ),
    "clip_l.safetensors": (
        ("text_encoders/clip_l.safetensors", "clip/clip_l.safetensors"),
        "text_encoders/clip_l.safetensors",
    ),
    "clip_g.safetensors": (
        ("text_encoders/clip_g.safetensors", "clip/clip_g.safetensors"),
        "text_encoders/clip_g.safetensors",
    ),
    "t5xxl_fp8_e4m3fn.safetensors": (
        (
            "text_encoders/t5xxl_fp8_e4m3fn.safetensors",
            "clip/t5xxl_fp8_e4m3fn.safetensors",
        ),
        "text_encoders/t5xxl_fp8_e4m3fn.safetensors",
    ),
}


def safetensors_header(path: Path) -> dict:
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError("file is too short for a safetensors header")
        (length,) = struct.unpack("<Q", length_bytes)
        if not 2 <= length <= 64 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header length: {length}")
        payload = handle.read(length)
        if len(payload) != length:
            raise ValueError("truncated safetensors header")
    header = json.loads(payload)
    if not isinstance(header, dict) or not any(k != "__metadata__" for k in header):
        raise ValueError("safetensors header contains no tensors")
    return header


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_hashes(token: str | None) -> dict[str, str]:
    url = f"https://huggingface.co/api/models/{MODEL_ID}/tree/main?recursive=true&expand=true"
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        entries = json.load(response)
    if not isinstance(entries, list):
        raise ValueError("unexpected Hugging Face file listing")
    return {
        item["path"]: item["lfs"]["oid"].removeprefix("sha256:")
        for item in entries
        if isinstance(item, dict)
        and isinstance(item.get("lfs"), dict)
        and isinstance(item["lfs"].get("oid"), str)
    }


def audit(model_root: Path, expected: dict[str, str]) -> int:
    failed = False
    for label, (local_names, official_name) in MODEL_FILES.items():
        path = next(
            (model_root / name for name in local_names if (model_root / name).is_file()),
            model_root / local_names[0],
        )
        if not path.is_file():
            print(f"MISSING {label}: {path}")
            failed = True
            continue
        try:
            header = safetensors_header(path)
            digest = sha256_file(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"INVALID {label}: {exc}")
            failed = True
            continue
        expected_digest = expected.get(official_name)
        status = "UNVERIFIED"
        if expected_digest:
            status = "MATCH" if digest == expected_digest else "MISMATCH"
            failed |= status == "MISMATCH"
        print(f"{status} {label}: {path.stat().st_size:,} bytes, sha256={digest}")
        if label == "sd3.5_medium.safetensors":
            tensor_names = [name for name in header if name != "__metadata__"]
            has_vae = any(
                name.startswith(("first_stage_model.", "vae."))
                for name in tensor_names
            )
            print(f"  checkpoint tensors={len(tensor_names)}, embedded_vae={has_vae}")

    external_vae = model_root / "vae" / "ae.safetensors"
    if external_vae.is_file():
        try:
            header = safetensors_header(external_vae)
            print(
                "EXTERNAL VAE ae.safetensors: "
                f"{external_vae.stat().st_size:,} bytes, "
                f"{len(header) - ('__metadata__' in header)} tensors; "
                "source not verified by this audit"
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"INVALID external VAE: {exc}")
            failed = True
    return 1 if failed else 0


def find_model_root(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.is_dir() else None
    candidates = [
        os.environ.get("COMFYUI_MODELS_DIR"),
        "/opt/ComfyUI/models",
        str(Path.home() / "ComfyUI" / "models"),
        str(Path.home() / "comfyui" / "models"),
        "/workspace/ComfyUI/models",
    ]
    return next((Path(path) for path in candidates if path and Path(path).is_dir()), None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path,
                        help="ComfyUI/models directory; otherwise search common locations")
    parser.add_argument("--offline", action="store_true",
                        help="show local hashes without checking official metadata")
    args = parser.parse_args()
    model_root = find_model_root(args.model_root)
    if model_root is None:
        print(
            "ComfyUI model directory not found. Run this script on the GPU "
            "host, or supply its real models directory with --model-root. "
            "The ComfyUI HTTP API does not expose model file paths or hashes.",
            file=sys.stderr,
        )
        return 2
    print(f"Checking ComfyUI models in {model_root}")
    expected: dict[str, str] = {}
    verification_unavailable = False
    if not args.offline:
        try:
            expected = official_hashes(os.environ.get("HF_TOKEN"))
            print(f"Official SHA-256 values available for {len(expected)} files")
        except (urllib.error.URLError, ValueError, OSError) as exc:
            print(f"Official metadata unavailable ({exc}); local hashes are unverified")
            verification_unavailable = True
        if not expected:
            verification_unavailable = True
    result = audit(model_root, expected)
    return result or (2 if verification_unavailable else 0)


if __name__ == "__main__":
    sys.exit(main())
