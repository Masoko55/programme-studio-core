import json
import os
import uuid
from pathlib import Path
from typing import Any

def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

def write_json(path: Path, value: Any) -> None:
    write_bytes(path, json.dumps(value, ensure_ascii=False, indent=2, default=str).encode())
