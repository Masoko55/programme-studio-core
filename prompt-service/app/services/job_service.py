import hashlib
import json
import secrets


def generate_reference_number() -> str:
    letters = secrets.token_hex(3).upper()
    numbers = secrets.randbelow(1_000_000)

    return f"{letters}-{numbers:06d}"


def calculate_sha256(data: dict) -> str:
    canonical_json = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()