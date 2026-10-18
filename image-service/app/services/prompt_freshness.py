"""Bounded source-prompt provenance for controlled renderer comparisons."""

from __future__ import annotations

import hashlib
import json


CURRENT_CREATIVE_DIRECTION_VERSION = "2.0"


def source_prompt_manifest(document: dict) -> dict:
    payload = dict(document)
    declared = payload.pop("prompt_generation_sha256", None)
    calculated = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                            ensure_ascii=False).encode("utf-8")).hexdigest()
    semantic_version = (document.get("brief") or {}).get("semantic_contract", {}).get("semantic_contract_version")
    direction_version = document.get("creative_direction_version")
    status = ("CURRENT" if direction_version == CURRENT_CREATIVE_DIRECTION_VERSION and
              declared == calculated and semantic_version else
              "LEGACY_COMPATIBLE" if not direction_version and semantic_version else
              "STALE_FOR_EXPERIMENT")
    return {"prompt_schema_version": document.get("prompt_schema_version") or document.get("schema_version"),
            "semantic_contract_version": semantic_version,
            "creative_direction_version": direction_version,
            "prompt_generation_sha256": declared or calculated,
            "hash_verified": declared == calculated if declared else None,
            "freshness_status": status,
            "created_at": document.get("created_at")}
