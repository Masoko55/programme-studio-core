"""Bounded manifest for reproducing a native candidate attempt."""

from __future__ import annotations

import hashlib
import json


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_reproducibility_manifest(record: dict) -> dict:
    compiled = record.get("compiled_prompt") or {}
    profile = record.get("sampling_profile") or {}
    effective = record.get("effective_workflow") or {}
    workflow = record.get("workflow") or {}
    models = [str(node.get("inputs", {}).get(key)) for node in workflow.values()
              if isinstance(node, dict) for key in ("ckpt_name", "unet_name", "clip_name1", "clip_name2", "clip_name3")
              if node.get("inputs", {}).get(key)]
    sampling = {key: record.get(key, profile.get(key)) for key in
                ("sampler_name", "scheduler", "cfg", "steps", "shift", "width", "height")}
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        for key in sampling:
            if sampling[key] is None and isinstance(inputs.get(key), (int, float, str)):
                sampling[key] = inputs[key]
    return {
        "reference": record.get("reference_number"),
        "source_reference": record.get("source_reference") or record.get("reference_number"),
        "semantic_contract_version": record.get("semantic_contract_version"),
        "semantic_contract_sha256": record.get("semantic_contract_sha256"),
        "candidate_spec_sha256": record.get("candidate_spec_sha256"),
        "engine": record.get("engine_id"), "engine_models": sorted(set(models)),
        "workflow_sha256": record.get("workflow_sha256"),
        "effective_workflow_values": {key: value for key, value in effective.items()
                                      if key not in {"positive_fields", "negative_fields"}},
        "direction": record.get("direction_id"),
        "sampling_profile": profile.get("name"), **sampling,
        "seed": record.get("seed"),
        "compiled_prompt_sha256": {key: _sha(str(compiled.get(key) or ""))
                                   for key in ("positive", "negative", "clip_l", "clip_g", "t5")},
        "conditioning_lengths": (compiled.get("prompt_budget") or {}),
        "vision_model": record.get("vision_model"),
        "ocr_config": record.get("ocr_config"),
        "validator_version": record.get("validator_version"),
        "retry_attempt": record.get("attempt_count"),
        "retry_stage": record.get("retry_stage"),
        "failure_history": record.get("failure_history"),
        "image_sha256": record.get("sha256"),
    }


def manifest_hash(manifest: dict) -> str:
    return _sha(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
