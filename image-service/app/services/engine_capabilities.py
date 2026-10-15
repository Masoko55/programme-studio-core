"""Conditioning capabilities of supported native workflows."""

CAPABILITIES = {
    "sdxl-1-0": {"negative_prompt_channel": True, "multi_encoder": True,
                  "long_context_encoder": False, "supports_shift": False},
    "sd-3-5-medium": {"negative_prompt_channel": True, "multi_encoder": True,
                      "long_context_encoder": True, "supports_shift": True},
    "flux-2": {"negative_prompt_channel": False, "multi_encoder": False,
               "long_context_encoder": False, "supports_shift": True},
}


def engine_capabilities(engine_id: str) -> dict:
    if engine_id not in CAPABILITIES:
        raise ValueError(f"Unknown engine capabilities: {engine_id}")
    return dict(CAPABILITIES[engine_id])
