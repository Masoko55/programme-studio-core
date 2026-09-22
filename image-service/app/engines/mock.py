"""Retired development adapter. Historical .runtime artifacts are preserved."""
from app.engines.base import ImageEngine

class MockImageEngine(ImageEngine):
    async def generate(self, *args, **kwargs):
        raise RuntimeError("Mock image generation has been retired; configure ComfyUI")
