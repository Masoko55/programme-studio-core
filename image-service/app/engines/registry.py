from app.config.settings import settings
from app.engines.base import EngineDescriptor, ImageEngine
from app.services.comfyui_client import generate_remote_image

class ComfyUIEngine(ImageEngine):
    async def generate(self, positive_prompt: str, negative_prompt: str, *,
                       reference_number: str, direction_id: str,
                       compiled_prompt=None, spec_sha256: str | None = None,
                       direction_role: str | None = None,
                       retry_stage: str | None = None,
                       failure_category: str | None = None) -> dict:
        return await generate_remote_image(reference_number, self.engine_id, direction_id,
                                           positive_prompt, negative_prompt,
                                           compiled_prompt=compiled_prompt,
                                           spec_sha256=spec_sha256,
                                           direction_role=direction_role,
                                           retry_stage=retry_stage,
                                           failure_category=failure_category)

def get_engine_descriptors() -> list[EngineDescriptor]:
    return [EngineDescriptor(settings.engine_1_id, settings.engine_1_label),
            EngineDescriptor(settings.engine_2_id, settings.engine_2_label),
            EngineDescriptor(settings.engine_3_id, settings.engine_3_label)]

def get_engine_descriptor(engine_id: str) -> EngineDescriptor:
    for descriptor in get_engine_descriptors():
        if descriptor.engine_id == engine_id:
            return descriptor
    raise ValueError(f"Unknown image engine: {engine_id}")

def get_engine(engine_id: str) -> ComfyUIEngine:
    return ComfyUIEngine(get_engine_descriptor(engine_id))
