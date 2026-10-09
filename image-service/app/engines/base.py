from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass


@dataclass(frozen=True)
class EngineDescriptor:
    engine_id: str
    label: str


class ImageEngine(ABC):
    def __init__(
        self,
        descriptor: EngineDescriptor,
    ):
        self.descriptor = descriptor

    @property
    def engine_id(self) -> str:
        return self.descriptor.engine_id

    @property
    def label(self) -> str:
        return self.descriptor.label

    @abstractmethod
    async def generate(
        self,
        positive_prompt: str,
        negative_prompt: str,
        *,
        reference_number: str,
        direction_id: str,
        compiled_prompt=None,
        spec_sha256: str | None = None,
        direction_role: str | None = None,
        retry_stage: str | None = None,
        failure_category: str | None = None,
        sampling_profile=None,
    ):
        raise NotImplementedError
