from datetime import datetime

from pydantic import BaseModel

from app.schemas.creative_direction import (
    CreativeDirectionOutput,
)


class ModelProvenance(BaseModel):
    provider: str
    model: str


class DirectionRecord(BaseModel):
    provenance: ModelProvenance
    direction: CreativeDirectionOutput


class PromptsDocument(BaseModel):
    schema_version: str
    prompt_schema_version: str = "1.0"
    creative_direction_version: str = "2.0"
    prompt_generation_sha256: str | None = None

    reference_number: str

    input_sha256: str

    created_at: datetime

    brief: dict

    directions: list[DirectionRecord]
