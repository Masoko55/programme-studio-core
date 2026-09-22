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

    reference_number: str

    input_sha256: str

    created_at: datetime

    brief: dict

    directions: list[DirectionRecord]