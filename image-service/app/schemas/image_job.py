from pydantic import BaseModel, Field


class ImageJobRequest(BaseModel):
    reference_number: str = Field(
        min_length=13,
        max_length=13,
    )