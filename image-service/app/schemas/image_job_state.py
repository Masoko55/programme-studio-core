from pydantic import BaseModel


class ImageOutputRecord(BaseModel):
    engine_id: str
    engine_label: str
    direction_id: str

    status: str

    output_path: str | None = None
    sha256: str | None = None

    error: str | None = None
    seed: int | None = None
    prompt_id: str | None = None
    workflow_sha256: str | None = None


class ImageJobState(BaseModel):
    reference_number: str

    status: str
    current_stage: str

    total_outputs: int
    completed_outputs: int

    outputs: list[ImageOutputRecord]

    error: str | None = None