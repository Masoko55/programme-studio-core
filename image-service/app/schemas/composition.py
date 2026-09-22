from pydantic import BaseModel


class CompositionResult(BaseModel):
    reference_number: str

    engine_id: str

    direction_id: str

    background_path: str

    output_path: str

    sha256: str

    width: int

    height: int

    dpi: int