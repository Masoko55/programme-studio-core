from datetime import datetime

from pydantic import BaseModel


class ArtifactProvenance(BaseModel):
    prompt_model: str
    image_engine: str
    direction_id: str


class QualityResult(BaseModel):
    passed: bool
    checks: dict[str, bool]
    errors: list[str]


class ManifestArtifact(BaseModel):
    artifact_key: str

    artifact_id: str | None = None

    engine_id: str
    direction_id: str

    input_sha256: str
    output_sha256: str

    width: int
    height: int
    dpi: int

    seed: int | None = None

    attempt_count: int

    provenance: ArtifactProvenance

    quality: QualityResult

    local_path: str

    retrieval_url: str | None = None


class ProgrammeManifest(BaseModel):
    schema_version: str

    reference_number: str

    input_sha256: str

    created_at: datetime

    expected_artifact_count: int

    completed_artifact_count: int

    artifacts: list[ManifestArtifact]

    complete: bool