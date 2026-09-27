import hashlib
import json
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path

from app.config.settings import settings
from app.schemas.manifest import (
    ArtifactProvenance,
    ManifestArtifact,
    ProgrammeManifest,
    QualityResult,
)
from app.services.artifact_validation import (
    validate_final_png,
)
from app.services.prompt_repository import (
    get_job_directory,
    load_prompts_document,
)


def calculate_file_sha256(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def get_prompt_model(
    prompts_document: dict,
    direction_id: str,
) -> str:
    for item in prompts_document[
        "directions"
    ]:
        direction = item[
            "direction"
        ]

        if (
            direction["direction_id"]
            == direction_id
        ):
            return item[
                "provenance"
            ]["model"]

    raise ValueError(
        "Prompt model not found for "
        f"direction {direction_id}."
    )


def load_selected_final(
    reference_number: str,
) -> dict:
    path = (
        get_job_directory(
            reference_number
        )
        / "selected-final.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            "No final background has been selected "
            f"for {reference_number}."
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def build_quality_result(
    final_path: Path,
) -> tuple[
    QualityResult,
    str,
]:
    if not final_path.exists():
        return (
            QualityResult(
                passed=False,
                checks={
                    "file_exists": False,
                    "png_valid": False,
                    "dimensions_valid": False,
                    "dpi_valid": False,
                    "sha256_valid": False,
                },
                errors=[
                    "Selected final programme "
                    "does not exist."
                ],
            ),
            "",
        )

    output_sha256 = (
        calculate_file_sha256(
            final_path
        )
    )

    try:
        validation = (
            validate_final_png(
                image_path=str(
                    final_path
                ),
                expected_sha256=(
                    output_sha256
                ),
            )
        )

        dimensions_valid = (
            validation["width"]
            == settings.final_image_width
            and validation["height"]
            == settings.final_image_height
        )

        dpi_valid = (
            validation["dpi"]
            == settings.final_image_dpi
        )

        passed = (
            dimensions_valid
            and dpi_valid
        )

        return (
            QualityResult(
                passed=passed,
                checks={
                    "file_exists": True,
                    "png_valid": True,
                    "dimensions_valid": (
                        dimensions_valid
                    ),
                    "dpi_valid": dpi_valid,
                    "sha256_valid": True,
                },
                errors=(
                    []
                    if passed
                    else [
                        "Selected final programme "
                        "failed print validation."
                    ]
                ),
            ),
            output_sha256,
        )

    except Exception as error:
        return (
            QualityResult(
                passed=False,
                checks={
                    "file_exists": True,
                    "png_valid": False,
                    "dimensions_valid": False,
                    "dpi_valid": False,
                    "sha256_valid": False,
                },
                errors=[
                    str(error)
                ],
            ),
            output_sha256,
        )


def build_manifest(
    reference_number: str,
) -> ProgrammeManifest:
    prompts_document = (
        load_prompts_document(
            reference_number
        )
    )

    selected = load_selected_final(
        reference_number
    )

    source = selected[
        "source_candidate"
    ]

    selected_output = selected[
        "selected_output"
    ]

    engine_id = source[
        "engine_id"
    ]

    direction_id = source[
        "direction_id"
    ]

    final_path = Path(
        selected_output[
            "output_path"
        ]
    )

    quality, output_sha256 = (
        build_quality_result(
            final_path
        )
    )

    prompt_model = (
        get_prompt_model(
            prompts_document,
            direction_id,
        )
    )

    artifact = ManifestArtifact(
        artifact_key="final-programme",
        artifact_id=None,
        engine_id=engine_id,
        direction_id=direction_id,
        input_sha256=(
            prompts_document[
                "input_sha256"
            ]
        ),
        output_sha256=(
            output_sha256
        ),
        width=(
            settings.final_image_width
        ),
        height=(
            settings.final_image_height
        ),
        dpi=(
            settings.final_image_dpi
        ),
        seed=None,
        attempt_count=1,
        provenance=(
            ArtifactProvenance(
                prompt_model=(
                    prompt_model
                ),
                image_engine=(
                    engine_id
                ),
                direction_id=(
                    direction_id
                ),
            )
        ),
        quality=quality,
        local_path=str(
            final_path
        ),
        retrieval_url=None,
    )

    return ProgrammeManifest(
        schema_version="2.0",
        reference_number=(
            reference_number.upper()
        ),
        input_sha256=(
            prompts_document[
                "input_sha256"
            ]
        ),
        created_at=datetime.now(
            timezone.utc
        ),
        expected_artifact_count=1,
        completed_artifact_count=(
            1
            if quality.passed
            else 0
        ),
        artifacts=[
            artifact
        ],
        complete=(
            quality.passed
        ),
    )


def persist_manifest(
    manifest: ProgrammeManifest,
) -> Path:
    output_path = (
        settings.programme_data_path
        / manifest.reference_number
        / "manifest.json"
    )

    temporary_path = (
        output_path.with_suffix(
            ".json.part"
        )
    )

    temporary_path.write_text(
        json.dumps(
            manifest.model_dump(
                mode="json"
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(
        output_path
    )

    return output_path