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
    load_prompts_document,
)
from app.services.job_persistence import (
    load_image_job_state,
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
        f"Prompt model not found for "
        f"direction {direction_id}."
    )


def get_final_artifact_path(
    reference_number: str,
    engine_id: str,
    direction_id: str,
) -> Path:
    return (
        settings.programme_data_path
        / reference_number
        / "final"
        / (
            f"{engine_id}-"
            f"{direction_id.lower()}.png"
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
                    "Final composed artifact "
                    "does not exist yet."
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

        quality = QualityResult(
            passed=True,
            checks={
                "file_exists": True,
                "png_valid": True,
                "dimensions_valid": (
                    validation["width"]
                    == settings.final_image_width
                    and validation["height"]
                    == settings.final_image_height
                ),
                "dpi_valid": (
                    validation["dpi"]
                    == settings.final_image_dpi
                ),
                "sha256_valid": True,
            },
            errors=[],
        )

        return (
            quality,
            output_sha256,
        )

    except Exception as error:
        checks = {
            "file_exists": True,
            "png_valid": False,
            "dimensions_valid": False,
            "dpi_valid": False,
            "sha256_valid": False,
        }

        error_message = str(
            error
        )

        if "width" in error_message.lower():
            checks[
                "png_valid"
            ] = True

        elif "height" in error_message.lower():
            checks[
                "png_valid"
            ] = True

        elif "dpi" in error_message.lower():
            checks[
                "png_valid"
            ] = True

            checks[
                "dimensions_valid"
            ] = True

        elif "sha-256" in error_message.lower():
            checks[
                "png_valid"
            ] = True

            checks[
                "dimensions_valid"
            ] = True

            checks[
                "dpi_valid"
            ] = True

        return (
            QualityResult(
                passed=False,
                checks=checks,
                errors=[
                    error_message
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

    input_sha256 = (
        prompts_document[
            "input_sha256"
        ]
    )

    artifacts = []
    image_job = load_image_job_state(
        reference_number
    )
    output_records = {
        (
            output.engine_id,
            output.direction_id,
        ): output
        for output in image_job.outputs
    }

    engines = [
        settings.engine_1_id,
        settings.engine_2_id,
        settings.engine_3_id,
    ]

    directions = [
        "A",
        "B",
        "C",
    ]

    for engine_id in engines:
        for direction_id in directions:
            artifact_key = (
                f"{engine_id}/"
                f"{direction_id.lower()}"
            )

            final_path = (
                get_final_artifact_path(
                    reference_number,
                    engine_id,
                    direction_id,
                )
            )

            prompt_model = (
                get_prompt_model(
                    prompts_document,
                    direction_id,
                )
            )

            (
                quality,
                output_sha256,
            ) = build_quality_result(
                final_path
            )

            artifact = ManifestArtifact(
                artifact_key=artifact_key,
                artifact_id=None,
                engine_id=engine_id,
                direction_id=direction_id,
                input_sha256=(
                    input_sha256
                ),
                output_sha256=(
                    output_sha256
                ),
                width=(
                    settings
                    .final_image_width
                ),
                height=(
                    settings
                    .final_image_height
                ),
                dpi=(
                    settings
                    .final_image_dpi
                ),
                seed=output_records[
                    (engine_id, direction_id)
                ].seed,
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

            artifacts.append(
                artifact
            )

    completed_count = sum(
        1
        for artifact in artifacts
        if artifact.quality.passed
    )

    return ProgrammeManifest(
        schema_version="1.0",
        reference_number=(
            reference_number
        ),
        input_sha256=input_sha256,
        created_at=datetime.now(
            timezone.utc
        ),
        expected_artifact_count=9,
        completed_artifact_count=(
            completed_count
        ),
        artifacts=artifacts,
        complete=(
            completed_count == 9
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
