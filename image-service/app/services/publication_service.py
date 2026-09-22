import hashlib
import logging

from app.schemas.manifest import (
    ProgrammeManifest,
)
from app.services.artifact_validation import (
    validate_final_png,
)
from app.services.manifest_service import (
    build_manifest,
    persist_manifest,
)
from app.services.repository_client import (
    get_repository_artifacts,
    download_repository_artifact,
    upload_artifact,
    upload_manifest,
)


logger = logging.getLogger(
    "uvicorn.error"
)


def get_existing_artifact_map(
    repository_artifacts: list,
) -> dict:
    return {
        artifact[
            "artifactKey"
        ]: artifact
        for artifact
        in repository_artifacts
    }


async def verify_retrieval(
    artifact,
) -> None:
    content = await download_repository_artifact(
        artifact.artifact_id
    )

    actual_sha256 = hashlib.sha256(
        content
    ).hexdigest()

    if actual_sha256 != artifact.output_sha256:
        raise RuntimeError(
            "Repository retrieval SHA-256 does not "
            f"match {artifact.artifact_key}."
        )

    artifact.quality.checks[
        "retrieval_verified"
    ] = True


async def publish_programme(
    reference_number: str,
) -> ProgrammeManifest:
    manifest = build_manifest(
        reference_number
    )

    if not manifest.complete:
        raise RuntimeError(
            "Programme cannot be published because "
            f"only {manifest.completed_artifact_count} "
            f"of {manifest.expected_artifact_count} "
            "final artifacts are available."
        )

    repository_artifacts = (
        await get_repository_artifacts(
            reference_number
        )
    )

    existing_artifacts = (
        get_existing_artifact_map(
            repository_artifacts
        )
    )

    for artifact in manifest.artifacts:
        validation = (
            validate_final_png(
                image_path=(
                    artifact.local_path
                ),
                expected_sha256=(
                    artifact.output_sha256
                ),
            )
        )

        if not validation[
            "valid"
        ]:
            raise RuntimeError(
                "Artifact validation failed for "
                f"{artifact.artifact_key}."
            )

        repository_key = (
            f"{artifact.engine_id}_"
            f"{artifact.direction_id.lower()}"
        )

        existing = (
            existing_artifacts.get(
                repository_key
            )
        )

        if existing is not None:
            if (
                existing["sha256"].lower()
                != artifact.output_sha256.lower()
            ):
                raise RuntimeError(
                    "Repository already contains "
                    f"{repository_key}, but its "
                    "SHA-256 does not match the "
                    "local artifact."
                )

            artifact.artifact_id = (
                existing[
                    "artifactId"
                ]
            )

            artifact.retrieval_url = (
                existing[
                    "downloadUrl"
                ]
            )

            await verify_retrieval(
                artifact
            )

            logger.info(
                "Reusing repository artifact "
                "%s with ID %s",
                repository_key,
                artifact.artifact_id,
            )

            persist_manifest(
                manifest
            )

            continue

        repository_result = (
            await upload_artifact(
                reference_number=(
                    reference_number
                ),
                engine_id=(
                    artifact.engine_id
                ),
                direction_id=(
                    artifact.direction_id
                ),
                file_path=(
                    artifact.local_path
                ),
                sha256=(
                    artifact.output_sha256
                ),
            )
        )

        artifact.artifact_id = (
            repository_result[
                "artifactId"
            ]
        )

        artifact.retrieval_url = (
            repository_result[
                "downloadUrl"
            ]
        )

        await verify_retrieval(
            artifact
        )

        logger.info(
            "Published %s as artifact %s",
            artifact.artifact_key,
            artifact.artifact_id,
        )

        persist_manifest(
            manifest
        )

    manifest.complete = True

    persist_manifest(
        manifest
    )

    await upload_manifest(
        reference_number=(
            reference_number
        ),
        manifest=(
            manifest.model_dump(
                mode="json"
            )
        ),
    )

    logger.info(
        "Published manifest for %s",
        reference_number,
    )

    return manifest
