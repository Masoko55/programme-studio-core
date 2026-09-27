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
    download_repository_artifact,
    get_repository_artifacts,
    upload_artifact,
    upload_manifest,
)


logger = logging.getLogger(
    "uvicorn.error"
)


async def verify_retrieval(
    artifact,
) -> None:
    content = (
        await download_repository_artifact(
            artifact.artifact_id
        )
    )

    actual_sha256 = hashlib.sha256(
        content
    ).hexdigest()

    if (
        actual_sha256
        != artifact.output_sha256
    ):
        raise RuntimeError(
            "Repository retrieval SHA-256 "
            "does not match the final programme."
        )

    artifact.quality.checks[
        "retrieval_verified"
    ] = True


async def publish_programme(
    reference_number: str,
) -> ProgrammeManifest:
    """Publish exactly one selected and composed final programme."""

    manifest = build_manifest(
        reference_number
    )

    if not manifest.complete:
        raise RuntimeError(
            "Programme cannot be published because "
            "the selected final programme has not "
            "passed validation."
        )

    if len(manifest.artifacts) != 1:
        raise RuntimeError(
            "A programme manifest must contain "
            "exactly one final artifact."
        )

    artifact = manifest.artifacts[0]

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

    if not validation["valid"]:
        raise RuntimeError(
            "Final programme validation failed."
        )

    repository_artifacts = (
        await get_repository_artifacts(
            reference_number
        )
    )

    repository_key = (
        f"{artifact.engine_id}_"
        f"{artifact.direction_id.lower()}"
    )

    existing = next(
        (
            item
            for item in repository_artifacts
            if item.get(
                "artifactKey"
            )
            == repository_key
        ),
        None,
    )

    if existing is not None:
        if (
            existing["sha256"].lower()
            != artifact.output_sha256.lower()
        ):
            raise RuntimeError(
                "Repository already contains "
                "the selected final key but its "
                "SHA-256 does not match."
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

    else:
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
        "Published selected final programme "
        "for %s using %s/%s",
        reference_number,
        artifact.engine_id,
        artifact.direction_id,
    )

    return manifest