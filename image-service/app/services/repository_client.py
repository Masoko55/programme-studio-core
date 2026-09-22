import logging
from pathlib import Path

import httpx

from app.config.settings import settings


logger = logging.getLogger(
    "uvicorn.error"
)


def get_repository_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=(
            settings
            .repository_connect_timeout_seconds
        ),
        read=(
            settings
            .repository_read_timeout_seconds
        ),
        write=300.0,
        pool=30.0,
    )


async def get_repository_health() -> dict:
    timeout = get_repository_timeout()

    try:
        async with httpx.AsyncClient(
            timeout=timeout
        ) as client:
            response = await client.get(
                f"{settings.repository_service_base_url}"
                "/health"
            )

            response.raise_for_status()

            return response.json()

    except (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
    ) as error:
        raise RuntimeError(
            "Repository Service is unavailable."
        ) from error

    except httpx.HTTPStatusError as error:
        raise RuntimeError(
            "Repository Service health check "
            f"returned HTTP "
            f"{error.response.status_code}."
        ) from error


def build_repository_artifact_key(
    engine_id: str,
    direction_id: str,
) -> str:
    return (
        f"{engine_id}_"
        f"{direction_id.lower()}"
    )


async def upload_artifact(
    reference_number: str,
    engine_id: str,
    direction_id: str,
    file_path: str,
    sha256: str,
) -> dict:
    path = Path(
        file_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {file_path}"
        )

    artifact_key = (
        build_repository_artifact_key(
            engine_id=engine_id,
            direction_id=direction_id,
        )
    )

    timeout = get_repository_timeout()

    logger.info(
        "Uploading artifact %s/%s "
        "to Repository Service",
        engine_id,
        direction_id,
    )

    try:
        async with httpx.AsyncClient(
            timeout=timeout
        ) as client:
            with path.open(
                "rb"
            ) as file:
                response = await client.put(
                    (
                        f"{settings.repository_service_base_url}"
                        f"/v1/references/"
                        f"{reference_number}"
                        f"/artifacts/"
                        f"{artifact_key}"
                    ),
                    headers={
                        "Content-Type": "image/png",
                        "X-SHA256": sha256,
                    },
                    content=file.read(),
                )

            response.raise_for_status()

            result = response.json()

    except (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
    ) as error:
        raise RuntimeError(
            "Failed to upload artifact to "
            "Repository Service."
        ) from error

    except httpx.HTTPStatusError as error:
        raise RuntimeError(
            "Repository artifact upload returned "
            f"HTTP {error.response.status_code}: "
            f"{error.response.text}"
        ) from error

    required_fields = {
        "artifactId",
        "referenceNumber",
        "artifactKey",
        "sha256",
        "downloadUrl",
    }

    missing_fields = (
        required_fields
        - result.keys()
    )

    if missing_fields:
        raise RuntimeError(
            "Repository artifact response is "
            "missing required fields: "
            + ", ".join(
                sorted(
                    missing_fields
                )
            )
        )

    if (
        result["referenceNumber"]
        != reference_number
    ):
        raise RuntimeError(
            "Repository returned an unexpected "
            "reference number."
        )

    if (
        result["sha256"].lower()
        != sha256.lower()
    ):
        raise RuntimeError(
            "Repository returned a SHA-256 "
            "that does not match the uploaded file."
        )

    logger.info(
        "Uploaded artifact %s/%s "
        "with ID %s",
        engine_id,
        direction_id,
        result["artifactId"],
    )

    return result


async def upload_manifest(
    reference_number: str,
    manifest: dict,
) -> dict:
    timeout = get_repository_timeout()

    logger.info(
        "Uploading manifest for %s",
        reference_number,
    )

    try:
        async with httpx.AsyncClient(
            timeout=timeout
        ) as client:
            response = await client.put(
                (
                    f"{settings.repository_service_base_url}"
                    f"/v1/references/"
                    f"{reference_number}"
                    f"/manifest"
                ),
                json=manifest,
            )

            response.raise_for_status()

            return response.json()

    except (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
    ) as error:
        raise RuntimeError(
            "Failed to upload manifest to "
            "Repository Service."
        ) from error

    except httpx.HTTPStatusError as error:
        raise RuntimeError(
            "Repository manifest upload returned "
            f"HTTP {error.response.status_code}: "
            f"{error.response.text}"
        ) from error


async def get_repository_artifacts(
    reference_number: str,
) -> list:
    timeout = get_repository_timeout()

    async with httpx.AsyncClient(
        timeout=timeout
    ) as client:
        response = await client.get(
            (
                f"{settings.repository_service_base_url}"
                f"/v1/references/"
                f"{reference_number}"
                f"/artifacts"
            )
        )

        response.raise_for_status()

        return response.json()


async def download_repository_artifact(
    artifact_id: str,
) -> bytes:
    timeout = get_repository_timeout()

    try:
        async with httpx.AsyncClient(
            timeout=timeout
        ) as client:
            response = await client.get(
                f"{settings.repository_service_base_url}"
                f"/v1/artifacts/{artifact_id}"
            )
            response.raise_for_status()
            return response.content

    except (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
    ) as error:
        raise RuntimeError(
            "Failed to retrieve artifact from "
            "Repository Service."
        ) from error

    except httpx.HTTPStatusError as error:
        raise RuntimeError(
            "Repository artifact retrieval returned "
            f"HTTP {error.response.status_code}: "
            f"{error.response.text}"
        ) from error


async def get_repository_manifest(
    reference_number: str,
) -> dict:
    timeout = get_repository_timeout()

    async with httpx.AsyncClient(
        timeout=timeout
    ) as client:
        response = await client.get(
            (
                f"{settings.repository_service_base_url}"
                f"/v1/references/"
                f"{reference_number}"
                f"/manifest"
            )
        )

        response.raise_for_status()

        return response.json()
