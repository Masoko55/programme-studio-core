import asyncio
import logging

import httpx

from app.config.settings import settings


logger = logging.getLogger(
    "uvicorn.error"
)


def image_service_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=(
            settings
            .image_service_connect_timeout_seconds
        ),
        read=(
            settings
            .image_service_read_timeout_seconds
        ),
        write=(
            settings
            .image_service_connect_timeout_seconds
        ),
        pool=(
            settings
            .image_service_connect_timeout_seconds
        ),
    )


async def create_image_job(
    client: httpx.AsyncClient,
    reference_number: str,
) -> dict:
    url = (
        settings
        .image_service_base_url
        .rstrip(
            "/"
        )
        + "/v1/image-jobs"
    )

    response = await client.post(
        url,
        json={
            "reference_number": (
                reference_number
            )
        },
    )

    response.raise_for_status()

    return response.json()


async def submit_workflow(
    client: httpx.AsyncClient,
    reference_number: str,
) -> dict:
    url = (
        settings
        .image_service_base_url
        .rstrip(
            "/"
        )
        + "/v1/image-jobs/"
        + reference_number
        + "/workflow"
    )

    delays = (
        2,
        4,
        8,
        16,
        30,
    )

    attempts = (
        len(
            delays
        )
        + 1
    )

    last_error = None

    for attempt in range(
        attempts
    ):
        try:
            response = await client.post(
                url
            )

            if (
                response.status_code
                == 503
            ):
                detail = (
                    response.text
                )

                logger.warning(
                    "Image Service workflow returned 503 "
                    "for %s on attempt %s/%s: %s",
                    reference_number,
                    attempt + 1,
                    attempts,
                    detail,
                )

                if (
                    attempt
                    >= len(
                        delays
                    )
                ):
                    response.raise_for_status()

                await asyncio.sleep(
                    delays[
                        attempt
                    ]
                )

                continue

            response.raise_for_status()

            return response.json()

        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
        ) as error:
            last_error = error

            logger.warning(
                "Image Service connection failed "
                "for %s on attempt %s/%s: %s",
                reference_number,
                attempt + 1,
                attempts,
                error,
            )

            if (
                attempt
                >= len(
                    delays
                )
            ):
                raise

            await asyncio.sleep(
                delays[
                    attempt
                ]
            )

    if last_error:
        raise last_error

    raise RuntimeError(
        "Image Service workflow did not return "
        "a successful response."
    )


async def run_image_workflow(
    reference_number: str,
) -> dict:
    timeout = (
        image_service_timeout()
    )

    try:
        async with httpx.AsyncClient(
            timeout=timeout
        ) as client:
            await create_image_job(
                client,
                reference_number,
            )

            logger.info(
                "Created image job for %s",
                reference_number,
            )

            result = await submit_workflow(
                client,
                reference_number,
            )

            logger.info(
                "Image workflow completed handoff "
                "for %s",
                reference_number,
            )

            return result

    except httpx.HTTPStatusError as error:
        detail = (
            error.response.text
            if error.response
            is not None
            else str(
                error
            )
        )

        raise RuntimeError(
            "Image Service handover failed for "
            f"{reference_number}: "
            f"HTTP "
            f"{error.response.status_code if error.response else 'unknown'} "
            f"{detail}"
        ) from error

    except (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
    ) as error:
        raise RuntimeError(
            "Image Service handover failed for "
            f"{reference_number}: {error}"
        ) from error