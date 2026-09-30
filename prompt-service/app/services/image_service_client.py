import asyncio
import logging

import httpx

from app.config.settings import settings


logger = logging.getLogger(
    "uvicorn.error"
)


RETRYABLE_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
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
                url,
                json={
                    "reference_number": (
                        reference_number
                    )
                },
            )

            if (
                response.status_code
                == 503
            ):
                logger.warning(
                    "Image Service job creation returned 503 "
                    "for %s on attempt %s/%s: %s",
                    reference_number,
                    attempt + 1,
                    attempts,
                    response.text,
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

        except RETRYABLE_TRANSPORT_ERRORS as error:
            last_error = error

            logger.warning(
                "Image Service job creation transport failure "
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
        "Image Service job creation did not return "
        "a successful response."
    )


async def get_workflow_state(
    client: httpx.AsyncClient,
    reference_number: str,
) -> dict | None:
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

    try:
        response = await client.get(
            url
        )

        if (
            response.status_code
            == 404
        ):
            return None

        response.raise_for_status()

        return response.json()

    except RETRYABLE_TRANSPORT_ERRORS:
        return None


def workflow_is_complete(
    state: dict | None,
) -> bool:
    if not state:
        return False

    image_job = (
        state.get(
            "image_job"
        )
        or {}
    )

    status = (
        image_job.get(
            "status"
        )
        or state.get(
            "status"
        )
    )

    stage = (
        image_job.get(
            "current_stage"
        )
        or state.get(
            "current_stage"
        )
    )

    if (
        status
        in {
            "awaiting_selection",
            "complete",
            "completed",
        }
    ):
        return True

    if (
        stage
        in {
            "generated",
            "awaiting_selection",
            "complete",
            "completed",
        }
    ):
        return True

    outputs = (
        image_job.get(
            "outputs"
        )
        or []
    )

    if not outputs:
        return False

    terminal = {
        "complete",
        "failed",
    }

    return all(
        item.get(
            "status"
        )
        in terminal
        for item
        in outputs
    )


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
                logger.warning(
                    "Image Service workflow returned 503 "
                    "for %s on attempt %s/%s: %s",
                    reference_number,
                    attempt + 1,
                    attempts,
                    response.text,
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

            if (
                response.status_code
                == 409
            ):
                state = (
                    await get_workflow_state(
                        client,
                        reference_number,
                    )
                )

                if (
                    workflow_is_complete(
                        state
                    )
                ):
                    logger.info(
                        "Image workflow for %s is already "
                        "complete after a 409 response.",
                        reference_number,
                    )

                    return state

                logger.warning(
                    "Image Service workflow returned 409 "
                    "for %s on attempt %s/%s: %s",
                    reference_number,
                    attempt + 1,
                    attempts,
                    response.text,
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

        except RETRYABLE_TRANSPORT_ERRORS as error:
            last_error = error

            logger.warning(
                "Image Service workflow transport failure "
                "for %s on attempt %s/%s: %s",
                reference_number,
                attempt + 1,
                attempts,
                error,
            )

            state = (
                await get_workflow_state(
                    client,
                    reference_number,
                )
            )

            if (
                workflow_is_complete(
                    state
                )
            ):
                logger.info(
                    "Image workflow for %s completed despite "
                    "the interrupted HTTP connection.",
                    reference_number,
                )

                return state

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

            result = (
                await submit_workflow(
                    client,
                    reference_number,
                )
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
            if (
                error.response
                is not None
            )
            else str(
                error
            )
        )

        status_code = (
            error.response.status_code
            if (
                error.response
                is not None
            )
            else "unknown"
        )

        raise RuntimeError(
            "Image Service handover failed for "
            f"{reference_number}: "
            f"HTTP {status_code} "
            f"{detail}"
        ) from error

    except RETRYABLE_TRANSPORT_ERRORS as error:
        raise RuntimeError(
            "Image Service handover failed for "
            f"{reference_number}: "
            f"{type(error).__name__}: "
            f"{error}"
        ) from error