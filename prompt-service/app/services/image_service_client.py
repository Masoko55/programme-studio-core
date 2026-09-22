import httpx

from app.config.settings import settings


async def run_image_workflow(reference_number: str) -> dict:
    """Submit one durable, idempotent image workflow for a frozen prompt job."""
    timeout = httpx.Timeout(
        connect=settings.image_service_connect_timeout_seconds,
        read=settings.image_service_read_timeout_seconds,
        write=settings.image_service_connect_timeout_seconds,
        pool=settings.image_service_connect_timeout_seconds,
    )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{settings.image_service_base_url.rstrip('/')}/v1/image-jobs/"
                f"{reference_number}/workflow"
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as error:
        raise RuntimeError(
            "Image Service handover failed for "
            f"{reference_number}: {error}"
        ) from error
