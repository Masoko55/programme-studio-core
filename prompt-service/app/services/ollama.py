import asyncio
import logging
import time

import httpx

from app.config.settings import (
    settings,
)

from app.services.gpu_lease import (
    GPULease,
)


logger = logging.getLogger(
    "uvicorn.error"
)


COMFYUI_RELEASE_TIMEOUT_SECONDS = 60
COMFYUI_USED_VRAM_THRESHOLD_BYTES = (
    1024
    * 1024
    * 1024
)


def get_ollama_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=(
            settings
            .ollama_connect_timeout_seconds
        ),
        read=(
            settings
            .ollama_read_timeout_seconds
        ),
        write=30.0,
        pool=30.0,
    )


async def get_models():
    timeout = (
        get_ollama_timeout()
    )

    try:
        async with httpx.AsyncClient(
            timeout=timeout
        ) as client:
            response = (
                await client.get(
                    f"{settings.ollama_base_url}"
                    "/api/tags"
                )
            )

            response.raise_for_status()

            return (
                response.json()
            )

    except httpx.ConnectTimeout as error:
        raise ValueError(
            "Timed out while connecting to "
            "the Ollama server."
        ) from error

    except httpx.ConnectError as error:
        raise ValueError(
            "Could not connect to the "
            "Ollama server."
        ) from error

    except httpx.HTTPStatusError as error:
        raise ValueError(
            "Ollama returned an HTTP error: "
            f"{error.response.status_code}"
        ) from error


def required_models() -> tuple[
    str,
    str,
    str,
]:
    return (
        settings.direction_a_model,
        settings.direction_b_model,
        settings.direction_c_model,
    )


async def assert_required_models_available() -> dict:
    payload = (
        await get_models()
    )

    installed = {
        item.get(
            "name"
        )
        for item
        in payload.get(
            "models",
            [],
        )
        if item.get(
            "name"
        )
    }

    missing = [
        model
        for model
        in required_models()
        if model not in installed
    ]

    if missing:
        raise ValueError(
            "Required Ollama model(s) are unavailable: "
            + ", ".join(
                missing
            )
        )

    return {
        "required_models": (
            list(
                required_models()
            )
        ),
        "installed_models": (
            sorted(
                installed
            )
        ),
    }


async def _generate_text(
    model: str,
    prompt: str,
    response_format=None,
):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": 0,
    }

    if (
        response_format
        is not None
    ):
        payload[
            "format"
        ] = (
            response_format
        )

    timeout = (
        get_ollama_timeout()
    )

    logger.info(
        "Sending generation request "
        "to Ollama model %s",
        model,
    )

    try:
        async with httpx.AsyncClient(
            timeout=timeout
        ) as client:
            response = (
                await client.post(
                    f"{settings.ollama_base_url}"
                    "/api/generate",
                    json=payload,
                )
            )

            response.raise_for_status()

            result = (
                response.json()
            )

            logger.info(
                "Received generation response "
                "from Ollama model %s",
                model,
            )

            return result

    except httpx.ReadTimeout as error:
        logger.error(
            "Ollama model %s exceeded the "
            "configured read timeout of "
            "%s seconds.",
            model,
            settings
            .ollama_read_timeout_seconds,
        )

        raise ValueError(
            f"Ollama model '{model}' did not "
            "finish within "
            f"{settings.ollama_read_timeout_seconds} "
            "seconds."
        ) from error

    except httpx.ConnectTimeout as error:
        raise ValueError(
            "Timed out while connecting to "
            f"Ollama for model '{model}'."
        ) from error

    except httpx.ConnectError as error:
        raise ValueError(
            "Could not connect to the Ollama "
            f"server for model '{model}'."
        ) from error

    except httpx.HTTPStatusError as error:
        raise ValueError(
            f"Ollama returned HTTP "
            f"{error.response.status_code} "
            f"while generating with "
            f"model '{model}'."
        ) from error


def _device_used_vram(
    device: dict,
) -> int:
    torch_total = int(
        device.get(
            "torch_vram_total",
            0,
        )
        or 0
    )

    torch_free = int(
        device.get(
            "torch_vram_free",
            0,
        )
        or 0
    )

    if (
        torch_total > 0
        and torch_free >= 0
    ):
        return max(
            0,
            torch_total
            - torch_free,
        )

    vram_total = int(
        device.get(
            "vram_total",
            0,
        )
        or 0
    )

    vram_free = int(
        device.get(
            "vram_free",
            0,
        )
        or 0
    )

    if (
        vram_total > 0
        and vram_free >= 0
    ):
        return max(
            0,
            vram_total
            - vram_free,
        )

    return 0


def _comfyui_memory_released(
    stats: dict,
) -> bool:
    devices = (
        stats.get(
            "devices",
            [],
        )
        or []
    )

    if not devices:
        return True

    used_values = [
        _device_used_vram(
            device
        )
        for device
        in devices
    ]

    return all(
        used
        <= COMFYUI_USED_VRAM_THRESHOLD_BYTES
        for used
        in used_values
    )


async def _wait_for_comfyui_release(
    client: httpx.AsyncClient,
) -> None:
    response = (
        await client.post(
            settings.comfyui_base_url
            + "/free",
            json={
                "unload_models": True,
                "free_memory": True,
            },
        )
    )

    response.raise_for_status()

    deadline = (
        time.monotonic()
        + COMFYUI_RELEASE_TIMEOUT_SECONDS
    )

    last_stats = None

    while True:
        stats_response = (
            await client.get(
                settings.comfyui_base_url
                + "/system_stats"
            )
        )

        stats_response.raise_for_status()

        last_stats = (
            stats_response.json()
        )

        if (
            _comfyui_memory_released(
                last_stats
            )
        ):
            logger.info(
                "ComfyUI GPU memory release confirmed"
            )

            return

        if (
            time.monotonic()
            >= deadline
        ):
            logger.warning(
                "ComfyUI memory release timed out; "
                "continuing after /free request. "
                "system_stats=%s",
                last_stats,
            )

            return

        await asyncio.sleep(
            1
        )


async def _release_ollama_models(
    client: httpx.AsyncClient,
) -> None:
    resident = (
        await client.get(
            settings.ollama_base_url
            + "/api/ps"
        )
    )

    resident.raise_for_status()

    for item in (
        resident.json()
        .get(
            "models",
            [],
        )
    ):
        model_name = (
            item.get(
                "name"
            )
        )

        if not model_name:
            continue

        response = (
            await client.post(
                settings.ollama_base_url
                + "/api/generate",
                json={
                    "model": model_name,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": 0,
                },
            )
        )

        response.raise_for_status()


async def generate_text(
    model: str,
    prompt: str,
    response_format=None,
):
    async with GPULease():
        async with httpx.AsyncClient(
            timeout=60
        ) as client:
            queue_response = (
                await client.get(
                    settings.comfyui_base_url
                    + "/queue"
                )
            )

            queue_response.raise_for_status()

            queue = (
                queue_response.json()
            )

            if (
                queue.get(
                    "queue_running"
                )
                or queue.get(
                    "queue_pending"
                )
            ):
                raise ValueError(
                    "ComfyUI has active jobs; retry "
                    "when its GPU work finishes"
                )

            await (
                _wait_for_comfyui_release(
                    client
                )
            )

            await (
                _release_ollama_models(
                    client
                )
            )

        return (
            await _generate_text(
                model,
                prompt,
                response_format,
            )
        )