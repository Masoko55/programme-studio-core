from fastapi import (
    FastAPI,
    HTTPException,
)
from fastapi.responses import (
    JSONResponse,
)

from app.schemas.image_job import (
    ImageJobRequest,
)
from app.schemas.final_selection import (
    FinalSelectionRequest,
)
from app.services.execution_plan import (
    build_execution_plan,
)
from app.services.image_execution import (
    execute_image_job,
)
from app.services.image_job_service import (
    create_image_job,
    get_image_job,
)
from app.services.image_runtime_client import (
    get_runtime_health,
)
from app.services.image_workflow import (
    run_image_workflow,
)
from app.services.manifest_service import (
    build_manifest,
    persist_manifest,
)
from app.services.prompt_repository import (
    load_prompts_document,
)
from app.services.publication_service import (
    publish_programme,
)
from app.services.repository_client import (
    get_repository_health,
)
from app.services.workflow_state import (
    load_workflow_state,
)
from app.services.final_selection_service import (
    select_final,
)


app = FastAPI(
    title=(
        "Nerdcode Programme Studio "
        "- Image Service"
    ),
    version="1.0.0",
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "image-service",
    }


@app.get("/ready")
async def ready():
    runtime_status = {
        "status": "unavailable",
    }

    repository_status = {
        "status": "unavailable",
    }

    runtime_available = False
    repository_available = False

    try:
        runtime = (
            await get_runtime_health()
        )

        runtime_status = {
            "status": "available",
            "details": runtime,
        }

        runtime_available = True

    except RuntimeError as error:
        runtime_status = {
            "status": "unavailable",
            "error": str(error),
        }

    try:
        repository = (
            await get_repository_health()
        )

        repository_status = {
            "status": "available",
            "details": repository,
        }

        repository_available = True

    except RuntimeError as error:
        repository_status = {
            "status": "unavailable",
            "error": str(error),
        }

    response = {
        "status": (
            "ready"
            if (
                runtime_available
                and repository_available
            )
            else "not_ready"
        ),
        "service": "image-service",
        "comfyui": (
            runtime_status
        ),
        "repository_service": (
            repository_status
        ),
    }

    if (
        runtime_available
        and repository_available
    ):
        return response

    return JSONResponse(
        status_code=503,
        content=response,
    )


@app.post(
    "/v1/image-jobs/plan"
)
async def create_execution_plan(
    request: ImageJobRequest,
):
    try:
        prompts_document = (
            load_prompts_document(
                request.reference_number
            )
        )

        plan = build_execution_plan(
            prompts_document
        )

        return {
            "status": "ready",
            "plan": plan,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


@app.post(
    "/v1/image-jobs"
)
async def create_job(
    request: ImageJobRequest,
):
    try:
        state = create_image_job(
            request.reference_number
        )

        return state.model_dump(
            mode="json"
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


@app.get(
    "/v1/image-jobs/{reference_number}"
)
async def get_job(
    reference_number: str,
):
    try:
        state = get_image_job(
            reference_number
        )

        return state.model_dump(
            mode="json"
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


@app.post(
    "/v1/image-jobs/{reference_number}/selection"
)
async def create_final_selection(
    reference_number: str,
    request: FinalSelectionRequest,
):
    """Create the client-selected A4 final, with optional headshot and logo."""
    try:
        return select_final(reference_number, request)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post(
    "/v1/image-jobs/{reference_number}/execute"
)
async def execute_job(
    reference_number: str,
):
    try:
        await get_runtime_health()

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        )

    try:
        state = await execute_image_job(
            reference_number
        )

        return state.model_dump(
            mode="json"
        )

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


@app.post(
    "/v1/image-jobs/{reference_number}/manifest"
)
async def create_manifest(
    reference_number: str,
):
    try:
        manifest = build_manifest(
            reference_number
        )

        manifest_path = (
            persist_manifest(
                manifest
            )
        )

        return {
            "manifest_path": str(
                manifest_path
            ),
            "manifest": (
                manifest.model_dump(
                    mode="json"
                )
            ),
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


@app.post(
    "/v1/image-jobs/{reference_number}/publish"
)
async def publish_job(
    reference_number: str,
):
    try:
        await get_repository_health()

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        )

    try:
        manifest = await publish_programme(
            reference_number
        )

        return {
            "status": "published",
            "reference_number": (
                reference_number
            ),
            "manifest": (
                manifest.model_dump(
                    mode="json"
                )
            ),
        }

    except RuntimeError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


@app.post(
    "/v1/image-jobs/{reference_number}/workflow"
)
async def run_workflow(
    reference_number: str,
):
    try:
        await get_runtime_health()

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        )

    try:
        await get_repository_health()

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        )

    try:
        result = (
            await run_image_workflow(
                reference_number
            )
        )

        return result

    except RuntimeError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


@app.get(
    "/v1/image-jobs/"
    "{reference_number}/workflow"
)
async def get_workflow(
    reference_number: str,
):
    state = load_workflow_state(
        reference_number
    )

    if state is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Image workflow state "
                "not found."
            ),
        )

    return state
