from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
)
from pydantic import BaseModel

from app.schemas.prompt_job import (
    PromptJobRequest,
)
from app.services.creative_direction import (
    generate_creative_direction,
)
from app.services.job_service import (
    calculate_sha256,
    generate_reference_number,
)
from app.services.ollama import (
    assert_required_models_available,
    generate_text,
    get_models,
)
from app.services.persistence import (
    get_job_status,
    load_brief,
    load_job_metadata,
    load_prompts_document,
    update_job_status,
)
from app.services.image_service_client import (
    run_image_workflow,
)
from app.services.prompt_workflow import (
    run_prompt_workflow,
)


app = FastAPI(
    title=(
        "Nerdcode Programme Studio "
        "- Prompt Service"
    ),
    version="1.0.0",
)


class GenerateRequest(BaseModel):
    model: str
    prompt: str


async def hand_off_to_image_service(
    reference_number: str,
) -> None:
    """Advance a frozen prompt job through the Image Service without user action."""
    update_job_status(
        reference_number,
        status="processing",
        current_stage="image_workflow_running",
        error=None,
    )

    try:
        await run_image_workflow(reference_number)
    except RuntimeError as error:
        update_job_status(
            reference_number,
            status="failed",
            current_stage="image_workflow_failed",
            error=str(error),
        )
        raise

    update_job_status(
        reference_number,
        status="complete",
        current_stage="complete",
        error=None,
    )


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "prompt-service",
    }


@app.get("/ready")
async def ready():
    try:
        model_status = await assert_required_models_available()
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return {
        "status": "ready",
        "service": "prompt-service",
        "models": model_status["required_models"],
    }


@app.get(
    "/internal/ollama/models"
)
async def ollama_models():
    return await get_models()


@app.post(
    "/internal/ollama/generate"
)
async def ollama_generate(
    request: GenerateRequest,
):
    return await generate_text(
        model=request.model,
        prompt=request.prompt,
    )


@app.post("/v1/prompt-jobs", status_code=202)
async def create_prompt_job(
    request: PromptJobRequest,
    background_tasks: BackgroundTasks,
):
    try:
        await assert_required_models_available()
        brief = request.model_dump()

        reference_number = (
            generate_reference_number()
        )

        input_sha256 = (
            calculate_sha256(
                brief
            )
        )

        result = (
            await run_prompt_workflow(
                reference_number=(
                    reference_number
                ),
                input_sha256=(
                    input_sha256
                ),
                brief=brief,
                resume=False,
            )
        )

        document = result[
            "prompts_document"
        ]

        background_tasks.add_task(
            hand_off_to_image_service,
            reference_number,
        )

        return {
            "status": "accepted",
            "reference_number": (
                reference_number
            ),
            "input_sha256": (
                input_sha256
            ),
            "brief_path": result[
                "brief_path"
            ],
            "prompts_path": result[
                "prompts_path"
            ],
            "prompts_document": (
                document.model_dump(
                    mode="json"
                )
            ),
        }

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        )


@app.post(
    "/v1/prompt-jobs/"
    "{reference_number}/resume"
)
async def resume_prompt_job(
    reference_number: str,
    background_tasks: BackgroundTasks,
):
    try:
        await assert_required_models_available()
        brief = load_brief(
            reference_number
        )

        metadata = (
            load_job_metadata(
                reference_number
            )
        )

        input_sha256 = metadata[
            "input_sha256"
        ]

        result = (
            await run_prompt_workflow(
                reference_number=(
                    reference_number
                ),
                input_sha256=(
                    input_sha256
                ),
                brief=brief,
                resume=True,
            )
        )

        document = result[
            "prompts_document"
        ]

        background_tasks.add_task(
            hand_off_to_image_service,
            reference_number,
        )

        return {
            "status": "complete",
            "reference_number": (
                reference_number.upper()
            ),
            "input_sha256": (
                input_sha256
            ),
            "brief_path": result[
                "brief_path"
            ],
            "prompts_path": result[
                "prompts_path"
            ],
            "prompts_document": (
                document.model_dump(
                    mode="json"
                )
            ),
        }

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


@app.get(
    "/v1/prompt-jobs/"
    "{reference_number}"
)
async def get_prompt_job(
    reference_number: str,
):
    try:
        status = get_job_status(
            reference_number
        )

        if status == "not_found":
            raise HTTPException(
                status_code=404,
                detail=(
                    "Prompt job not found."
                ),
            )

        brief = load_brief(
            reference_number
        )

        response = {
            "reference_number": (
                reference_number.upper()
            ),
            "status": status,
            "brief": brief,
        }

        try:
            metadata = (
                load_job_metadata(
                    reference_number
                )
            )

            response[
                "current_stage"
            ] = metadata.get(
                "current_stage"
            )

            response[
                "input_sha256"
            ] = metadata.get(
                "input_sha256"
            )

            response[
                "error"
            ] = metadata.get(
                "error"
            )

            response[
                "updated_at"
            ] = metadata.get(
                "updated_at"
            )

        except FileNotFoundError:
            pass

        return response

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
    "/v1/prompt-jobs/"
    "{reference_number}/prompts"
)
async def get_prompt_job_prompts(
    reference_number: str,
):
    try:
        document = (
            load_prompts_document(
                reference_number
            )
        )

        return document.model_dump(
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
    "/v1/prompt-jobs/"
    "directions/{direction_id}"
)
async def create_creative_direction(
    direction_id: str,
    request: PromptJobRequest,
):
    try:
        brief = request.model_dump()

        direction = (
            await generate_creative_direction(
                direction_id=(
                    direction_id
                ),
                brief=brief,
            )
        )

        return direction.model_dump(
            mode="json"
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )
