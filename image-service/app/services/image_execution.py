import asyncio
import json
import logging
from pathlib import Path
import httpx
from app.config.settings import settings
from app.engines.registry import get_engine
from app.services.comfyui_client import ComfyUIClient, ComfyUIError
from app.services.execution_plan import build_execution_plan
from app.services.gpu_lease import GPULease
from app.services.job_persistence import load_image_job_state, persist_image_job_state, update_output
from app.services.prompt_repository import get_direction, load_prompts_document

logger = logging.getLogger("uvicorn.error")


def _candidate_was_rejected(reference_number: str, engine_id: str, direction_id: str) -> bool:
    """Only retry a completed generation rejected by local background QA.

    Transport, validation, and submission failures can be ambiguous.  They must
    remain resumable failures rather than being resubmitted automatically.
    """
    record_path = (
        settings.programme_data_path
        / reference_number
        / "backgrounds"
        / engine_id
        / f"image-{direction_id.lower()}.json"
    )
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return record.get("status") == "rejected" and bool(record.get("rejection_reason"))


async def execute_image_job(reference_number: str, max_outputs: int | None = None):
    """Run in frozen engine-first order; max_outputs supports staged verification."""
    async with GPULease():
        state = load_image_job_state(reference_number)
        document = load_prompts_document(reference_number)
        expected = [(s['engine_id'], s['direction']) for s in build_execution_plan(document)['execution_order']]
        if [(o.engine_id, o.direction_id) for o in state.outputs] != expected:
            raise ValueError("Historical engine plan differs from current configuration. Preserve this job and use a new reference for FLUX.2.")
        state.status, state.current_stage, state.error = "processing", "generating", None
        persist_image_job_state(state)
        # Do not overlap ComfyUI with a resident Ollama model.
        try:
            async with httpx.AsyncClient(timeout=60) as http:
                response = await http.get(settings.ollama_base_url + '/api/ps')
                response.raise_for_status()
                if response.json().get('models'):
                    raise RuntimeError("Ollama has resident models; unload them before image generation")
            async with ComfyUIClient() as client:
                current_engine = None
                generated = 0
                try:
                    for output in state.outputs:
                        if current_engine != output.engine_id:
                            queue = (await client.request('GET', '/queue')).json()
                            entries = queue.get('queue_running', []) + queue.get('queue_pending', [])
                            # A surviving remote job is resumed by its persisted submission token.
                            own_path = settings.programme_data_path / reference_number / 'backgrounds' / output.engine_id / f'image-{output.direction_id.lower()}.json'
                            own = json.loads(own_path.read_text()) if own_path.exists() else {}
                            if entries:
                                if not own or any(e[3].get('programme_request_id') != own.get('request_id') for e in entries):
                                    raise RuntimeError('ComfyUI has another active job; retry later')
                            else:
                                await client.release_models()
                            current_engine = output.engine_id
                        state.current_stage = f'{output.engine_id}:{output.direction_id}'
                        persist_image_job_state(state)
                        direction = get_direction(document, output.direction_id)
                        try:
                            for retry_count in range(settings.max_candidate_retries + 1):
                                try:
                                    result = await get_engine(output.engine_id).generate(
                                        reference_number=reference_number,
                                        direction_id=output.direction_id,
                                        positive_prompt=direction['positive_prompt'],
                                        negative_prompt=direction.get('negative_prompt', ''),
                                    )
                                    break
                                except ComfyUIError as error:
                                    if (
                                        retry_count >= settings.max_candidate_retries
                                        or not _candidate_was_rejected(
                                            reference_number,
                                            output.engine_id,
                                            output.direction_id,
                                        )
                                    ):
                                        raise
                                    delay_seconds = 2 ** retry_count
                                    logger.warning(
                                        "event=candidate_retry reference=%s engine=%s direction=%s retry=%s/%s delay_seconds=%s reason=%s",
                                        reference_number,
                                        output.engine_id,
                                        output.direction_id,
                                        retry_count + 1,
                                        settings.max_candidate_retries,
                                        delay_seconds,
                                        error,
                                    )
                                    await asyncio.sleep(delay_seconds)
                            output.seed = result['seed']
                            output.prompt_id = result['prompt_id']
                            output.workflow_sha256 = result['workflow_sha256']
                            update_output(state, output.engine_id, output.direction_id, 'complete', result['output_path'], result['sha256'])
                            logger.info('event=candidate_complete reference=%s engine=%s direction=%s reused=%s', reference_number, output.engine_id, output.direction_id, result['reused'])
                            if not result['reused']:
                                generated += 1
                            if max_outputs is not None and generated >= max_outputs:
                                break
                        except Exception as error:
                            update_output(state, output.engine_id, output.direction_id, 'failed', error=str(error))
                            raise
                finally:
                    # On timeout an active remote job is kept and reconciled on resume.
                    queue = (await client.request('GET', '/queue')).json()
                    if not queue.get('queue_running') and not queue.get('queue_pending'):
                        await client.release_models()
            state.status = 'generated' if state.completed_outputs == 9 else 'paused'
            state.current_stage = 'generation_complete' if state.completed_outputs == 9 else 'generation_checkpoint'
            persist_image_job_state(state)
            return state
        except Exception as error:
            state.status, state.current_stage, state.error = 'failed', 'generation_failed', str(error)
            persist_image_job_state(state)
            raise
