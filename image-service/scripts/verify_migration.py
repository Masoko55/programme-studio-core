"""Real integration driver. Never fabricates model outputs or modifies source jobs."""
import argparse
import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config.settings import settings
from app.services.atomic import write_json
from app.services.image_job_service import create_image_job
from app.services.image_execution import execute_image_job
from app.services.comfyui_client import get_runtime_health
from app.services.prompt_repository import load_prompts_document

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-reference')
    parser.add_argument('--reference')
    parser.add_argument('--max-outputs', type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(await get_runtime_health()), flush=True)
    reference = args.reference
    if args.source_reference:
        document = load_prompts_document(args.source_reference)
        reference = uuid.uuid4().hex[:6].upper() + '-' + str(uuid.uuid4().int % 1000000).zfill(6)
        target = settings.programme_data_path / reference
        target.mkdir(exist_ok=False)
        document['reference_number'] = reference
        write_json(target/'prompts.json', document)
        write_json(target/'brief.json', document['brief'])
        write_json(target/'migration-source.json', {'source_reference': args.source_reference, 'purpose': 'ComfyUI migration verification'})
        print('NEW_REFERENCE='+reference, flush=True)
    if not reference:
        parser.error('Supply --reference or --source-reference')
    create_image_job(reference)
    state = await execute_image_job(reference, max_outputs=args.max_outputs)
    print(json.dumps(state.model_dump(mode='json'), indent=2), flush=True)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    asyncio.run(main())
