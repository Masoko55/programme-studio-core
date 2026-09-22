import hashlib
import json
from pathlib import Path
from app.config.settings import settings
from app.services.atomic import write_json
from app.services.artifact_validation import validate_final_png
from app.services.comfyui_client import validate_background
from app.services.composer import compose_programme
from app.services.prompt_repository import get_direction, load_prompts_document, get_job_directory

COMPOSER_VERSION = '2.1-content-sized-panels'

def get_final_output_path(reference_number: str, engine_id: str, direction_id: str) -> Path:
    from app.engines.registry import get_engine_descriptor
    get_engine_descriptor(engine_id)
    if direction_id not in {'A','B','C'}:
        raise ValueError('Invalid direction')
    return get_job_directory(reference_number) / 'final' / f'{engine_id}-{direction_id.lower()}.png'


def composition_inputs(reference_number: str, direction_id: str, background_path: str) -> tuple[dict, dict, str]:
    document = load_prompts_document(reference_number)
    direction = get_direction(document, direction_id)
    layout = direction.get('layout_contract') or direction.get('layout_guidance')
    if not layout:
        raise ValueError('Direction has no layout_contract or layout_guidance')
    background = validate_background(Path(background_path))
    assets = {name: hashlib.sha256(Path(document['brief'][name]).read_bytes()).hexdigest()
              for name in ['headshot_path','logo_path'] if document['brief'].get(name)}
    evidence = {'brief':document['brief'], 'layout':layout, 'background_sha256':background['sha256'],
                'assets':assets, 'composer_version':COMPOSER_VERSION}
    fingerprint = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    return document['brief'], layout, fingerprint


def validate_composition(reference_number: str, engine_id: str, direction_id: str) -> dict:
    path = get_final_output_path(reference_number, engine_id, direction_id)
    evidence_path = path.with_suffix('.composition.json')
    evidence = json.loads(evidence_path.read_text())
    background_path = evidence['background_path']
    _, _, fingerprint = composition_inputs(reference_number, direction_id, background_path)
    if evidence['input_fingerprint'] != fingerprint or not evidence.get('composition_completed'):
        raise ValueError('Composition evidence does not match current inputs')
    return validate_final_png(str(path), evidence['sha256'])


def compose_generated_background(reference_number: str, engine_id: str, direction_id: str,
                                 background_path: str) -> dict:
    path = get_final_output_path(reference_number, engine_id, direction_id)
    brief, layout, fingerprint = composition_inputs(reference_number, direction_id, background_path)
    try:
        validation = validate_composition(reference_number, engine_id, direction_id)
        return {'reused':True, 'composition':json.loads(path.with_suffix('.composition.json').read_text()), 'validation':validation}
    except (OSError, ValueError, KeyError):
        pass
    result = compose_programme(reference_number, engine_id, direction_id, background_path, brief, layout)
    validation = validate_final_png(result.output_path, result.sha256)
    evidence = {**result.model_dump(mode='json'), 'input_fingerprint':fingerprint,
                'composition_completed':True, 'programme_rows_placed':len(brief['programme']),
                'title_placed':True, 'headshot_placed':bool(brief.get('headshot_path')),
                'logo_placed':bool(brief.get('logo_path')), 'composer_version':COMPOSER_VERSION}
    write_json(path.with_suffix('.composition.json'), evidence)
    return {'reused':False, 'composition':evidence, 'validation':validation}
