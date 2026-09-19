"""Frozen duration intervention over the existing capacity comparison inputs."""
import json
from pathlib import Path

from . import incremental_capacity as base, reference_data as data

ROOT = Path(__file__).resolve().parents[3]
PLAN = 'config/experiments/knowledge-rehearsal.json'
PREPARED = 'config/experiments/knowledge-rehearsal-prepared.json'
FORMAT = 'neuroshard-knowledge-rehearsal-v1'
SOURCES = (*base.SOURCES, 'src/neuroshard/evolution/rehearsal.py',
           'src/neuroshard/evolution/sharded/rehearsal_job.py',
           'src/neuroshard/evolution/sharded/features.py',
           'src/neuroshard/evolution/sharded/feature_bank.py',
           'src/neuroshard/evolution/sharded/feature_probe.py',
           'scripts/run_knowledge_rehearsal.py')


def read(path):
    return json.loads(Path(path).read_bytes())


def plan():
    original, prepared = base.validate_prepared(ROOT / 'config/experiments/incremental-capacity.json',
        ROOT / 'config/experiments/incremental-capacity-prepared.json')
    value = json.loads(base.committed(ROOT / PLAN))
    expected = {**original['training'], 'steps': 1024, 'learning_rate': .00005}
    if (value['format'] != FORMAT or value['parent'] != original['parent']['checkpoint']
            or value['base_prepared'] != data.identity(prepared)
            or value['arms'] != ['append', 'tail-control'] or value['epochs'] != 8
            or value['recipe'] != expected or value['checkpoints'] != [0, 512, 1024]
            or value['selection_path'] != 'config/experiments/knowledge-rehearsal-selection.json'
            or value['feature_prerequisite'] != 'config/experiments/frozen-feature-results.json'
            or value['prior_selection'] != 'config/experiments/incremental-capacity-selection.json'):
        raise ValueError('Unsupported isolated training-duration intervention')
    for name in SOURCES:
        base.committed(ROOT / name)
    return value, original, prepared


def prerequisites(value):
    proof = json.loads(base.committed(ROOT / value['feature_prerequisite']))
    selection = json.loads(base.committed(ROOT / value['prior_selection']))
    if (proof['passed'] is not True or proof['source_commit'] != value['feature_source_commit']
            or proof['parent'] != value['parent'] or proof['prepared'] != value['base_prepared']
            or set(proof['arms']) != set(value['arms'])
            or any(proof['arms'][arm]['exact_updates'] != 8 for arm in value['arms'])
            or selection['prepared'] != value['base_prepared']
            or selection['selected'] != {arm: None for arm in value['arms']}
            or len(selection['candidates']) != 4
            or any(row['decision']['passed'] for row in selection['candidates'])):
        raise ValueError('Complete the numerical prerequisite and close the original comparison first')
    return {name: data.sha256(ROOT / value[name]) for name in ('feature_prerequisite', 'prior_selection')}


def preparation():
    value, original, prepared = plan()
    return {'format': FORMAT + '/prepared', 'plan': data.identity(value),
            'base_prepared': data.identity(prepared), 'prerequisites': prerequisites(value),
            'sources': {name: data.sha256(ROOT / name) for name in SOURCES},
            'schedule': schedule(value, prepared)}


def validate():
    value, original, prepared = plan()
    bound = json.loads(base.committed(ROOT / PREPARED))
    if bound != preparation():
        raise ValueError('Rehearsal differs from its committed source, prerequisites or exact batches')
    return value, original, prepared, bound


def schedule(value, prepared):
    original = prepared['schedule']
    if len(original) != 128 or value['epochs'] != 8 or value['recipe']['steps'] != 1024:
        raise ValueError('Rehearsal must repeat exactly the declared complete schedule')
    return [index for _ in range(value['epochs']) for index in range(len(original))]


def job(value, bound, arm):
    if arm not in value['arms']:
        raise ValueError('Unknown rehearsal arm')
    return data.identity({'format': FORMAT + '/job', 'plan': data.identity(value),
                          'prepared': data.identity(bound), 'arm': arm})


def selection(value, bound, candidates):
    if len(candidates) != 2 or {row['arm'] for row in candidates} != set(value['arms']):
        raise ValueError('Both completed arms are required before final selection')
    for row in candidates:
        if row['step'] != value['recipe']['steps'] or row['job'] != job(value, bound, row['arm']):
            raise ValueError('Only the declared terminal checkpoint may be selected')
    return {'format': FORMAT + '/selection', 'plan': data.identity(value), 'prepared': data.identity(bound),
        'candidates': candidates, 'selected': {row['arm']: (row if row['decision']['passed'] else None)
                                              for row in candidates}}
