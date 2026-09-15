"""A frozen, post-training composition test of an added transformer expert."""
import json
import math
from pathlib import Path

from . import incremental_capacity as base, reference_data as data

ROOT = base.ROOT
PLAN = ROOT / 'config/experiments/branch-growth.json'
PREPARED = ROOT / 'config/experiments/branch-growth-prepared.json'
SELECTION = ROOT / 'config/experiments/branch-growth-selection.json'
SOURCES = sorted(set(base.SOURCES) | {'src/neuroshard/evolution/branch_experiment.py',
    'src/neuroshard/evolution/sharded/branch.py', 'src/neuroshard/evolution/sharded/branch_job.py',
    'scripts/run_branch_growth.py'})


def graph(plan, parent, expert):
    if data.identity(parent) != plan['parent'] or data.identity(expert) != plan['expert']:
        raise ValueError('Wrong fixed model or learned expert')
    added = {name: spec for name, spec in expert['tensors'].items()
             if name.startswith('model.layers.') and int(name.split('.')[2]) >= plan['split']}
    return {'format': plan['format'] + '/graph', 'parent': plan['parent'], 'expert': plan['expert'],
        'split': plan['split'], 'parent_layout': plan['parent_layout'], 'expert_layout': plan['expert_layout'],
        'selector': plan['selector'], 'parent_parameters': sum(math.prod(spec['shape']) for spec in parent['tensors'].values()),
        'added_parameters': sum(math.prod(spec['shape']) for spec in added.values()), 'added_tensors': added}


def validate():
    plan = json.loads(base.committed(PLAN))
    prepared = json.loads(base.committed(PREPARED))
    if (plan['format'] != 'neuroshard-branch-growth-v1' or plan['split'] != 22
            or plan['parent_layout'] != [0, 6, 15, 24] or plan['expert_layout'] != [0, 6, 15, 22, 24]
            or prepared['plan'] != data.identity(plan)
            or prepared['sources'] != {name: data.sha256(ROOT / name) for name in SOURCES}):
        raise ValueError('Branch execution differs from its frozen preparation')
    if json.loads(base.committed(PLAN, prepared['source_commit'])) != plan:
        raise ValueError('Freeze the graph, questions and gates before evaluation')
    for name in SOURCES:
        base.committed(ROOT / name)
        base.committed(ROOT / name, prepared['source_commit'])
    return plan, prepared


def rows(prepared, home, role):
    spec = prepared['roles'][role]
    result = data.read_records(Path(home) / (role + '.jsonl'), spec['sha256'])
    if len(result) != spec['count'] or data.identity([row['id'] for row in result]) != spec['ids']:
        raise ValueError('Wrong fixed ordered questions or retention probes')
    return result
