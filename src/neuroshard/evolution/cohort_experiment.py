"""Frozen preparation and gates for a second, separately trained expert."""
import json
from pathlib import Path

from . import branch_experiment, cohort_questions, incremental_capacity as base, reference_data as data
from .sharded.branch_groups import OrderedRoutes

ROOT = base.ROOT
PLAN = ROOT / 'config/experiments/branch-cohort-learning.json'
PREPARED = ROOT / 'config/experiments/branch-cohort-learning-prepared.json'
SELECTION = ROOT / 'config/experiments/branch-cohort-learning-selection.json'
FORMAT = 'neuroshard-branch-cohort-learning-v1'
SOURCES = sorted(set(branch_experiment.SOURCES) | {
    'scripts/run_branch_cohort.py', 'src/neuroshard/evolution/cohort_experiment.py',
    'src/neuroshard/evolution/cohort_questions.py',
    *(f'src/neuroshard/evolution/sharded/{name}.py' for name in
      ('branch_groups', 'cohort_features', 'cohort_state', 'cohort_job', 'features', 'feature_bank', 'feature_probe'))})


def validate():
    plan = json.loads(base.committed(PLAN))
    prepared = json.loads(base.committed(PREPARED))
    if (plan['format'] != FORMAT or plan['split'] != 22 or plan['parent_layout'] != [0, 6, 15, 24]
            or plan['expert_layout'] != [0, 6, 15, 22, 24] or plan['training']['steps'] != 560
            or plan['batch_documents'] != 32 or plan['microbatch'] != 8
            or plan['checkpoints'] != [0, 280, 560] or prepared['plan'] != data.identity(plan)
            or prepared['sources'] != {name: data.sha256(ROOT / name) for name in SOURCES}
            or prepared['schedule'] != list(range(28)) * 20):
        raise ValueError('Cohort differs from its complete frozen preparation')
    routes = OrderedRoutes(plan['rules'])
    routes.require_extension_of(OrderedRoutes([{'id': 'directory', 'needle': 'fictional luma directory', 'owner': 3}]))
    if list(routes.rules) != [{'id': 'directory', 'needle': 'fictional luma directory', 'owner': 3},
                        {'id': 'protocol', 'needle': 'neuroshard 0.4.0', 'owner': 4}]:
        raise ValueError('Require exactly the original expert and one new scoped expert')
    if json.loads(base.committed(PLAN, prepared['source_commit'])) != plan:
        raise ValueError('Freeze the complete recipe before preparing model inputs')
    for name in SOURCES:
        if base.committed(ROOT / name) != base.committed(ROOT / name, prepared['source_commit']):
            raise ValueError('Cohort numerical source differs from its prepared commit')
    previous = prepared['prerequisite']
    if (previous['graph'] != plan['previous_graph'] or not previous['passed']
            or not previous['artifacts_preserved'] or len(previous['result_sha256']) != 64):
        raise ValueError('Require the actual passing and preserved first-cohort result')
    return plan, prepared


def job(plan, prepared):
    return data.identity({'format': FORMAT, 'plan': data.identity(plan), 'prepared': data.identity(prepared)})


def graph(plan, parent, first, second):
    from .sharded import incremental_state
    if data.identity(parent) != plan['parent'] or data.identity(first) != plan['expert']:
        raise ValueError('The established model or first expert changed')
    for expert in (first, second):
        incremental_state.validate(expert, parent)
        if expert['mode'] != 'tail-control' or expert['frozen_layers'] != plan['split']:
            raise ValueError('Each separate expert must preserve the same prefix')
    return {'format': FORMAT + '/graph', 'parent': plan['parent'], 'split': plan['split'],
            'experts': [{'id': 'directory', 'checkpoint': data.identity(first)},
                        {'id': 'protocol', 'checkpoint': data.identity(second)}],
            'rules': plan['rules'], 'parent_layout': plan['parent_layout'],
            'expert_layout': plan['expert_layout'], 'tokenizer': plan['tokenizer']}


def rows(prepared, home, role, tokenizer=None, max_length=256):
    spec = prepared['roles'][role]
    values = data.read_records(Path(home) / (role + '.jsonl'), spec['sha256'])
    if len(values) != spec['count'] or data.identity([row['id'] for row in values]) != spec['ids']:
        raise ValueError('Require the fixed complete ordered cohort')
    if role in ('train', 'dev', 'test'):
        cohort_questions.validate_rows(values, tokenizer, max_length)
    if role == 'train':
        batches = prepared['batches']
        if (len(batches) != 28 or any(len(batch) != 32 for batch in batches)
                or sorted(index for batch in batches for index in batch) != list(range(len(values)))):
            raise ValueError('Feature production must cover every training example once')
    return values


def final_selection(plan, prepared):
    selection = json.loads(base.committed(SELECTION))
    if (selection['format'] != FORMAT + '/selection' or not selection['eligible']
            or selection['plan'] != data.identity(plan) or selection['prepared'] != data.identity(prepared)
            or selection['step'] != plan['training']['steps']):
        raise ValueError('Only the committed eligible terminal expert may open finals')
    return selection
