import copy
import hashlib
import json
import subprocess

import pytest

from neuroshard.evolution.milestone import (
    PLAN_PATH, decide_learning, load, selection_path, training_allowed, validate,
)
from neuroshard.evolution import milestone
from neuroshard.dataflow.store import canonical

PLAN = json.loads(PLAN_PATH.read_text())
PLAN['status'] = 'plan-frozen'


@pytest.fixture
def repository(tmp_path, monkeypatch):
    path = tmp_path/'config/experiments/learning-milestone.json'
    path.parent.mkdir(parents=True)
    path.write_bytes(canonical(PLAN))
    monkeypatch.setattr(milestone, 'PLAN_PATH', path)
    monkeypatch.setattr(milestone, 'implementation_digest', lambda: 'a'*64)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args], stderr=subprocess.DEVNULL).decode().strip()
    git('init', '-q')
    git('config', 'user.name', 'Test')
    git('config', 'user.email', 'test@example.invalid')
    git('add', '.')
    git('commit', '-qm', 'Freeze plan')
    return path, git


def selection_fixture(repository):
    path, git = repository
    def hashed(value):
        return hashlib.sha256(canonical(value)).hexdigest()
    sources = []
    for name, role in (('heldout', 'heldout'), ('train', 'train')):
        spec = {k: PLAN['data'][k] for k in ('repo', 'revision', 'license')}
        spec.update(split=PLAN['learning'][name+'_split'], role=role)
        start = PLAN['learning'][name+'_start']
        sources.append(dict(source=hashed(spec), spec=spec, start=start,
                            end=start+PLAN['learning'][name+'_scan_limit']))
    documents = {}
    for role, count in (('retention', 64), ('fresh', 64), ('test', 64), ('train', 256)):
        role_index = ('retention', 'fresh', 'test', 'train').index(role)
        source = sources[int(role == 'train')]
        documents[role] = [dict(id=f'{3*(i+1)+role_index if role != "train" else 10000+i:064x}',
                                source=source['source'], row=source['start']+role_index*64+i,
                                object=hashed([role, i, 'object']), windows=[hashed([role, i, 'window'])],
                                omitted_targets=0) for i in range(count)]
    windows = [d['windows'][0] for d in documents['train']]
    return dict(format='neuroshard-learning-milestone-selection-v1',
                baseline='b'*64, imported_model_root=PLAN['seed']['imported_model_root'], tokenizer_root='c'*64,
                plan_commit=git('rev-parse', 'HEAD'), plan_digest=hashlib.sha256(path.read_bytes()).hexdigest(),
                implementation_digest='a'*64, source_windows=sources, documents=documents,
                training_batches=[windows[i:i+2] for i in range(0, 256, 2)],
                generation_prompts=[dict(document=d['id'], prompt='A public prompt')
                                    for d in sorted(documents['test'], key=lambda d: d['id'])[:20]])


def seal(repository, selection):
    path, git = repository
    milestone.selection_path(PLAN).write_bytes(canonical(selection))
    git('add', '.')
    git('commit', '-qm', 'Commit sealed selection')
    plan = copy.deepcopy(PLAN)
    plan['status'] = 'selection-committed'
    path.write_bytes(canonical(plan))
    git('add', '.')
    git('commit', '-qm', 'Record selection commitment')
    return plan


def test_frozen_plan_loads_and_forbids_training(repository):
    plan = load()
    assert plan['status'] == 'plan-frozen'
    assert plan['seed']['growth_layers'] == 0
    assert plan['optimizer']['recipes'] == 1
    assert training_allowed(plan) is False
    assert not selection_path(plan).is_file()


def test_quality_margins_match_existing_gate():
    from neuroshard.evolution.evaluation import decide
    evaluation = PLAN['evaluation']
    baseline = [1.2] * 64
    better = [1.19] * 64
    worse = [1.3] * 64
    reference = decide(baseline, better, baseline, better,
                       evaluation['retention_margin'], evaluation['fresh_min_gain'])
    assert reference['retention']['margin'] == evaluation['retention_margin']
    assert reference['fresh']['margin'] == -evaluation['fresh_min_gain']
    rejected = decide(baseline, worse, baseline, worse,
                      evaluation['retention_margin'], evaluation['fresh_min_gain'])
    assert rejected['promote'] is False


def test_fresh_cannot_rescue_or_veto_the_sealed_set():
    good = [1.2] * 64
    improved = [1.1] * 64
    worse = [1.3] * 64
    rescued = decide_learning({
        'baseline': {'test': good, 'retention': good, 'fresh': good},
        'candidate': {'test': [1.1999] * 64, 'retention': improved, 'fresh': improved},
    }, PLAN)
    assert rescued['fresh']['passes'] is True
    assert rescued['pass'] is False
    vetoed = decide_learning({
        'baseline': {'test': good, 'retention': good, 'fresh': good},
        'candidate': {'test': improved, 'retention': improved, 'fresh': worse},
    }, PLAN)
    assert vetoed['fresh']['passes'] is False
    assert vetoed['pass'] is True


def test_stop_rules_reject_growth_and_margin_changes():
    plan = copy.deepcopy(PLAN)
    plan['seed']['growth_layers'] = 4
    with pytest.raises(ValueError, match='no growth'):
        validate(plan)
    plan = copy.deepcopy(PLAN)
    plan['evaluation']['fresh_min_gain'] = 0.0001
    with pytest.raises(ValueError, match='cannot be relaxed'):
        validate(plan)
    plan = copy.deepcopy(PLAN)
    plan['continual']['blocked_on'] = 'optional'
    with pytest.raises(ValueError, match='only after a learning pass'):
        validate(plan)
    plan = copy.deepcopy(PLAN)
    plan['open_source']['secret_evaluation'] = True
    with pytest.raises(ValueError, match='secret evaluation'):
        validate(plan)


def test_training_status_without_selection_is_invalid(repository):
    plan = copy.deepcopy(PLAN)
    plan['status'] = 'learning-running'
    with pytest.raises(ValueError, match='committed sealed-set'):
        validate(plan)


def test_selection_must_be_committed_not_merely_present_or_staged(repository):
    path, git = repository
    selection = selection_fixture(repository)
    milestone.selection_path(PLAN).write_bytes(canonical(selection))
    with pytest.raises(ValueError, match='committed to Git'):
        milestone.committed_selection(PLAN)
    git('add', '.')
    with pytest.raises(ValueError, match='committed to Git'):
        milestone.committed_selection(PLAN)
    plan = seal(repository, selection)
    assert training_allowed(plan)
    milestone.selection_path(plan).write_bytes(canonical(selection)+b'\n')
    git('add', '.')
    with pytest.raises(ValueError, match='unchanged Git-committed'):
        training_allowed(plan)


@pytest.mark.parametrize('field', ['plan_digest', 'implementation_digest'])
def test_committing_a_false_plan_or_implementation_binding_does_not_authorize_training(repository, field):
    selection = selection_fixture(repository)
    selection[field] = 'd'*64
    plan = seal(repository, selection)
    with pytest.raises(ValueError, match='frozen plan bytes|implementation changed'):
        training_allowed(plan)


def test_plan_changes_after_selection_cannot_hide_behind_a_valid_commit(repository):
    selection = selection_fixture(repository)
    plan = seal(repository, selection)
    plan['data']['revision'] = 'e'*40
    repository[0].write_bytes(canonical(plan))
    repository[1]('add', '.')
    repository[1]('commit', '-qm', 'Attempt source substitution')
    with pytest.raises(ValueError, match='source ranges|constants changed'):
        training_allowed(plan)


def test_sealed_test_windows_and_repeated_work_cannot_enter_training(repository):
    selection = selection_fixture(repository)
    milestone.validate_selection(selection, PLAN)
    altered = copy.deepcopy(selection)
    altered['training_batches'][0][0] = altered['documents']['test'][0]['windows'][0]
    with pytest.raises(ValueError, match='training pool'):
        milestone.validate_selection(altered, PLAN)
    altered = copy.deepcopy(selection)
    altered['training_batches'][0][0] = altered['training_batches'][1][0]
    with pytest.raises(ValueError, match='repeat'):
        milestone.validate_selection(altered, PLAN)
    altered = copy.deepcopy(selection)
    altered['documents']['test'][0]['row'] = PLAN['learning']['heldout_start']-1
    with pytest.raises(ValueError, match='cursor window'):
        milestone.validate_selection(altered, PLAN)


def test_decision_requires_all_64_documents_and_finite_scores():
    scores = {side: {role: [1.]*64 for role in ('test', 'retention', 'fresh')}
              for side in ('baseline', 'candidate')}
    scores['candidate']['test'].pop()
    with pytest.raises(ValueError, match='every sealed document'):
        decide_learning(scores, PLAN)
    scores['candidate']['test'].append(float('nan'))
    with pytest.raises(ValueError):
        decide_learning(scores, PLAN)
