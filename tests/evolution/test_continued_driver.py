"""Exercise the actual artifact/driver boundaries, using a disposable Git repository."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from neuroshard.evolution import continued, grounded_tasks as tasks, reference_data as data


def commit(root):
    subprocess.run(['git', '-C', str(root), 'add', '.'], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(root), '-c', 'user.name=Fixture', '-c',
                    'user.email=fixture@example.invalid', 'commit', '-qm', 'Fixture'],
                   check=True, capture_output=True)
    return subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    original = continued.repo_root()
    plan = continued.load()
    plan['status'] = 'plan-frozen'
    for name in (*continued.SOURCE_PATHS, plan['prior_exclusion']['inputs_path']):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / name, target)
    path = tmp_path / 'config/experiments/continued-learning.json'
    data.save(path, plan)
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True, capture_output=True)
    plan_commit = commit(tmp_path)
    monkeypatch.setattr(continued, 'PLAN_PATH', path)
    prior = json.loads((tmp_path / plan['prior_exclusion']['inputs_path']).read_bytes())
    replay = prior['roles']['train-a']['ids'][:3072]
    sources = {name: data.sha256(tmp_path / name) for name in continued.SOURCE_PATHS}
    prepared = {
        'format': continued.PREPARED_FORMAT,
        'plan': {k: v for k, v in plan.items() if k != 'status'},
        'plan_digest': data.sha256(path), 'plan_commit': plan_commit,
        'implementation_digest': data.identity(sources), 'sources': sources,
        'parent_checkpoint': plan['parent']['checkpoint'],
        'parent_state_root': plan['parent']['state_root'],
        'reference_checkpoint': plan['reference']['checkpoint'],
        'tokenizer': plan['tokenizer'], 'config_sha256': plan['config_sha256'],
        'runtime': plan['runtime'], 'trained_replay_ids': replay[:1536],
        'conversation_replay_ids': replay[1536:], 'roles': {},
        'schedule': [{'role': 'train', 'indices': list(range(i*32, (i+1)*32)) +
                     list(range(3072+i*32, 3072+(i+1)*32))} for i in range(96)],
    }
    for role, count in {'train': 3072, 'dev-new': 64, 'dev-prior': 64, 'dev-retention': 64,
                        'test-new': 256, 'test-prior': 128, 'retention': 128}.items():
        ids = [hashlib.sha256(f'fixture:{role}:{i}'.encode()).hexdigest() for i in range(count)]
        if role == 'train':
            ids += replay
        prepared['roles'][role] = {'file': role+'.jsonl', 'sha256': 'f'*64,
                                   'count': len(ids), 'ids': ids}
    continued.validate_prepared(prepared, plan)
    data.save(continued.prepared_path(plan), prepared)
    plan['status'] = 'prepared-committed'
    data.save(path, plan)
    commit(tmp_path)
    return plan, prepared, tmp_path


def selection_fixture(plan, prepared, step=224):
    checkpoint = {'job': continued.job_identity(prepared, plan['runtime']), 'step': step,
                  'boundaries': plan['parent']['boundaries'], 'transition': None}
    losses = [{'id': name, 'loss': 1.0} for name in prepared['roles']['dev-retention']['ids']]
    selection = {'format': continued.SELECTION_FORMAT, 'prepared': data.identity(prepared),
                 'plan_digest': prepared['plan_digest'],
                 'implementation_digest': prepared['implementation_digest'],
                 'baseline': plan['parent']['checkpoint'], 'candidate': data.identity(checkpoint),
                 'step': 224, 'frozen_before_final_evaluation': True,
                 'development': continued.development_decision(
                     plan, prepared, data.identity(checkpoint), losses, losses)}
    return selection, checkpoint


def test_prepared_allows_only_declared_trained_replay(frozen):
    plan, prepared, _ = frozen
    assert continued.committed_prepared(plan, prepared)
    bad = copy.deepcopy(prepared)
    prior = json.loads((continued.repo_root() / plan['prior_exclusion']['inputs_path']).read_bytes())
    bad['roles']['train']['ids'][0] = prior['roles']['test-a']['ids'][0]
    with pytest.raises(ValueError, match='Only the declared trained replay'):
        continued.validate_prepared(bad, plan)
    bad = copy.deepcopy(prepared)
    bad['roles']['test-new']['ids'][0] = prior['roles']['test-b']['ids'][0]
    with pytest.raises(ValueError, match='Fresh evaluation overlaps'):
        continued.validate_prepared(bad, plan)


def test_substituted_inputs_and_modified_source_are_rejected(frozen):
    plan, prepared, root = frozen
    bad = copy.deepcopy(prepared)
    bad['roles']['train']['sha256'] = 'a'*64
    with pytest.raises(ValueError, match='Supplied prepared artifact'):
        continued.committed_prepared(plan, bad)
    source = root / continued.SOURCE_PATHS[0]
    source.write_text(source.read_text() + '\n# changed after freeze\n')
    with pytest.raises(ValueError, match='implementation changed'):
        continued.committed_prepared(plan, prepared)


def test_schedule_cannot_duplicate_or_skip_trained_windows(frozen):
    plan, prepared, _ = frozen
    bad = copy.deepcopy(prepared)
    bad['schedule'][1] = copy.deepcopy(bad['schedule'][0])
    with pytest.raises(ValueError, match='exactly once'):
        continued.validate_prepared(bad, plan)


def test_selection_checks_actual_endpoint_and_evaluator(frozen):
    plan, prepared, _ = frozen
    selection, checkpoint = selection_fixture(plan, prepared)
    continued.validate_selection(selection, plan, prepared, checkpoint)
    bad, early = selection_fixture(plan, prepared, 160)
    with pytest.raises(ValueError, match='actual fixed-size final checkpoint'):
        continued.validate_selection(bad, plan, prepared, early)
    bad = copy.deepcopy(selection)
    bad['implementation_digest'] = 'a'*64
    with pytest.raises(ValueError, match='another plan or evaluator'):
        continued.validate_selection(bad, plan, prepared, checkpoint)


def test_selection_must_be_the_committed_file(frozen):
    plan, prepared, root = frozen
    selection, checkpoint = selection_fixture(plan, prepared)
    data.save(continued.selection_path(plan), selection)
    plan['status'] = 'selection-committed'
    data.save(continued.PLAN_PATH, plan)
    commit(root)
    assert continued.final_evaluation_allowed(plan)
    assert continued.committed_selection(plan, selection)
    alternative, _ = selection_fixture(plan, prepared, 192)
    with pytest.raises(ValueError, match='Supplied selection differs'):
        continued.committed_selection(plan, alternative)
    plan['status'] = 'learning-running'
    assert not continued.training_allowed(plan)


def test_resume_refuses_unprepared_plan_before_loading_cuda(tmp_path, monkeypatch):
    from neuroshard.evolution.sharded import continued_job
    plan = continued.load()
    plan['status'] = 'plan-frozen'
    paths = {}
    for name, value in {'plan': plan, 'prepared': {}, 'parent': {}, 'resume': {'step':160}}.items():
        paths[name] = tmp_path / (name+'.json')
        data.save(paths[name], value)
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid resume reached tokenizer/CUDA initialization')
    monkeypatch.setattr(continued_job.AutoTokenizer, 'from_pretrained', forbidden)
    monkeypatch.setattr(continued_job.reference, 'configure', forbidden)
    argv = ['train', '--seed', str(tmp_path), '--home', str(tmp_path/'run')]
    for name, path in paths.items():
        argv += ['--'+name, str(path)]
    with pytest.raises(ValueError, match='Git-committed prepared freeze'):
        continued_job.main(argv, tmp_path)
    assert not (tmp_path/'run').exists()


def test_content_exclusion_ignores_row_ids_and_hidden_task_fields():
    index = continued.ContentExclusion()
    case = tasks.make_case(1, 'test', 2)
    messages = [{'role':'user', 'content':tasks.prompt(case)}, {'role':'assistant', 'content':'{}'}]
    assert index.add(messages)
    other = copy.deepcopy(case)
    other['rows'][0]['city'] = 'invisible change'
    assert tasks.task_identity(other) != tasks.task_identity(case)
    assert tasks.prompt(other) == tasks.prompt(case)
    assert not index.add([{'role':'user', 'content':tasks.prompt(other)},
                          {'role':'assistant', 'content':'different answer'}])


def test_development_abort_cannot_be_promoted_by_a_selection(frozen):
    plan, prepared, _ = frozen
    selection, checkpoint = selection_fixture(plan, prepared)
    before = selection['development']['before']
    after = [{**row, 'loss':row['loss']+.06} for row in before]
    selection['development'] = continued.development_decision(
        plan, prepared, data.identity(checkpoint), before, after)
    assert not selection['development']['passed']
    with pytest.raises(ValueError, match='failed its frozen development'):
        continued.validate_selection(selection, plan, prepared, checkpoint)
