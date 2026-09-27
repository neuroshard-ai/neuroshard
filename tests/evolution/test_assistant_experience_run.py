import json

import pytest

from neuroshard.evolution import assistant_experience_run as run
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution.modular_reference_execution import ROOT, read

from test_assistant_experience import SPEC, TEMPLATE, tiny_model
from test_assistant_workflow import envelope, policy, reference_texts, reply
from test_granite_tokenizer import granite_like, load_tiny

torch = pytest.importorskip('torch')
CASES = [data.make_case('train', 'copy', 0), data.make_case('train', 'latest', 1), data.make_case('train', 'sum', 2)]
STUBBORN = CASES[1]['id']
EXECUTION = {'device': 'cpu', 'max_batch': 4, 'workers': 3, 'samples_per_case': 2, 'temperature': .8,
             'top_p': .95, 'seed': 5, 'integration_samples': 2, 'replay_seed': 27092029,
             'replay_max_input_tokens': 100000, 'replay_max_new_tokens': 4}


class ScriptedBatcher:
    """Goal-directed stand-in for sampling; one case succeeds only with the coaching card."""
    card = read(ROOT / run.PLAN)['coaching']['card']

    def __init__(self, model, tokenizer, generation, **options):
        self.batches = [1]

    def close(self):
        pass

    def respond(self, messages, tools):
        first = next(m['content'] for m in messages if m['role'] == 'user')
        case = next((c for c in CASES if c['turns'][0]['user'] == first), None)
        if case is None:
            return reply('Okay.')
        texts = reference_texts(case)
        if case['id'] == STUBBORN and self.card not in messages[0]['content']:
            texts[2] = envelope('save_draft', {**case['turns'][0]['expected'], 'total': 0})
        return reply(texts[sum(m['role'] == 'assistant' for m in messages)])


@pytest.fixture
def setup(tmp_path, monkeypatch):
    directory = granite_like(tmp_path / 'granite')
    (directory / 'chat_template.jinja').write_text(TEMPLATE)
    tokenizer = load_tiny(directory)[0]
    monkeypatch.setattr(run.rollout, 'Batcher', ScriptedBatcher)
    monkeypatch.setattr(run, 'split_cases', lambda plan, split: CASES)
    plan = {**read(ROOT / run.PLAN), 'training': {**SPEC, 'steps': 3}}
    return tokenizer, plan, tmp_path / 'home'


def test_gpu_inventory_pins_contracts_sources_and_packages_without_launch_authority():
    import importlib.util
    import subprocess
    import sys

    execution = read(ROOT / run.EXECUTION)
    for name, digest in execution['contracts'].items():
        assert run.sha256(ROOT / name) == digest
    assert set(execution['contracts']) <= set(execution['sources'])
    probe = ('import os, sys; import neuroshard.evolution.assistant_experience_run as m; '
             'assert "torch" not in sys.modules; '
             'import neuroshard.evolution.assistant_experience_train, neuroshard.evolution.assistant_selector; '
             'root = os.path.abspath("src"); '
             'print("\\n".join(sorted(os.path.relpath(x.__file__) for x in list(sys.modules.values()) '
             'if getattr(x, "__file__", None) and os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(execution['sources'])
    pinned = [line.split('==')[0] for line in (ROOT / 'docs/assistant-experience-requirements.txt').read_text().splitlines()
              if line and not line.startswith('#') and '==' in line]
    assert set(pinned) | {'torch'} == set(execution['packages'])
    assert execution['environment'] == execution['worker_environment']
    assert not execution['training_authorized'] and not execution['gpu_launch_authorized']
    spec = importlib.util.spec_from_file_location('experience_cloud', ROOT / 'scripts/modular_reference_cloud.py')
    cloud = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cloud)
    resources = cloud.resources(run.PROFILE)
    assert resources['gpu'] and resources['instance_type'] == 'g6e.xlarge' and resources['planning_cap_usd'] <= 15
    assert resources['hours'] * resources['price']['usd_per_hour'] + 3 <= resources['planning_cap_usd']
    assert cloud.remote_command(run.PROFILE)[1].endswith(run.SCRIPT)
    assert cloud.GRANITE_PROFILES[run.PROFILE] == ('assistant_experience_run', 'docs/assistant-experience-requirements.txt')


def test_accelerator_phases_never_open_evaluation_goals():
    plan = read(ROOT / run.PLAN)
    assert len(run.split_cases(plan, 'train')) == 256 and len(run.split_cases(plan, 'integration')) == 64
    for split in ('development', 'confirmation'):
        with pytest.raises(ValueError, match='may not access'):
            run.split_cases(plan, split)


def test_collection_coaches_only_unsolved_cases_then_trains_and_fits_gates(setup):
    tokenizer, plan, home = setup
    home.mkdir()
    model = tiny_model(tokenizer)
    rows, report = run.collect(model, tokenizer, plan, policy(), EXECUTION, home)
    assert report['coached_cases'] == [STUBBORN]
    summary = report['summary']
    assert summary['cases_without_experience'] == [] and summary['rollouts'] == 3 * 2 + 2
    assert summary['by_family']['latest'] == {'cases': 1, 'complete': 1, 'coached_only': 1}
    # Identical scripted texts deduplicate to one trajectory per case.
    assert len(rows) == 3 and report['near_policy_ceiling'] is not None
    replay_rows = run.record_replay(model, tokenizer, plan, EXECUTION, home)
    assert len(replay_rows) == 256 and read(home / 'replay.json')['terminated'] == 256
    manifests = run.train_arms(lambda: tiny_model(tokenizer), plan, EXECUTION, rows, replay_rows, home)
    assert set(manifests) == {'update', 'addition'}
    assert manifests['addition']['trainable_parameters'] < manifests['update']['trainable_parameters']
    assert manifests['addition']['schedule_sha256'] == manifests['update']['schedule_sha256']
    gates = run.integrate(lambda: tiny_model(tokenizer), tokenizer, plan, policy(), EXECUTION, home)
    # Scripted parent and arms both succeed except the stubborn case, so every rate ties.
    assert {arm: value['gate']['rule'] for arm, value in gates.items()} == {
        'update': 'constant-parent', 'addition': 'constant-parent'}
    integration = read(home / 'integration.json')
    assert integration['parent_outcomes'][STUBBORN] == [False] * 3
    assert json.dumps(integration).count('"gate"') == 2
