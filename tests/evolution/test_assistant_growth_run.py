import importlib.util
import subprocess
import sys

import pytest

from neuroshard.evolution import assistant_growth_run as growth
from neuroshard.evolution import assistant_schedule_data as schedule
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256

from test_assistant_calendar import schedule_texts
from test_assistant_experience import SPEC, TEMPLATE, tiny_model
from test_assistant_workflow import envelope, reference_texts, reply
from test_granite_tokenizer import granite_like, load_tiny

torch = pytest.importorskip('torch')
PLAN = read(ROOT / growth.PLAN)
SCHEDULING = [schedule.make_case('train', 'slot', 0), schedule.make_case('train', 'invite', 1),
              schedule.make_case('cross-train', 'handoff', 0)]
DRAFTING = [data.make_case('train', 'copy', 0), data.make_case('train', 'recipient', 1)]
STUBBORN = SCHEDULING[0]['id']


class ScriptedBatcher:
    """Goal-directed stand-in for sampling; one scheduling case succeeds only with the coaching card."""
    card = PLAN['collection']['coaching']['card']

    def __init__(self, model, tokenizer, generation, **options):
        self.batches = [1]

    def close(self):
        pass

    def respond(self, messages, tools):
        first = next(m['content'] for m in messages if m['role'] == 'user')
        case = next((c for c in SCHEDULING + DRAFTING if c['turns'][0]['user'] == first), None)
        if case is None:
            return reply('Okay.')
        texts = schedule_texts(case) if case['id'].startswith('schedule-') else reference_texts(case)
        if case['id'] == STUBBORN and self.card not in messages[0]['content']:
            goal = case['turns'][0]['expected']['meeting']
            texts[-2] = envelope('save_meeting', {**goal, 'start_time': '16:30', 'duration_minutes': 30})
        done = sum(m['role'] == 'assistant' for m in messages)
        return reply(texts[done] if done < len(texts) else 'Done.')


@pytest.fixture
def setup(tmp_path, monkeypatch):
    directory = granite_like(tmp_path / 'granite')
    (directory / 'chat_template.jinja').write_text(TEMPLATE)
    tokenizer = load_tiny(directory)[0]
    monkeypatch.setattr(growth.rollout, 'Batcher', ScriptedBatcher)
    monkeypatch.setattr(growth, 'scheduling_cases', lambda plan, split: [
        c for c in SCHEDULING if c['split'] == split] if split in schedule.TRAINING else [])
    monkeypatch.setattr(growth, 'drafting_cases', lambda learning, split: DRAFTING)
    learning = {**read(ROOT / 'config/experiments/assistant-experience-learning.json'), 'training': SPEC}
    plan = {**PLAN, 'training': {**PLAN['training'], 'steps': 2}}
    return tokenizer, plan, learning, tmp_path


def test_stage1_inventory_pins_contracts_sources_runtime_and_a_bounded_gpu_host():
    execution = read(ROOT / growth.EXECUTION)
    for name, digest in execution['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert set(execution['contracts']) <= set(execution['sources'])
    probe = ('import os, sys; import neuroshard.evolution.assistant_growth_run; assert "torch" not in sys.modules; '
             'import neuroshard.evolution.assistant_experience_run, neuroshard.evolution.assistant_experience_train, '
             'neuroshard.evolution.assistant_selector, neuroshard.evolution.assistant_calendar; '
             'root = os.path.abspath("src"); print("\\n".join(sorted(os.path.relpath(x.__file__) '
             'for x in list(sys.modules.values()) if getattr(x, "__file__", None) and '
             'os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(execution['sources'])
    a2 = read(ROOT / 'config/experiments/assistant-experience-execution.json')
    assert all(execution[k] == a2[k] for k in ('python', 'gpus', 'environment', 'packages', 'tokenizer_pipeline_sha256'))
    assert execution['drafting_collection']['files'] == a2['collection']['files']
    spec = importlib.util.spec_from_file_location('growth_cloud', ROOT / 'scripts/modular_reference_cloud.py')
    cloud = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cloud)
    resources = cloud.resources(growth.PROFILE)
    candidates = resources['candidates']
    assert resources['gpu'] and resources['planning_cap_usd'] <= PLAN['budget']['ceiling_usd']
    assert resources['hours'] * max(c['price']['usd_per_hour'] for c in candidates) + 3 <= resources['planning_cap_usd']
    assert {c['gpu_name'] for c in candidates} == set(execution['gpus'])
    assert execution['worker_seconds'] + resources['setup_seconds'] + 600 <= resources['hours'] * 3600
    assert cloud.remote_command(growth.PROFILE)[1].endswith(growth.SCRIPT)
    assert cloud.UPLOAD_PROFILES[growth.PROFILE] == growth.UPLOADED
    assert set(resources['upload']['files']) == set(execution['drafting_collection']['files']) | {
        'update-checkpoint/manifest.json', 'update-checkpoint/trainable.safetensors'}
    assert PLAN['training_execution_authorized'] and PLAN['gpu_launch_authorized'] and not PLAN['native_promotion_authorized']


def test_stage1_reads_only_training_and_integration_goals():
    for split in ('development', 'confirmation', 'cross-development', 'cross-confirmation'):
        with pytest.raises(ValueError, match='may not access'):
            growth.scheduling_cases(PLAN, split)
    assert len(growth.scheduling_cases(PLAN, 'train')) == 256 and len(growth.scheduling_cases(PLAN, 'cross-integration')) == 8
    spec = growth.stage_spec(PLAN, read(ROOT / 'config/experiments/assistant-experience-learning.json'))
    assert spec['mixture'] == {'experience': 4, 'drafting': 2, 'replay': 2} and spec['steps'] == 128
    assert spec['learning_rates'] == {'update': 1e-05, 'addition': 0.0003} and spec['layers'] == list(range(32, 40))


def test_collection_coaches_where_the_sampler_never_succeeds_and_keeps_only_scheduling_experience(setup):
    tokenizer, plan, learning, tmp_path = setup
    policies = growth.contracts(plan)[2]
    home = tmp_path / 'home'
    home.mkdir()
    rows, report = growth.collect(tiny_model(tokenizer), tokenizer, plan, learning, policies,
                                  {'device': 'cpu', 'max_batch': 4, 'workers': 2}, home)
    assert report['coached_cases'] == [STUBBORN]
    assert report['drafting_reference'] == {'cases': 2, 'natural_successes': 4}
    assert report['near_policy_ceiling'] is not None and rows
    from neuroshard.evolution import assistant_experience_run as run

    chosen = run.read_rows(home / 'trajectories.jsonl.gz')
    assert {t['case_id'] for t in chosen} <= {c['id'] for c in SCHEDULING}
    assert any(t['coached'] for t in chosen) or STUBBORN not in {t['case_id'] for t in chosen}
    assert all(t['messages'][0]['content'] == policies['scheduling']['system_instruction'] for t in chosen)


def test_the_mixture_fills_every_update_with_its_declared_counts_and_leaves_the_a2_schedule_alone():
    from neuroshard.evolution import assistant_experience_train as trainer

    spec = {**SPEC, 'steps': 5, 'gradient_accumulation': 8, 'mixture': {'experience': 4, 'drafting': 2, 'replay': 2}}
    sources = {'experience': [0] * 3, 'drafting': [0] * 5, 'replay': [0] * 2}
    updates = trainer.mixture_schedule(sources, spec)
    assert len(updates) == 5
    assert all(sorted(kind for kind, _ in batch) == ['drafting'] * 2 + ['experience'] * 4 + ['replay'] * 2
               for batch in updates)
    assert all(index < len(sources[kind]) for batch in updates for kind, index in batch)
    assert updates == trainer.mixture_schedule(sources, spec)
    with pytest.raises(ValueError, match='mixture'):
        trainer.mixture_schedule({**sources, 'drafting': []}, spec)
    with pytest.raises(ValueError, match='mixture'):
        trainer.mixture_schedule(sources, {**spec, 'gradient_accumulation': 6})
    plain = {k: v for k, v in spec.items() if k != 'mixture'}
    assert trainer.schedule([0] * 3, [0] * 2, plain) == trainer.schedule([0] * 3, [0] * 2, plain)


def test_both_arms_train_from_the_accepted_update_on_the_declared_mixture(setup, monkeypatch):
    from neuroshard.evolution import assistant_experience_train as trainer

    tokenizer, plan, learning, tmp_path = setup
    accepted_dir = tmp_path / 'accepted'
    model = tiny_model(tokenizer)
    trainable = trainer.prepare(model, 'update', SPEC)
    manifest = trainer.checkpoint(accepted_dir / 'update-checkpoint', trainable,
                                  {'arm': 'update', 'optimizer_state': {}, 'trainable_parameters': 1, 'steps': 0,
                                   'schedule_sha256': '', 'losses': [0.0]}, {})
    monkeypatch.setattr(growth, 'UPLOADED', str(accepted_dir))

    def sequence(text):
        item = {'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': text}],
                'trainable': [False, True]}
        return trainer.encode(tokenizer, item, [])

    rows = {'scheduling': [sequence('a'), sequence('b')], 'drafting': [sequence('c')], 'replay': [sequence('d')]}
    home = tmp_path / 'home'
    home.mkdir()
    manifests = growth.train(lambda: tiny_model(tokenizer), plan, learning, rows, {'device': 'cpu'}, home,
                             manifest['trainable_sha256'])
    assert set(manifests) == {'U2', 'L2'} and manifests['U2']['steps'] == manifests['L2']['steps'] == 2
    assert read(home / 'update-checkpoint' / 'manifest.json')['arm'] == 'update'
    assert read(home / 'module-checkpoint' / 'manifest.json')['arm'] == 'addition'
    loaders = growth.unit_loaders(lambda: tiny_model(tokenizer), growth.stage_spec(plan, learning), home,
                                  manifest['trainable_sha256'])
    assert all(loader() is not None for loader in loaders.values())
    with pytest.raises(ValueError, match='pinned digest'):
        growth.train(lambda: tiny_model(tokenizer), plan, learning, rows, {'device': 'cpu'}, tmp_path / 'other', 'x' * 64)


def test_integration_fits_each_versions_turn_selector_from_single_route_outcomes(setup, monkeypatch):
    tokenizer, plan, learning, tmp_path = setup
    policies = growth.contracts(plan)[2]
    monkeypatch.setattr(growth, 'integration_cases', lambda plan, learning: SCHEDULING + DRAFTING)
    home = tmp_path / 'home'
    home.mkdir()
    loaders = {unit: (lambda: tiny_model(tokenizer)) for unit in ('U1', 'U2', 'L2')}
    gates, outcomes = growth.integrate(lambda: tiny_model(tokenizer), loaders, tokenizer, plan, learning, policies,
                                       {'device': 'cpu', 'max_batch': 4, 'workers': 2}, home)
    assert set(gates) == set(growth.VERSIONS) and set(outcomes) == {f'{u}-{r}' for u, r in growth.ROUTE_RUNS}
    handoff = SCHEDULING[2]['id']
    # The drafting route cannot book, so the handoff's scheduling follow-up succeeds only on the scheduling route.
    assert not any(run[1] for run in outcomes['U1-drafting'][handoff] if len(run) > 1)
    assert all(run == [True, True] for run in outcomes['U2-scheduling'][handoff])
    saved = read(home / 'integration.json')
    assert saved['gates'] == gates and len(saved['features_sha256']) == 64
    assert all(gate['rule'] in ('logistic', 'constant-parent', 'constant-arm') for gate in gates.values())
