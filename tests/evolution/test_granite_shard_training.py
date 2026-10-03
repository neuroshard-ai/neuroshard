import copy
import importlib.util
import subprocess
import sys

import pytest

from neuroshard.evolution import granite_shard_training as training
from neuroshard.evolution.modular_reference_execution import ROOT, read, save, sha256

PLAN = read(ROOT / training.PLAN)


def cloud_module():
    spec = importlib.util.spec_from_file_location('training_cloud', ROOT / 'scripts/modular_reference_cloud.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sequences_and_spec_are_pinned_and_the_arm_lives_on_the_last_owner():
    experience, replay, pairs = training.sequences(PLAN)
    assert (len(experience), len(replay), len(pairs)) == (12, 4, 4)
    assert all(len(s['input_ids']) <= 2048 for s in experience + replay)
    assert all(any(label != -100 for label in s['labels']) for s in experience + replay)
    learning = read(ROOT / 'config/experiments/assistant-experience-learning.json')
    for key in ('layers', 'rank', 'alpha', 'betas', 'epsilon', 'gradient_clip', 'weight_decay'):
        assert PLAN['spec'][key] == learning['training'][key]
    assert PLAN['spec']['learning_rates'] == learning['goal_guided_repairs']['training']['learning_rates']
    begin, end = PLAN['boundaries'][-2:]
    assert all(begin <= layer < end for layer in PLAN['spec']['layers'])
    with pytest.raises(ValueError, match='sequences changed'):
        training.sequences({**PLAN, 'sequences': {**PLAN['sequences'], 'sha256': '0' * 64}})


def test_phase_jobs_resume_from_the_arm_checkpoint_and_saved_references(tmp_path):
    home, store = tmp_path / 'home', tmp_path / 'store'
    fresh = training.job(PLAN, 'train', 2, home, store)
    assert 'fail' not in fresh and 'start' not in fresh and fresh['checkpoints'].endswith('arm-train')
    assert training.job(PLAN, 'outage', 2, home, store)['fail'] == {'step': 3}
    (home / 'outage').mkdir(parents=True, exist_ok=True)
    save(home / 'outage' / 'references.json', [[-1.0, -2.0]] * 4)
    driver = training.job(PLAN, 'resume', 0, home, store)
    holder = training.job(PLAN, 'resume', 2, home, store)
    assert driver['start'] == holder['start'] == 3 and driver['references'] == [[-1.0, -2.0]] * 4
    assert holder['resume_from'] == str(store / 'arm-outage') and 'resume_from' not in driver


def test_the_second_attempt_warms_up_every_owner_and_the_reference(tmp_path):
    torch = pytest.importorskip('torch')
    from test_granite_partition import canonical, prompt, tiny_checkpoint

    assert PLAN['warm_up'] and PLAN['attempt'] == 2
    assert all(training.job(PLAN, phase, 0, tmp_path, tmp_path)['warm_up'] for phase in ('train', 'outage'))
    model = canonical(tiny_checkpoint(tmp_path / 'granite'))
    ids = prompt()
    with torch.inference_mode():
        before = model(ids).logits
    training.warm_up(model, lengths=(12, 1))
    with torch.inference_mode():
        assert torch.equal(model(ids).logits, before)


def passing_evidence():
    digests = {f'model.layers.{i}.self_attn.q_proj.lora_a': f'{i:064x}' for i in range(32, 40)}
    losses = [3.1, 3.0, 2.9, 2.8, 2.7, 2.6]
    margins = [0.0, 0.0, 0.1, 0.2, 0.3, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.8]
    control = {'completed': True, 'trainable_sha256': digests, 'peak_rss_bytes': 20 * 2 ** 30, 'seconds': 600,
               'receipt': {'losses': losses, 'preference_margins': margins}}
    owner = {'completed': True, 'peak_rss_bytes': 9 * 2 ** 30, 'seconds': 700, 'sent_bytes': 10}
    phases = {
        'train': [{**owner, 'receipt': {'losses': losses, 'preference_margins': margins}}, owner,
                  {**owner, 'trainable_sha256': digests, 'trainable_parameters': 1048576}],
        'outage': [{**owner, 'completed': False}, {**owner, 'completed': False}, None],
        'resume': [{**owner, 'receipt': {'losses': losses[3:], 'preference_margins': margins[6:]}}, owner,
                   {**owner, 'trainable_sha256': digests}],
    }
    fetches = [{'completed': True}] * 3
    return fetches, phases, control


def test_assessment_requires_bit_identical_tensors_losses_and_recovery():
    fetches, phases, control = passing_evidence()
    report = training.assess(PLAN, fetches, phases, control)
    assert report['passed'] and report['trainable_parameters'] == 1048576
    drift = copy.deepcopy(phases)
    drift['train'][2]['trainable_sha256'] = {**control['trainable_sha256'],
                                             'model.layers.32.self_attn.q_proj.lora_a': 'f' * 64}
    assert not training.assess(PLAN, fetches, drift, control)['checks']['tensors']
    loss = copy.deepcopy(phases)
    loss['train'][0]['receipt']['losses'][4] += 1e-12
    assert not training.assess(PLAN, fetches, loss, control)['checks']['losses']
    unrecovered = copy.deepcopy(phases)
    unrecovered['resume'][0]['receipt']['losses'] = [0.0, 0.0, 0.0]
    assert not training.assess(PLAN, fetches, unrecovered, control)['checks']['recovery']
    heavy = copy.deepcopy(phases)
    heavy['resume'][2]['peak_rss_bytes'] = control['peak_rss_bytes']
    assert not training.assess(PLAN, fetches, heavy, control)['checks']['memory']
    survived = copy.deepcopy(phases)
    survived['outage'][2] = {'completed': True}
    assert not training.assess(PLAN, fetches, survived, control)['checks']['outage_injected']
    assert not training.assess(PLAN, fetches, phases, {**control, 'completed': False})['passed']


def test_training_profile_is_bounded_and_the_freeze_covers_every_imported_source():
    for name, digest in PLAN['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert set(PLAN['contracts']) <= set(PLAN['sources'])
    cloud = cloud_module()
    resources = cloud.resources(training.PROFILE)
    assert resources['instance_type'] == 'r7i.4xlarge' and not resources['gpu'] and resources['purpose'] == training.PROFILE
    assert 4 * resources['planning_cap_usd'] <= 32
    owner_seconds = sum(PLAN['phase_seconds'][p] for p in training.PHASES)
    reference_seconds = sum(PLAN['phase_seconds'][p] for p in training.REFERENCE_PHASES)
    assert max(owner_seconds, reference_seconds) + resources['setup_seconds'] + resources['copy_seconds'] \
        <= resources['hours'] * 3600
    assert cloud.GRANITE_PROFILES[training.PROFILE][0] == 'granite_shard_training'
    probe = ('import os, sys; import neuroshard.evolution.granite_shard_training, '
             'neuroshard.evolution.sharded.granite_training, neuroshard.evolution.assistant_experience_train; '
             'root = os.path.abspath("src"); '
             'print("\\n".join(sorted(os.path.relpath(x.__file__) for x in list(sys.modules.values()) '
             'if getattr(x, "__file__", None) and os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(PLAN['sources']), set(imported) - set(PLAN['sources'])
    assert PLAN['training_authorized'] and not PLAN['gpu_launch_authorized'] and PLAN['attempts'] == 1


def test_importing_the_training_execution_does_not_load_torch():
    probe = 'import sys, neuroshard.evolution.granite_shard_training; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', probe], check=True, cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'src')})
