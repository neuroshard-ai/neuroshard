import json
import random
import shutil

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

from neuroshard.evolution import assistant_experience_train as trainer
from neuroshard.evolution.sharded import granite

from test_granite_partition import BOUNDARIES, canonical, tiny_checkpoint
from test_granite_pipeline import WORLD, free_port

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SPEC = {'layers': [3], 'rank': 4, 'alpha': 8, 'seed': 5, 'steps': 3, 'gradient_accumulation': 4,
        'experience_per_replay': 3, 'preference_per_update': 1, 'beta': 0.1, 'preference_weight': 1.0,
        'learning_rates': {'addition': 1e-2, 'update': 1e-3}, 'warmup_steps': 2, 'gradient_clip': 1.0,
        'weight_decay': 0.01, 'betas': [0.9, 0.999], 'epsilon': 1e-8}


def sequence(rng, length):
    ids = [rng.randrange(2, 96) for _ in range(length)]
    cut = length // 2
    return {'input_ids': ids, 'labels': [-100] * cut + ids[cut:], 'sha256': str(rng.random())}


def data(seed=11):
    rng = random.Random(seed)
    experience = [sequence(rng, rng.randrange(10, 18)) for _ in range(5)]
    replay = [sequence(rng, rng.randrange(10, 18)) for _ in range(2)]
    pairs = []
    for _ in range(2):
        prefix = [rng.randrange(2, 96) for _ in range(8)]
        chosen, rejected = [rng.randrange(2, 96) for _ in range(4)], [rng.randrange(2, 96) for _ in range(4)]
        pairs.append({'chosen': {'input_ids': prefix + chosen, 'labels': [-100] * 8 + chosen},
                      'rejected': {'input_ids': prefix + rejected, 'labels': [-100] * 8 + rejected}})
    return experience, replay, pairs


def single_host(checkpoint, arm, experience, replay, pairs):
    """The single-host trainer in a fresh process with the owners' numerical environment."""
    from safetensors.torch import load_file

    work = Path(checkpoint).parent / f'single-{arm}'
    work.mkdir()
    (work / 'job.json').write_text(json.dumps({'arm': arm, 'spec': SPEC, 'experience': experience,
                                               'replay': replay, 'pairs': pairs}))
    code = ('import json, sys, torch; from pathlib import Path; '
            'from neuroshard.evolution import assistant_experience_train as trainer; '
            'from transformers import AutoModelForCausalLM; from safetensors.torch import save_file; '
            'torch.set_num_threads(1); job = json.loads(Path(sys.argv[2]).read_text()); '
            'model = AutoModelForCausalLM.from_pretrained(sys.argv[1], dtype=torch.bfloat16, '
            'attn_implementation="eager", local_files_only=True).eval(); '
            'trainable, receipt = trainer.train(model, job["arm"], job["experience"], job["replay"], job["spec"], '
            'pairs=job["pairs"]); '
            'save_file({k: v.detach().contiguous() for k, v in trainable.items()}, sys.argv[3] + "/trainable.safetensors"); '
            'receipt.pop("optimizer_state"); Path(sys.argv[3], "receipt.json").write_text(json.dumps(receipt))')
    subprocess.run([sys.executable, '-c', code, str(checkpoint), str(work / 'job.json'), str(work)], check=True,
                   env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')})
    return load_file(work / 'trainable.safetensors'), json.loads((work / 'receipt.json').read_text())


def launch(config_dir, shards, jobs, home):
    home.mkdir(parents=True, exist_ok=True)
    port = free_port()
    code = ('import os, sys; from neuroshard.evolution.sharded import granite_training as t; '
            't.run_owner(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), "127.0.0.1", '
            'int(sys.argv[5]), sys.argv[6], sys.argv[7], timeout=15); os._exit(0)')
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    processes = []
    for rank in range(WORLD):
        job_path = home / f'job-{rank}.json'
        job_path.write_text(json.dumps(jobs[rank]))
        processes.append(subprocess.Popen([sys.executable, '-c', code, str(config_dir), str(shards), str(rank),
                                           str(WORLD), str(port), str(job_path), str(home / f'result-{rank}.json')],
                                          env=env))
    for process in processes:
        try:
            process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            process.kill()
    return [json.loads((home / f'result-{r}.json').read_text()) if (home / f'result-{r}.json').exists() else None
            for r in range(WORLD)]


@pytest.fixture
def deployed(tmp_path):
    checkpoint = tiny_checkpoint(tmp_path / 'granite')
    shards = tmp_path / 'shards'
    for rank in range(WORLD):
        granite.export(checkpoint, BOUNDARIES, rank, shards)
    config_dir = tmp_path / 'config'
    config_dir.mkdir()
    shutil.copy(checkpoint / 'config.json', config_dir / 'config.json')
    return checkpoint, config_dir, shards


def job(arm, experience, replay, pairs, checkpoints, **extra):
    return {'arm': arm, 'spec': SPEC, 'experience': experience, 'replay': replay, 'pairs': pairs,
            'max_tokens': 64, 'checkpoints': str(checkpoints), **extra}


def trained(directory):
    from safetensors.torch import load_file
    return load_file(Path(directory) / 'trainable.safetensors')


@pytest.mark.parametrize('arm', ['addition', 'update'])
def test_training_across_owners_reproduces_the_single_host_trainer(deployed, tmp_path, arm):
    checkpoint, config_dir, shards = deployed
    experience, replay, pairs = data()
    expected, receipt = single_host(checkpoint, arm, experience, replay, pairs)
    checkpoints = tmp_path / 'arm'
    results = launch(config_dir, shards, [job(arm, experience, replay, pairs, checkpoints)] * WORLD, tmp_path / 'run')
    assert all(r and r['completed'] for r in results), results
    assert results[0]['receipt']['schedule_sha256'] == receipt['schedule_sha256']
    assert results[0]['trainable_parameters'] == 0 and results[1]['trainable_parameters'] == 0
    assert results[2]['trainable_parameters'] == receipt['trainable_parameters']
    actual = trained(checkpoints)
    assert set(actual) == set(expected)
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name
    assert results[0]['receipt']['losses'] == receipt['losses']
    assert results[0]['receipt']['preference_margins'] == receipt['preference_margins']


def test_a_lost_arm_owner_resumes_from_its_last_checkpoint(deployed, tmp_path):
    checkpoint, config_dir, shards = deployed
    experience, replay, pairs = data()
    expected, _ = single_host(checkpoint, 'addition', experience, replay, pairs)
    checkpoints = tmp_path / 'arm'
    first = launch(config_dir, shards, [job('addition', experience, replay, pairs, checkpoints,
                                            fail={'step': 2})] * WORLD, tmp_path / 'outage')
    assert first[2] is None and not first[0]['completed']
    state = json.loads((checkpoints / 'state.json').read_text())
    assert state['completed_steps'] == 2
    references = json.loads((tmp_path / 'outage' / 'references.json').read_text()) if (
        tmp_path / 'outage' / 'references.json').exists() else None
    if references is None:
        with torch.no_grad():
            model = canonical(checkpoint)
            references = [(float(trainer.sequence_logprob(model, p['chosen'], 'cpu')),
                           float(trainer.sequence_logprob(model, p['rejected'], 'cpu'))) for p in pairs]
    resumed = launch(config_dir, shards, [job('addition', experience, replay, pairs, checkpoints, start=2,
                                              references=references, resume_from=str(checkpoints))] * WORLD,
                     tmp_path / 'resume')
    assert all(r and r['completed'] for r in resumed), resumed
    actual = trained(checkpoints)
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name
