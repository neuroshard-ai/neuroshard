import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

from neuroshard.evolution.sharded import granite

from test_granite_partition import BOUNDARIES, canonical, prompt, tiny_checkpoint

ROOT = Path(__file__).resolve().parents[2]
WORLD = len(BOUNDARIES) - 1


def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def launch(config_dir, shards, job, home):
    home.mkdir(parents=True, exist_ok=True)
    job_path = home / 'job.json'
    job_path.write_text(json.dumps(job))
    port = free_port()
    code = ('import os, sys; from neuroshard.evolution.sharded import granite_pipeline as p; '
            'p.run_owner(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), "127.0.0.1", '
            'int(sys.argv[5]), sys.argv[6], sys.argv[7], timeout=15); os._exit(0)')
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    processes = [subprocess.Popen([sys.executable, '-c', code, str(config_dir), str(shards), str(rank), str(WORLD),
                                   str(port), str(job_path), str(home / f'result-{rank}.json')], env=env)
                 for rank in range(WORLD)]
    for process in processes:
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()
    results = []
    for rank in range(WORLD):
        path = home / f'result-{rank}.json'
        results.append(json.loads(path.read_text()) if path.exists() else None)
    return results


@pytest.fixture
def deployed(tmp_path):
    checkpoint = tiny_checkpoint(tmp_path / 'granite')
    shards = tmp_path / 'shards'
    for rank in range(WORLD):
        granite.export(checkpoint, BOUNDARIES, rank, shards)
    config_dir = tmp_path / 'config'
    config_dir.mkdir()
    (config_dir / 'config.json').write_text((checkpoint / 'config.json').read_text())
    return checkpoint, config_dir, shards


def expected_tokens(checkpoint, ids, count):
    with torch.inference_mode():
        output = canonical(checkpoint).generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                                num_beams=1, use_cache=True, max_new_tokens=count,
                                                eos_token_id=None, pad_token_id=0)
    return output[0, ids.shape[1]:].tolist()


def test_owner_processes_generate_the_canonical_tokens_holding_only_their_shards(deployed, tmp_path):
    checkpoint, config_dir, shards = deployed
    requests = [{'id': f'r{seed}', 'input_ids': prompt(seed=seed)[0].tolist(), 'max_new_tokens': 12}
                for seed in range(2)]
    results = launch(config_dir, shards, {'requests': requests, 'eos_ids': [], 'max_tokens': 64}, tmp_path / 'run')
    assert all(r and r['completed'] for r in results), results
    for request, output in zip(requests, results[0]['outputs']):
        assert output['token_ids'] == expected_tokens(checkpoint, torch.tensor([request['input_ids']]), 12)
    total = sum(r['resident_bytes'] for r in results)
    assert all(r['resident_bytes'] < total for r in results)
    hidden = 64 * 2
    prefill = sum(len(r['input_ids']) for r in requests)
    steps = 2 * 12
    decode = steps - 2
    assert results[0]['sent_bytes'] == (prefill + decode) * hidden
    assert results[1]['sent_bytes'] == results[0]['sent_bytes']
    assert results[2]['sent_bytes'] == steps * hidden


def test_a_lost_owner_is_relaunched_and_the_continuation_is_unchanged(deployed, tmp_path):
    checkpoint, config_dir, shards = deployed
    ids = prompt(seed=5)
    request = {'id': 'r', 'input_ids': ids[0].tolist(), 'max_new_tokens': 14}
    job = {'requests': [request], 'eos_ids': [], 'max_tokens': 64, 'fail': {'rank': 2, 'at': 6}}
    first = launch(config_dir, shards, job, tmp_path / 'outage')
    assert first[2] is None and not first[0]['completed']
    committed = first[0]['inflight']['token_ids']
    assert len(committed) == 5
    resumed = launch(config_dir, shards, {**job, 'fail': None, 'requests': [{**request, 'committed': committed}]},
                     tmp_path / 'resume')
    assert all(r and r['completed'] for r in resumed), resumed
    assert resumed[0]['outputs'][0]['token_ids'] == expected_tokens(checkpoint, ids, 14)
    assert resumed[2]['steps'] == 14
