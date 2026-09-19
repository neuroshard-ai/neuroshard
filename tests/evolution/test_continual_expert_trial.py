"""Run the actual bounded continued-learning driver on four tiny shard owners."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil

import torch.multiprocessing as mp

from neuroshard.evolution.reference_data import identity, save, sha256
from neuroshard.evolution.sharded import graph_quality
from test_expert_data import prepared

SOURCE = Path(__file__).resolve().parents[2]


def worker(rank, home):
    os.environ.update(RANK=str(rank), WORLD_SIZE='4', MASTER_ADDR='127.0.0.1')
    spec = importlib.util.spec_from_file_location('continued_trial', SOURCE/'scripts/run_continual_expert_trial.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run(Path(home)/str(rank), SOURCE)


def test_four_owners_train_fresh_job_and_leave_final_closed_after_failed_gate(prepared, monkeypatch):
    import socket
    home, _, job, _, store, _, _, plan, inputs = prepared
    parent = job['work']['parent']
    baseline = job['lifecycle']['serving_graph']
    trial_home = home/'trial'
    trial_home.mkdir()
    shared = trial_home/'inputs'
    shutil.copytree(home/'general-inputs', shared)
    plan = copy.deepcopy(plan)
    plan['seed_expert'] = {'name': 'protocol', 'checkpoint': baseline['experts']['protocol']}
    plan['tokenizer'] = baseline['tokenizer']['root']
    plan['tokenizer_files'] = baseline['tokenizer']['files']
    inputs = {**copy.deepcopy(inputs), 'plan': identity(plan)}
    quality = store.json(job['lifecycle']['quality']['policy_root'])
    example_rows = [json.loads(line) for line in store.get(quality['roles']['test']['sha256']).splitlines()]
    evaluation = {}
    for role in ('dev', 'retained', 'test'):
        rows = copy.deepcopy(example_rows)
        for row in rows:
            row['answers'] = ['impossible-unseen-vocabulary'] * len(row['answers'])
            row['messages'][-1]['content'] = '; '.join(row['answers'])
        path = shared/(role+'-questions.jsonl')
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        evaluation[role] = {'file': path.name, 'sha256': sha256(path), 'count': len(rows),
                            'ids': identity([row['id'] for row in rows]), 'release_scope': False}
    trial = {'format': 'neuroshard-continual-expert-trial-v1', 'training': plan['training'],
        'objective': plan['objective'], 'max_seconds': 120, 'max_tokens': 4,
        'evaluation': evaluation, 'gates': quality['gates'], 'numerical_profile': job['work']['numerical_profile']}
    freeze = {'trial': identity(trial), 'plan': identity(plan), 'prepared': identity(inputs),
              'driver': sha256(SOURCE/'scripts/run_continual_expert_trial.py'), 'sources': {}}
    for name, value in [('parent', parent), ('trial', trial), ('plan', plan), ('prepared', inputs), ('freeze', freeze)]:
        save(shared/(name+'.json'), value)
    for rank in range(4):
        folder = trial_home/str(rank)
        folder.mkdir()
        for name, target in [('inputs', shared), ('objects', home/'objects'), ('seed', home/'seed')]:
            (folder/name).symlink_to(target, target_is_directory=True)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        monkeypatch.setenv('MASTER_PORT', str(listener.getsockname()[1]))
    mp.spawn(worker, args=(str(trial_home),), nprocs=4, join=True)
    results = [json.loads((trial_home/str(rank)/'results/development.json').read_bytes()) for rank in range(4)]
    assert all(row == results[0] for row in results)
    result = results[0]
    assert result['checkpoint']['step'] == plan['training']['steps']
    assert not result['decision']['passed'] and result['final_opened'] is False
    assert result['native_activated'] is False and result['tokens_issued'] == 0
    assert (trial_home/'3/results/window-000000.json').is_file()
    assert not any(trial_home.glob('*/results/before-test.json'))
    assert len(list((trial_home/'3/results/checkpoints').iterdir())) == 1
