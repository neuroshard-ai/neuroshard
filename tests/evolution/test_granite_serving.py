import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

from neuroshard.evolution import assistant_experience_eval as evaluation
from neuroshard.evolution import assistant_experience_run as accelerator
from neuroshard.evolution import assistant_experience_train as trainer
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution import granite_tokenizer
from neuroshard.evolution.assistant_serving import cached_responder
from neuroshard.evolution.sharded import granite

from test_assistant_experience import TEMPLATE
from test_assistant_workflow import policy
from test_granite_pipeline import free_port
from test_granite_tokenizer import granite_like, load_tiny

ROOT = Path(__file__).resolve().parents[2]
BOUNDARIES = (0, 1, 3, 4)
SPEC = {'layers': [3], 'rank': 4, 'alpha': 8, 'seed': 27092026}


def bounded():
    value = copy.deepcopy(policy())
    value['generation'].update(max_input_tokens=100000, max_new_tokens=6)
    return value


@pytest.fixture
def world(tmp_path):
    """A tiny Granite-shaped assistant, its saved addition arm, owner shards and a mixed gate."""
    from transformers import GraniteConfig, GraniteForCausalLM

    directory = granite_like(tmp_path / 'granite')
    (directory / 'chat_template.jinja').write_text(TEMPLATE)
    tokenizer = load_tiny(directory)[0]
    torch.manual_seed(3)
    config = GraniteConfig(vocab_size=len(tokenizer.runtime), hidden_size=32, intermediate_size=64,
                           num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                           tie_word_embeddings=True, eos_token_id=tokenizer.eos_token_id,
                           pad_token_id=tokenizer.pad_token_id, logits_scaling=2.0, embedding_multiplier=4.0,
                           residual_multiplier=0.5, initializer_range=0.2)
    checkpoint = tmp_path / 'checkpoint'
    GraniteForCausalLM(config).to(torch.bfloat16).save_pretrained(checkpoint, safe_serialization=True)
    model = load(checkpoint)
    trainable = trainer.prepare(model, 'addition', SPEC)
    generator = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for name, value in trainable.items():
            if name.endswith('lora_b'):
                value.copy_(torch.randn(value.shape, generator=generator))
    arm = tmp_path / 'arm'
    trainer.checkpoint(arm, trainable, {'arm': 'addition', 'optimizer_state': {}, 'trainable_parameters': 1,
                                        'steps': 0, 'schedule_sha256': '', 'losses': [0.0]}, {})
    shards = tmp_path / 'shards'
    for rank in range(3):
        granite.export(checkpoint, BOUNDARIES, rank, shards)
    config_dir = tmp_path / 'config'
    config_dir.mkdir()
    (config_dir / 'config.json').write_text((checkpoint / 'config.json').read_text())
    cases = data.cases('development')[:4]
    features = [accelerator.boundary_feature(load(checkpoint), tokenizer, bounded(), case, 'cpu') for case in cases]
    # A gate through the middle of these features selects the arm for some episodes and the parent for others.
    center = torch.nn.functional.normalize(torch.tensor(features), dim=1)
    direction = center[0] - center[1]
    bias = -float(((center[0] + center[1]) / 2) @ direction)
    gate = {'rule': 'logistic', 'weight': direction.tolist(), 'bias': bias, 'epsilon': 1e-6, 'threshold': 0.5}
    return {'directory': directory, 'tokenizer': tokenizer, 'checkpoint': checkpoint, 'arm': arm, 'shards': shards,
            'config': config_dir, 'cases': cases, 'gate': gate}


def load(checkpoint):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16, attn_implementation='eager',
                                                local_files_only=True).eval()


def single_host(world):
    """The development evaluator's served system in a fresh process with the owners' numerical environment."""
    work = world['checkpoint'].parent / 'single'
    work.mkdir()
    job = {'checkpoint': str(world['checkpoint']), 'arm': str(world['arm']), 'spec': SPEC, 'gate': world['gate'],
           'tokenizer': str(world['directory']), 'case_ids': [c['id'] for c in world['cases']], 'policy': bounded()}
    (work / 'job.json').write_text(json.dumps(job))
    code = '''
import json, sys, torch
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from test_granite_tokenizer import load_tiny
from test_granite_serving import load
from neuroshard.evolution import assistant_experience_eval as evaluation, assistant_experience_run as accelerator
from neuroshard.evolution import assistant_experience_train as trainer, assistant_workflow_data as data
from neuroshard.evolution.assistant_serving import cached_responder
torch.set_num_threads(1)
job = json.loads(Path(sys.argv[1]).read_text())
tokenizer = load_tiny(Path(job["tokenizer"]))[0]
parent, model = load(job["checkpoint"]), load(job["checkpoint"])
trainer.load_trainable(model, "addition", job["spec"], job["arm"])
by_id = {c["id"]: c for c in data.cases("development")}
cases = [by_id[k] for k in job["case_ids"]]
feature = lambda case: accelerator.boundary_feature(parent, tokenizer, job["policy"], case, "cpu")
rows = evaluation.evaluate_arm(parent, model, tokenizer, job["gate"], feature, cases, job["policy"], {"tasks": []},
                               cached_responder)["episodes"]
Path(sys.argv[2]).write_text(json.dumps(rows))
'''
    subprocess.run([sys.executable, '-c', code, str(work / 'job.json'), str(work / 'rows.json'),
                    str(ROOT / 'tests/evolution')], check=True, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')})
    return json.loads((work / 'rows.json').read_text())


def sharded(world, home, conversations=None, episodes=True, streams=None, log=False, fault=None):
    home.mkdir()
    reference = Path(world['directory']) / 'tokenizer.json'
    job = {'spec': SPEC, 'arm': str(world['arm']), 'gate': world['gate'], 'tokenizer': str(world['directory']),
           'parent_tokenizer_digest': granite_tokenizer.blob_sha1(reference), 'policy': bounded(),
           'eos_ids': [world['tokenizer'].eos_token_id], 'max_tokens': 100000}
    if episodes:
        job.update(split='development', case_ids=[c['id'] for c in world['cases']])
    if conversations:
        job['conversations'] = conversations
    if streams:
        job['streams'] = streams
    if log:
        job['log'] = True
    if fault:
        job['fault'] = fault
    (home / 'job.json').write_text(json.dumps(job))
    port = free_port()
    code = ('import os, sys; from neuroshard.evolution.sharded import granite_serving as s; '
            's.run_owner(sys.argv[1], sys.argv[2], int(sys.argv[3]), 3, "127.0.0.1", int(sys.argv[4]), sys.argv[5], '
            'sys.argv[6], timeout=30); os._exit(0)')
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    processes = [subprocess.Popen([sys.executable, '-c', code, str(world['config']), str(world['shards']), str(rank),
                                   str(port), str(home / 'job.json'), str(home / f'result-{rank}.json')], env=env)
                 for rank in range(3)]
    for process in processes:
        try:
            process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            process.kill()
    return [json.loads((home / f'result-{r}.json').read_text()) for r in range(3)]


def test_the_served_assistant_on_owners_reproduces_single_host_serving(world, tmp_path):
    expected = single_host(world)
    results = sharded(world, tmp_path / 'served')
    assert all(r['completed'] for r in results), [r.get('error') for r in results]
    rows = results[0]['episodes']
    assert {row['selected'] for row in expected} == {'arm', 'parent'}
    assert [row['selected'] for row in rows] == [row['selected'] for row in expected]
    for got, want in zip(rows, expected):
        assert got['score'] == want['score'] and got['calls'] == want['calls']
        assert len(got['generations']) == len(want['generations'])
        for a, b in zip(got['generations'], want['generations']):
            for key in ('input_token_ids', 'token_ids', 'text', 'terminated', 'prompt_sha256', 'reused_prefix_tokens'):
                assert a[key] == b[key], key
    assert results[2]['arm_sha256'] and results[0]['arm_sha256'] is None


def test_concurrent_streams_serve_every_episode_exactly_as_alone(world, tmp_path):
    expected = single_host(world)
    results = sharded(world, tmp_path / 'streams', streams=3)
    assert all(r['completed'] for r in results), [r.get('error') for r in results]
    rows = results[0]['episodes']
    assert results[0]['streams'] == 3 and results[0]['peak_in_flight'] >= 2
    assert [row['id'] for row in rows] == [row['id'] for row in expected]
    assert [row['selected'] for row in rows] == [row['selected'] for row in expected]
    for got, want in zip(rows, expected):
        assert got['score'] == want['score'] and len(got['generations']) == len(want['generations'])
        for a, b in zip(got['generations'], want['generations']):
            for key in ('input_token_ids', 'token_ids', 'text', 'terminated', 'prompt_sha256', 'reused_prefix_tokens'):
                assert a[key] == b[key], key


def audit(world, logs, rank):
    """An auditor holding only one owner's shard replays its log in the owners' runtime."""
    code = '''
import json, sys, torch
from pathlib import Path
from neuroshard.evolution.sharded import granite, granite_audit
from neuroshard.evolution.sharded.granite_serving import Adapter
torch.set_num_threads(1)
config_dir, shards, log_dir, rank, arm, spec = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], json.loads(sys.argv[6])
partition, _ = granite.load_partition(granite.load_config(config_dir), shards, rank)
adapter = Adapter(partition, spec, arm) if rank == 2 else None
partition.warm_up(lengths=(16, 1))
record, payloads = granite_audit.load(log_dir)
report = granite_audit.replay(partition, record, payloads, adapter)
if report['valid']:
    forwards = [i for i, e in enumerate(record['entries']) if 'output' in e]
    forged = {'record': record, 'mismatch': forwards[0], 'inputs': {i: p for i, p in payloads.items() if i <= forwards[0]}}
    report['forged_proof_accepted'] = granite_audit.check_fraud_proof(partition, forged, record['public_key'], adapter)
else:
    granite_audit.save_proof(granite_audit.fraud_proof(record, payloads, report), Path(log_dir) / 'proof')
    proof, manifest = granite_audit.load_proof(Path(log_dir) / 'proof')
    report['proof_accepted'] = granite_audit.check_fraud_proof(partition, proof, record['public_key'], adapter)
    report['proof_inputs'] = len(proof['inputs'])
report['signed'] = granite_audit.signed_by(record, record['public_key'])
print(json.dumps(report))
'''
    output = subprocess.check_output([sys.executable, '-c', code, str(world['config']), str(world['shards']),
                                      str(logs / f'log-{rank}'), str(rank), str(world['arm']), json.dumps(SPEC)],
                                     env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')}, text=True)
    return json.loads(output.strip().splitlines()[-1])


def test_replay_audits_confirm_honest_owners_and_catch_a_one_bit_fault(world, tmp_path):
    from neuroshard.evolution.sharded import granite_audit

    honest = sharded(world, tmp_path / 'honest', log=True)
    assert all(r['completed'] for r in honest), [r.get('error') for r in honest]
    logs = tmp_path / 'honest'
    for rank in (1, 2):
        report = audit(world, logs, rank)
        assert report['valid'] and report['forwards_checked'] == honest[rank]['log']['forwards'] > 0
        assert report['signed'] and not report['forged_proof_accepted']
        record = granite_audit.load(logs / f'log-{rank}')[0]
        assert record['public_key'] == honest[rank]['log']['public_key']
        tampered = {**record, 'entries': record['entries'][:-1]}
        assert not granite_audit.signed_by(tampered, record['public_key'])
    assert granite_audit.continuity(granite_audit.load(logs / 'log-1')[0]['entries'],
                                    granite_audit.load(logs / 'log-2')[0]['entries']) is None
    cheated = sharded(world, tmp_path / 'cheated', log=True, fault={'rank': 1, 'at': 3})
    assert all(r['completed'] for r in cheated), [r.get('error') for r in cheated]
    report = audit(world, tmp_path / 'cheated', 1)
    record, _ = granite_audit.load(tmp_path / 'cheated' / 'log-1')
    forwards = [i for i, e in enumerate(record['entries']) if 'output' in e]
    assert not report['valid'] and report['first_mismatch'] == forwards[3]
    assert report['proof_accepted'] and report['proof_inputs'] == 4 and report['signed']
    assert audit(world, tmp_path / 'cheated', 2)['valid']
    upstream, downstream = granite_audit.load(tmp_path / 'cheated' / 'log-1')[0], granite_audit.load(tmp_path / 'cheated' / 'log-2')[0]
    assert granite_audit.continuity(upstream['entries'], downstream['entries']) is None


def test_owner_caches_crop_to_the_shared_prefix_like_the_single_host_responder(world, tmp_path):
    from neuroshard.evolution import assistant_workspace as workspace

    from test_assistant_serving import conversation

    requests = list(conversation(world['cases'][0]))
    tokenizer = world['tokenizer']
    parent, armed = load(world['checkpoint']), load(world['checkpoint'])
    trainer.load_trainable(armed, 'addition', SPEC, world['arm'])
    expected = {}
    for name, model in (('parent', parent), ('arm', armed)):
        respond = cached_responder(model, tokenizer, bounded())
        expected[name] = [respond(messages, workspace.TOOLS) for messages in requests]
    conversations = {name: {'arm': name == 'arm', 'tools': workspace.TOOLS, 'requests': requests}
                     for name in ('parent', 'arm')}
    results = sharded(world, tmp_path / 'conversation', conversations=conversations, episodes=False)
    assert all(r['completed'] for r in results), [r.get('error') for r in results]
    for name in ('parent', 'arm'):
        got = results[0]['conversations'][name]
        for a, b in zip(got, expected[name]):
            for key in ('input_token_ids', 'token_ids', 'text', 'terminated', 'prompt_sha256', 'reused_prefix_tokens'):
                assert a[key] == b[key], (name, key)
        assert [g['reused_prefix_tokens'] for g in got][0] == 0 and all(g['reused_prefix_tokens'] > 0 for g in got[1:])
    assert expected['arm'] != expected['parent']


def test_switching_the_arm_off_restores_the_parent_bit_for_bit(world):
    from neuroshard.evolution.sharded.granite_serving import Adapter

    read, _ = granite.checkpoint_reader(world['checkpoint'])
    config = granite.load_config(world['checkpoint'])
    parent = granite.Partition(config, BOUNDARIES, 2).load(read)
    served = granite.Partition(config, BOUNDARIES, 2).load(read)
    adapter = Adapter(served, SPEC, world['arm'])
    hidden = torch.randn(1, 7, 32, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    with torch.inference_mode():
        assert not torch.equal(served(hidden), parent(hidden))
        adapter.set(False)
        assert torch.equal(served(hidden), parent(hidden))
        adapter.set(True)
        assert not torch.equal(served(hidden), parent(hidden))
