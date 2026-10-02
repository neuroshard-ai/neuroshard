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


def sharded(world, home, conversations=None, episodes=True, streams=None, log=False, fault=None, keys=None,
            session=None):
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
    if keys:
        job['keys'] = {str(rank): str(path) for rank, path in keys.items()}
    if session:
        job['session'] = session
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


VALIDATOR = '''
import json, sys, torch
from pathlib import Path
from neuroshard.evolution.sharded import granite, granite_audit
from neuroshard.inference import optimistic as ledger
torch.set_num_threads(1)
config_dir, shards, store, plan_path = sys.argv[1:5]
plan = json.loads(Path(plan_path).read_text())
partition, _ = granite.load_partition(granite.load_config(config_dir), shards, 1)
partition.warm_up(lengths=(16, 1))
check = granite_audit.challenge_checker(store, {1: partition})
state, outcomes = plan["genesis"], []
for block in plan["blocks"]:
    state = ledger.advance(state, state["height"] + 1)
    for envelope in block:
        try:
            state = ledger.transition(state, envelope, check)
            outcomes.append("accepted")
        except ValueError as error:
            outcomes.append("rejected: " + str(error))
print(json.dumps({"root": ledger.root(state), "outcomes": outcomes, "state": state}))
'''


def signing_keys(directory, ranks=(0, 1, 2)):
    """Ed25519 key files and public keys: rank 0 holds the user's session key, ranks 1 and 2 owner log keys."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    paths, public = {}, {}
    for rank in ranks:
        private = Ed25519PrivateKey.generate()
        paths[rank] = Path(directory) / f'signing-{rank}.key'
        paths[rank].write_text(private.private_bytes_raw().hex() + '\n')
        public[rank] = private.public_key().public_bytes_raw().hex()
    return paths, public


def session(envelope, chain):
    """The serving session of an opened job, as every owner receives it before serving."""
    from neuroshard.inference import optimistic as ledger

    body = envelope['body']
    return {'chain_id': chain, 'job_id': ledger.transaction_id(envelope), 'request_root': body['request_root'],
            'session_key': body['session_key'], 'log_keys': body['owners']}


def log_commit(account, key_path, chain, job_id, log_dir):
    """An owner's commitment, to ``job_id``, of the bound log in ``log_dir``."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from neuroshard.evolution.sharded import granite_audit
    from neuroshard.inference import optimistic as ledger

    private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(Path(key_path).read_text().strip()))
    commitment = granite_audit.commitment(granite_audit.load(log_dir)[0])
    signature = private.sign(ledger.commitment_message(chain, job_id, commitment['statement_root'])).hex()
    return account.sign('log_commit', job_id=job_id, log_signature=signature, **commitment)


def test_the_ledger_settles_honest_serving_and_slashes_a_fault_proven_by_replay(world, tmp_path):
    import hashlib
    import shutil

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from neuroshard.evolution.sharded import granite_audit
    from neuroshard.inference import optimistic as ledger

    from test_optimistic_serving import CHAIN, PARAMS, Account

    keys, public = signing_keys(tmp_path)
    people = {name: Account(name) for name in ('user', 'owner-1', 'owner-2', 'auditor', 'accuser')}
    model = hashlib.sha256((world['config'] / 'config.json').read_bytes()).hexdigest()
    genesis = ledger.genesis(CHAIN, model, 3, {p.public: 20_000_000 for p in people.values()}, PARAMS)

    def open_job(request):
        return people['user'].sign('serve_open', model_root=model, owners=[public[1], public[2]],
                                   request_root=hashlib.sha256(request.encode()).hexdigest(), session_key=public[0],
                                   price=1_000_000)

    # Jobs open before serving, so every message of a pass is signed for its job.
    open_honest, open_cheated = open_job('honest pass'), open_job('cheated pass')
    honest_job, cheated_job = ledger.transaction_id(open_honest), ledger.transaction_id(open_cheated)
    honest = sharded(world, tmp_path / 'honest', log=True, keys=keys, session=session(open_honest, CHAIN))
    cheated = sharded(world, tmp_path / 'cheated', log=True, fault={'rank': 1, 'at': 3}, keys=keys,
                      session=session(open_cheated, CHAIN))
    assert all(r['completed'] for r in honest + cheated)
    # Signing every link changes no served token.
    expected = single_host(world)
    assert len(honest[0]['episodes']) == len(expected)
    for got, want in zip(honest[0]['episodes'], expected):
        assert [g['token_ids'] for g in got['generations']] == [g['token_ids'] for g in want['generations']]
    for rank in (1, 2):
        record = granite_audit.load(tmp_path / 'honest' / f'log-{rank}')[0]
        assert record['session']['job_id'] == honest_job and not granite_audit.unattested(record)
    # The user's device keeps the last owner's signature over everything it sent back.
    held = honest[0]['attestation']
    assert held['key'] == public[2] and not granite_audit.equivocates(
        granite_audit.load(tmp_path / 'honest' / 'log-2')[0], held)
    assert audit(world, tmp_path / 'cheated', 1)['proof_accepted']
    store = tmp_path / 'store'
    proof = tmp_path / 'cheated' / 'log-1' / 'proof'
    real = granite_audit.bundle_root(proof)
    shutil.copytree(proof, store / real)
    record, payloads = granite_audit.load(tmp_path / 'honest' / 'log-1')
    first = next(i for i, e in enumerate(record['entries']) if 'output' in e)
    granite_audit.save_proof({'record': record, 'mismatch': first,
                              'inputs': {i: p for i, p in payloads.items() if i <= first}}, tmp_path / 'forged')
    forged = granite_audit.bundle_root(tmp_path / 'forged')
    shutil.copytree(tmp_path / 'forged', store / forged)

    def bond(rank):
        private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(keys[rank].read_text().strip()))
        account, amount = people[f'owner-{rank}'], PARAMS['owner_bond_minimum']
        possession = private.sign(ledger.possession_message(CHAIN, account.public, public[rank], rank, amount,
                                                            account.nonce)).hex()
        return account.sign('owner_bond', model_root=model, shard=rank, log_key=public[rank], amount=amount,
                            possession=possession)

    def commit(rank, job_id, pass_name):
        return log_commit(people[f'owner-{rank}'], keys[rank], CHAIN, job_id, tmp_path / pass_name / f'log-{rank}')

    blocks = [[bond(1), bond(2)], [open_honest], [commit(1, honest_job, 'honest'), commit(2, honest_job, 'honest')],
              [open_cheated]]
    # Owner 1 first offers its honest job's log, internally correct, for the cheated job.
    reused = commit(1, cheated_job, 'honest')
    people['owner-1'].nonce -= 1
    blocks += [[reused, commit(1, cheated_job, 'cheated'), commit(2, cheated_job, 'cheated')],
               [people['accuser'].sign('challenge', job_id=honest_job, log_key=public[1], proof_root=forged)],
               [people['auditor'].sign('challenge', job_id=cheated_job, log_key=public[1], proof_root=real)],
               [], [], []]
    (tmp_path / 'plan.json').write_text(json.dumps({'genesis': genesis, 'blocks': blocks}))
    replicas = []
    for _ in range(2):
        output = subprocess.check_output([sys.executable, '-c', VALIDATOR, str(world['config']), str(world['shards']),
                                          str(store), str(tmp_path / 'plan.json')],
                                         env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')}, text=True)
        replicas.append(json.loads(output.strip().splitlines()[-1]))
    assert replicas[0]['root'] == replicas[1]['root']
    outcomes, state = replicas[0]['outcomes'], replicas[0]['state']
    assert outcomes == (['accepted'] * 6 + ['rejected: Log is not bound to this job request'] + ['accepted'] * 2
                        + ['rejected: Fraud proof does not verify', 'accepted'])
    assert state['results'][honest_job]['status'] == 'settled' and state['results'][cheated_job]['status'] == 'fraud'
    assert state['owners'][public[1]]['status'] == 'slashed' and state['owners'][public[2]]['status'] == 'active'
    reward = PARAMS['owner_bond_minimum'] * PARAMS['auditor_share_ppm'] // 1_000_000
    assert state['accounts'][people['auditor'].public]['balance'] == 20_000_000 - PARAMS['fee'] + reward
    assert state['accounts'][people['owner-2'].public]['balance'] == (20_000_000 - 3 * PARAMS['fee']
                                                                     - PARAMS['owner_bond_minimum'] + 500_000)


ROLE = '''
import json, sys
from neuroshard.evolution import granite_shard_audit as audited, granite_shard_settlement as settlement
audited.freeze = lambda plan_path=None: {"commit": "rehearsal"}
role, phase, home, store = sys.argv[1:5]
if role.startswith("owner-"):
    result = settlement.owner(int(role.split("-")[1]), "", 0, phase, home, store)
else:
    result = settlement.auditor(phase, home, store)
print(json.dumps({"completed": result["completed"]}))
'''


def test_settlement_roles_sign_every_declared_transaction_and_validators_settle_them(world, tmp_path):
    import hashlib
    import shutil

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from neuroshard.evolution import granite_shard_settlement as settlement
    from neuroshard.evolution.modular_reference_execution import read, sha256
    from neuroshard.inference import optimistic as ledger

    plan = read(ROOT / settlement.PLAN)
    terms = plan['ledger']
    homes = {name: tmp_path / name / 'home' for name in ('owner-1', 'owner-2', 'auditor')}
    stores = {name: tmp_path / name / 'store' for name in homes}
    for name in homes:
        homes[name].mkdir(parents=True)
        stores[name].mkdir(parents=True)
    for rank in (1, 2):
        (stores[f'owner-{rank}'] / 'owner.key').write_text(Ed25519PrivateKey.generate().private_bytes_raw().hex() + '\n')
    keys = {rank: stores[f'owner-{rank}'] / 'owner.key' for rank in (1, 2)}
    # Owner 0 holds the user's session key, as the user's own device would.
    session_keys, session_public = signing_keys(tmp_path, ranks=(0,))
    keys[0] = session_keys[0]

    def role(name, phase, **request):
        if request:
            (homes[name] / f'{phase}-request.json').write_text(json.dumps({'chain_id': terms['chain_id'], **request}))
        output = subprocess.check_output([sys.executable, '-c', ROLE, name, phase, str(homes[name]), str(stores[name])],
                                         env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')}, text=True)
        assert json.loads(output.strip().splitlines()[-1])['completed']
        return read(homes[name] / phase / 'result.json')

    model = sha256(ROOT / settlement.MODEL_INVENTORY)
    bonds = [role(f'owner-{r}', 'sign-bond', nonce=0, model_root=model, amount=terms['owner_bond'])['envelope'] for r in (1, 2)]
    log_keys = [envelope['body']['log_key'] for envelope in bonds]
    user_key, user = settlement.account(tmp_path, 'user')
    jobs, opened = {}, {}
    for nonce, label in enumerate(('honest', 'cheat')):
        opened[label] = settlement.signed(user_key, {'kind': 'serve_open', 'chain_id': terms['chain_id'], 'nonce': nonce,
                                                     'model_root': model, 'owners': log_keys,
                                                     'request_root': hashlib.sha256(label.encode()).hexdigest(),
                                                     'session_key': session_public[0], 'price': terms['price']})
        jobs[label] = ledger.transaction_id(opened[label])
    for label, fault in (('honest', None), ('cheat', {'rank': 1, 'at': 3})):
        results = sharded(world, tmp_path / label, log=True, fault=fault, keys=keys,
                          session=session(opened[label], terms['chain_id']))
        assert all(r['completed'] for r in results)
        for rank in (1, 2):
            shutil.copytree(tmp_path / label / f'log-{rank}', homes[f'owner-{rank}'] / f'serve-{label}' / f'log-{rank}')
        shutil.copytree(tmp_path / label / 'log-1', homes['auditor'] / f'serve-{label}' / 'log-1')
    assert audit(world, tmp_path / 'cheat', 1)['proof_accepted']
    shutil.copytree(tmp_path / 'cheat' / 'log-1' / 'proof', homes['auditor'] / 'audit-cheat' / 'proof')
    commits = {label: [role(f'owner-{r}', f'commit-{label}', nonce=nonce, job_id=jobs[label])['envelope'] for r in (1, 2)]
               for nonce, label in ((1, 'honest'), (2, 'cheat'))}
    challenge = role('auditor', 'challenge', honest_job=jobs['honest'], cheated_job=jobs['cheat'], log_key=log_keys[0])
    for rank in (1, 2):
        for label in ('honest', 'cheat'):
            assert not (homes[f'owner-{rank}'] / f'serve-{label}' / f'log-{rank}' / 'inputs.safetensors').exists()
    assert not (homes['auditor'] / 'serve-honest' / 'log-1' / 'inputs.safetensors').exists()
    parties = {'user': user, 'owner-1': bonds[0]['public_key'], 'owner-2': bonds[1]['public_key'],
               'auditor': challenge['proven']['public_key'], 'accuser': challenge['framing']['public_key']}
    genesis = ledger.genesis(terms['chain_id'], model, terms['shards'],
                             {account: terms['allocation'] for account in parties.values()}, terms['params'])
    blocks = [bonds, [opened['honest']], commits['honest'], [opened['cheat']], commits['cheat'],
              [challenge['framing']], [], [], [challenge['proven']], [], []]
    (tmp_path / 'plan.json').write_text(json.dumps({'genesis': genesis, 'blocks': blocks}))
    output = subprocess.check_output([sys.executable, '-c', VALIDATOR, str(world['config']), str(world['shards']),
                                      str(homes['auditor'] / 'bundles'), str(tmp_path / 'plan.json')],
                                     env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')}, text=True)
    replica = json.loads(output.strip().splitlines()[-1])
    assert replica['outcomes'] == ['accepted'] * 8 + ['rejected: Fraud proof does not verify', 'accepted']
    state = replica['state']
    assert {a: state['accounts'][a]['balance'] for a in parties.values()} == settlement.expected_balances(plan, parties)
    assert state['results'][jobs['honest']]['status'] == 'settled' and state['results'][jobs['cheat']]['status'] == 'fraud'


def test_cometbft_validators_settle_honest_serving_and_slash_a_proven_fault(world, tmp_path):
    import hashlib
    import shutil
    import time

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from neuroshard.evolution.sharded import granite_audit
    from neuroshard.inference import optimistic as ledger
    from neuroshard.inference import optimistic_app as app
    from neuroshard.inference import optimistic_network as network

    from test_optimistic_serving import CHAIN, Account

    try:
        network.engine_path()
    except ValueError:
        pytest.skip('CometBFT v0.38.26 is not installed')
    keys, public = signing_keys(tmp_path)
    people = {name: Account(name) for name in ('user', 'owner-1', 'owner-2', 'auditor', 'accuser')}
    model = hashlib.sha256((world['config'] / 'config.json').read_bytes()).hexdigest()

    def open_job(label):
        envelope = people['user'].sign('serve_open', model_root=model, owners=[public[1], public[2]],
                                       request_root=hashlib.sha256(label.encode()).hexdigest(), session_key=public[0],
                                       price=1_000_000)
        return envelope, ledger.transaction_id(envelope)

    jobs = {label: open_job(label) for label in ('honest', 'cheat')}
    for label, fault in (('honest', None), ('cheat', {'rank': 1, 'at': 3})):
        assert all(r['completed'] for r in sharded(world, tmp_path / label, log=True, fault=fault, keys=keys,
                                                   session=session(jobs[label][0], CHAIN)))
    assert audit(world, tmp_path / 'cheat', 1)['proof_accepted']
    proven = granite_audit.bundle_root(tmp_path / 'cheat' / 'log-1' / 'proof')
    record, payloads = granite_audit.load(tmp_path / 'honest' / 'log-1')
    first = next(i for i, e in enumerate(record['entries']) if 'output' in e)
    granite_audit.save_proof({'record': record, 'mismatch': first, 'inputs': {i: p for i, p in payloads.items() if i <= first}},
                             tmp_path / 'forged')
    forged = granite_audit.bundle_root(tmp_path / 'forged')
    # Each validator keeps its own bundle store, so a bundle can reach some validators before others.
    stores = [tmp_path / f'store-{i}' for i in range(4)]
    for store in stores:
        shutil.copytree(tmp_path / 'forged', store / forged)

    params = {**ledger.PARAMS, 'challenge_blocks': 12, 'job_blocks': 400}
    terms = {'chain_id': CHAIN, 'model_root': model, 'shards': 3, 'params': params,
             'allocations': {p.public: 20_000_000 for p in people.values()}}
    nodes = [{'shards': {'1': {'config': str(world['config']), 'shard': str(world['shards'])}}, 'bundles': str(store),
              'threads': 1, 'warm_up_lengths': [16, 1]} for store in stores]
    config = network.initialize(tmp_path / 'chain', terms, nodes, base_port=29650, block_seconds=0.5)

    def submit(envelope, validator=0):
        admission = network.broadcast(network.url(config, validator), envelope)
        if admission['code'] == 0:
            return network.wait_included(config, envelope)
        return admission

    def bond(rank):
        private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(keys[rank].read_text().strip()))
        account, amount = people[f'owner-{rank}'], params['owner_bond_minimum']
        possession = private.sign(ledger.possession_message(CHAIN, account.public, public[rank], rank, amount, account.nonce))
        return account.sign('owner_bond', model_root=model, shard=rank, log_key=public[rank], amount=amount,
                            possession=possession.hex())

    def commit(rank, job, label):
        return log_commit(people[f'owner-{rank}'], keys[rank], CHAIN, job, tmp_path / label / f'log-{rank}')

    def agreed():
        while True:
            states = [network.state(network.url(config, i)) for i in range(4)]
            if len({s['height'] for s in states}) == 1:
                return states
            time.sleep(0.1)

    try:
        network.start(config, timeout=300)
        for rank in (1, 2):
            submit(bond(rank), rank)
        opened, honest = jobs['honest']
        submit(opened)
        committed = max(submit(commit(rank, honest, 'honest'), rank) for rank in (1, 2))
        framing = submit(people['accuser'].sign('challenge', job_id=honest, log_key=public[1], proof_root=forged), 3)
        assert framing == {**framing, 'code': 1, 'log': 'Fraud proof does not verify'}
        people['accuser'].nonce -= 1
        network.wait_height(config, committed + params['challenge_blocks'] + 1)
        assert agreed()[0]['results'][honest]['status'] == 'settled'
        opened, cheated = jobs['cheat']
        submit(opened)
        for rank in (1, 2):
            submit(commit(rank, cheated, 'cheat'), rank)
        # The proof reaches three of the four validators: more than two thirds of the voting power.
        for store in stores[:3]:
            shutil.copytree(tmp_path / 'cheat' / 'log-1' / 'proof', store / proven)
        challenge = people['auditor'].sign('challenge', job_id=cheated, log_key=public[1], proof_root=proven)
        lacking = submit(challenge, 3)
        assert lacking['code'] == app.NO_VERDICT and 'not held here' in lacking['log']
        submit(challenge, 2)
        states = agreed()
    finally:
        network.stop(config)
    # The fourth validator gave no verdict, so it voted for no block holding the challenge; it executed the
    # committed block all the same, reaching the others' state without the bundle.
    assert 'has no verdict on this validator' in (Path(config['home']) / 'logs' / 'app3.log').read_text()
    assert not (stores[3] / proven).exists()
    assert len({s['root'] for s in states}) == 1
    state = states[0]
    assert state['results'][cheated]['status'] == 'fraud' and state['owners'][public[1]]['status'] == 'slashed'
    fee, bond_amount = params['fee'], params['owner_bond_minimum']
    assert {name: state['accounts'][p.public]['balance'] for name, p in people.items()} == {
        'user': 20_000_000 - 2 * fee - 1_000_000, 'owner-1': 20_000_000 - 3 * fee - bond_amount + 500_000,
        'owner-2': 20_000_000 - 3 * fee - bond_amount + 500_000, 'auditor': 20_000_000 - fee + bond_amount // 2,
        'accuser': 20_000_000}


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
