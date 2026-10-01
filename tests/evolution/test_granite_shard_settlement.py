import copy
import hashlib
import importlib.util
import subprocess
import sys

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from neuroshard.evolution import granite_shard_settlement as settlement
from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256
from neuroshard.inference import optimistic as ledger

from test_granite_shard_serving import passing_evidence
from test_optimistic_serving import Account

PLAN = read(ROOT / settlement.PLAN)
TERMS = PLAN['ledger']


def cloud_module():
    spec = importlib.util.spec_from_file_location('settlement_cloud', ROOT / 'scripts/modular_reference_cloud.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def declared_chain():
    """The declared block sequence with real signatures; proofs are stubbed by their roots."""
    chain_id = TERMS['chain_id']
    people = {name: Account(name) for name in ('user', 'owner-1', 'owner-2', 'auditor', 'accuser')}
    for person in people.values():
        person.sign = (lambda account: lambda kind, **fields: settlement.signed(
            account.key, {'kind': kind, 'chain_id': chain_id, 'nonce': fields.pop('nonce'), **fields}))(person)
    logs = {r: Ed25519PrivateKey.generate() for r in (1, 2)}
    keys = {r: logs[r].public_key().public_bytes_raw().hex() for r in (1, 2)}
    model = sha256(ROOT / settlement.MODEL_INVENTORY)
    genesis = ledger.genesis(chain_id, model, TERMS['shards'], {p.public: TERMS['allocation'] for p in people.values()},
                             TERMS['params'])

    def bond(r):
        account = people[f'owner-{r}']
        possession = logs[r].sign(ledger.possession_message(chain_id, account.public, keys[r], r, TERMS['owner_bond'], 0))
        return account.sign('owner_bond', nonce=0, model_root=model, shard=r, log_key=keys[r], amount=TERMS['owner_bond'],
                            possession=possession.hex())

    def commit(r, job, nonce):
        statement = hashlib.sha256(f'{job}:{r}'.encode()).hexdigest()
        return people[f'owner-{r}'].sign('log_commit', nonce=nonce, job_id=job, statement_root=statement,
                                         log_signature=logs[r].sign(ledger.commitment_message(chain_id, job, statement)).hex())

    def open_job(nonce):
        envelope = people['user'].sign('serve_open', nonce=nonce, model_root=model, owners=[keys[1], keys[2]],
                                       request_root='cd' * 32, price=TERMS['price'])
        return envelope, ledger.transaction_id(envelope)

    honest_open, honest = open_job(0)
    cheat_open, cheated = open_job(1)
    challenge = {'log_key': keys[1], 'nonce': 0}
    blocks = [[bond(1), bond(2)], [honest_open], [commit(1, honest, 1), commit(2, honest, 1)], [cheat_open],
              [commit(1, cheated, 2), commit(2, cheated, 2)],
              [people['accuser'].sign('challenge', job_id=honest, proof_root='0f' * 32, **challenge)],
              [], [], [people['auditor'].sign('challenge', job_id=cheated, proof_root='aa' * 32, **challenge)], [], []]
    parties = {name: p.public for name, p in people.items()}
    return genesis, blocks, parties, keys, {'honest': honest, 'cheated': cheated}


def test_the_declared_sequence_leaves_exactly_the_declared_balances():
    genesis, blocks, parties, keys, jobs = declared_chain()
    report = settlement.replay_blocks(genesis, blocks, lambda state, request: request['proof_root'] == 'aa' * 32)
    assert report['outcomes'] == ['accepted'] * 8 + ['rejected: Fraud proof does not verify', 'accepted']
    state = report['state']
    assert {a: state['accounts'][a]['balance'] for a in parties.values()} == settlement.expected_balances(PLAN, parties)
    assert state['results'][jobs['honest']]['status'] == 'settled' and state['results'][jobs['cheated']]['status'] == 'fraud'
    assert len(report['challenge_seconds']) == 2


def evidence():
    genesis, blocks, parties, keys, jobs = declared_chain()
    report = settlement.replay_blocks(genesis, blocks, lambda state, request: request['proof_root'] == 'aa' * 32)
    owner_fetches, _, served = passing_evidence()
    fetches = {'owners': [owner_fetches[0], {**owner_fetches[1], 'log_key': keys[1]}, {**owner_fetches[2], 'log_key': keys[2]}]}
    validator = {**report, 'completed': True, 'load_seconds': 9.0}
    phases = {'serve-honest': served, 'audit-cheat': {'fault_forward': PLAN['fault']['at']},
              'validate': {'validator-1': validator, 'validator-2': copy.deepcopy(validator)}}
    return fetches, phases, parties, jobs


def test_assessment_requires_agreeing_validators_a_rejected_framing_a_proven_fault_and_exact_settlement():
    fetches, phases, parties, jobs = evidence()
    report = settlement.assess(PLAN, fetches, phases, parties, jobs)
    assert report['passed'], report['checks']
    split = copy.deepcopy(phases)
    split['validate']['validator-2']['root'] = '00' * 32
    assert not settlement.assess(PLAN, fetches, split, parties, jobs)['checks']['replicas_agree']
    framed = copy.deepcopy(phases)
    for name in ('validator-1', 'validator-2'):
        framed['validate'][name]['outcomes'][8] = 'accepted'
    assert not settlement.assess(PLAN, fetches, framed, parties, jobs)['checks']['framing_rejected']
    misnamed = copy.deepcopy(phases)
    misnamed['audit-cheat']['fault_forward'] = 199
    assert not settlement.assess(PLAN, fetches, misnamed, parties, jobs)['checks']['fault_named']
    shorted = copy.deepcopy(phases)
    for name in ('validator-1', 'validator-2'):
        shorted['validate'][name]['state']['accounts'][parties['auditor']]['balance'] -= 1
    assert not settlement.assess(PLAN, fetches, shorted, parties, jobs)['checks']['balances_exact']
    unsettled = copy.deepcopy(phases)
    for name in ('validator-1', 'validator-2'):
        unsettled['validate'][name]['state']['results'][jobs['honest']]['status'] = 'expired'
    assert not settlement.assess(PLAN, fetches, unsettled, parties, jobs)['checks']['honest_settled']


def test_the_settlement_plan_reuses_the_audited_serving_target_and_fault():
    audit = read(ROOT / 'config/experiments/granite-shard-audit.json')
    for key in ('model', 'learning', 'runtime', 'boundaries', 'arm', 'upload', 'target', 'fault', 'signing_packages'):
        assert PLAN[key] == audit[key]
    assert PLAN['audited_owners'] == [1] and TERMS['module'] == 'src/neuroshard/inference/optimistic.py'
    assert TERMS['params']['challenge_blocks'] == 4 and TERMS['shards'] == len(PLAN['boundaries']) - 1


def test_the_settlement_profile_is_bounded_and_the_freeze_covers_every_imported_source():
    import ast
    import re

    for name, digest in PLAN['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert set(PLAN['contracts']) <= set(PLAN['sources'])
    cloud = cloud_module()
    resources = cloud.resources(settlement.PROFILE)
    assert resources['purpose'] == settlement.PROFILE and 6 * resources['planning_cap_usd'] <= 60
    seconds = PLAN['phase_seconds']
    timeline = (seconds['fetch'] + seconds['sign'] + 2 * (seconds['serve'] + seconds['commit'] + seconds['audit'])
                + seconds['challenge'] + seconds['validate'])
    assert timeline + resources['setup_seconds'] + resources['copy_seconds'] <= resources['hours'] * 3600
    module, requirements = cloud.GRANITE_PROFILES[settlement.PROFILE]
    assert module == 'granite_shard_settlement' and requirements == 'docs/granite-shard-audit-requirements.txt'
    probe = ('import os, sys; import neuroshard.evolution.granite_shard_settlement, neuroshard.inference.optimistic, '
             'neuroshard.core.crypto.ecdsa, neuroshard.evolution.sharded.granite_audit, '
             'neuroshard.evolution.sharded.granite_serving, neuroshard.evolution.sharded.granite_training, '
             'neuroshard.evolution.assistant_experience_run, neuroshard.evolution.granite_tokenizer, '
             'neuroshard.evolution.assistant_selector, neuroshard.evolution.assistant_workflow, '
             'neuroshard.evolution.assistant_experience_gate; root = os.path.abspath("src"); '
             'print("\\n".join(sorted(os.path.relpath(x.__file__) for x in list(sys.modules.values()) '
             'if getattr(x, "__file__", None) and os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(PLAN['sources']), set(imported) - set(PLAN['sources'])
    lines = [line.split('#')[0].strip() for line in (ROOT / requirements).read_text().splitlines()]
    pinned = {re.split('[=<>@ ]', line)[0].lower().replace('_', '-') for line in lines if line}
    third_party = set()
    for name in [n for n in PLAN['sources'] if n.startswith('src/') or n == settlement.SCRIPT]:
        for node in ast.walk(ast.parse((ROOT / name).read_text())):
            if isinstance(node, ast.Import):
                third_party.update(a.name.split('.')[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                third_party.add(node.module.split('.')[0])
    third_party -= set(sys.stdlib_module_names) | {'neuroshard', 'granite_switch'}
    assert {{'huggingface_hub': 'huggingface-hub'}.get(t, t) for t in third_party} <= pinned, third_party
    spec = importlib.util.spec_from_file_location('settlement_controller', ROOT / 'scripts/granite_shard_settlement_cloud.py')
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    for phase in ('sign-bond', 'commit-honest', 'audit-cheat', 'serve-cheat', 'challenge', 'validate', 'fetch'):
        assert controller.base.seconds(PLAN, phase) == seconds[phase.split('-')[0]]


def test_every_registered_execution_module_exports_what_bootstrap_calls():
    import importlib

    for profile, (module, _) in cloud_module().GRANITE_PROFILES.items():
        loaded = importlib.import_module(f'neuroshard.evolution.{module}')
        assert callable(getattr(loaded, 'configure', None)) and callable(getattr(loaded, 'freeze', None)), profile


def test_importing_the_settlement_execution_does_not_load_torch():
    probe = 'import sys, neuroshard.evolution.granite_shard_settlement; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', probe], check=True, cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'src')})
