import copy
import hashlib
import importlib.util
import subprocess
import sys

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from neuroshard.evolution import granite_shard_settlement as settlement
from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256
from neuroshard.inference import optimistic as ledger

from test_granite_shard_serving import passing_evidence
from test_optimistic_serving import Account

FROZEN = read(ROOT / settlement.PLAN)
# The published plan predates metered settlement and challenge deposits; these tests give its ledger
# a position budget, a deposit and a one-block proof window.
BUDGET, SERVED, DEPOSIT = 64, 40, 1_000_000
PLAN = {**FROZEN, 'ledger': {**FROZEN['ledger'], 'positions': BUDGET, 'params': {
    **FROZEN['ledger']['params'], 'challenge_deposit': DEPOSIT, 'proof_blocks': 1}}}
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
    # Key 0 is the user's session key; keys 1 and 2 are the owners' log keys.
    logs = {r: Ed25519PrivateKey.generate() for r in (0, 1, 2)}
    keys = {r: logs[r].public_key().public_bytes_raw().hex() for r in (0, 1, 2)}
    model = sha256(ROOT / settlement.MODEL_INVENTORY)
    genesis = ledger.genesis(chain_id, model, TERMS['shards'], {p.public: TERMS['allocation'] for p in people.values()},
                             TERMS['params'])

    def bond(r):
        account = people[f'owner-{r}']
        possession = logs[r].sign(ledger.possession_message(chain_id, account.public, keys[r], r, TERMS['owner_bond'], 0))
        return account.sign('owner_bond', nonce=0, model_root=model, shard=r, log_key=keys[r], amount=TERMS['owner_bond'],
                            possession=possession.hex())

    def commit(r, job, nonce):
        # Each log is bound to its job: its upstream sender (the user, then owner 1) signed its inputs' transcript.
        head, entries_root = (hashlib.sha256(f'{job}:{r}:{part}'.encode()).hexdigest() for part in ('head', 'entries'))
        upstream = {'key': keys[r - 1], 'entries': 1, 'head': head, 'positions': SERVED,
                    'signature': logs[r - 1].sign(ledger.link_message(chain_id, job, r - 1, 1, head, SERVED)).hex()}
        header = {'format': ledger.OWNER_LOG_FORMAT, 'rank': r, 'public_key': keys[r], 'upstream': upstream,
                  'session': {'chain_id': chain_id, 'job_id': job, 'request_root': 'cd' * 32}}
        statement = ledger.log_statement(header, entries_root)
        return people[f'owner-{r}'].sign('log_commit', nonce=nonce, job_id=job, statement_root=statement,
                                         entries_root=entries_root, header=header,
                                         log_signature=logs[r].sign(ledger.commitment_message(chain_id, job, statement)).hex())

    def open_job(nonce):
        envelope = people['user'].sign('serve_open', nonce=nonce, model_root=model, owners=[keys[1], keys[2]],
                                       request_root='cd' * 32, session_key=keys[0], price=TERMS['price'],
                                       positions=BUDGET)
        return envelope, ledger.transaction_id(envelope)

    honest_open, honest = open_job(0)
    cheat_open, cheated = open_job(1)
    framing, framing_prove = settlement.challenge_pair(people['accuser'].key, chain_id, honest, keys[1], '0f' * 32)
    proven, proven_prove = settlement.challenge_pair(people['auditor'].key, chain_id, cheated, keys[1], 'aa' * 32)
    blocks = [[bond(1), bond(2)], [honest_open], [commit(1, honest, 1), commit(2, honest, 1)], [cheat_open],
              [commit(1, cheated, 2), commit(2, cheated, 2)]] + settlement.challenge_blocks(TERMS['params'], {
                  'framing': framing, 'framing_prove': framing_prove, 'proven': proven, 'proven_prove': proven_prove})
    parties = {name: p.public for name, p in people.items()}
    return genesis, blocks, parties, keys, {'honest': honest, 'cheated': cheated,
                                            'framing': ledger.transaction_id(framing)}


def test_the_declared_sequence_leaves_exactly_the_declared_balances():
    genesis, blocks, parties, keys, jobs = declared_chain()
    report = settlement.replay_blocks(genesis, blocks, lambda state, request: request['proof_root'] == 'aa' * 32)
    assert report['outcomes'] == ['accepted'] * 9 + ['rejected: Fraud proof does not verify', 'accepted', 'accepted']
    state, served = report['state'], {1: SERVED, 2: SERVED}
    assert {a: state['accounts'][a]['balance'] for a in parties.values()} == settlement.expected_balances(
        PLAN, parties, served)
    assert state['results'][jobs['honest']]['status'] == 'settled' and state['results'][jobs['cheated']]['status'] == 'fraud'
    # The framing lapses and the honest job settles at block 8, before the fraud proof lands at block 9.
    assert state['results'][jobs['framing']] == {**state['results'][jobs['framing']], 'status': 'forfeited',
                                                 'deposit': DEPOSIT, 'height': 8}
    assert state['results'][jobs['honest']]['height'] == 8 and state['results'][jobs['cheated']]['height'] == 9
    assert state['burned'] == 11 * TERMS['params']['fee'] + DEPOSIT + TERMS['owner_bond'] // 2
    paid = TERMS['price'] * SERVED // (BUDGET * 2)
    assert state['results'][jobs['honest']]['paid'] == settlement.honest_payments(PLAN, served) == [paid, paid]
    assert state['results'][jobs['honest']]['refunded'] == TERMS['price'] - 2 * paid > 0
    assert len(report['challenge_seconds']) == 2


def test_honest_payments_never_exceed_what_the_user_sent_or_the_budget_bought():
    price = TERMS['price']
    assert settlement.honest_payments(PLAN, {1: BUDGET, 2: BUDGET}) == [price // 2, price // 2]
    assert settlement.honest_payments(PLAN, {1: 10, 2: 30}) == [price * 10 // (BUDGET * 2)] * 2
    assert settlement.honest_payments(PLAN, {1: 3 * BUDGET, 2: 2 * BUDGET}) == [price // 2, price // 2]
    with pytest.raises(KeyError):
        settlement.honest_payments(FROZEN, {1: 1, 2: 1})


def evidence():
    genesis, blocks, parties, keys, jobs = declared_chain()
    report = settlement.replay_blocks(genesis, blocks, lambda state, request: request['proof_root'] == 'aa' * 32)
    owner_fetches, _, served = passing_evidence()
    fetches = {'owners': [owner_fetches[0], {**owner_fetches[1], 'log_key': keys[1]}, {**owner_fetches[2], 'log_key': keys[2]}]}
    validator = {**report, 'completed': True, 'load_seconds': 9.0}
    phases = {'serve-honest': served, 'audit-cheat': {'fault_forward': PLAN['fault']['at']},
              'commit-honest': {f'owner-{r}': {'positions': SERVED, 'completed': True} for r in (1, 2)},
              'validate': {'validator-1': validator, 'validator-2': copy.deepcopy(validator)}}
    return fetches, phases, parties, jobs


def test_assessment_requires_agreeing_validators_a_forfeited_framing_a_proven_fault_and_exact_settlement():
    fetches, phases, parties, jobs = evidence()
    report = settlement.assess(PLAN, fetches, phases, parties, jobs)
    assert report['passed'], report['checks']
    split = copy.deepcopy(phases)
    split['validate']['validator-2']['root'] = '00' * 32
    assert not settlement.assess(PLAN, fetches, split, parties, jobs)['checks']['replicas_agree']
    framed = copy.deepcopy(phases)
    for name in ('validator-1', 'validator-2'):
        framed['validate'][name]['outcomes'][9] = 'accepted'
    assert not settlement.assess(PLAN, fetches, framed, parties, jobs)['checks']['framing_forfeited']
    kept = copy.deepcopy(phases)
    for name in ('validator-1', 'validator-2'):
        del kept['validate'][name]['state']['results'][jobs['framing']]
    assert not settlement.assess(PLAN, fetches, kept, parties, jobs)['checks']['framing_forfeited']
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
    # Owners that report covering other positions than the ledger paid them for, or report none.
    for rank, miscounted in ((1, {'positions': SERVED + 1}), (2, {'positions': SERVED - 1}), (2, {}),
                             (1, {'positions': str(SERVED)})):
        claimed = copy.deepcopy(phases)
        claimed['commit-honest'][f'owner-{rank}'] = miscounted
        checks = settlement.assess(PLAN, fetches, claimed, parties, jobs)['checks']
        assert not checks['honest_settled'] and not checks['balances_exact']


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
