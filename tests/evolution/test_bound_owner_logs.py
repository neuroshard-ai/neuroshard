"""Bound owner logs and the challenge checker on a tiny Granite model, served in one process through signed links."""
import json
import shutil

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from neuroshard.evolution.sharded import granite, granite_audit
from neuroshard.evolution.sharded.granite_pipeline import FORWARD, RESET
from neuroshard.evolution.sharded.granite_serving import ADAPTER, CROP, FEATURE
from neuroshard.inference import optimistic as ledger

from test_optimistic_serving import CHAIN, PARAMS, Account, settle

BOUNDARIES = (0, 1, 3, 4)
# Two episodes: a feature pass, a prefill, decoding, a crop to a shared prefix, and a trailing arm switch.
PLAN = [(ADAPTER, 0), (FEATURE, 4), (RESET, 0), (ADAPTER, 1), (FORWARD, 5), (FORWARD, 1), (FORWARD, 1), (CROP, 4),
        (FORWARD, 2), (FORWARD, 1), (ADAPTER, 0)]
SERVED = sum(value for op, value in PLAN if op in (FORWARD, FEATURE))


@pytest.fixture(scope='module')
def partitions(tmp_path_factory):
    from transformers import GraniteConfig, GraniteForCausalLM

    checkpoint = tmp_path_factory.mktemp('granite') / 'checkpoint'
    torch.manual_seed(3)
    config = GraniteConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                           num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True, logits_scaling=2.0,
                           embedding_multiplier=4.0, residual_multiplier=0.5, initializer_range=0.2)
    GraniteForCausalLM(config).to(torch.bfloat16).save_pretrained(checkpoint, safe_serialization=True)
    read, _ = granite.checkpoint_reader(checkpoint)
    return {rank: granite.Partition(granite.load_config(checkpoint), BOUNDARIES, rank).load(read) for rank in (1, 2)}


@pytest.fixture
def market():
    """Bonded owners 1 and 2, the user's session key (key 0), an auditor, the ledger state and a job opener."""
    keys = [Ed25519PrivateKey.generate() for _ in range(3)]
    public = [granite_audit.public_hex(key) for key in keys]
    people = {name: Account(name) for name in ('user', 'owner-1', 'owner-2', 'auditor')}
    state = ledger.genesis(CHAIN, 'ab' * 32, 3, {p.public: 20_000_000 for p in people.values()}, PARAMS)
    for rank in (1, 2):
        owner = people[f'owner-{rank}']
        possession = keys[rank].sign(ledger.possession_message(CHAIN, owner.public, public[rank], rank,
                                                               PARAMS['owner_bond_minimum'], owner.nonce)).hex()
        state = ledger.transition(state, owner.sign('owner_bond', model_root='ab' * 32, shard=rank, log_key=public[rank],
                                                    amount=PARAMS['owner_bond_minimum'], possession=possession), None)
    value = {'keys': keys, 'public': public, 'people': people, 'state': state}

    def open_job(request, positions=SERVED):
        envelope = people['user'].sign('serve_open', model_root='ab' * 32, owners=public[1:], request_root=request,
                                       session_key=public[0], price=1_000_000, positions=positions)
        value['state'] = ledger.transition(value['state'], envelope, None)
        return {'chain_id': CHAIN, 'job_id': ledger.transaction_id(envelope), 'request_root': request}

    value['open'] = open_job
    return value


def serve(partitions, session, keys, fault=None, seed=5, history=None):
    """Owners 1 and 2 serve ``PLAN`` for one job through signed links; random states stand in for owner 0's layers.

    ``history``, if given, collects every upstream signature each owner received, in order.
    """
    from transformers import DynamicCache

    public = [granite_audit.public_hex(key) for key in keys]
    links = [granite_audit.Link(session, rank, 3, keys[rank], public[(rank - 1) % 3]) for rank in range(3)]
    logs, caches = {r: granite_audit.OwnerLog(r) for r in (1, 2)}, {r: DynamicCache() for r in (1, 2)}
    generator, forwards = torch.Generator().manual_seed(seed), 0
    for op, value in PLAN:
        if op not in (FORWARD, FEATURE):
            for link in links:
                link.command(op, value)
            for rank in (1, 2):
                logs[rank].command(op, value)
                caches[rank] = DynamicCache() if op == RESET else caches[rank]
                if op == CROP:
                    caches[rank].crop(value)
            continue
        message = torch.randn((1, value, 32), generator=generator).to(torch.bfloat16)
        signature = links[0].send(op, value, message)
        for rank in (1, 2):
            attested = links[rank].receive(op, value, message, signature)
            if history is not None:
                history.setdefault(rank, []).append(attested)
            with torch.inference_mode():
                if op == FORWARD:
                    mask = torch.ones((1, caches[rank].get_seq_length() + value), dtype=torch.long)
                    out = partitions[rank](message, mask, caches[rank])
                else:
                    out = partitions[rank](message, None, DynamicCache())
            sent = out[:, -1:] if rank == 2 else out
            if fault == (rank, forwards):
                sent = sent.clone()
                sent.view(torch.int16).view(-1)[0] ^= 1
            logs[rank].forward(op, value, message, sent, attested)
            signature = links[rank].send(op, value, sent)
            message = sent
        links[0].receive(op, value, message, signature)
        forwards += 1
    return logs, links


def saved(logs, keys, session, directory):
    records = {}
    for rank, log in logs.items():
        log.save(directory / f'log-{rank}', keys[rank], session)
        records[rank] = granite_audit.load(directory / f'log-{rank}')
    return records


def commit(market, rank, session, record):
    """Owner ``rank``'s log commitment for the job: the header, entries digest and statement of ``record``."""
    commitment = granite_audit.commitment(record)
    signature = market['keys'][rank].sign(ledger.commitment_message(CHAIN, session['job_id'], commitment['statement_root']))
    return market['people'][f'owner-{rank}'].sign('log_commit', job_id=session['job_id'], log_signature=signature.hex(),
                                                  **commitment)


def verdict(state, store, partitions, auditor, session, log_key, proof):
    """The checker's verdict on ``proof`` saved as a bundle and named by an auditor's challenge."""
    staging = store.parent / 'staging'
    granite_audit.save_proof(proof, staging)
    root = granite_audit.bundle_root(staging)
    if (store / root).exists():
        shutil.rmtree(staging)
    else:
        shutil.move(str(staging), str(store / root))
    return granite_audit.challenge_checker(store, partitions)(*proof_request(state, auditor, session, log_key, root))


def proof_request(state, auditor, session, log_key, root):
    """The next block's state with an auditor's challenge naming ``root`` open, and what a proof of it must establish.

    Neither transaction is kept: the auditor's nonce is restored.
    """
    opening = auditor.sign('challenge', job_id=session['job_id'], log_key=log_key, proof_root=root)
    opened = ledger.transition(state, opening, None)
    opened = ledger.advance(opened, opened['height'] + 1)
    request = ledger.admit(opened, auditor.sign('prove', job_id=session['job_id'],
                                                challenge_id=ledger.transaction_id(opening)))[2]
    auditor.nonce -= 2
    return opened, request


def committed(market, session, records):
    for rank in (1, 2):
        market['state'] = ledger.transition(market['state'], commit(market, rank, session, records[rank][0]), None)
    return market['state']


def test_honest_bound_logs_commit_and_no_claim_against_them_verifies(partitions, market, tmp_path):
    session = market['open']('cd' * 32)
    logs, links = serve(partitions, session, market['keys'])
    records = saved(logs, market['keys'], session, tmp_path)
    state = committed(market, session, records)
    assert state['jobs'][session['job_id']]['deadline'] is not None
    store, auditor = tmp_path / 'store', market['people']['auditor']
    store.mkdir()
    for rank, (record, payloads) in records.items():
        # The trailing arm switch changes no output, so the bound log ends at the last signed message.
        assert len(record['entries']) == len(PLAN) - 1 == record['upstream']['entries']
        assert not granite_audit.unattested(record) and granite_audit.signed_by(record, market['public'][rank])
        assert granite_audit.replay(partitions[rank], record, payloads)['valid']
        forwards = [i for i, entry in enumerate(record['entries']) if 'output' in entry]
        framing = {'record': record, 'mismatch': forwards[0], 'inputs': {i: p for i, p in payloads.items() if i <= forwards[0]}}
        assert verdict(state, store, partitions, auditor, session, market['public'][rank], framing) is False
        assert verdict(state, store, partitions, auditor, session, market['public'][rank],
                       granite_audit.claim_proof(record, 'unattested')) is False
    # What each owner signed passing results on is exactly what its log says it sent.
    for rank, held in ((1, records[2][0]['upstream']), (2, links[0].received)):
        assert not granite_audit.equivocates(records[rank][0], held)
        assert verdict(state, store, partitions, auditor, session, market['public'][rank],
                       granite_audit.claim_proof(records[rank][0], 'equivocation', held)) is False


def test_settlement_pays_each_owner_for_the_positions_its_signed_log_covers(partitions, market, tmp_path):
    whole = market['open']('cd' * 32)
    records = saved(serve(partitions, whole, market['keys'])[0], market['keys'], whole, tmp_path / 'whole')
    for record, _ in records.values():
        assert granite_audit.positions(record) == record['upstream']['positions'] == SERVED
    state = market['state'] = settle(committed(market, whole, records), whole['job_id'])
    assert state['results'][whole['job_id']] == {**state['results'][whole['job_id']], 'positions': [SERVED, SERVED],
                                                 'paid': [500_000, 500_000], 'refunded': 0}
    # Another job served in full, but each owner commits only the prefix its first upstream signature covers.
    prefix, history = market['open']('ef' * 32), {}
    logs, _ = serve(partitions, prefix, market['keys'], history=history)
    for rank in (1, 2):
        logs[rank].attested = history[rank][0]
    records = saved(logs, market['keys'], prefix, tmp_path / 'prefix')
    first = PLAN[1][1]
    for record, _ in records.values():
        assert granite_audit.positions(record) == record['upstream']['positions'] == first
        assert not granite_audit.unattested(record)
    state = settle(committed(market, prefix, records), prefix['job_id'])
    paid = 1_000_000 * first // (SERVED * 2)
    assert state['results'][prefix['job_id']] == {**state['results'][prefix['job_id']], 'positions': [first, first],
                                                  'paid': [paid, paid], 'refunded': 1_000_000 - 2 * paid}


def test_an_owner_that_signs_a_count_its_log_contradicts_is_proven_by_its_own_signature(partitions, market, tmp_path):
    session = market['open']('cd' * 32)
    records = saved(serve(partitions, session, market['keys'])[0], market['keys'], session, tmp_path)
    record, held = records[1][0], records[2][0]['upstream']
    assert held['positions'] == SERVED and not granite_audit.equivocates(record, held)
    inflated = {**held, 'positions': SERVED + 1, 'signature': market['keys'][1].sign(ledger.link_message(
        CHAIN, session['job_id'], 1, held['entries'], held['head'], SERVED + 1)).hex()}
    assert granite_audit.equivocates(record, inflated)


def test_a_wrong_output_is_proven_by_replay_even_with_a_broken_record_signature(partitions, market, tmp_path):
    session = market['open']('cd' * 32)
    logs, _ = serve(partitions, session, market['keys'], fault=(1, 3))
    records = saved(logs, market['keys'], session, tmp_path)
    state = committed(market, session, records)
    record, payloads = records[1]
    report = granite_audit.replay(partitions[1], record, payloads)
    forwards = [i for i, entry in enumerate(record['entries']) if 'output' in entry]
    assert not report['valid'] and report['first_mismatch'] == forwards[3]
    store = tmp_path / 'store'
    store.mkdir()
    proof = granite_audit.fraud_proof(record, payloads, report)
    auditor = market['people']['auditor']
    assert verdict(state, store, partitions, auditor, session, market['public'][1], proof) is True
    # The commitment signs the statement, so an owner cannot escape by publishing its record with a broken signature.
    broken = {**proof, 'record': {**record, 'signature': '00' * 64}}
    assert verdict(state, store, partitions, auditor, session, market['public'][1], broken) is True
    assert verdict(state, store, partitions, auditor, session, market['public'][2],
                   {**proof, 'record': records[2][0]}) is False


def test_an_old_log_is_refused_and_relabelled_old_work_is_provable_without_replay(partitions, market, tmp_path):
    first = market['open']('cd' * 32)
    old = saved(serve(partitions, first, market['keys'])[0], market['keys'], first, tmp_path / 'old')
    committed(market, first, old)
    session = market['open']('cd' * 32)
    fresh = saved(serve(partitions, session, market['keys'], seed=6)[0], market['keys'], session, tmp_path / 'new')
    state, record = market['state'], old[1][0]
    with pytest.raises(ValueError, match='not bound to this job request'):
        ledger.transition(state, commit(market, 1, session, record), None)
    market['people']['owner-1'].nonce -= 1
    # Relabelled with this job's session, the old work still carries the old session's signature.
    relabelled = {**record, 'session': fresh[1][0]['session']}
    with pytest.raises(ValueError, match='not signed by their upstream sender'):
        ledger.transition(state, commit(market, 1, session, relabelled), None)
    market['people']['owner-1'].nonce -= 1
    # With this session's real signature over old entries, the commitment is admitted, and provably fraud.
    swapped = granite_audit.bind(1, record['entries'], record['inputs_sha256'], market['keys'][1],
                                 fresh[1][0]['session'], fresh[1][0]['upstream'])
    assert granite_audit.unattested(swapped)
    state = ledger.transition(state, commit(market, 1, session, swapped), None)
    state = ledger.transition(state, commit(market, 2, session, fresh[2][0]), None)
    store = tmp_path / 'store'
    store.mkdir()
    proof = granite_audit.claim_proof(swapped, 'unattested')
    # No shard is needed: the claim is judged by hashing the log.
    assert verdict(state, store, {}, market['people']['auditor'], session, market['public'][1], proof) is True
    (root,) = [path.name for path in store.iterdir()]
    auditor = market['people']['auditor']
    opening = auditor.sign('challenge', job_id=session['job_id'], log_key=market['public'][1], proof_root=root)
    state = ledger.transition(state, opening, None)
    state = ledger.advance(state, state['height'] + 1)
    proven = auditor.sign('prove', job_id=session['job_id'], challenge_id=ledger.transaction_id(opening))
    state = ledger.transition(state, proven, granite_audit.challenge_checker(store, {}))
    assert state['results'][session['job_id']]['status'] == 'fraud'
    assert state['owners'][market['public'][1]]['status'] == 'slashed'


def corrected(partition, record, payloads):
    """The log a cheating owner would like to commit: every output replaced by what its shard computes."""
    from transformers import DynamicCache

    entries, cache, last = [dict(entry) for entry in record['entries']], DynamicCache(), partition.rank == 2
    for index, entry in enumerate(entries):
        if entry['op'] == RESET:
            cache = DynamicCache()
        elif entry['op'] == CROP:
            cache.crop(entry['value'])
        elif entry['op'] in (FORWARD, FEATURE):
            with torch.inference_mode():
                if entry['op'] == FORWARD:
                    mask = torch.ones((1, cache.get_seq_length() + entry['value']), dtype=torch.long)
                    out = partition(payloads[index], mask, cache)
                else:
                    out = partition(payloads[index], None, DynamicCache())
            entry['output'] = granite_audit.digest(out[:, -1:] if last else out)
    return entries


@pytest.mark.parametrize('rank', [1, 2])
def test_an_owner_that_sent_other_outputs_than_its_log_shows_is_proven_by_its_own_signature(partitions, market, tmp_path,
                                                                                            rank):
    session = market['open']('cd' * 32)
    logs, links = serve(partitions, session, market['keys'], fault=(rank, 2))
    records = saved(logs, market['keys'], session, tmp_path)
    record, payloads = records[rank]
    # It sent a wrong output but commits a log of correct computation on the inputs it was sent.
    clean = granite_audit.bind(rank, corrected(partitions[rank], record, payloads), record['inputs_sha256'],
                               market['keys'][rank], session, record['upstream'])
    assert granite_audit.replay(partitions[rank], clean, payloads)['valid'] and not granite_audit.unattested(clean)
    state = market['state']
    for owner in (1, 2):
        state = ledger.transition(state, commit(market, owner, session, clean if owner == rank else records[owner][0]),
                                  None)
    # Owner 1's signature reaches owner 2's committed header; the last owner's reaches the user's device.
    held = records[2][0]['upstream'] if rank == 1 else links[0].received
    assert granite_audit.equivocates(clean, held)
    store = tmp_path / 'store'
    store.mkdir()
    proof = granite_audit.claim_proof(clean, 'equivocation', held)
    assert verdict(state, store, partitions, market['people']['auditor'], session, market['public'][rank], proof) is True
    # An attestation the owner never signed proves nothing.
    forged = {**held, 'signature': market['keys'][0].sign(b'other').hex()}
    assert verdict(state, store, partitions, market['people']['auditor'], session, market['public'][rank],
                   granite_audit.claim_proof(clean, 'equivocation', forged)) is False


def test_an_upstream_signature_is_checked_before_a_message_is_used(market):
    session = {'chain_id': CHAIN, 'job_id': '11' * 32, 'request_root': 'cd' * 32}
    keys, public = market['keys'], market['public']
    sender = granite_audit.Link(session, 0, 3, keys[0], public[2])
    receiver = granite_audit.Link(session, 1, 3, keys[1], public[0])
    message = torch.ones((1, 2, 32), dtype=torch.bfloat16)
    signature = sender.send(FORWARD, 2, message)
    head = ledger.link_step(ledger.link_seed(CHAIN, session['job_id'], 0), FORWARD, 2, granite_audit.digest(message))
    miscounted = keys[0].sign(ledger.link_message(CHAIN, session['job_id'], 0, 1, head, 3))
    for tensor, signed in ((message * 2, signature), (message, keys[2].sign(b'x')),
                           (message, granite_audit.Link({**session, 'job_id': '22' * 32}, 0, 3, keys[0], public[2])
                            .send(FORWARD, 2, message)), (message, miscounted)):
        with pytest.raises(ValueError, match='did not sign'):
            receiver.receive(FORWARD, 2, tensor, signed)
    assert receiver.receive(FORWARD, 2, message, signature)['entries'] == 1


def test_a_bundle_is_judged_only_from_bytes_matching_its_address(partitions, market, tmp_path):
    session = market['open']('cd' * 32)
    logs, _ = serve(partitions, session, market['keys'], fault=(1, 1))
    records = saved(logs, market['keys'], session, tmp_path)
    state = committed(market, session, records)
    record, payloads = records[1]
    proof = granite_audit.fraud_proof(record, payloads, granite_audit.replay(partitions[1], record, payloads))
    staging, store = tmp_path / 'staging', tmp_path / 'store'
    granite_audit.save_proof(proof, staging)
    root = granite_audit.bundle_root(staging)
    state, request = proof_request(state, market['people']['auditor'], session, market['public'][1], root)
    check = granite_audit.challenge_checker(store, partitions)
    with pytest.raises(ledger.ProofUnavailable):
        check(state, request)
    shutil.copytree(staging, store / root)
    (store / root / 'record.json').unlink()
    with pytest.raises(ledger.ProofUnavailable):
        check(state, request)
    shutil.copy(staging / 'record.json', store / root / 'record.json')
    (store / root / 'proof.json').write_text('{}')
    with pytest.raises(ledger.ProofUnavailable):
        check(state, request)
    with pytest.raises(ledger.ProofUnavailable):
        granite_audit.challenge_checker(None, partitions)(state, request)
    shutil.copy(staging / 'proof.json', store / root / 'proof.json')
    # Held in full, the proof needs the shard to replay: without it this validator has no verdict.
    with pytest.raises(ledger.NoVerdict) as raised:
        granite_audit.challenge_checker(store, {})(state, request)
    assert type(raised.value) is ledger.NoVerdict
    assert check(state, request) is True


def test_malformed_bundles_and_logs_fail_the_same_way_everywhere(partitions, market, tmp_path):
    session = market['open']('cd' * 32)
    records = saved(serve(partitions, session, market['keys'], fault=(1, 1))[0], market['keys'], session, tmp_path)
    state = committed(market, session, records)
    record, payloads = records[1]
    store, auditor = tmp_path / 'store', market['people']['auditor']
    store.mkdir()
    forwards = [i for i, entry in enumerate(record['entries']) if 'output' in entry]
    proof = {'record': record, 'mismatch': forwards[1], 'inputs': {i: p for i, p in payloads.items() if i <= forwards[1]}}
    assert verdict(state, store, partitions, auditor, session, market['public'][1], proof) is True
    for broken in ({**proof, 'mismatch': len(record['entries'])}, {**proof, 'mismatch': -1},
                   {**proof, 'inputs': {i: p for i, p in payloads.items() if i < forwards[1]}},
                   {**proof, 'inputs': {**proof['inputs'], forwards[1]: payloads[forwards[0]].clone()}},
                   {**proof, 'claim': 'nonsense'}):
        assert verdict(state, store, partitions, auditor, session, market['public'][1], broken) is False
    # A bundle at a valid address whose contents are not a proof at all.
    staging = tmp_path / 'garbage'
    staging.mkdir()
    for name, data in (('proof.json', b'[1, 2'), ('record.json', b'{}'), ('inputs.safetensors', b'\x00')):
        (staging / name).write_bytes(data)
    root = granite_audit.bundle_root(staging)
    shutil.move(str(staging), str(store / root))
    assert granite_audit.challenge_checker(store, partitions)(
        *proof_request(state, auditor, session, market['public'][1], root)) is False
    # Replay checks each logged command against what serving could have sent.
    for entries in ([{'op': CROP, 'value': 3}], [{'op': ADAPTER, 'value': 2}], [{'op': 99, 'value': 0}],
                    [{'op': FORWARD, 'value': 10 ** 9, 'input': 'x', 'output': 'y'}], [{'op': True, 'value': 0}]):
        with pytest.raises(ValueError):
            granite_audit.replay(partitions[1], {**record, 'entries': entries}, payloads)
    misshapen = {**record, 'entries': [{**record['entries'][forwards[0]], 'value': 2}]}
    assert granite_audit.replay(partitions[1], misshapen, {0: payloads[forwards[0]]})['input_mismatch'] == 0


def test_unbound_logs_keep_their_format_for_standalone_audits(tmp_path):
    log = granite_audit.OwnerLog(1)
    message = torch.ones((1, 2, 32), dtype=torch.bfloat16)
    log.forward(FORWARD, 2, message, message)
    key = Ed25519PrivateKey.generate()
    log.save(tmp_path / 'log', key)
    record, _ = granite_audit.load(tmp_path / 'log')
    assert set(record) == {'format', 'rank', 'entries', 'inputs_sha256', 'public_key', 'signature'}
    assert record['format'] == granite_audit.LOG_FORMAT and granite_audit.signed_by(record, record['public_key'])
    manifest = granite_audit.save_proof({'record': record, 'mismatch': 0, 'inputs': {0: message}}, tmp_path / 'proof')
    assert manifest['format'] == granite_audit.PROOF_FORMAT and 'claim' not in manifest
    assert json.loads((tmp_path / 'proof' / 'proof.json').read_text()) == manifest
