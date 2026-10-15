import json
from pathlib import Path

from neuroshard.evolution import assistant_chat_data as chat
from neuroshard.evolution import assistant_documents_data as docs
from neuroshard.evolution import assistant_documents_demonstration as demo
from neuroshard.evolution import sealed_evaluation as sealed
from neuroshard.inference import beacon, growth, optimistic as ledger
from neuroshard.evolution.modular_reference_execution import ROOT, read

ROUNDS = json.loads((Path(__file__).with_name('drand_quicknet_rounds.json')).read_text())


def test_recorded_beacons_verify_and_forgeries_do_not():
    first, second = ROUNDS
    assert beacon.verify(first['round'], first['signature'], first['randomness'])
    assert beacon.verify(second['round'], second['signature'])
    assert not beacon.verify(second['round'], first['signature'])
    assert not beacon.verify(first['round'], first['signature'], second['randomness'])
    assert not beacon.verify(first['round'], '00' * 48)
    assert beacon.round_at(beacon.time_of(first['round'])) == first['round']
    assert beacon.round_at(beacon.time_of(first['round']) + 1) == first['round'] + 1


def test_a_sealed_set_is_fixed_by_its_beacon_new_every_round_and_solvable():
    first, second = ROUNDS
    a = sealed.sealed(first['round'], first['signature'], first['randomness'])
    assert a == sealed.sealed(first['round'], first['signature'], first['randomness'])
    b = sealed.sealed(second['round'], second['signature'], second['randomness'])
    assert a['sha256'] != b['sha256'] and len(a['sets']['documents']) == 132 and len(a['sets']['chat']) == 96
    policy = read(ROOT / 'config/experiments/assistant-general-policy-documents5.json')
    for case in a['sets']['documents'][::11]:
        assert demo.demonstration({**case, 'split': 'generated-train'}, policy)['passed']
    import pytest
    with pytest.raises(ValueError, match='does not verify'):
        sealed.sealed(first['round'], second['signature'], first['randomness'])


def test_sealed_cases_share_no_workspace_with_any_published_split():
    first = ROUNDS[0]
    cases = sealed.sealed(first['round'], first['signature'], first['randomness'])['sets']
    worlds = {json.dumps(c['world'], sort_keys=True) for c in cases['documents']}
    for split in docs.SPLITS:
        published = {json.dumps(c['world'], sort_keys=True) for c in docs.cases(split)}
        assert not worlds & published, split
    texts = {c['turns'][0]['user'] for c in cases['chat']}
    for split in ('confirmation7', 'generated-train', 'generated-development5'):
        assert len(texts & {c['turns'][0]['user'] for c in chat.cases(split)}) <= 2


def test_a_settled_job_names_a_seal_round_after_its_settlement_time():
    from test_optimistic_serving import CHAIN, PARAMS, Account

    sponsor, worker = Account('s'), Account('w')
    state = ledger.genesis(CHAIN, 'ab' * 32, 3, {sponsor.public: 40_000_000, worker.public: 40_000_000}, PARAMS)
    opened = sponsor.sign('growth_open', recipe_root='cd' * 32, start_root='ef' * 32, steps=1, segment_steps=1,
                          reward=1_000_000)
    job = ledger.transaction_id(opened)
    state = ledger.transition(state, opened, None)
    state = ledger.transition(state, worker.sign('growth_commit', job_id=job, segment=0, step_roots=['01' * 32]), None)
    unix = 1_800_000_000
    for _ in range(growth.PARAMS['growth_window_blocks'] + 1):
        unix += 1
        state = ledger.advance(state, state['height'] + 1, unix)
    result = state['results'][job]
    assert result['settled_unix'] == unix and beacon.time_of(result['seal_round']) > unix
    assert result['seal_round'] == beacon.round_at(unix) + growth.SEAL_MARGIN
