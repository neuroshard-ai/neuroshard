import pytest

from neuroshard.inference import optimistic as ledger
from neuroshard.inference.optimistic_app import promotion_or_fraud
from test_optimistic_serving import CHAIN, PARAMS, Account


def test_promotion_requires_the_authority_and_approved_quality_and_preserves_supply():
    authority, outsider = Account('authority'), Account('outsider')
    initial = ledger.genesis(CHAIN, 'ab' * 32, 3,
                             {authority.public: 20_000_000, outsider.public: 20_000_000}, PARAMS)
    checker = promotion_or_fraud({'promotion_authority': authority.public,
                                 'approved_models': {'cd' * 32: {'quality_root': 'ef' * 32}}}, None)
    with pytest.raises(ValueError, match='not authorized'):
        ledger.transition(initial, outsider.sign('model_promote', model_root='cd' * 32, quality_root='ef' * 32), checker)
    with pytest.raises(ValueError, match='not authorized'):
        ledger.transition(initial, authority.sign('model_promote', model_root='cd' * 32, quality_root='11' * 32), checker)
    authority.nonce = 0
    promoted = ledger.transition(initial, authority.sign('model_promote', model_root='cd' * 32, quality_root='ef' * 32), checker)
    assert promoted['model_root'] == 'cd' * 32 and promoted['model_history'][0]['previous'] == 'ab' * 32
    assert initial['model_root'] == 'ab' * 32 and promoted['initial_supply'] == initial['initial_supply']
    assert promoted['burned'] - initial['burned'] == PARAMS['fee']
    ledger.invariant(promoted)


def test_promotion_cannot_change_the_model_while_a_job_is_open():
    authority = Account('authority')
    state = ledger.genesis(CHAIN, 'ab' * 32, 3, {authority.public: 20_000_000}, PARAMS)
    state['jobs']['open'] = {}
    calls = []
    with pytest.raises(ValueError, match='Drain'):
        ledger.transition(state, authority.sign('model_promote', model_root='cd' * 32, quality_root='ef' * 32),
                          lambda *args: calls.append(args))
    assert calls == []


def test_promotion_through_abci_admission_proposal_and_commit(tmp_path):
    from neuroshard.demo import abci_pb2 as pb
    from neuroshard.inference.optimistic_app import Application

    authority, outsider = Account('authority'), Account('outsider')
    checker = promotion_or_fraud({'promotion_authority': authority.public,
                                 'approved_models': {'cd' * 32: {'quality_root': 'ef' * 32}}}, None)
    app = Application(tmp_path / 'chain.sqlite', checker)
    app.state = ledger.genesis(CHAIN, 'ab' * 32, 3,
                               {authority.public: 20_000_000, outsider.public: 20_000_000}, PARAMS)
    bad = ledger.canonical(outsider.sign('model_promote', model_root='cd' * 32, quality_root='ef' * 32))
    raw = ledger.canonical(authority.sign('model_promote', model_root='cd' * 32, quality_root='ef' * 32))
    try:
        assert app.CheckTx(pb.RequestCheckTx(tx=bad), None).code == 1
        assert app.CheckTx(pb.RequestCheckTx(tx=raw), None).code == 0
        assert list(app.PrepareProposal(pb.RequestPrepareProposal(height=1, txs=[bad, raw]), None).txs) == [raw]
        assert app.ProcessProposal(pb.RequestProcessProposal(height=1, txs=[raw]), None).status == 1
        response = app.FinalizeBlock(pb.RequestFinalizeBlock(height=1, txs=[raw]), None)
        assert response.tx_results[0].code == 0
        app.Commit(pb.RequestCommit(), None)
        assert app.state['model_root'] == 'cd' * 32
        ledger.invariant(app.state)
    finally:
        app.check.worker.shutdown(wait=True)
        app.db.close()
