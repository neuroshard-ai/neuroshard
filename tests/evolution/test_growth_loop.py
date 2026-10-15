"""The whole growth loop without a person: agents with their own keys open, train, check, seal, evaluate, propose,
vote and enact, and one of them cheats and is caught.

The ledger is the real state machine (``optimistic.transition``/``advance``), driven block by block with BFT
times; training is the real protocol trainer (``growth_replay``) on a tiny seeded Granite model; the beacon is
real drand quicknet (recorded rounds, verified signatures); the evaluation set is derived from it.
"""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from neuroshard.evolution import growth_agent, growth_replay
from neuroshard.evolution import sealed_evaluation
from neuroshard.inference import beacon, growth as ledger_growth, optimistic as ledger
from neuroshard.inference.optimistic_app import promotion_or_fraud
from test_network_growth import RECIPE, tiny
from test_owner_governance import bond_at
from test_optimistic_serving import CHAIN, PARAMS, Account

BASE = {'vocab_size': 64, 'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 2, 'num_attention_heads': 4,
        'num_key_value_heads': 2, 'tie_word_embeddings': True, 'eos_token_id': 1, 'pad_token_id': 0, 'logits_scaling': 2.0}
# Two consecutive real quicknet rounds, recorded once (verified in the test, so nothing here is trusted).
ROUNDS = Path(__file__).with_name('drand_quicknet_rounds.json')


class Chain:
    """The real state machine behind a submit/state interface, one block per submitted transaction."""

    def __init__(self, state, checker, start_unix):
        self.state_value, self.checker, self.unix = state, checker, start_unix

    def state(self):
        return json.loads(json.dumps(self.state_value))

    def block(self, seconds=1):
        self.unix += seconds
        self.state_value = ledger.advance(self.state_value, self.state_value['height'] + 1, self.unix)

    def submit(self, account, kind, **fields):
        account.nonce = self.state_value['accounts'].get(account.public, {'nonce': 0})['nonce']
        envelope = account.sign(kind, **fields)
        self.block()
        self.state_value = ledger.transition(self.state_value, envelope, self.checker)
        return envelope


class Trainer:
    """A contributor's training and a checker's replay, on the real protocol trainer. ``cheat`` corrupts roots."""

    def __init__(self, store, cheat=False):
        self.store, self.cheat = Path(store), cheat

    def _state(self, recipe, job, index):
        model = tiny()
        if index == 0:
            return model, *growth_replay.start(model, recipe)
        return model, *growth_replay.load(model, recipe, (self.store / f'{index - 1}.state').read_bytes())

    def segment(self, recipe, job_id, start, first, last):
        model = tiny()
        index = first // (last - first) if first else 0
        trainable, optimizer = (growth_replay.start(model, recipe) if first == 0 else
                                growth_replay.load(model, recipe, (self.store / f'{index - 1}.state').read_bytes()))
        assert growth_replay.state_root(trainable, optimizer) == start
        roots = growth_replay.train_segment(model, trainable, optimizer, recipe, first, last)
        self.store.mkdir(parents=True, exist_ok=True)
        (self.store / f'{index}.state').write_bytes(growth_replay.save(trainable, optimizer))
        return ['ab' * 32] * len(roots) if self.cheat else roots

    def replay(self, recipe, job, index, step):
        model, trainable, optimizer = self._state(recipe, job, index)
        first = index * job['segment_steps']
        if step:
            growth_replay.train_segment(model, trainable, optimizer, recipe, first, first + step)
        bundle = growth_replay.bundle(growth_replay.save(trainable, optimizer), first + step)
        growth_replay.step(model, trainable, optimizer, recipe, first + step)
        return growth_replay.state_root(trainable, optimizer), bundle


class Evaluator:
    """Runs a grown unit on the sealed cases; here a deterministic stand-in that passes the declared bar."""

    def evaluate(self, result, cases):
        return {'job_id_final_root': result['final_root'], 'cases_sha256': cases['sha256'], 'round': cases['round'],
                'documents': len(cases['sets']['documents']), 'chat': len(cases['sets']['chat']), 'passed': True,
                'model_root': hashlib.sha256(('grown:' + result['final_root']).encode()).hexdigest()}


@pytest.mark.skipif(not ROUNDS.exists(), reason='recorded drand rounds missing')
def test_agents_grow_seal_propose_vote_and_enact_with_a_cheater_caught(tmp_path):
    torch.set_num_threads(1)
    rounds = json.loads(ROUNDS.read_text())
    seal = rounds[0]
    assert beacon.verify(seal['round'], seal['signature'], seal['randomness'])
    steward, sponsor, honest, cheater, checker, owner1, owner2 = (Account(n) for n in (
        'steward', 'sponsor', 'honest', 'cheater', 'checker', 'owner1', 'owner2'))
    state = ledger.genesis(CHAIN, 'ab' * 32, 3, {a.public: 40_000_000 for a in (
        steward, sponsor, honest, cheater, checker, owner1, owner2)}, PARAMS)
    recipe = {**RECIPE, 'base': 'tiny-granite', 'base_config': BASE, 'base_seed': 0, 'model_root': 'ab' * 32}
    root = growth_replay.recipe_root(recipe, recipe['model_root'])
    store = tmp_path / 'store'
    (store / 'recipes').mkdir(parents=True)
    (store / 'recipes' / f'{root}.json').write_text(json.dumps(recipe))
    (store / 'bundles').mkdir()
    checker_fn = promotion_or_fraud({'promotion_authority': steward.public, 'approved_models': {},
                                     'growth_recipes': str(store / 'recipes'), 'bundles': str(store / 'bundles')}, None)
    # Seal round must come after settlement: put the chain clock just before the recorded seal round.
    chain = Chain(state, checker_fn, beacon.time_of(seal['round']) - ledger_growth.SEAL_MARGIN * beacon.PERIOD - 200)
    keys = {}
    for account, shard, name in ((owner1, 1, 'owner1-log'), (owner2, 2, 'owner2-log')):
        key, envelope = bond_at(account, shard, name, 5_000_000, chain.state_value)
        chain.block()
        chain.state_value = ledger.transition(chain.state_value, envelope, checker_fn)
        keys[key] = account
    chain.submit(steward, 'governance_handover', approval_ppm=666_667, vote_blocks=400)

    model = tiny()
    trainable, optimizer = growth_replay.start(model, recipe)
    opened = chain.submit(sponsor, 'growth_open', recipe_root=root, start_root=growth_replay.state_root(trainable, optimizer),
                          steps=6, segment_steps=3, reward=6_000_000)
    job_id = ledger.transaction_id(opened)

    def deliver(proof_root, payload):
        (store / 'bundles' / proof_root).write_bytes(payload)

    shared = tmp_path / 'shared-states'  # committed segment states are public bytes (any contributor can fetch them)
    cheat = growth_agent.Agent(chain, cheater, tmp_path / 'cheater', recipes={root: recipe},
                               trainer=Trainer(shared, cheat=True))
    good = growth_agent.Agent(chain, honest, tmp_path / 'honest', recipes={root: recipe}, trainer=Trainer(shared))
    watcher = growth_agent.Agent(chain, checker, tmp_path / 'checker', recipes={root: recipe}, trainer=Trainer(shared),
                                 deliver=deliver, rng=__import__('random').Random(0),
                                 evaluator=Evaluator(), beacon=lambda r: (seal['round'], seal['signature'],
                                                                           seal['randomness']),
                                 clock=lambda: chain.unix)
    voters = growth_agent.Agent(chain, owner1, tmp_path / 'owners', owner_keys=keys, evaluator=Evaluator(),
                                beacon=lambda r: (seal['round'], seal['signature'], seal['randomness']),
                                clock=lambda: chain.unix)

    assert [a['kind'] for a in cheat.tick()] == ['commit']           # the cheater commits segment 0, wrong
    caught = watcher.tick()                                             # the checker replays one step and challenges
    assert [a['kind'] for a in caught] == ['challenge']
    chain.block()
    # Proven: the cheater's bond is slashed and the segment reopens; the checker, a contributor too, retrains it.
    assert [a['kind'] for a in watcher.tick()] == ['prove', 'commit']
    s = chain.state()
    assert s['growth_faults'][-1]['contributor'] == cheater.public and len(s['growth'][job_id]['commits']) == 1
    assert s['growth'][job_id]['commits'][0]['contributor'] == checker.public
    # The honest contributor checks the checker's segment and trains the last one.
    assert [a['kind'] for a in good.tick()] == ['checked', 'commit']
    assert [a['kind'] for a in watcher.tick()] == ['checked']
    for _ in range(ledger_growth.PARAMS['growth_window_blocks'] + 2):
        chain.block()
    result = chain.state()['results'][job_id]
    assert result['status'] == 'grown' and result['seal_round'] >= beacon.round_at(result['settled_unix'])

    chain.unix = beacon.time_of(seal['round']) + 1                      # the seal round now exists
    chain.state_value['results'][job_id]['seal_round'] = seal['round']   # recorded beacon stands in for the live one
    actions = watcher.tick()
    assert [a['kind'] for a in actions] == ['evaluated', 'propose']
    proposal_id, proposal = next(iter(chain.state()['governance']['proposals'].items()))
    assert [a['kind'] for a in voters.tick()] == ['vote']               # owners recompute the sealed result themselves
    assert voters.memory['actions'][-1]['approve'] is True
    chain.submit(sponsor, 'model_enact', proposal_id=proposal_id)
    final = chain.state()
    assert final['model_root'] == proposal['model_root'] and final['model_history'][-1]['authority'] == 'owners'
    ledger.invariant(final)
    # Each honest contributor was paid for the segment it trained; the cheater lost its bond.
    assert final['accounts'][honest.public]['balance'] > 40_000_000 + 2_900_000
    assert final['accounts'][cheater.public]['balance'] < 40_000_000 - ledger_growth.PARAMS['growth_bond'] + 1
    sealed = sealed_evaluation.sealed(seal['round'], seal['signature'], seal['randomness'])
    assert json.loads((tmp_path / 'checker' / f'sealed-{job_id}.json').read_text())['cases_sha256'] == sealed['sha256']



def test_owners_vote_no_when_their_own_sealed_evaluation_disagrees(tmp_path):
    rounds = json.loads(ROUNDS.read_text())
    seal = rounds[0]

    class Lying(Evaluator):
        def evaluate(self, result, cases):
            return {**super().evaluate(result, cases), 'passed': True, 'note': 'inflated'}

    class Chainless:
        def state(self):
            return {}

    owner = Account('owner-x')
    agent = growth_agent.Agent(Chainless(), owner, tmp_path, owner_keys={'k': owner}, evaluator=Evaluator(),
                               beacon=lambda r: (seal['round'], seal['signature'], seal['randomness']),
                               clock=lambda: beacon.time_of(seal['round']) + 1)
    result = {'status': 'grown', 'final_root': 'cd' * 32, 'seal_round': seal['round']}
    honest_digest, honest = agent.sealed_report(result)
    lying = Lying().evaluate(result, sealed_evaluation.sealed(seal['round'], seal['signature'], seal['randomness']))
    lying_digest = hashlib.sha256(json.dumps(lying, sort_keys=True).encode()).hexdigest()
    state = {'results': {'j': result}}
    assert agent.verdict({'model_root': honest['model_root'], 'quality_root': honest_digest}, state) is True
    assert agent.verdict({'model_root': honest['model_root'], 'quality_root': lying_digest}, state) is False
    early = growth_agent.Agent(Chainless(), owner, tmp_path / 'early', owner_keys={'k': owner}, evaluator=Evaluator(),
                               beacon=lambda r: (seal['round'], seal['signature'], seal['randomness']),
                               clock=lambda: beacon.time_of(seal['round']) - 10)
    assert early.verdict({'model_root': honest['model_root'], 'quality_root': honest_digest}, state) is None


def test_state_trainers_share_committed_states_through_any_store_and_refuse_tampered_bytes(tmp_path):
    import http.server
    import threading

    from neuroshard.evolution.growth_runtime import StateTrainer

    recipe = {**RECIPE, 'base': 'tiny-granite', 'base_config': BASE, 'base_seed': 0, 'model_root': 'ab' * 32}
    first = StateTrainer(tmp_path / 'a', tiny)
    model = tiny()
    start = growth_replay.state_root(*growth_replay.start(model, recipe))
    roots = first.segment(recipe, 'job', start, 0, 3)
    served = tmp_path / 'store'
    served.mkdir()
    (served / f'{roots[-1]}.state').write_bytes(first.path(roots[-1]).read_bytes())
    bogus = growth_replay.save(*growth_replay.start(tiny(), recipe))
    (served / f'{"ee" * 32}.state').write_bytes(bogus)

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(served), **k)

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Quiet)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f'http://127.0.0.1:{server.server_address[1]}'
        second = StateTrainer(tmp_path / 'b', tiny, [url])
        more = second.segment(recipe, 'job', roots[-1], 3, 6)
        job = {'start_root': start, 'segment_steps': 3, 'commits': [{'step_roots': roots}, {'step_roots': more}]}
        assert second.replay(recipe, job, 1, 2)[0] == more[2]       # a checker recomputes a step of segment 1
        import pytest
        with pytest.raises(FileNotFoundError):
            StateTrainer(tmp_path / 'c', tiny, [url]).state(recipe, 'ee' * 32)  # bytes not hashing to the root
    finally:
        server.shutdown()
