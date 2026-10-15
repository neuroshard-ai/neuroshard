"""Growth driven by use and by nodes, with no person in the loop.

Users pay for conversations; a governed share of every owner payment fills the growth pool. An agent probes
the serving version on a beacon-sealed set, finds its weak family, and proposes a recipe; owners' agents
re-verify the recipe and vote it into the queue. When usage has paid for it, the chain opens the job
itself. Contributor agents train its segments, the job settles, its evaluation is sealed by a later drand
round, and the version is proposed, recomputed, voted and enacted.

Then the same with more users and more contributor nodes: the pool fills sooner and the job finishes in
fewer blocks.
"""

import json
from pathlib import Path

import torch

from neuroshard.evolution import growth_agent, growth_replay
from neuroshard.inference import beacon, growth as ledger_growth, optimistic as ledger
from neuroshard.inference.optimistic_app import promotion_or_fraud
from test_growth_loop import BASE, ROUNDS, Chain, Evaluator, Trainer, RECIPE, tiny
from test_owner_governance import bond_at
from test_optimistic_serving import CHAIN, PARAMS, Account

REWARD, STEPS, SEGMENT = 3_000_000, 6, 1


def paid_conversations(chain, user, owners, count):
    """``count`` users' served jobs settling in the same block (users act concurrently). Their prices fund the pool."""
    state = chain.state_value
    for index in range(count):
        state['jobs'][f"job-{state['height']}-{index}"] = {
            'user': user.public, 'owners': list(owners), 'request_root': 'aa' * 32, 'session_key': 'bb' * 32,
            'price': 1_000_000, 'positions': 100, 'opened': state['height'], 'expires': state['height'] + 50,
            'commits': {k: 'cc' * 32 for k in owners}, 'served': {k: 100 for k in owners}, 'challenges': {},
            'deadline': state['height']}
        state['accounts'][user.public]['balance'] -= 1_000_000
    chain.block()


def network(tmp_path, contributors):
    rounds = json.loads(ROUNDS.read_text())
    seal = rounds[0]
    steward, proposer, user, owner1, owner2 = (Account(n) for n in ('steward', 'proposer', 'user', 'owner1', 'owner2'))
    workers = [Account(f'worker{i}') for i in range(contributors)]
    accounts = [steward, proposer, user, owner1, owner2, *workers]
    state = ledger.genesis(CHAIN, 'ab' * 32, 3, {a.public: 400_000_000 for a in accounts}, PARAMS)
    recipe = {**RECIPE, 'base': 'tiny-granite', 'base_config': BASE, 'base_seed': 0, 'model_root': 'ab' * 32}
    root = growth_replay.recipe_root(recipe, recipe['model_root'])
    store = tmp_path / 'store'
    (store / 'recipes').mkdir(parents=True)
    (store / 'recipes' / f'{root}.json').write_text(json.dumps(recipe))
    (store / 'bundles').mkdir()
    checker = promotion_or_fraud({'promotion_authority': steward.public, 'approved_models': {},
                                  'growth_recipes': str(store / 'recipes'), 'bundles': str(store / 'bundles')}, None)
    chain = Chain(state, checker, beacon.time_of(seal['round']) - 2000)
    keys = {}
    for account, shard, name in ((owner1, 1, 'owner1-log'), (owner2, 2, 'owner2-log')):
        key, envelope = bond_at(account, shard, name, 5_000_000, chain.state_value)
        chain.block()
        chain.state_value = ledger.transition(chain.state_value, envelope, checker)
        keys[key] = account
    chain.submit(steward, 'governance_handover', approval_ppm=666_667, vote_blocks=400)
    # The probe found three weak families: one recipe each (the data differs per family, here by seed).
    planned, roots = [], []
    for index, family in enumerate(('claim-deadline', 'not-found-return', 'invoice-sum')):
        variant = {**recipe, 'spec': {**recipe['spec'], 'seed': recipe['spec']['seed'] + index}}
        variant_start = growth_replay.state_root(*growth_replay.start(tiny(), variant))
        planned.append({**variant, 'start_root': variant_start, 'steps': STEPS, 'segment_steps': SEGMENT,
                        'reward': REWARD, 'families': [family]})
        roots.append(growth_replay.recipe_root(variant, variant['model_root']))
        (store / 'recipes' / f'{roots[-1]}.json').write_text(json.dumps(variant))
    fixed = lambda r: (seal['round'], seal['signature'], seal['randomness'])  # noqa: E731
    shared = tmp_path / 'states'
    planner_agent = growth_agent.Agent(chain, proposer, tmp_path / 'planner', planner=lambda randomness: planned,
                                       beacon=fixed, clock=lambda: chain.unix)
    owners = growth_agent.Agent(chain, owner1, tmp_path / 'owners', owner_keys=keys, evaluator=Evaluator(),
                                beacon=fixed, clock=lambda: chain.unix)
    for item, item_root in zip(planned, roots):
        owners.recipes[item_root] = item
    owners.recipe_check = lambda r: r['sources'] == recipe['sources']  # re-verifies the data it was proposed with
    by_root = {item_root: {k: v for k, v in item.items() if k not in ('start_root', 'steps', 'segment_steps', 'reward',
                                                                      'families')} for item, item_root in zip(planned, roots)}
    workers = [growth_agent.Agent(chain, w, tmp_path / f'w{i}', recipes=dict(by_root), trainer=Trainer(shared),
                                  evaluator=Evaluator(), beacon=fixed, clock=lambda: chain.unix,
                                  rng=__import__('random').Random(i)) for i, w in enumerate(workers)]
    return chain, user, keys, planner_agent, owners, workers, roots, seal


def grow(tmp_path, users, contributors):
    torch.set_num_threads(1)
    chain, user, keys, planner, owners, workers, roots, seal = network(tmp_path, contributors)
    # Owners vote a growth levy of 20% of their payments; nothing else is funded by anyone.
    owner1 = owners.owner_keys[next(iter(owners.owner_keys))]
    levy = chain.submit(owner1, 'growth_levy', levy_ppm=200_000)
    lid = ledger.transaction_id(levy)
    for key, account in owners.owner_keys.items():
        chain.submit(account, 'model_vote', proposal_id=lid, log_key=key, approve=True)
    chain.submit(owner1, 'model_enact', proposal_id=lid)
    assert [a['kind'] for a in planner.tick()] == ['recipe'] * 3     # each weak family becomes a recipe
    assert [a['kind'] for a in owners.tick()] == ['vote'] * 3        # owners re-verify them and vote them in
    for proposal_id in list(chain.state()['governance']['proposals']):
        chain.submit(owner1, 'model_enact', proposal_id=proposal_id)
    assert [q['recipe_root'] for q in chain.state()['growth_queue']] == roots and not chain.state().get('growth')
    opened_at = None
    blocks = 0  # rounds: in each, every user and every node acts once
    def grown():
        return [r for r in chain.state_value.get('results', {}).values() if r.get('status') == 'grown']

    while len(grown()) < len(roots):
        # Each block, people use the assistant (usage fills the pool) and every node takes the next segment it can.
        if opened_at is None and chain.state_value.get('growth'):
            opened_at = blocks
        if chain.state_value.get('growth_queue'):
            paid_conversations(chain, user, list(keys), users)
        else:
            chain.block()
        for worker in workers:
            worker.tick()
        blocks += 1
        assert blocks < 5000
    jobs = grown()
    finished = blocks
    chain.unix = max(chain.unix, beacon.time_of(seal['round']) + 1)
    for result in chain.state_value['results'].values():
        if result.get('status') == 'grown':
            result['seal_round'] = seal['round']
    actions = [a['kind'] for worker in workers for a in worker.tick()]
    assert actions.count('propose') == len(roots)                  # each grown unit is evaluated and proposed
    votes = owners.tick()
    assert [a['kind'] for a in votes] == ['vote'] * len(roots) and all(a['approve'] for a in votes)
    enact = next(p for p, v in chain.state()['governance']['proposals'].items() if 'model_root' in v)
    chain.submit(owner1, 'model_enact', proposal_id=enact)
    final = chain.state()
    assert final['model_history'][-1]['authority'] == 'owners' and final['model_root'] != 'ab' * 32
    ledger.invariant(final)
    return {'funded_after_rounds': opened_at, 'trained_in_rounds': finished - opened_at,
            'jobs': jobs, 'contributors_paid': len({c for j in jobs for c in j['contributors']}),
            'pool': final.get('growth_pool', 0)}


def test_usage_funds_growth_and_more_nodes_train_it_faster_with_no_person(tmp_path):
    few = grow(tmp_path / 'few', users=1, contributors=1)
    many = grow(tmp_path / 'many', users=4, contributors=3)
    assert len(few['jobs']) == len(many['jobs']) == 3
    # More users: the pool pays for the queued recipe sooner. More nodes: segments train in parallel.
    assert many['funded_after_rounds'] < few['funded_after_rounds']
    assert many['trained_in_rounds'] < few['trained_in_rounds']
    assert many['contributors_paid'] > few['contributors_paid'] == 1


def test_ledger_rules_for_the_pool_and_the_queue(tmp_path):
    import pytest

    chain, user, keys, planner, owners, workers, roots, seal = network(tmp_path, 1)
    owner1 = next(iter(keys.values()))
    with pytest.raises(ValueError):
        chain.submit(owner1, 'growth_levy', levy_ppm=ledger.MAX_GROWTH_LEVY_PPM + 1)
    before = chain.state()
    paid_conversations(chain, user, list(keys), 2)                # no levy voted yet: owners keep everything
    assert chain.state().get('growth_pool', 0) == 0
    settled = [r for r in chain.state()['results'].values() if r.get('status') == 'settled']
    assert settled and all('pooled' not in r for r in settled)   # records unchanged where no levy applies
    levy = chain.submit(owner1, 'growth_levy', levy_ppm=100_000)
    for key, account in keys.items():
        chain.submit(account, 'model_vote', proposal_id=ledger.transaction_id(levy), log_key=key, approve=True)
    chain.submit(owner1, 'model_enact', proposal_id=ledger.transaction_id(levy))
    paid_conversations(chain, user, list(keys), 3)
    pool = chain.state()['growth_pool']
    assert pool == sum(r['pooled'] for r in chain.state()['results'].values() if 'pooled' in r) > 0
    ledger.invariant(chain.state())                              # the pool is counted in conservation
    assert before['height'] < chain.state()['height']
    # Queue order is admission order, and a job opens only when the pool covers its whole reward.
    planner.tick()
    owners.tick()
    for proposal_id in list(chain.state()['governance']['proposals']):
        chain.submit(owner1, 'model_enact', proposal_id=proposal_id)
    while chain.state_value.get('growth_queue'):
        pool, queued = chain.state_value.get('growth_pool', 0), len(chain.state_value['growth_queue'])
        paid_conversations(chain, user, list(keys), 1)
        opened = queued - len(chain.state_value.get('growth_queue', []))
        assert opened == 0 or pool + 200_000 >= REWARD * opened - 1
    jobs = chain.state_value['growth']
    assert [j['recipe_root'] for j in sorted(jobs.values(), key=lambda j: j['opened'])] == roots
    assert all(j['sponsor'] is None and j['funded_by'] == 'pool' for j in jobs.values())


def test_recipes_travel_through_any_store_and_are_checked_against_the_chain(tmp_path):
    import pytest
    import urllib.error
    import urllib.request

    from neuroshard.assistant import audit, network as client

    recipe = {**RECIPE, 'base': 'tiny-granite', 'base_config': BASE, 'base_seed': 0, 'model_root': 'ab' * 32,
              'spec': {**RECIPE['spec'], 'seed': 99}}
    root = growth_replay.recipe_root(recipe, recipe['model_root'])
    payload = json.dumps(recipe, sort_keys=True).encode()
    store = tmp_path / 'store'
    unnamed = {'governance': {'proposals': {}}, 'jobs': {}}
    named = {'governance': {'proposals': {'p': {'kind': 'recipe', 'recipe_root': root}}}, 'jobs': {}}
    with pytest.raises(ValueError):
        audit.receive_proof(payload, root, unnamed, store)                  # nobody funded a proposal naming it
    with pytest.raises(ValueError, match='content address'):
        audit.receive_proof(payload.replace(b'99', b'98'), root, named, store)
    assert audit.receive_proof(payload, root, named, store) == root

    class Faucet:
        class chain:  # noqa: N801
            @staticmethod
            def state():
                return named

    server = client.serve_faucet(Faucet, '127.0.0.1', 0, store)
    base = f'http://127.0.0.1:{server.server_port}/proofs'
    try:
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f'{base}/../store/{root}.json')
        agent = growth_agent.Agent(None, None, tmp_path / 'agent')
        agent.recipe_stores = ['http://127.0.0.1:1/proofs', base]          # a dead store first: the next serves it
        assert agent.fetch_recipe(root) == json.loads(payload)
        assert agent.fetch_recipe('cd' * 32) is None
    finally:
        server.shutdown()
        server.server_close()
