import copy

import pytest
import torch

from neuroshard.evolution import assistant_experience_train as trainer
from neuroshard.evolution import growth_replay as growth
from neuroshard.inference import growth as ledger_growth
from neuroshard.inference import optimistic as ledger
from test_optimistic_serving import CHAIN, PARAMS, Account

SPEC = {'layers': [0, 1], 'rank': 4, 'alpha': 8, 'seed': 27092026, 'steps': 6, 'gradient_accumulation': 4,
        'experience_per_replay': 3, 'gradient_clip': 1.0, 'weight_decay': 0.01, 'betas': [0.9, 0.999],
        'epsilon': 1e-8, 'warmup_steps': 2, 'learning_rates': {'update': 1e-3, 'addition': 1e-2}}


def tiny():
    from transformers import GraniteConfig, GraniteForCausalLM

    torch.manual_seed(0)
    config = GraniteConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True, eos_token_id=1,
                           pad_token_id=0, logits_scaling=2.0)
    return GraniteForCausalLM(config).eval()


def sequences(seed, count):
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(count):
        ids = torch.randint(2, 64, (12,), generator=generator).tolist()
        rows.append({'input_ids': ids, 'labels': [-100] * 6 + ids[6:]})
    return rows


RECIPE = {'arm': 'addition', 'spec': SPEC, 'sources': {'experience': sequences(1, 6), 'replay': sequences(2, 3)}}


def test_segments_and_one_step_replay_reproduce_the_unchanged_trainer_bit_for_bit():
    torch.set_num_threads(1)
    reference_model = tiny()
    reference, _ = trainer.train(reference_model, 'addition', RECIPE['sources']['experience'],
                                 RECIPE['sources']['replay'], SPEC)
    model = tiny()
    trainable, optimizer = growth.start(model, RECIPE)
    start_root = growth.state_root(trainable, optimizer)
    roots = growth.train_segment(model, trainable, optimizer, RECIPE, 0, 3)
    saved = growth.save(trainable, optimizer)
    # A second contributor continues from the committed state on another machine (here, another model).
    other = tiny()
    resumed, resumed_optimizer = growth.load(other, RECIPE, saved)
    assert growth.state_root(resumed, resumed_optimizer) == roots[-1]
    roots += growth.train_segment(other, resumed, resumed_optimizer, RECIPE, 3, 6)
    assert trainer_digests(resumed) == trainer_digests(reference)
    assert len(set(roots)) == 6 and start_root not in roots

    request = {'step': 3, 'before_root': roots[2], 'claimed_root': roots[3]}
    honest = growth.bundle(saved, 3)
    import hashlib
    assert not growth.judge(tiny, RECIPE, honest, {**request, 'proof_root': hashlib.sha256(honest).hexdigest()})
    forged = {**request, 'claimed_root': 'ab' * 32}
    assert growth.judge(tiny, RECIPE, honest, {**forged, 'proof_root': hashlib.sha256(honest).hexdigest()})
    wrong_before = {**forged, 'before_root': roots[1], 'proof_root': hashlib.sha256(honest).hexdigest()}
    assert not growth.judge(tiny, RECIPE, honest, wrong_before)  # a bundle must start from the committed state


def trainer_digests(trainable):
    from neuroshard.evolution.sharded.granite_training import digests

    return digests(trainable)


def growth_chain():
    sponsor, alice, bob, challenger = (Account(n) for n in ('sponsor', 'alice', 'bob', 'challenger'))
    state = ledger.genesis(CHAIN, 'ab' * 32, 3, {a.public: 40_000_000 for a in (sponsor, alice, bob, challenger)},
                           PARAMS)
    return state, (sponsor, alice, bob, challenger)


def run(state, envelope, execute=None, blocks=0):
    for _ in range(blocks):
        state = ledger.advance(state, state['height'] + 1)
    return ledger.transition(state, envelope, execute)


def test_contributors_are_paid_per_step_and_the_job_names_the_grown_unit():
    state, (sponsor, alice, bob, _) = growth_chain()
    roots = [f'{i:02x}' * 32 for i in range(1, 7)]
    opened = sponsor.sign('growth_open', recipe_root='cd' * 32, start_root='ef' * 32, steps=6, segment_steps=3,
                          reward=6_000_000)
    job_id = ledger.transaction_id(opened)
    state = run(state, opened)
    with pytest.raises(ValueError, match='in order'):
        run(state, bob.sign('growth_commit', job_id=job_id, segment=1, step_roots=roots[3:]))
    bob.nonce = state['accounts'][bob.public]['nonce']
    with pytest.raises(ValueError, match='one state root per step'):
        run(state, alice.sign('growth_commit', job_id=job_id, segment=0, step_roots=roots[:2]))
    alice.nonce = state['accounts'][alice.public]['nonce']
    state = run(state, alice.sign('growth_commit', job_id=job_id, segment=0, step_roots=roots[:3]))
    state = run(state, bob.sign('growth_commit', job_id=job_id, segment=1, step_roots=roots[3:]))
    ledger.invariant(state)
    before = {a.public: state['accounts'][a.public]['balance'] for a in (sponsor, alice, bob)}
    for _ in range(PARAMS.get('growth_window_blocks', ledger_growth.PARAMS['growth_window_blocks']) + 1):
        state = ledger.advance(state, state['height'] + 1)
    assert job_id not in state.get('growth', {})
    result = state['results'][job_id]
    assert result['status'] == 'grown' and result['final_root'] == roots[-1]
    bond = ledger_growth.PARAMS['growth_bond']
    assert state['accounts'][alice.public]['balance'] == before[alice.public] + 3_000_000 + bond
    assert state['accounts'][bob.public]['balance'] == before[bob.public] + 3_000_000 + bond
    ledger.invariant(state)


def test_a_proven_wrong_step_slashes_its_segment_and_reopens_it_and_a_framing_forfeits():
    state, (sponsor, alice, bob, challenger) = growth_chain()
    opened = sponsor.sign('growth_open', recipe_root='cd' * 32, start_root='ef' * 32, steps=4, segment_steps=2,
                          reward=4_000_000)
    job_id = ledger.transaction_id(opened)
    state = run(state, opened)
    state = run(state, alice.sign('growth_commit', job_id=job_id, segment=0, step_roots=['01' * 32, '02' * 32]))
    state = run(state, bob.sign('growth_commit', job_id=job_id, segment=1, step_roots=['03' * 32, '04' * 32]))
    challenge = challenger.sign('growth_challenge', job_id=job_id, segment=0, step=1, proof_root='aa' * 32)
    cid = ledger.transaction_id(challenge)
    state = run(state, challenge)
    seen = []

    def judge(_, request):
        seen.append(request)
        return request['claimed_root'] == '02' * 32

    with pytest.raises(ValueError, match='later block'):
        run(state, challenger.sign('growth_prove', job_id=job_id, challenge_id=cid), judge)
    challenger.nonce = state['accounts'][challenger.public]['nonce']
    paid = state['accounts'][challenger.public]['balance']
    proven = run(state, challenger.sign('growth_prove', job_id=job_id, challenge_id=cid), judge, blocks=1)
    assert seen[-1]['before_root'] == '01' * 32 and seen[-1]['step'] == 1 and seen[-1]['recipe_root'] == 'cd' * 32
    assert seen[-1]['proof_root'] == 'aa' * 32 and ledger_growth.open_proof(state, 'aa' * 32)
    job = proven['growth'][job_id]
    assert job['commits'] == [] and proven['growth_faults'][-1]['contributor'] == alice.public
    bond, deposit = ledger_growth.PARAMS['growth_bond'], ledger_growth.PARAMS['growth_challenge_deposit']
    assert proven['accounts'][challenger.public]['balance'] == paid - PARAMS['fee'] + deposit + bond // 2
    ledger.invariant(proven)

    # A challenge whose proof never verifies forfeits its deposit when the proof window closes.
    framing = copy.deepcopy(state)
    burned = framing['burned']
    for _ in range(ledger_growth.PARAMS['growth_proof_blocks'] + 1):
        framing = ledger.advance(framing, framing['height'] + 1)
    assert framing['burned'] == burned + deposit
    assert job_id not in framing.get('growth', {}) or not framing['growth'][job_id]['challenges']
    ledger.invariant(framing)


def test_validators_judge_a_growth_proof_from_their_recipe_and_bundle_stores(tmp_path):
    import hashlib
    import json

    from neuroshard.inference.optimistic_app import promotion_or_fraud

    base_config = {'vocab_size': 64, 'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 2,
                   'num_attention_heads': 4, 'num_key_value_heads': 2, 'tie_word_embeddings': True,
                   'eos_token_id': 1, 'pad_token_id': 0, 'logits_scaling': 2.0}
    recipe = {**RECIPE, 'base': 'tiny-granite', 'base_config': base_config, 'base_seed': 0, 'model_root': 'ab' * 32}
    root = growth.recipe_root(recipe, recipe['model_root'])
    (tmp_path / 'recipes').mkdir()
    (tmp_path / 'recipes' / f'{root}.json').write_text(json.dumps(recipe))
    (tmp_path / 'bundles').mkdir()
    model = tiny()
    trainable, optimizer = growth.start(model, recipe)
    first = growth.train_segment(model, trainable, optimizer, recipe, 0, 1)
    saved = growth.save(trainable, optimizer)
    honest = growth.train_segment(model, trainable, optimizer, recipe, 1, 2)
    payload = growth.bundle(saved, 1)
    proof_root = hashlib.sha256(payload).hexdigest()
    (tmp_path / 'bundles' / proof_root).write_bytes(payload)
    check = promotion_or_fraud({'growth_recipes': str(tmp_path / 'recipes'), 'bundles': str(tmp_path / 'bundles')},
                               None)
    request = {'kind': 'growth_step', 'recipe_root': root, 'step': 1, 'before_root': first[0],
               'proof_root': proof_root}
    assert check({}, {**request, 'claimed_root': 'ab' * 32}) is True
    assert check({}, {**request, 'claimed_root': honest[0]}) is False
    with pytest.raises(ledger.NoVerdict):
        promotion_or_fraud({}, None)({}, {**request, 'claimed_root': 'ab' * 32})
    with pytest.raises(ledger.ProofUnavailable):
        check({}, {**request, 'recipe_root': 'ee' * 32, 'claimed_root': 'ab' * 32})
