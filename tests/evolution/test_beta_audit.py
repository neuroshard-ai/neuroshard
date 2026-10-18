"""Regression tests for the October 18, 2026 beta-readiness audit (.neuroshard/beta-audit-20261018).

Each test reproduces one reported failure and asserts the safe outcome instead.
"""
import copy
import hashlib
import io

import pytest
import torch

from neuroshard.evolution import growth_replay as g
from neuroshard.evolution.growth_model import model_factory

from test_network_growth import RECIPE, tiny

torch.set_num_threads(1)


def _honest_step():
    model = tiny()
    trainable, optimizer = g.start(model, RECIPE)
    g.train_segment(model, trainable, optimizer, RECIPE, 0, 1)
    before_root, before = g.state_root(trainable, optimizer), g.save(trainable, optimizer)
    after_root = g.train_segment(model, trainable, optimizer, RECIPE, 1, 2)[0]
    return before_root, before, after_root


def _verdict(before, before_root, claimed_root, recipe=RECIPE, make=tiny):
    bundle = g.bundle(before, 1)
    return g.judge(make, recipe, bundle, {'step': 1, 'before_root': before_root, 'claimed_root': claimed_root,
                                          'proof_root': hashlib.sha256(bundle).hexdigest()})


def _rewrite(before, change):
    saved = torch.load(io.BytesIO(before), weights_only=True)
    change(saved)
    buffer = io.BytesIO()
    torch.save(saved, buffer)
    return buffer.getvalue()


# Finding 1: uncommitted optimizer settings must not turn honest work into fraud.

@pytest.mark.parametrize('key,value', [('weight_decay', 100.0), ('eps', 1.0), ('betas', (0.1, 0.2)),
                                       ('amsgrad', True), ('maximize', True), ('foreach', True)])
def test_altered_optimizer_settings_cannot_prove_an_honest_step_wrong(key, value):
    before_root, before, after_root = _honest_step()
    assert _verdict(before, before_root, after_root) is False
    def change(saved):
        saved['optimizer']['param_groups'][0][key] = value
    forged = _rewrite(before, change)
    assert g.bytes_root(forged) == before_root  # the setting is outside the committed root
    assert _verdict(forged, before_root, after_root) is False


def test_a_reordered_or_regrouped_state_proves_nothing():
    before_root, before, after_root = _honest_step()
    reordered = _rewrite(before, lambda saved: saved['order'].reverse())
    regrouped = _rewrite(before, lambda saved: saved['optimizer']['param_groups'].append(
        dict(saved['optimizer']['param_groups'][0], params=[])))
    assert _verdict(reordered, before_root, after_root) is False
    assert _verdict(regrouped, before_root, after_root) is False


def test_a_real_forgery_is_still_proven():
    before_root, before, after_root = _honest_step()
    assert _verdict(before, before_root, 'ff' * 32) is True


# Finding 3: a recipe root binds everything that affects execution.

def test_the_base_model_seed_and_configuration_are_bound_by_the_recipe_root():
    recipe = {**copy.deepcopy(RECIPE), 'base': 'tiny-granite', 'base_seed': 0, 'model_root': 'ab' * 32,
              'base_config': {'vocab_size': 64, 'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 2,
                              'num_attention_heads': 4, 'num_key_value_heads': 2, 'tie_word_embeddings': True}}
    root = g.recipe_root(recipe, recipe['model_root'])
    assert g.recipe_root({**recipe, 'base_seed': 1}, recipe['model_root']) != root
    assert g.recipe_root({**recipe, 'base_config': {**recipe['base_config'], 'hidden_size': 64}},
                         recipe['model_root']) != root
    # A parent recipe carrying a base configuration or seed would ignore them, so it is refused outright.
    with pytest.raises(ValueError):
        g.recipe_root({**recipe, 'base': 'granite-parent'}, recipe['model_root'])


def test_parent_recipe_roots_are_unchanged_and_extra_fields_are_not_executed():
    parent = {'arm': 'addition', 'spec': {'seed': 1}, 'sources': {'experience': []}, 'model_root': 'cd' * 32}
    legacy = hashlib.sha256(__import__('json').dumps(
        {'model_root': 'cd' * 32, 'arm': 'addition', 'spec': {'seed': 1}, 'sources': {'experience': []}},
        sort_keys=True).encode()).hexdigest()
    assert g.recipe_root(parent, parent['model_root']) == legacy
    assert g.recipe_root({**parent, 'base': 'granite-parent', 'route': 'documents', 'kind': 'x'},
                         parent['model_root']) == legacy
    assert set(g.execution({**parent, 'route': 'documents', 'whatever': 1})) == {'model_root', 'arm', 'spec', 'sources'}


def test_validators_judge_only_the_pinned_parent_unless_configured():
    with pytest.raises(ValueError):
        model_factory({'base': 'tiny-granite', 'base_config': {}}, allowed={'granite-parent'})
    model_factory({'base': 'granite-parent'}, allowed={'granite-parent'})
