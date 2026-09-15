"""New-tail recovery preserves its Adam state without copying the parent model."""
import copy

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from neuroshard.evolution import reference
from neuroshard.evolution.reference_data import identity
from neuroshard.evolution.sharded import checkpoint, cohort_state, incremental, incremental_state, portable
from neuroshard.evolution.sharded.model import Partition

RECIPE = {'steps': 2, 'warmup_steps': 0, 'learning_rate': .0003, 'weight_decay': .01, 'clip_norm': 1.}


def prepare(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(14)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=6,
                        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                        tie_word_embeddings=True, attention_dropout=0.)
    config._attn_implementation = 'sdpa'
    model = LlamaForCausalLM(config).float()
    optimizer = reference.optimizer_for(model, RECIPE)
    tensors = {}
    objects = tmp_path / 'owned-parent-objects'
    objects.mkdir()
    for name, parameter in model.named_parameters():
        values = {'weight': parameter.detach(), 'step': torch.tensor(7.),
                  'exp_avg': torch.ones_like(parameter) * .001,
                  'exp_avg_sq': torch.ones_like(parameter) * .002}
        temporary = tmp_path / 'tensor.pending'
        spec = checkpoint.tensor_file(temporary, values)
        if name.startswith('model.layers.5.'):
            temporary.replace(portable.tensor_path(objects, spec['sha256']))
        else:
            temporary.unlink()
        tensors[name] = {**spec, 'shape': list(parameter.shape), 'born': 0, 'group': int(parameter.ndim < 2)}
    parent = {'format': portable.FORMAT, 'job': identity({'test': 'parent'}), 'step': 7,
              'config': portable.configuration(config), 'optimizer': portable.recipe(optimizer),
              'tensors': tensors, 'boundaries': [0, 2, 4, 6], 'shards': [], 'parent': None, 'transition': None}
    parent['state_root'] = portable.learned_root(parent)
    portable.validate(parent)
    shard = Partition(config, [0, 2, 4, 5, 6], 3)
    incremental_state.initialize(shard, parent, objects, 'tail-control', 5)
    return parent, objects, shard, incremental.configure(shard, 5, RECIPE)


def update(shard, optimizer, step):
    optimizer.zero_grad(set_to_none=True)
    for index, (_, parameter) in enumerate(shard.named_owned_parameters()):
        parameter.grad = torch.full_like(parameter, .01 * (index + 1) * (step + 1))
    optimizer.step()


def test_tail_checkpoint_and_recovery_need_no_unowned_parent_tensors(tmp_path):
    parent, objects, shard, optimizer = prepare(tmp_path)
    original = copy.deepcopy(parent)
    home = tmp_path / 'run'
    initial = cohort_state.commit_tail(home, shard, optimizer, parent, objects, 'a' * 64, 0, RECIPE, 5)
    update(shard, optimizer, 0)
    midpoint = cohort_state.commit_tail(home, shard, optimizer, parent, objects, 'a' * 64, 1, RECIPE, 5)
    update(shard, optimizer, 1)
    final = cohort_state.commit_tail(home, shard, optimizer, parent, objects, 'a' * 64, 2, RECIPE, 5)
    restored = Partition(shard.config, list(shard.boundaries), 3)
    restored_optimizer = incremental.configure(restored, 5, RECIPE)
    incremental_state.load(home, restored, restored_optimizer, midpoint, parent, 'a' * 64, RECIPE)
    update(restored, restored_optimizer, 1)
    replay = cohort_state.commit_tail(tmp_path / 'replay', restored, restored_optimizer, parent,
                                     objects, 'a' * 64, 2, RECIPE, 5)
    assert replay == final and parent == original
    inherited = incremental_state.records(parent)
    for name, spec in final['tensors'].items():
        if name.startswith('model.layers.5.'):
            assert initial['tensors'][name]['optimizer_step'] == 0 and spec['optimizer_step'] == 2
        else:
            assert spec == inherited[name] and spec['optimizer_step'] == 7
    # Identical norm weights/moments can legitimately share content hashes
    # with other layers. Only objects referenced by the owned tail are staged.
    owned_hashes = {spec['sha256'] for name, spec in inherited.items() if name.startswith('model.layers.5.')}
    assert {path.stem for path in objects.iterdir()} == owned_hashes
    assert len(owned_hashes) < len({spec['sha256'] for spec in inherited.values()})
    for name, parameter in restored.named_owned_parameters():
        expected = dict(shard.named_owned_parameters())[name]
        assert torch.equal(parameter, expected)
        assert all(torch.equal(value, optimizer.state[expected][key])
                   for key, value in restored_optimizer.state[parameter].items())
    changed = copy.deepcopy(final)
    changed['tensors']['model.norm.weight']['optimizer_step'] = 2
    changed['state_root'] = incremental_state.state_root(changed)
    with pytest.raises(ValueError, match='Frozen parent state changed'):
        incremental_state.validate(changed, parent)


def test_tail_checkpoint_rejects_wrong_actual_adam_age(tmp_path):
    parent, objects, shard, optimizer = prepare(tmp_path)
    update(shard, optimizer, 0)
    optimizer.state[next(shard.parameters())]['step'].fill_(7)
    with pytest.raises(ValueError, match='Adam age'):
        cohort_state.commit_tail(tmp_path / 'bad', shard, optimizer, parent, objects,
                                 'a' * 64, 1, RECIPE, 5)
