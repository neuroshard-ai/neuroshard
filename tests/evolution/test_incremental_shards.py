"""Compare a frozen distributed prefix and trainable tail with full autograd."""
from datetime import timedelta
import copy
import json
import math
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import LlamaConfig, LlamaForCausalLM

from neuroshard.evolution import reference
from neuroshard.evolution.reference_data import identity, save
from neuroshard.evolution.sharded import incremental, incremental_state, portable, checkpoint
from neuroshard.evolution.sharded.guarded import correct_margin, objective
from neuroshard.evolution.sharded.model import Partition, batch_tensors
from neuroshard.evolution.sharded.training import generate
from neuroshard.evolution.sharded.wire import Wire


RECIPE = {'steps': 4, 'warmup_steps': 1, 'learning_rate': .0003,
          'weight_decay': .01, 'clip_norm': .1}


def config(layers):
    value = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=1,
                        max_position_embeddings=64, tie_word_embeddings=True, attention_dropout=0.)
    value._attn_implementation = 'eager'
    return value


def models():
    torch.manual_seed(71)
    parent = LlamaForCausalLM(config(4)).float().eval().requires_grad_(False)
    grown = LlamaForCausalLM(config(6)).float()
    grown.load_state_dict(parent.state_dict(), strict=False)
    for layer in grown.model.layers[4:]:
        layer.load_state_dict(parent.model.layers[-1].state_dict())
        with torch.no_grad():
            layer.self_attn.o_proj.weight.zero_()
            layer.mlp.down_proj.weight.zero_()
    names = incremental.trainable_names(grown.config, 4)
    for name, parameter in grown.named_parameters():
        parameter.requires_grad_(name in names)
    return parent, grown


def records():
    return [{'id': str(i), 'input_ids': [1, 3, 4] + [7 + i] * (i + 1) + [2],
             'labels': [-100] * 3 + [7 + i] * (i + 1) + [2], 'targets': i + 2,
             'loss_weight': 1 / (i + 2), 'distill': i != 1} for i in range(5)]


def control_step(parent, model, optimizer, index):
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = reference.learning_rate(RECIPE, index)
    rows = records()
    denominator = sum(r['targets'] * r['loss_weight'] for r in rows)
    anchors = sum(r['targets'] for r in rows if r['distill'])
    total = 0.
    for offset in range(0, len(rows), 2):
        batch = rows[offset:offset + 2]
        ids, labels, mask, weights = batch_tensors(batch, 'cpu')
        active = labels[:, 1:] != -100
        target = labels[:, 1:][active]
        with torch.no_grad():
            teacher = parent(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1][active]
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1][active]
        anchor_rows = torch.tensor([r['distill'] for r in batch], dtype=torch.bool)
        anchor_mask = anchor_rows[:, None].expand_as(active)[active]
        loss, _, _ = objective(logits, target, weights[:, None].expand_as(active)[active],
                                teacher, anchor_mask, denominator, anchors, 2.)
        loss = loss + .3 * correct_margin(logits, target, teacher, anchor_mask, anchors, .5, 2.)
        total += float(loss.detach())
        loss.backward()
    pairs = [(name, p) for name, p in sorted(model.named_parameters()) if p.requires_grad]
    norm = math.sqrt(math.fsum(float(torch.linalg.vector_norm(p.grad, dtype=torch.float64).square())
                              for _, p in pairs))
    scale = min(1., RECIPE['clip_norm'] / (norm + 1e-6))
    for _, parameter in pairs:
        parameter.grad.mul_(scale)
    optimizer.step()
    return total, norm


def worker(rank, rendezvous, folder, boundaries, growing=True, resume=False):
    torch.set_num_threads(1)
    parent, control = models()
    first_trainable = 4 if growing else 2
    if not growing:
        control = copy.deepcopy(parent)
        names = incremental.trainable_names(control.config, first_trainable)
        for name, parameter in control.named_parameters():
            parameter.requires_grad_(name in names)
    original = {name: p.detach().clone() for name, p in control.named_parameters()}
    expected = dict(control.named_parameters())
    groups = [[p for p in control.parameters() if p.requires_grad and p.ndim >= 2],
              [p for p in control.parameters() if p.requires_grad and p.ndim < 2]]
    control_optimizer = torch.optim.AdamW([
        {'params': groups[0], 'weight_decay': RECIPE['weight_decay']},
        {'params': groups[1], 'weight_decay': 0.},
    ], lr=RECIPE['learning_rate'], betas=(.9, .95), eps=1e-8, foreach=False)
    # Full models above are test oracles. The implementation may allocate only
    # its owned parameters, including when that owner has no optimizer at all.
    def forbidden(*args, **kwargs):
        raise AssertionError('Incremental worker attempted whole-model allocation')
    LlamaForCausalLM.__init__ = forbidden
    shard = Partition(config(control.config.num_hidden_layers), boundaries, rank)
    parent_state = json.loads(Path(folder, 'parent.json').read_bytes())
    mode = 'append' if growing else 'tail-control'
    frozen_sources = incremental_state.initialize(shard, parent_state, Path(folder, 'parent'),
                                                   mode, first_trainable)
    assert all(torch.equal(parameter, expected[name]) for name, parameter in shard.named_owned_parameters())
    teacher_tail = None if growing else incremental.reference_tail(shard, first_trainable)
    optimizer = incremental.configure(shard, first_trainable, RECIPE)
    world = len(boundaries) - 1
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank,
                            world_size=world, timeout=timedelta(seconds=60))
    wire = Wire(rank, world)
    home = Path(folder) / f'worker-{rank}'
    job = identity({'parent': identity(parent_state), 'recipe': RECIPE, 'rows': records(), 'mode': mode})
    try:
        start = 0
        if resume:
            before = json.loads((home / 'commit-000002.json').read_bytes())
            original_result = json.loads((home / 'commit-000004.json').read_bytes())
            frozen_sources = incremental_state.load(home, shard, optimizer, before, parent_state, job, RECIPE)
            for index in (0, 1):
                control_step(parent, control, control_optimizer, index)
            home = home / 'fresh-process'
            start = 2
        else:
            first_tokens = generate(shard, wire, [1, 5, 7], 4, -1)
            tokens = [1, 5, 7]
            with torch.no_grad():
                for _ in range(4):
                    tokens.append(int(parent(torch.tensor([tokens])).logits[0, -1].argmax()))
            assert first_tokens == tokens[3:]
        incremental_state.commit(home, shard, optimizer, wire, parent_state, frozen_sources,
                                 job, start, RECIPE, mode, first_trainable)
        for index in range(start, 4):
            result = incremental.train_step(shard, optimizer, wire, records(), RECIPE,
                                            index, 2, first_trainable, margin_strength=.3,
                                            reference_layers=teacher_tail)
            loss, norm = control_step(parent, control, control_optimizer, index)
            assert result['loss'] == pytest.approx(loss, rel=2e-6, abs=2e-7)
            assert result['gradient_norm'] == pytest.approx(norm, rel=2e-5, abs=2e-7)
            for name, parameter in shard.named_owned_parameters():
                if not parameter.requires_grad:
                    assert parameter.grad is None
                    assert torch.equal(parameter, original[name])
                else:
                    torch.testing.assert_close(parameter, expected[name], rtol=3e-5, atol=3e-7)
                    for key, value in control_optimizer.state[expected[name]].items():
                        torch.testing.assert_close(optimizer.state[parameter][key], value,
                                                   rtol=3e-5, atol=3e-7)
            if index + 1 in (2, 4):
                common = incremental_state.commit(home, shard, optimizer, wire, parent_state,
                                                   frozen_sources, job, index + 1, RECIPE, mode, first_trainable)
        assert any(not torch.equal(p, original[name]) for name, p in control.named_parameters()
                   if p.requires_grad)
        expected_root = identity(common)
        earlier = json.loads((home / 'commit-000002.json').read_bytes())
        frozen_sources = incremental_state.load(home, shard, optimizer, earlier, parent_state, job, RECIPE)
        for index in (2, 3):
            incremental.train_step(shard, optimizer, wire, records(), RECIPE,
                                   index, 2, first_trainable, margin_strength=.3, reference_layers=teacher_tail)
        recovered = incremental_state.commit(home / 'recovered', shard, optimizer, wire, parent_state,
                                              frozen_sources, job, 4, RECIPE, mode, first_trainable)
        assert identity(recovered) == expected_root
        if resume:
            assert recovered == original_result
        active = incremental.trainable_names(shard.config, first_trainable)
        assert all(spec['optimizer_step'] == (4 if name in active else 7)
                   for name, spec in recovered['tensors'].items())
        inference_shard = Partition(shard.config, boundaries, rank).requires_grad_(False)
        incremental_state.load(home / 'recovered', inference_shard, None, recovered,
                               parent_state, job, RECIPE, restore_optimizer=False)
        assert all(torch.equal(p, dict(shard.named_owned_parameters())[name])
                   for name, p in inference_shard.named_owned_parameters())
        assert generate(inference_shard, wire, [1, 5, 7], 4, -1) == generate(shard, wire, [1, 5, 7], 4, -1)
        with pytest.raises(ValueError, match='portable state'):
            portable.validate(recovered)
        if rank == 0:
            changed = copy.deepcopy(recovered)
            changed['tensors']['model.embed_tokens.weight']['sha256'] = '0' * 64
            changed['state_root'] = incremental_state.state_root(changed)
            with pytest.raises(ValueError, match='Frozen parent state changed'):
                incremental_state.validate(changed, parent_state)
            changed = copy.deepcopy(recovered)
            changed['tensors'][sorted(active)[0]]['optimizer_step'] += 1
            changed['state_root'] = incremental_state.state_root(changed)
            with pytest.raises(ValueError, match='Adam age'):
                incremental_state.validate(changed, parent_state)
        Path(folder, f'rank-{rank}.txt').write_text(str(shard.resident_parameters))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('boundaries', [[0, 2, 4, 6], [0, 2, 4, 5, 6], [0, 3, 5, 6]])
def test_added_blocks_match_full_autograd_and_leave_parent_unchanged(tmp_path, boundaries):
    prepare_parent(tmp_path)
    mp.spawn(worker, args=(str(tmp_path / 'rendezvous'), str(tmp_path), boundaries),
             nprocs=len(boundaries) - 1, join=True)
    if boundaries == [0, 2, 4, 6]:
        mp.spawn(worker, args=(str(tmp_path / 'restart-rendezvous'), str(tmp_path), boundaries, True, True),
                 nprocs=len(boundaries) - 1, join=True)
    _, model = models()
    counts = [int(path.read_text()) for path in tmp_path.glob('rank-*.txt')]
    total = sum(p.numel() for p in model.parameters())
    assert sum(counts) == total and all(n < total for n in counts)


def test_frozen_original_tail_control_matches_full_model_reference(tmp_path):
    prepare_parent(tmp_path)
    mp.spawn(worker, args=(str(tmp_path / 'rendezvous'), str(tmp_path), [0, 1, 2, 4], False),
             nprocs=3, join=True)
    mp.spawn(worker, args=(str(tmp_path / 'restart-rendezvous'), str(tmp_path), [0, 1, 2, 4], False, True),
             nprocs=3, join=True)


def prepare_parent(folder):
    parent, _ = models()
    directory = folder / 'parent'
    directory.mkdir()
    optimizer = reference.optimizer_for(parent, RECIPE)
    tensors = {}
    for name, parameter in parent.named_parameters():
        values = {'weight': parameter.detach(), 'step': torch.tensor(7.),
                  'exp_avg': torch.ones_like(parameter) * .001,
                  'exp_avg_sq': torch.ones_like(parameter) * .002}
        temporary = directory / 'pending.safetensors'
        spec = checkpoint.tensor_file(temporary, values)
        temporary.replace(portable.tensor_path(directory, spec['sha256']))
        tensors[name] = {**spec, 'shape': list(parameter.shape), 'born': 0,
                         'group': int(parameter.ndim < 2)}
    common = {'format': portable.FORMAT, 'job': identity({'test': 'parent'}), 'step': 7,
              'config': portable.configuration(parent.config), 'optimizer': portable.recipe(optimizer),
              'tensors': tensors, 'boundaries': [0, 2, 4], 'shards': [], 'parent': None, 'transition': None}
    common['state_root'] = portable.learned_root(common)
    portable.validate(common)
    save(folder / 'parent.json', common)


def test_identity_initialization_and_invalid_training_boundary():
    torch.set_num_threads(1)
    parent, grown = models()
    tokens = torch.tensor([[1, 5, 7, 9], [1, 7, 3, 2]])
    with torch.no_grad():
        assert torch.equal(parent(tokens).logits, grown(tokens).logits)
    shard = Partition(config(6), [0, 3, 6], 0)
    for invalid in (True, 0, -1, 6, 7):
        with pytest.raises(ValueError, match='nonempty tail'):
            incremental.configure(shard, invalid, RECIPE)
    with pytest.raises(ValueError, match='output-head owner'):
        incremental.configure(shard, 2, RECIPE)
    assert incremental.configure(shard, 4, RECIPE) is None
    assert all(not p.requires_grad for p in shard.parameters())
