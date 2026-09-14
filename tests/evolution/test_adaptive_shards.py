"""Numerical equivalence and real process-group changes over portable state."""
from datetime import timedelta
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import LlamaConfig, LlamaForCausalLM

from neuroshard.evolution import reference
from neuroshard.evolution.reference_data import identity
from neuroshard.evolution.sharded import guarded, portable, expansion, transcript
from neuroshard.evolution.sharded.model import Partition, batch_tensors, owner
from neuroshard.evolution.sharded.training import generate
from neuroshard.evolution.sharded.wire import Wire

RECIPE = {'steps': 4, 'warmup_steps': 1, 'learning_rate': .0003,
          'weight_decay': .01, 'clip_norm': .1}
JOB = identity({'test': 'portable-guarded-v1'})


def config(layers=6):
    cfg = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=1,
                      max_position_embeddings=64, tie_word_embeddings=True, attention_dropout=0.)
    cfg._attn_implementation = 'eager'
    return cfg


def records():
    return [{'id': str(i), 'input_ids': [1, 3, 4]+[7+i]*(i+1)+[2],
             'labels': [-100]*3+[7+i]*(i+1)+[2], 'targets': i+2,
             'loss_weight': 8 if i % 2 else 1, 'distill': i % 2 == 0} for i in range(5)]


def models():
    torch.manual_seed(53)
    teacher = LlamaForCausalLM(config()).float().eval().requires_grad_(False)
    student = LlamaForCausalLM(config()).float()
    student.load_state_dict(teacher.state_dict())
    with torch.no_grad():
        student.model.layers[2].mlp.down_proj.weight.add_(.003)
    return student, teacher


def full_step(student, teacher, optimizer, index):
    rows = records()
    denom = sum(r['targets']*r['loss_weight'] for r in rows)
    anchors = sum(r['targets'] for r in rows if r['distill'])
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = reference.learning_rate(RECIPE, index)
    loss_sum = 0.
    for offset in range(0, len(rows), 2):
        batch = rows[offset:offset+2]
        ids, labels, mask, weights = batch_tensors(batch, 'cpu')
        active = labels[:, 1:] != -100
        with torch.no_grad():
            old = teacher(ids, attention_mask=mask, use_cache=False).logits[:, :-1][active]
        new = student(ids, attention_mask=mask, use_cache=False).logits[:, :-1][active]
        token_weights = weights[:, None].expand_as(active)[active]
        selected = torch.tensor([r['distill'] for r in batch])[:, None].expand_as(active)[active]
        # Independent formula, including the global (not microbatch) normalizers.
        ce = (torch.nn.functional.cross_entropy(new, labels[:, 1:][active], reduction='none')*token_weights).sum()/denom
        probability = old[selected].softmax(-1)
        kl = (probability*(probability.log()-new[selected].log_softmax(-1))).sum()/anchors
        loss = ce+2*kl
        loss_sum += float(loss.detach())
        loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(student.parameters(), RECIPE['clip_norm']))
    optimizer.step()
    return loss_sum, norm


def worker(rank, world, rendezvous, root, tag, boundaries, start, end, resume):
    torch.set_num_threads(1)
    random.seed(53+rank)
    np.random.seed(53+rank)
    student, teacher = models()
    control_optimizer = reference.optimizer_for(student, RECIPE)
    for index in range(start):
        full_step(student, teacher, control_optimizer, index)
    shard = Partition(config(), boundaries, rank)
    reference_shard = Partition(config(), boundaries, rank).eval().requires_grad_(False)
    old = dict(teacher.named_parameters())
    expected = dict(student.named_parameters())
    with torch.no_grad():
        for name, parameter in shard.named_owned_parameters():
            parameter.copy_(expected[name])
        for name, parameter in reference_shard.named_owned_parameters():
            parameter.copy_(old[name])
    optimizer = reference.optimizer_for(shard, RECIPE)
    home = Path(root)/tag/f'rank-{rank}'
    dist.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(seconds=90))
    wire = Wire(rank, world)
    births, parent = {}, None
    try:
        if resume:
            source = Path(root)/resume/f'rank-{rank}'
            common = json.loads((source/f'commit-{start:06d}.json').read_bytes())
            births = portable.load(source, shard, optimizer, common, JOB)
            parent = identity(common)
        else:
            parent = identity(portable.commit(home, shard, optimizer, wire, JOB, 0, None))
        capture = transcript.Recorder(wire, home/'transcript') if tag == 'baseline' else wire
        original = parent
        for index in range(start, end):
            row = guarded.train_step(shard, reference_shard, optimizer, capture, records(), RECIPE, index, 2, kl_strength=2.)
            loss, norm = full_step(student, teacher, control_optimizer, index)
            assert row['loss'] == pytest.approx(loss, abs=2e-6)
            assert row['gradient_norm'] == pytest.approx(norm, rel=3e-6)
            for name, parameter in shard.named_owned_parameters():
                torch.testing.assert_close(parameter, expected[name], rtol=2e-5, atol=3e-7)
                for key, value in control_optimizer.state[expected[name]].items():
                    torch.testing.assert_close(optimizer.state[parameter][key], value, rtol=3e-5, atol=3e-7)
            common = portable.commit(home, shard, optimizer, wire, JOB, index+1, parent, births)
            parent = identity(common)
        if tag == 'baseline':
            capture.finish({'job': JOB, 'start': start, 'end': end, 'input': original, 'output': parent})
        tokens = generate(shard, wire, [1, 3, 4], 3, -1)
        (home/'result.json').write_text(json.dumps({'state_root': common['state_root'], 'tokens': tokens,
            'resident_parameters': shard.resident_parameters}))
    finally:
        dist.destroy_process_group()


def run(root, tag, boundaries, start, end, resume=None):
    mp.spawn(worker, args=(len(boundaries)-1, str(root/(tag+'-rendezvous')), str(root), tag,
                           boundaries, start, end, resume), nprocs=len(boundaries)-1, join=True)
    return json.loads((root/tag/'rank-0'/f'commit-{end:06d}.json').read_bytes())


def migrate(root, source, common, tag, boundaries):
    proposal = portable.repartition(common, boundaries)
    sources, metas = {}, []
    for rank in range(len(common['boundaries'])-1):
        folder = portable.directory(root/source/f'rank-{rank}', common['step'])
        meta = json.loads((folder/'manifest.json').read_bytes())
        for name, spec in meta['tensors'].items():
            sources[name] = portable.tensor_path(folder, spec['sha256'])
    old_folder = portable.directory(root/source/'rank-0', common['step'])
    old_meta = json.loads((old_folder/'manifest.json').read_bytes())
    for rank in range(len(boundaries)-1):
        subset = {n: p for n, p in sources.items() if owner(n, boundaries) == rank}
        metas.append(portable.install_partition(root/tag/f'rank-{rank}', rank, proposal, subset,
                                                old_meta, old_folder/'rng.safetensors'))
    result = portable.assemble(metas, identity(common), proposal['transition'])
    assert result['state_root'] == common['state_root']
    for rank in range(len(boundaries)-1):
        (root/tag/f'rank-{rank}'/f'commit-{common["step"]:06d}.json').write_text(json.dumps(result))
    return result


def test_guarded_autograd_and_two_three_two_process_groups_preserve_adam(tmp_path):
    baseline = run(tmp_path, 'baseline', [0, 3, 6], 0, 4)
    first = json.loads((tmp_path/'baseline/rank-0/commit-000001.json').read_bytes())
    migrate(tmp_path, 'baseline', first, 'joined', [0, 2, 4, 6])
    third = run(tmp_path, 'three', [0, 2, 4, 6], 1, 3, 'joined')
    migrate(tmp_path, 'three', third, 'left', [0, 3, 6])
    final = run(tmp_path, 'two', [0, 3, 6], 3, 4, 'left')
    assert final['state_root'] == baseline['state_root']
    a = json.loads((tmp_path/'baseline/rank-0/result.json').read_bytes())
    b = json.loads((tmp_path/'two/rank-0/result.json').read_bytes())
    assert a['tokens'] == b['tokens']
    portable.validate(final)
    damaged = json.loads(json.dumps(final))
    damaged['tensors'].pop(next(iter(damaged['tensors'])))
    damaged['state_root'] = portable.learned_root(damaged)
    with pytest.raises(ValueError, match='Incomplete'):
        portable.validate(damaged)
    traces = [json.loads((tmp_path/'baseline'/f'rank-{rank}'/'transcript/transcript.json').read_bytes()) for rank in range(2)]
    transcript.validate(traces)
    # Each auditor can replay the whole window one partition at a time, without
    # constructing the full student. The tiny teacher below is a test fixture.
    def replay_rank(rank, trace):
        torch.set_num_threads(1)
        _, teacher = models()
        shard = Partition(config(), [0, 3, 6], rank)
        old = Partition(config(), [0, 3, 6], rank).eval().requires_grad_(False)
        source = dict(teacher.named_parameters())
        for name, p in old.named_owned_parameters():
            p.data.copy_(source[name])
        optimizer = reference.optimizer_for(shard, RECIPE)
        home = tmp_path/'baseline'/f'rank-{rank}'
        initial = json.loads((home/'commit-000000.json').read_bytes())
        portable.load(home, shard, optimizer, initial, JOB)
        wire = transcript.Replay(home/'transcript', trace)
        for index in range(4):
            guarded.train_step(shard, old, optimizer, wire, records(), RECIPE, index, 2, kl_strength=2.)
        wire.finish()
        meta = portable.write(tmp_path/f'audit-{rank}', shard, optimizer, JOB, 4)
        assert meta['tensors'] == json.loads((portable.directory(home, 4)/'manifest.json').read_bytes())['tensors']
    for rank in range(2):
        replay_rank(rank, traces[rank])
    import copy
    from safetensors.torch import load_file
    corrupted = copy.deepcopy(traces)
    sent = next(e for e in corrupted[0]['events'] if e['kind'] == 'send')
    received = next(e for e in corrupted[1]['events'] if e['kind'] == 'receive' and e['peer'] == 0)
    value = load_file(portable.tensor_path(tmp_path/'baseline/rank-0/transcript', sent['tensor']['sha256']))['value']+1
    from neuroshard.evolution.sharded.checkpoint import tensor_file
    for rank in range(2):
        folder = tmp_path/'baseline'/f'rank-{rank}'/'transcript'
        temporary = folder/'forged.pending'
        spec = tensor_file(temporary, {'value': value})
        temporary.replace(portable.tensor_path(folder, spec['sha256']))
    sent['tensor'] = received['tensor'] = {**spec, 'shape': list(value.shape)}
    transcript.validate(corrupted)  # Self-consistent forged witnesses are not proof.
    with pytest.raises(ValueError, match='forward value or backward gradient'):
        replay_rank(0, corrupted[0])


def test_capacity_assignment_and_impossible_join_are_bounded():
    cfg = config()
    result = portable.layout(cfg, [50000, 60000, 50000], bytes_per_parameter=4, reserve_bytes=0)
    assert result[0] == 0 and result[-1] == 6 and len(result) == 4
    with pytest.raises(ValueError, match='capacity'):
        portable.layout(cfg, [2, 2, 2], reserve_bytes=0)


def test_growth_preserves_function_and_old_moments_with_new_adam_ages(tmp_path):
    torch.set_num_threads(1)
    student, teacher = models()
    optimizer = reference.optimizer_for(student, RECIPE)
    full_step(student, teacher, optimizer, 0)
    metas, paths = [], {}
    for rank in range(2):
        shard = Partition(config(), [0, 3, 6], rank)
        local = reference.optimizer_for(shard, RECIPE)
        old = dict(student.named_parameters())
        for name, p in shard.named_owned_parameters():
            p.data.copy_(old[name])
            local.state[p] = {k: v.clone() for k, v in optimizer.state[old[name]].items()}
        home = tmp_path/f'old-{rank}'
        meta = portable.write(home, shard, local, JOB, 1)
        metas.append(meta)
        paths.update({n: portable.tensor_path(portable.directory(home, 1), s['sha256'])
                      for n, s in meta['tensors'].items()})
    common = portable.assemble(metas, None)
    template = {n: p for n, p in paths.items() if n.startswith('model.layers.5.')}
    added = expansion.materialize(common, 2, template, tmp_path/'added')
    proposal = expansion.propose(common, 2, added, [4*1024**3]*3)
    assert proposal['step'] == 1
    assert all(proposal['tensors'][n] == s for n, s in common['tensors'].items())
    assert set(proposal['tensors'])-set(common['tensors']) == set(added)
    # A full model is used only as the independent, tiny CPU oracle.
    grown = LlamaForCausalLM(config(8)).float()
    from safetensors.torch import load_file
    with torch.no_grad():
        for n, p in grown.named_parameters():
            path = paths[n] if n in paths else portable.tensor_path(tmp_path/'added', added[n]['sha256'])
            p.copy_(load_file(path)['weight'])
        ids, _, mask, _ = batch_tensors(records(), 'cpu')
        before = student(ids, attention_mask=mask).logits
        after = grown(ids, attention_mask=mask).logits
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    grown_opt = reference.optimizer_for(grown, RECIPE)
    old = dict(student.named_parameters())
    for n, p in grown.named_parameters():
        if n in old:
            grown_opt.state[p] = {k: v.clone() for k, v in optimizer.state[old[n]].items()}
    full_step(grown, teacher, grown_opt, 1)
    for n, p in grown.named_parameters():
        assert float(grown_opt.state[p]['step']) == (2 if n in old else 1)
    assert grown.model.layers[6].mlp.down_proj.weight.count_nonzero() > 0
    assert grown.model.layers[7].self_attn.o_proj.weight.count_nonzero() > 0
    # All three new owners can load the same proposal without resetting old state.
    paths.update({n: portable.tensor_path(tmp_path/'added', s['sha256']) for n, s in added.items()})
    rng_path = portable.directory(tmp_path/'old-0', 1)/'rng.safetensors'
    installed = []
    for rank in range(3):
        subset = {n: p for n, p in paths.items() if owner(n, proposal['boundaries']) == rank}
        installed.append(portable.install_partition(tmp_path/f'new-{rank}', rank, proposal, subset, metas[0], rng_path))
    committed = portable.assemble(installed, identity(common), proposal['transition'])
    for rank in range(3):
        shard = Partition(config(8), proposal['boundaries'], rank)
        local = reference.optimizer_for(shard, RECIPE)
        births = portable.load(tmp_path/f'new-{rank}', shard, local, committed, JOB)
        for n, p in shard.named_owned_parameters():
            if n in old:
                for k, value in optimizer.state[old[n]].items():
                    torch.testing.assert_close(local.state[p][k], value, rtol=0, atol=0)
            else:
                assert births[n] == 1 and not local.state.get(p)
