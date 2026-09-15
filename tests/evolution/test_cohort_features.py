"""Five processes: learn a new tail while retaining both earlier serving paths."""
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from transformers import LlamaConfig

from neuroshard.evolution.reference import learning_rate
from neuroshard.evolution.sharded import cohort_features, feature_bank, features, incremental
from neuroshard.evolution.sharded.branch_groups import GroupWire, OrderedRoutes, RoutedNetwork
from neuroshard.evolution.sharded.guarded import response_logits
from neuroshard.evolution.sharded.model import Partition, batch_tensors
from neuroshard.evolution.sharded.wire import Wire

RULES = [{'id': 'first', 'needle': 'first domain', 'owner': 3},
         {'id': 'second', 'needle': 'second domain', 'owner': 4}]
RECIPE = {'steps': 8, 'warmup_steps': 1, 'learning_rate': .002,
          'weight_decay': .01, 'clip_norm': .01}


class Tokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return [7, 11, 19]

    def decode(self, values, **kwargs):
        return ','.join(map(str, values))


def initialize(shard, variant='parent'):
    with torch.no_grad():
        for name, parameter in shard.named_owned_parameters():
            suffix = '/' + variant if name.startswith('model.layers.5.') else ''
            seed = int(hashlib.sha256((name + suffix).encode()).hexdigest()[:8], 16)
            parameter.copy_(torch.randn(parameter.shape, generator=torch.Generator().manual_seed(seed)) * .1)
            parameter.requires_grad_(False)
    shard.eval()


def wait_for(path):
    deadline = time.monotonic() + 45
    while not path.exists():
        assert time.monotonic() < deadline, str(path)
        time.sleep(.01)


def oracle_step(model, optimizer, rows, index):
    """Independent full autograd calculation; only present in this tiny test."""
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = learning_rate(RECIPE, index)
    denominator = sum(row['targets'] * row['loss_weight'] for row in rows)
    for offset in range(0, len(rows), 2):
        ids, labels, mask, weights = batch_tensors(rows[offset:offset + 2], 'cpu')
        logits, targets, active = response_logits(model, model(ids, mask), labels)
        ce = F.cross_entropy(logits.float(), targets, reduction='none')
        loss = (ce * weights[:, None].expand_as(active)[active]).sum() / denominator
        loss.backward()
    parameters = [p for _, p in model.named_owned_parameters() if p.requires_grad]
    norm = math.sqrt(math.fsum(float(torch.linalg.vector_norm(p.grad, dtype=torch.float64).square())
                              for p in parameters))
    scale = min(1., RECIPE['clip_norm'] / (norm + 1e-6))
    for parameter in parameters:
        parameter.grad.mul_(scale)
    optimizer.step()
    return norm


def process(rank, rendezvous, output):
    torch.set_num_threads(1)
    home = Path(output)
    config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=6,
        num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=True, attention_dropout=0,
        max_position_embeddings=64)
    config._attn_implementation = 'sdpa'
    shard = Partition(config, [0, 2, 4, 6] if rank < 3 else [0, 2, 4, 5, 6], min(rank, 3))
    initialize(shard, 'first' if rank == 3 else 'parent')
    initial = {name: p.detach().clone() for name, p in shard.named_owned_parameters()}
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=5,
                            timeout=timedelta(seconds=60))
    parent = dist.new_group([0, 1, 2], timeout=timedelta(seconds=60))
    groups = {rule['id']: dist.new_group([0, 1, 2, rule['owner']], timeout=timedelta(seconds=60))
              for rule in RULES}
    graph = RoutedNetwork(rank, shard, Tokenizer(), 5, OrderedRoutes(RULES), parent, groups)
    everyone = Wire(rank, 5)
    rows = [{'id': str(i), 'input_ids': [7, 11, 19] + [20 + i] * (i + 1) + [2],
             'labels': [-100] * 3 + [20 + i] * (i + 1) + [2],
             'targets': i + 2, 'loss_weight': 1. / (i + 2)} for i in range(5)]
    try:
        before = {question: graph.answer(question, 4) for question in ('general question', 'first domain')}
        everyone.exchange('earlier paths recorded')
        bank_root = None
        binding = {'purpose': 'concurrent-cohort-cpu-test'}
        if rank != 3:
            wire = GroupWire(rank, [0, 1, 2, 4], groups['second'])
            bank_root = cohort_features.produce(shard, wire, rows, [list(range(5))],
                home / 'features', binding, 5, 2)
        bank_root = everyone.exchange(bank_root)[4]
        if rank == 4:
            bank = feature_bank.Reader(home / 'features', bank_root, binding, config, 2, 1)
            packets = bank.batch(0, rows, 'cpu')
            oracle = Partition(config, [0, 6], 0)
            initialize(oracle)
            # Compare actual distributed cached representations against the
            # independently evaluated complete parent, including padding.
            captured = []
            hook = oracle.layers['4'].register_forward_hook(lambda module, inputs, value: captured.append(value.detach()))
            with torch.no_grad():
                for packet in packets:
                    expected = oracle(packet['ids'], packet['mask'])
                    assert torch.equal(packet['prefix'], captured.pop())
                    assert torch.equal(packet['reference'], expected)
            hook.remove()
            head = features.FrozenHead(config, oracle.embedding.weight.detach(), oracle.norm.weight.detach())
            optimizer = incremental.configure(shard, 5, RECIPE)
            active = []
            for name, parameter in oracle.named_owned_parameters():
                parameter.requires_grad_(name.startswith('model.layers.5.'))
                if parameter.requires_grad:
                    active.append(parameter)
            oracle_optimizer = torch.optim.AdamW([
                {'params': [p for p in active if p.ndim >= 2], 'weight_decay': RECIPE['weight_decay']},
                {'params': [p for p in active if p.ndim < 2], 'weight_decay': 0.},
            ], lr=RECIPE['learning_rate'], betas=(.9, .95), eps=1e-8, foreach=False)
            began = time.time()
            for index in range(RECIPE['steps']):
                report = features.train_step(shard, head, optimizer, packets, rows, RECIPE, index, 2,
                                             kl_strength=0., margin_strength=0.)
                norm = oracle_step(oracle, oracle_optimizer, rows, index)
                assert report['gradient_norm'] == norm
                oracle_parameters = dict(oracle.named_owned_parameters())
                for name, parameter in shard.named_owned_parameters():
                    expected = oracle_parameters[name]
                    assert torch.equal(parameter, expected), name
                    assert all(torch.equal(value, oracle_optimizer.state[expected][key])
                               for key, value in optimizer.state[parameter].items())
                if index == 0:
                    (home / 'first-update').write_text(str(time.time()))
                if index == 3:
                    wait_for(home / 'served-during-learning')
            assert any(not torch.equal(parameter, initial[name]) for name, parameter in shard.named_owned_parameters())
            assert head.unchanged()
            (home / 'learned.json').write_text(json.dumps({'steps': 8, 'first': began, 'last': time.time(),
                'clipping_active': norm > RECIPE['clip_norm'], 'exact_weights_and_adam': True}))
            shard.requires_grad_(False)
        else:
            wait_for(home / 'first-update')
            for question, expected in before.items():
                assert graph.answer(question, 4) == expected
            graph.networks['first'].wire.exchange('earlier paths answered during the learning job')
            if rank == 0:
                (home / 'served-during-learning').write_text(str(time.time()))
        everyone.exchange('new expert learned; retained paths served')
        for question, expected in before.items():
            assert graph.answer(question, 4) == expected
        graph.answer('second domain', 4)
        if rank < 4:
            assert all(torch.equal(parameter, initial[name]) for name, parameter in shard.named_owned_parameters())
        (home / f'owner-{rank}').write_text('passed')
    finally:
        dist.destroy_process_group()


def test_new_expert_learns_exactly_while_existing_shards_serve(tmp_path):
    mp.spawn(process, args=(str(tmp_path / 'group'), str(tmp_path)), nprocs=5, join=True)
    report = json.loads((tmp_path / 'learned.json').read_bytes())
    served = float((tmp_path / 'served-during-learning').read_text())
    assert report['first'] < served < report['last']
    assert report['exact_weights_and_adam'] and report['clipping_active']
    assert all((tmp_path / f'owner-{rank}').read_text() == 'passed' for rank in range(5))
