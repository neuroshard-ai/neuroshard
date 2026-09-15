import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import LlamaConfig, LlamaForCausalLM

from neuroshard.evolution import rehearsal, reference, reference_data as data
from neuroshard.evolution.sharded import incremental, incremental_state, portable, checkpoint
from neuroshard.evolution.sharded import rehearsal_job as runner
from neuroshard.evolution.sharded.model import Partition
from neuroshard.evolution.sharded.wire import Wire


def small_config(layers):
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=layers,
                        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                        tie_word_embeddings=True, attention_dropout=0.)
    config._attn_implementation = 'sdpa'
    return config


def prepare_parent(folder):
    torch.manual_seed(81)
    model = LlamaForCausalLM(small_config(4)).float().eval()
    recipe = {'learning_rate': .00005, 'weight_decay': .01}
    optimizer = reference.optimizer_for(model, recipe)
    objects = folder / 'objects'
    objects.mkdir()
    tensors = {}
    for name, parameter in model.named_parameters():
        path = objects / 'pending.safetensors'
        spec = checkpoint.tensor_file(path, {'weight': parameter.detach()})
        path.replace(portable.tensor_path(objects, spec['sha256']))
        tensors[name] = {**spec, 'shape': list(parameter.shape), 'born': 0, 'group': int(parameter.ndim < 2)}
    parent = {'format': portable.FORMAT, 'job': 'fixture', 'step': 0,
        'config': portable.configuration(model.config), 'optimizer': portable.recipe(optimizer),
        'tensors': tensors, 'boundaries': [0, 2, 4], 'shards': [], 'parent': None, 'transition': None}
    parent['state_root'] = portable.learned_root(parent)
    portable.validate(parent)
    data.save(folder / 'parent.json', parent)


class Tokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [1, 3, 4]

    def decode(self, ids, skip_special_tokens):
        return '{}'


def worker(rank, folder, port, arm):
    folder = Path(folder)
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE='3', MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port),
                      PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    parent = json.loads((folder / 'parent.json').read_bytes())
    recipe = {'steps': 8, 'batch_documents': 3, 'warmup_steps': 1, 'weight_decay': .01,
              'clip_norm': 1., 'learning_rate': .00005}
    layouts = {'append': {'layers': 6, 'boundaries': [0, 2, 4, 6], 'frozen_layers': 4},
               'tail-control': {'layers': 4, 'boundaries': [0, 1, 2, 4], 'frozen_layers': 2}}
    plan = {'arms': list(layouts), 'parent': data.identity(parent), 'recipe': recipe,
            'checkpoints': [0, 8], 'selection_path': 'absent-selection.json'}
    original = {'runtime': {}, 'threads': 1, 'max_length': 32, 'max_seconds': 300,
        'seeds': {'training': 87}, 'arms': layouts, 'microbatch': 2, 'parameter_limit': 1000000,
        'reference': {'kl_strength': 2., 'margin_strength': 1., 'margin_min': .5, 'margin_max': 2.},
        'generation': {'knowledge': 3, 'skills': 3}}
    rows = [{'id': str(i), 'input_ids': [1, 3, 4, 8 + i, 2], 'labels': [-100, -100, -100, 8 + i, 2],
        'targets': 2, 'loss_weight': .5, 'distill': i != 1,
        'messages': [{'role': 'user', 'content': 'input'}, {'role': 'assistant', 'content': 'output'}],
        'task': {'family': 'directory', 'expected': 'value'}} for i in range(3)]
    prepared = {'schedule': [[0, 1, 2] for _ in range(128)]}
    bound = {'schedule': list(range(8))}
    rehearsal.ROOT = folder
    rehearsal.validate = lambda: (plan, original, prepared, bound)
    runner.tokenizer_for = lambda original, seed: Tokenizer()
    runner.base.schedule = lambda original, actual: prepared['schedule']
    runner.base.read_role = lambda prepared, inputs, role, tokenizer, maximum: rows if role != 'dev-skills' else []
    args = SimpleNamespace(command='train', arm=arm, parent=folder / 'parent.json',
        objects=folder / 'objects', inputs=folder, seed=folder, home=folder / f'rank-{rank}', resume=None)
    runner.run(args, device='cpu')
    common = json.loads((args.home / 'commit-000008.json').read_bytes())
    report = json.loads((args.home / 'evaluation.json').read_bytes())
    assert len(report['outcomes']['dev-knowledge']['answers']) == 3
    assert len(report['outcomes']['dev-conversation']['losses']) == 3
    assert report['feature_root'] and report['checkpoint'] == data.identity(common)
    # Compare the entire runner, durable bank and repeated local learner with
    # the existing distributed gradient path. The test changes only small
    # fixture preparation, never either numerical training implementation.
    layout = layouts[arm]
    shard = Partition(small_config(layout['layers']), layout['boundaries'], rank)
    incremental_state.initialize(shard, parent, args.objects, arm, layout['frozen_layers'])
    teacher = incremental.reference_tail(shard, layout['frozen_layers']) if arm == 'tail-control' else None
    optimizer = incremental.configure(shard, layout['frozen_layers'], recipe)
    os.environ['MASTER_PORT'] = str(port + 1)
    dist.init_process_group('gloo', timeout=timedelta(seconds=60))
    wire = Wire(rank, 3)
    try:
        for index in range(8):
            incremental.train_step(shard, optimizer, wire, rows, recipe, index, 2,
                layout['frozen_layers'], reference_layers=teacher, **original['reference'])
        observed = Partition(small_config(layout['layers']), layout['boundaries'], rank)
        observed_optimizer = incremental.configure(observed, layout['frozen_layers'], recipe)
        incremental_state.load(args.home, observed, observed_optimizer, common, parent,
                               rehearsal.job(plan, bound, arm), recipe)
        for (name, left), (other, right) in zip(shard.named_owned_parameters(), observed.named_owned_parameters()):
            assert name == other and torch.equal(left, right)
            if left.requires_grad:
                assert all(torch.equal(value, observed_optimizer.state[right][key])
                           for key, value in optimizer.state[left].items())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('arm', ['append', 'tail-control'])
def test_rehearsal_driver_reuses_durable_features_and_matches_distributed_weights_and_adam(tmp_path, arm):
    prepare_parent(tmp_path)
    with socket.socket() as port:
        port.bind(('127.0.0.1', 0))
        number = port.getsockname()[1]
    mp.spawn(worker, args=(str(tmp_path), number, arm), nprocs=3, join=True)


def test_rehearsal_schedule_and_selection_cannot_shorten_training_or_drop_a_control():
    plan = {'arms': ['append', 'tail-control'], 'epochs': 8, 'recipe': {'steps': 1024}}
    prepared = {'schedule': [[index] for index in range(128)]}
    schedule = rehearsal.schedule(plan, prepared)
    assert len(schedule) == 1024 and all(schedule.count(index) == 8 for index in range(128))
    bound = {'schedule': schedule}
    candidates = [{'arm': arm, 'job': rehearsal.job(plan, bound, arm), 'step': 1024,
                   'checkpoint': arm, 'decision': {'passed': False}} for arm in plan['arms']]
    assert rehearsal.selection(plan, bound, candidates)['selected'] == {'append': None, 'tail-control': None}
    with pytest.raises(ValueError, match='Both completed arms'):
        rehearsal.selection(plan, bound, candidates[:1])
    changed = copy.deepcopy(candidates)
    changed[0]['step'] = 512
    with pytest.raises(ValueError, match='terminal'):
        rehearsal.selection(plan, bound, changed)


def test_failed_development_never_reads_the_final_inputs(tmp_path, monkeypatch):
    plan = {'selection_path': 'selection.json'}
    bound = {'data': 'prepared'}
    selected = {'plan': data.identity(plan), 'prepared': data.identity(bound),
                'selected': {'append': None, 'tail-control': None}}
    data.save(tmp_path / 'selection.json', selected)
    monkeypatch.setattr(rehearsal, 'ROOT', tmp_path)
    monkeypatch.setattr(rehearsal, 'validate', lambda: (plan, {}, {}, bound))
    monkeypatch.setattr(runner.base, 'committed', lambda path: Path(path).read_bytes())
    monkeypatch.setattr(runner.base, 'read_role', lambda *args: pytest.fail('A failed candidate opened final inputs'))
    args = SimpleNamespace(command='score', candidate_reports=[], home=tmp_path / 'scored')
    runner.decide(args)
    result = json.loads((args.home / 'results.json').read_bytes())
    assert result['finals_opened'] is False and result['passed'] is False
