"""Exercise the actual operated job and receipts with five small CPU owners."""
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp
from transformers import LlamaConfig, LlamaForCausalLM

from neuroshard.evolution import cohort_experiment as contract, reference_data as data
from neuroshard.evolution.sharded import checkpoint, cohort_job, cohort_state, incremental_state, portable
from neuroshard.evolution.sharded.model import Partition, batch_tensors, weighted_loss
from test_expert_cohort_state import prepare, update, RECIPE


class Tokenizer:
    eos_token_id = 2
    all_special_tokens = ['<eos>']

    def apply_chat_template(self, messages, **kwargs):
        return [1, 4, 9] if len(messages) == 1 else [1, 4, 9, 12, 2]

    def decode(self, values, **kwargs):
        return ','.join(map(str, values))


def reference_answer(model, question):
    ids, output = [1, 4, 9], []
    with torch.no_grad():
        for _ in range(4):
            tokens = torch.tensor([ids])
            token = int(model.logits(model(tokens, torch.ones_like(tokens)))[:, -1].argmax(-1))
            ids.append(token)
            output.append(token)
            if token == 2:
                break
    return {'ids': output, 'text': Tokenizer().decode(output),
            'route': 'expert' if 'fictional luma directory' in question.casefold() else 'parent'}


def inputs(home):
    parent, owned, first_shard, first_optimizer = prepare(home)
    update(first_shard, first_optimizer, 0)
    first = cohort_state.commit_tail(home / 'first', first_shard, first_optimizer, parent,
                                    owned, 'b' * 64, 1, RECIPE, 5)
    # Reconstruct the same tiny full parent solely for this test's oracle and
    # per-owner object staging. The real GPU driver never does this.
    torch.manual_seed(14)
    config = LlamaConfig(**parent['config'])
    config._attn_implementation = 'sdpa'
    model = LlamaForCausalLM(config).float()
    objects = home / 'objects'
    objects.mkdir()
    for name, parameter in model.named_parameters():
        values = {'weight': parameter.detach(), 'step': torch.tensor(7.),
                  'exp_avg': torch.ones_like(parameter) * .001,
                  'exp_avg_sq': torch.ones_like(parameter) * .002}
        temporary = home / 'pending'
        spec = checkpoint.tensor_file(temporary, values)
        assert spec['sha256'] == parent['tensors'][name]['sha256']
        temporary.replace(portable.tensor_path(objects, spec['sha256']))
    oracle = Partition(config, [0, 6], 0).requires_grad_(False)
    first_oracle = Partition(config, [0, 6], 0).requires_grad_(False)
    with torch.no_grad():
        for target, expert in ((oracle, False), (first_oracle, True)):
            for name, parameter in target.named_owned_parameters():
                record = incremental_state.records(parent)[name]
                source = objects
                if expert and name.startswith('model.layers.5.'):
                    record, source = first['tensors'][name], home / 'first/shard-000001'
                parameter.copy_(incremental_state.tensor_values(portable.tensor_path(source, record['sha256']), record)['weight'])
    plan = {'format': contract.FORMAT, 'parent': data.identity(parent), 'expert': data.identity(first),
        'previous_graph': 'c' * 64, 'parent_layout': [0, 2, 4, 6], 'expert_layout': [0, 2, 4, 5, 6],
        'split': 5, 'threads': 1, 'runtime': {'device': 'cpu', 'threads': 1, 'allocator': 'test'},
        'parameter_limit': 100000, 'max_seconds': 120, 'max_length': 64, 'tokenizer': 'd' * 64,
        'training': {**RECIPE, 'steps': 4}, 'checkpoints': [0, 2, 4], 'microbatch': 8,
        'objective': {'kl_strength': 0., 'margin_strength': 0.},
        'generation': {'new': 4, 'retained_knowledge': 4, 'retained_skills': 4},
        'rules': [{'id': 'directory', 'needle': 'fictional luma directory', 'owner': 3},
                  {'id': 'protocol', 'needle': 'neuroshard 0.4.0', 'owner': 4}],
        'gate': {'single_accuracy': .75, 'composed_accuracy': .5, 'gain_lower': .1,
                 'bootstrap_samples': 100, 'bootstrap_seed': 42, 'confidence': .95}}
    data.save(home / 'parent.json', parent)
    data.save(home / 'first.json', first)
    data.save(home / 'plan.json', plan)
    directory = home / 'inputs'
    directory.mkdir()

    def row(index, question, topics=('first-fact',), answers=('12',)):
        messages = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': '; '.join(answers)}]
        encoded = data.conversation(Tokenizer(), messages, 64)
        return {'id': str(index), 'stratum': 'single' if len(topics) == 1 else 'composed',
                'topics': list(topics), 'answers': list(answers), 'messages': messages,
                **encoded, 'loss_weight': 1. / encoded['targets']}

    roles = {'train': [row(i, 'NeuroShard 0.4.0 training ' + str(i)) for i in range(896)],
        'dev': [row(900, 'NeuroShard 0.4.0 question one'),
                row(901, 'NeuroShard 0.4.0 question two', ('second-fact',)),
                row(902, 'NeuroShard 0.4.0 two questions', ('first-fact', 'second-fact'), ('12', '12'))]}
    cache = {'graph': plan['previous_graph'], 'roles': {}}
    for suffix, question in [('knowledge', 'fictional Luma directory question'),
                             ('skills', 'An established general question'),
                             ('conversation', 'An established conversation')]:
        role = 'retained-dev-' + suffix
        value = row(role, question)
        roles[role] = [value]
        outcome = {'answers': [], 'losses': []}
        if suffix == 'conversation':
            ids, labels, mask, weights = batch_tensors([value], 'cpu')
            with torch.no_grad():
                loss = float(weighted_loss(oracle.logits(oracle(ids, mask)), labels, torch.ones_like(weights))) / value['targets']
            outcome['losses'] = [{'id': value['id'], 'targets': value['targets'], 'loss': loss}]
        else:
            outcome['answers'] = [{'id': value['id'], **reference_answer(first_oracle if suffix == 'knowledge' else oracle, question)}]
        cache['roles'][role] = outcome
    data.save(directory / 'retention-cache.json', cache)
    specs = {}
    for role, rows in roles.items():
        path = directory / (role + '.jsonl')
        path.write_text(''.join(json.dumps(value) + '\n' for value in rows))
        specs[role] = {'count': len(rows), 'ids': data.identity([value['id'] for value in rows]), 'sha256': data.sha256(path)}
    prepared = {'roles': specs, 'retention_cache': data.identity(cache),
                'batches': [list(range(i, i + 32)) for i in range(0, 896, 32)], 'schedule': [0, 1, 2, 3]}
    data.save(home / 'prepared.json', prepared)
    return plan, prepared


def worker(rank, home, address, port):
    home = Path(home)
    os.environ.update(RANK=str(rank), WORLD_SIZE='5', MASTER_ADDR=address, MASTER_PORT=str(port),
                      PYTORCH_CUDA_ALLOC_CONF='test')
    plan, prepared = cohort_job.read(home / 'plan.json'), cohort_job.read(home / 'prepared.json')
    # Only the frozen GPU profile/large-model contract is replaced. The actual
    # worker, neural computations, data validation, checkpoint and receipt code
    # execute unchanged against the small CPU fixture.
    contract.validate = lambda: (plan, prepared)
    contract.SELECTION = home / 'selection.json'
    cohort_job.tokenizer_for = lambda *args: Tokenizer()
    args = SimpleNamespace(command='train', inputs=home / 'inputs', parent=home / 'parent.json',
        first=home / 'first.json', second=None, objects=home / 'objects',
        first_objects=home / 'first/shard-000001', seed=home / 'seed', home=home / f'rank-{rank}')
    cohort_job.run(args, device='cpu')


def test_actual_training_job_serves_and_survives_learner_exit(tmp_path):
    import socket
    plan, prepared = inputs(tmp_path)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    context = mp.spawn(worker, args=(str(tmp_path), '127.0.0.1', port), nprocs=5, join=False)
    began, sent_start, sent_service, sent_finish, sent_exit = time.monotonic(), False, False, False, False
    try:
        while any(process.is_alive() for process in context.processes):
            assert time.monotonic() - began < 90
            assert all(process.exitcode in (None, 0) for process in context.processes)
            first_path = tmp_path / 'rank-4/first-update.json'
            if first_path.exists() and not sent_start:
                first = cohort_job.read(first_path)
                binding = first['binding']
                for rank in range(4):
                    data.save(tmp_path / f'rank-{rank}/new-learner-started.json', {'binding': binding})
                sent_start = True
            service_path = tmp_path / 'rank-0/service.jsonl'
            if sent_start and service_path.exists() and not sent_service:
                complete = service_path.read_text().splitlines()
                values = [json.loads(line) for line in complete]
                counts = {route: sum(value['answer']['route'] == route for value in values) for route in ('parent', 'expert')}
                if all(counts.values()):
                    data.save(tmp_path / 'rank-4/earlier-paths-served.json', {'binding': binding,
                        'parent_answers': counts['parent'], 'first_expert_answers': counts['expert']})
                    sent_service = True
            if (tmp_path / 'rank-4/training-complete.json').exists() and not sent_finish:
                data.save(tmp_path / 'rank-0/new-learner-finished.json', {'binding': binding})
                sent_finish = True
            if context.processes[4].exitcode == 0 and not sent_exit:
                for rank in range(4):
                    data.save(tmp_path / f'rank-{rank}/new-expert-exited.json',
                              {'binding': binding, 'exit_code': 0, 'process_exit_observed': True})
                sent_exit = True
            time.sleep(.02)
        while not context.join(timeout=5):
            pass
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
    assert sent_start and sent_service and sent_finish and sent_exit
    results = [cohort_job.read(tmp_path / f'rank-{rank}/result.json') for rank in range(5)]
    assert len({result['graph'] for result in results}) == 1
    assert all(result['learning']['steps'] == 4 and result['tokens_issued'] == 0 for result in results)
    assert all(cohort_job.read(tmp_path / f'rank-{rank}/survival.json')['passed'] for rank in range(4))
