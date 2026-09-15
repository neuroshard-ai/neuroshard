"""Operated second-cohort training with the earlier model kept available."""
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from transformers import LlamaConfig

from .. import cohort_experiment as contract, cohort_questions, reference, reference_data as data
from . import cohort_features, cohort_state, feature_bank, features, incremental, incremental_state, portable
from .branch_groups import GroupWire, OrderedRoutes, RoutedNetwork
from .feature_probe import load_head
from .incremental_job import tokenizer_for
from .model import Partition, batch_tensors, weighted_loss
from .wire import Wire


def read(path):
    return json.loads(Path(path).read_bytes())


def answer_equal(left, right):
    return (left['id'] == right['id'] and left['ids'] == right['ids']
            and left['text'] == right['text'] and left['route'] == right['route'])


def signal(home, name, binding, timeout=900):
    path = home / name
    began = time.monotonic()
    while not path.exists():
        if time.monotonic() - began > timeout:
            raise TimeoutError('No controller receipt for ' + name)
        time.sleep(.1)
    result = read(path)
    if result['binding'] != binding:
        raise ValueError('Controller receipt belongs to a different learning job')
    return result


def run(args, device='cuda', *, experiment=contract, network_factory=None):
    contract = experiment
    plan, prepared = contract.validate()
    final = args.command == 'final'
    if args.command not in ('train', 'final'):
        raise ValueError('Require training/development or the selected final')
    selection = contract.final_selection(plan, prepared) if final else None
    if not final and contract.SELECTION.exists():
        raise ValueError('Training is closed after selecting this cohort')
    job = contract.job(plan, prepared)
    parent, first = read(args.parent), read(args.first)
    if data.identity(parent) != plan['parent'] or data.identity(first) != plan['expert']:
        raise ValueError('Established model identities changed')
    incremental_state.validate(first, parent)
    selected = read(args.second) if args.second else None
    if final:
        if (selected is None or data.identity(selected) != selection['checkpoint']
                or selected['job'] != job or selected['step'] != plan['training']['steps']
                or data.identity(contract.graph(plan, parent, first, selected)) != selection['graph']):
            raise ValueError('Restore exactly the selected terminal graph')
    elif selected is not None:
        raise ValueError('This fixed learning run starts one new expert from its frozen parent')
    cache = read(args.inputs / 'retention-cache.json')
    if data.identity(cache) != prepared['retention_cache'] or cache['graph'] != plan['previous_graph']:
        raise ValueError('Retained outcomes differ from the actual first-cohort result')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != 5 or not 0 <= rank < world:
        raise ValueError('Require three parent owners and two separate expert owners')
    args.home.mkdir(parents=True, exist_ok=False)
    tokenizer = tokenizer_for(plan, args.seed)
    runtime = reference.configure(device, plan['threads'])
    runtime['allocator'] = os.environ['PYTORCH_CUDA_ALLOC_CONF']
    if {key: runtime[key] for key in plan['runtime']} != plan['runtime']:
        raise ValueError('Cohort numerical runtime changed')
    config = LlamaConfig(**parent['config'])
    config._attn_implementation = 'sdpa'
    shard = Partition(config, plan['parent_layout'] if rank < 3 else plan['expert_layout'],
                      min(rank, 3), device, plan['parameter_limit'])
    records = incremental_state.records(parent) if rank != 3 else first['tensors']
    objects = args.first_objects if rank == 3 else args.objects
    with torch.no_grad():
        for name, parameter in shard.named_owned_parameters():
            values = incremental_state.tensor_values(portable.tensor_path(objects, records[name]['sha256']), records[name])
            parameter.copy_(values['weight'])
            parameter.requires_grad_(False)
            del values
    optimizer = None
    if rank == 4 and final:
        incremental_state.load(args.second.parent, shard, None, selected, parent, job,
                               plan['training'], restore_optimizer=False)
    shard.eval()
    versions = tuple(parameter._version for _, parameter in shard.named_owned_parameters())
    dist.init_process_group('gloo', timeout=timedelta(seconds=900))
    parent_group = dist.new_group([0, 1, 2], timeout=timedelta(seconds=900))
    groups = {rule['id']: dist.new_group([0, 1, 2, rule['owner']], timeout=timedelta(seconds=900))
              for rule in plan['rules']}
    routes = OrderedRoutes(plan['rules'])
    net = RoutedNetwork(rank, shard, tokenizer, plan['split'], routes, parent_group, groups)
    fixed_path = network_factory(args, plan, prepared, net) if network_factory else None
    additional_parameters = fixed_path.additional_parameters if fixed_path else 0
    all_owners = Wire(rank, 5)
    binding = {'plan': data.identity(plan), 'prepared': data.identity(prepared), 'job': job,
               'previous_graph': plan['previous_graph'], 'retention_cache': prepared['retention_cache']}
    started = time.monotonic()
    try:
        declarations = all_owners.exchange({'binding': binding, 'rank': rank, 'runtime': plan['runtime']})
        if declarations != [{'binding': binding, 'rank': i, 'runtime': plan['runtime']} for i in range(5)]:
            raise ValueError('Owners disagree on the complete learning contract')
        data.save(args.home / 'started.json', {**binding, 'rank': rank, 'runtime': runtime,
            'owned_parameters': shard.resident_parameters + additional_parameters, 'tokens_issued': 0})

        def expired():
            if time.monotonic() - started > plan['max_seconds']:
                raise TimeoutError('Frozen cohort numerical deadline expired')

        def answer(question, cap, new=True):
            expired()
            if new:
                value = net.answer(question, cap)
            elif rank < 3:
                value = net.networks['directory'].generate(question, cap, False)
            else:
                value = None
            values = all_owners.exchange(value)
            if values[0] is None or any(item is not None and item != values[0] for item in values):
                raise ValueError('Owners disagree on the actual generated answer')
            return values[0]

        prefix = 'test' if final else 'dev'
        new_rows = contract.rows(prepared, args.inputs, prefix, tokenizer, plan['max_length'])
        before = []
        for row in new_rows:
            value = answer(row['messages'][0]['content'], plan['generation']['new'], False)
            before.append({'id': row['id'], **value})
        data.save(args.home / 'new-baseline.json', before)
        feature_root, production, learning = None, None, None
        if not final:
            training = contract.rows(prepared, args.inputs, 'train', tokenizer, plan['max_length'])
            feature_binding = {**binding, 'cut': plan['split'], 'batches': data.identity(prepared['batches']),
                               'runtime': plan['runtime']}
            began = time.monotonic()
            if rank != 3:
                feature_wire = GroupWire(rank, [0, 1, 2, 4], groups['protocol'])
                feature_root = cohort_features.produce(shard, feature_wire, training, prepared['batches'],
                    args.home / 'features', feature_binding, plan['split'], plan['microbatch'])
                production = {'root': feature_root, 'seconds': time.monotonic() - began,
                              'sent_tensor_bytes': feature_wire.sent_tensor_bytes}
            feature_root = all_owners.exchange(feature_root)[4]
            data.save(args.home / 'feature-production.json', {'root': feature_root, 'local': production})
            if rank == 4:
                bank = feature_bank.Reader(args.home / 'features', feature_root, feature_binding,
                                           config, plan['microbatch'], len(prepared['batches']))
                head = load_head(parent, args.objects, device)
                optimizer = incremental.configure(shard, plan['split'], plan['training'])
                selected = cohort_state.commit_tail(args.home, shard, optimizer, parent, args.objects,
                    job, 0, plan['training'], plan['split'])
                began = time.time()
                for index, batch_index in enumerate(prepared['schedule']):
                    expired()
                    rows = [training[i] for i in prepared['batches'][batch_index]]
                    packets = bank.batch(batch_index, rows, device)
                    result = features.train_step(shard, head, optimizer, packets, rows,
                        plan['training'], index, plan['microbatch'], **plan['objective'])
                    result.update(time=time.time(), batch=batch_index)
                    with (args.home / 'updates.jsonl').open('a') as output:
                        output.write(json.dumps(result, sort_keys=True) + '\n')
                        output.flush()
                    if index == 0:
                        data.save(args.home / 'first-update.json', {'binding': binding, **result})
                    if index + 1 == plan['checkpoints'][1]:
                        receipt = signal(args.home, 'earlier-paths-served.json', binding)
                        if receipt['parent_answers'] < 1 or receipt['first_expert_answers'] < 1:
                            raise ValueError('Both established paths must answer during learning')
                    if index + 1 in plan['checkpoints']:
                        selected = cohort_state.commit_tail(args.home, shard, optimizer, parent,
                            args.objects, job, index + 1, plan['training'], plan['split'])
                    if (index + 1) % 32 == 0:
                        print(json.dumps({'event': 'trained', **result}), flush=True)
                    del packets
                if not head.unchanged():
                    raise ValueError('The replicated frozen output head changed')
                learning = {'binding': binding, 'checkpoint': data.identity(selected),
                    'first': began, 'last': time.time(), 'steps': plan['training']['steps'],
                    'feature_root': feature_root, 'tokens_issued': 0}
                data.save(args.home / 'training-complete.json', learning)
                del optimizer, head, bank
                optimizer = None
                shard.requires_grad_(False)
                shard.eval()
            else:
                signal(args.home, 'new-learner-started.json', binding)
                service_wire = net.networks['directory'].wire
                probes = []
                for suffix in ('knowledge', 'skills'):
                    role = 'retained-dev-' + suffix
                    examples = contract.rows(prepared, args.inputs, role)
                    expected = cache['roles'][role]['answers']
                    probes.append((examples[0], expected[0], plan['generation']['retained_' + suffix]))
                index = 0
                while True:
                    expired()
                    stop = None
                    if rank == 0:
                        path = args.home / 'new-learner-finished.json'
                        stop = path.exists()
                        if stop and read(path)['binding'] != binding:
                            raise ValueError('Learning completion belongs to a different job')
                    if service_wire.exchange(stop)[0]:
                        break
                    row, expected, cap = probes[index % len(probes)]
                    began = time.time()
                    value = net.answer(row['messages'][0]['content'], cap)
                    values = service_wire.exchange(value)
                    if values[0] is None or any(item is not None and item != values[0] for item in values):
                        raise ValueError('Established owners disagree while the new peer trains')
                    value = values[0]
                    actual = {'id': row['id'], **value}
                    if not answer_equal(actual, expected):
                        raise ValueError('Established answer changed during new learning')
                    with (args.home / 'service.jsonl').open('a') as output:
                        output.write(json.dumps({'binding': binding, 'started': began, 'finished': time.time(),
                                                 'answer': actual, 'index': index}, sort_keys=True) + '\n')
                        output.flush()
                    index += 1
                data.save(args.home / 'service-complete.json', {'binding': binding, 'answers': index})
            selected = all_owners.exchange(selected)[4]
            learning = all_owners.exchange(learning)[4]
            if selected['job'] != job or selected['step'] != plan['training']['steps']:
                raise ValueError('The new learner did not finish the fixed schedule')
            incremental_state.validate(selected, parent)
            data.save(args.home / 'selected-checkpoint.json', selected)
        descriptor = contract.graph(plan, parent, first, selected)
        outcomes = []
        for index, row in enumerate(new_rows):
            value = answer(row['messages'][0]['content'], plan['generation']['new'])
            if value['expert'] != 'protocol':
                raise ValueError('New documentation question took the wrong model path')
            outcomes.append({'id': row['id'], **value})
            if (index + 1) % 16 == 0:
                print(json.dumps({'event': 'new-answers', 'rank': rank, 'count': index + 1}), flush=True)
        decision = cohort_questions.decision(new_rows, before, outcomes, plan['gate'])
        data.save(args.home / 'new-answers.json', {'before': before, 'after': outcomes, 'decision': decision})
        retained = {}
        for suffix in ('knowledge', 'skills', 'conversation'):
            role = 'retained-' + prefix + '-' + suffix
            examples = contract.rows(prepared, args.inputs, role)
            expected = cache['roles'][role]
            result = {'answers': [], 'losses': []}
            for index, row in enumerate(examples):
                expired()
                if suffix == 'conversation':
                    if routes.select(row['messages'][0]['content']) is not None:
                        raise ValueError('An expert captured an original conversation')
                    value = None
                    if rank < 3:
                        hidden = net.networks['directory'].forward(row['input_ids'], False)
                        if rank == 0:
                            _, labels, _, weights = batch_tensors([row], device)
                            with torch.no_grad(), reference.autocast(device):
                                value = float(weighted_loss(shard.logits(hidden), labels, torch.ones_like(weights))) / row['targets']
                    loss = {'id': row['id'], 'targets': row['targets'], 'loss': all_owners.exchange(value)[0]}
                    if loss != expected['losses'][index]:
                        raise ValueError('An established conversation loss changed')
                    result['losses'].append(loss)
                else:
                    value = answer(row['messages'][0]['content'], plan['generation']['retained_' + suffix])
                    actual = {'id': row['id'], **value}
                    if not answer_equal(actual, expected['answers'][index]):
                        raise ValueError('An established generated answer changed')
                    result['answers'].append(actual)
                if (index + 1) % 32 == 0:
                    print(json.dumps({'event': 'retained', 'rank': rank, 'role': role, 'count': index + 1}), flush=True)
            retained[role] = result
            data.save(args.home / (role + '.json'), result)
        if rank < 4 and tuple(p._version for _, p in shard.named_owned_parameters()) != versions:
            raise ValueError('An established parameter was modified')
        if fixed_path:
            fixed_path.verify_unchanged()
        digest = data.identity({'before': before, 'new': outcomes, 'retained': retained})
        if any(item != digest for item in all_owners.exchange(digest)):
            raise ValueError('Owners disagree on complete second-cohort outcomes')
        decision['checks']['exact_retained_answers_and_losses'] = True
        decision['checks']['established_parameters_unchanged'] = True
        data.save(args.home / 'result.json', {**binding, 'rank': rank, 'graph': data.identity(descriptor),
            'descriptor': descriptor, 'checkpoint': data.identity(selected), 'decision': decision,
            'answer_identity': digest, 'learning': learning, 'feature_root': feature_root,
            'seconds': time.monotonic() - started,
            'owned_parameters': shard.resident_parameters + additional_parameters,
            'tokens_issued': 0, 'native_activated': False})
        all_owners.exchange('new expert may exit; prior paths remain available')
        if rank == 4:
            return
        receipt = signal(args.home, 'new-expert-exited.json', binding, timeout=300)
        if receipt['exit_code'] != 0 or not receipt['process_exit_observed']:
            raise ValueError('Require the controller-observed new expert exit')
        survival = []
        for suffix in ('knowledge', 'skills'):
            role = 'retained-' + prefix + '-' + suffix
            row = contract.rows(prepared, args.inputs, role)[0]
            value = net.answer(row['messages'][0]['content'], plan['generation']['retained_' + suffix])
            values = net.networks['directory'].wire.exchange(value)
            if values[0] is None or any(item is not None and item != values[0] for item in values):
                raise ValueError('Established owners disagree after the new expert exited')
            value = values[0]
            actual = {'id': row['id'], **value}
            if not answer_equal(actual, cache['roles'][role]['answers'][0]):
                raise ValueError('Established path changed after the new expert exited')
            survival.append(actual)
        if any(item != survival for item in net.networks['directory'].wire.exchange(survival)):
            raise ValueError('Established owners disagree after departure')
        if fixed_path:
            fixed_path.verify_unchanged()
        data.save(args.home / 'survival.json', {'binding': binding, 'rank': rank, 'passed': True,
                                             'answers': survival, 'exit': receipt})
    finally:
        dist.destroy_process_group()
