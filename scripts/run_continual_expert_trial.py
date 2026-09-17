#!/usr/bin/env python3
"""One frozen fresh-data trial on three parent shards and a continued expert.

This measures the learning method directly. It does not claim automatic routing,
independent operators, native admission or an approved public serving model.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
import time

import torch
import torch.distributed as dist
from transformers import LlamaConfig

from neuroshard.evolution import cohort_questions, expert_checkpoint, expert_data, expert_work, reference
from neuroshard.evolution.reference_data import identity, save, sha256
from neuroshard.evolution.sharded import cohort_features, cohort_state, expert_execution
from neuroshard.evolution.sharded import incremental, incremental_state, portable
from neuroshard.evolution.sharded.branch import Network, ParentWire
from neuroshard.evolution.sharded.expert_commitment import snapshot
from neuroshard.evolution.sharded.expert_replay import batch_identity
from neuroshard.evolution.sharded.incremental_job import tokenizer_for
from neuroshard.evolution.sharded.model import Partition
from neuroshard.evolution.sharded.wire import Wire

FORMAT = 'neuroshard-continual-expert-trial-v1'
ISOLATED = 'neuroshard-isolated-expert-training-v1'


def run(home, source):
    inputs = home/'inputs'
    read = lambda name: json.loads((inputs/name).read_bytes())
    trial, parent, plan, prepared, freeze = [read(name+'.json') for name in
                                           ('trial', 'parent', 'plan', 'prepared', 'freeze')]
    isolated = trial['format'] == ISOLATED
    if (trial['format'] not in (FORMAT, ISOLATED) or identity(trial) != freeze['trial']
            or identity(plan) != freeze['plan'] or identity(prepared) != freeze['prepared']
            or sha256(source/'scripts/run_continual_expert_trial.py') != freeze['driver']
            or plan['parent'] != identity(parent) or not plan.get('seed_expert')
            or plan['training'] != trial['training'] or plan['objective'] != trial['objective']):
        raise ValueError('Freeze the entire continued-learning prescription before execution')
    if isolated and (trial.get('retain_boundaries') != 2 or trial.get('evaluation_mode') != 'separate-frozen-serving'):
        raise ValueError('Isolated training retains the terminal replay boundary and requires a separate serving gate')
    for path, expected in freeze['sources'].items():
        if Path(path).is_absolute() or '..' in Path(path).parts or sha256(source/path) != expected:
            raise ValueError('Frozen trial execution source changed')
    expert_data.validate_prepared(prepared, plan, parent['config']['vocab_size'])
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != 4 or not 0 <= rank < world:
        raise ValueError('This trial requires three parent owners and one separate learner')
    output = home/'results'
    output.mkdir(exist_ok=False)
    runtime = reference.configure(plan['runtime']['device'], plan['threads'])
    runtime['allocator'] = os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
    if any(runtime.get(key) != value for key, value in plan['runtime'].items()):
        raise ValueError('The frozen numerical runtime changed')
    tokenizer = tokenizer_for(plan, home/'seed')
    config = LlamaConfig(**parent['config'])
    config._attn_implementation = 'sdpa'
    shard = Partition(config, plan['parent_layout'] if rank < 3 else plan['expert_layout'], rank,
                      runtime['device'], plan['parameter_limit'])
    seed = plan['seed_expert']['checkpoint']
    if rank == 3:
        cohort_state.initialize_tail(shard, parent, home/'objects', plan['split'], seed)
    else:
        inherited = incremental_state.records(parent)
        with torch.no_grad():
            for name, parameter in shard.named_owned_parameters():
                spec = inherited[name]
                values = incremental_state.tensor_values(portable.tensor_path(home/'objects', spec['sha256']), spec)
                parameter.copy_(values['weight'])
                del values
    shard.eval().requires_grad_(False)
    started = time.monotonic()
    dist.init_process_group('gloo', timeout=timedelta(seconds=1800))
    try:
        owners = Wire(rank, world)
        declaration = {'freeze': identity(freeze), 'runtime': plan['runtime']}
        if owners.exchange(declaration) != [declaration] * world:
            raise ValueError('Owners disagree on the committed experiment')
        group = dist.new_group([0, 1, 2], timeout=timedelta(seconds=300))
        network = Network(shard, owners, ParentWire(rank, group) if rank < 3 else None,
                          tokenizer, plan['split'])
        binding = {'plan': identity(plan), 'prepared': identity(prepared),
            'job': expert_data.job_identity(plan, prepared), 'previous_graph': plan['previous_graph'],
            'retention_cache': prepared['retention_cache'], 'cut': plan['split'],
            'batches': identity(prepared['batches']), 'runtime': plan['runtime']}
        save(output/'started.json', {'freeze': identity(freeze), 'rank': rank,
            'owned_parameters': shard.resident_parameters, 'runtime': runtime})

        def deadline():
            if time.monotonic()-started > trial['max_seconds']:
                raise TimeoutError('The frozen learning experiment deadline expired')

        def rows(role):
            spec = trial['evaluation'][role]
            path = inputs/spec['file']
            if path.name != spec['file'] or sha256(path) != spec['sha256']:
                raise ValueError('The frozen evaluation role changed')
            values = [json.loads(line) for line in path.read_bytes().splitlines()]
            if len(values) != spec['count'] or identity([row['id'] for row in values]) != spec['ids']:
                raise ValueError('Evaluation identities or coverage changed')
            cohort_questions.validate_rows(values, release_scope=spec['release_scope'])
            return values

        def evaluate(role, arm):
            values, outcomes = rows(role), []
            for row in values:
                deadline()
                before = owners.sent_tensor_bytes
                began = time.monotonic()
                result = network.generate(row['messages'][0]['content'], trial['max_tokens'], True)
                observed = owners.exchange({'result': result, 'seconds': time.monotonic()-began,
                                             'sent_tensor_bytes': owners.sent_tensor_bytes-before})
                if any(item['result'] != observed[0]['result'] for item in observed):
                    raise ValueError('Shard owners disagree on a generated answer')
                outcomes.append({'id': row['id'], **result, 'measurements': observed})
            save(output/(arm+'-'+role+'.json'), outcomes)
            return values, outcomes

        before = {} if isolated else {role: evaluate(role, 'before') for role in ('dev', 'retained')}
        initial = None
        if rank == 3:
            optimizer = incremental.configure(shard, plan['split'], plan['training'])
            initial = snapshot(shard, optimizer, parent, binding['job'], 0, plan['training'])
            del optimizer
            shard.requires_grad_(False)
        initial = owners.exchange(initial)[3]
        save(output/'initial-checkpoint.json', initial)
        records = expert_execution.training_records(plan, prepared, inputs, parent)
        # Every owner sends only its own forward activations. The final parent
        # additionally holds the counted accepted reference, never the backbone.
        feature_root = cohort_features.produce(shard, owners, records, prepared['batches'],
            output/'features', binding, plan['split'], plan['microbatch'], reference_expert=seed,
            parent=parent, objects=home/'objects', resident_parameter_limit=plan['parameter_limit'])
        batch_roots = None
        if rank == 3:
            index = json.loads((output/'features/index.json').read_bytes())
            batch_roots = [batch_identity(batch) for batch in index['batches']]
        batch_roots = owners.exchange(batch_roots)[3]
        profile = {'format': expert_work.FORMAT, 'parent': parent, 'checkpoint': initial,
            'prepared': identity(prepared), 'feature_root': feature_root,
            'feature_stages': 3*sum((len(batch)+plan['microbatch']-1)//plan['microbatch'] for batch in prepared['batches']),
            'batch_roots': batch_roots, 'schedule': prepared['schedule'],
            'numerical_profile': trial['numerical_profile'], 'seed_expert': plan['seed_expert']}
        expert_work.validate_profile(profile)
        save(output/'execution-profile.json', profile)
        current = initial
        while current['step'] < plan['training']['steps']:
            deadline()
            receipt = None
            if rank == 3:
                receipt = expert_execution.produce_training(current,
                    min(4, plan['training']['steps']-current['step']), profile, plan, prepared,
                    inputs=inputs, objects=home/'objects', bank_home=output/'features',
                    checkpoint_store=output/'checkpoints', max_seconds=300)
                save(output/f'window-{current["step"]:06d}.json', receipt)
                if isolated:
                    # No ledger settlement is claimed by this method study.
                    # Keep all window metadata but only the final replay pair;
                    # do not accumulate hundreds of GB of unsettled tensors.
                    keep = {receipt['window'][key]['checkpoint'] for key in ('input', 'output')}
                    for path in (output/'checkpoints').iterdir():
                        if path.is_dir() and len(path.name) == 64 and path.name not in keep:
                            shutil.rmtree(path)
            receipt = owners.exchange(receipt)[3]
            current = receipt['window']['output']
            save(output/'progress.json', {'checkpoint': current['checkpoint'], 'step': current['step']})
            if rank == 0:
                print(json.dumps({'event': 'trained', 'step': current['step']}), flush=True)
        save(output/'terminal-checkpoint.json', current)
        if isolated:
            save(output/'training.json', {'format': ISOLATED+'/result', 'freeze': identity(freeze),
                'checkpoint': current, 'steps': current['step'], 'seconds': time.monotonic()-started,
                'quality_evaluated': False, 'native_activated': False, 'tokens_issued': 0})
            return
        if rank == 3:
            incremental_state.load(output/'checkpoints'/current['checkpoint'], shard, None,
                expert_checkpoint.unpack(parent, current),
                parent, binding['job'], plan['training'], restore_optimizer=False)
            shard.eval().requires_grad_(False)
        after = {role: evaluate(role, 'after') for role in ('dev', 'retained')}
        decision = cohort_questions.decision(before['dev'][0], before['dev'][1], after['dev'][1],
                                             trial['gates'], release_scope=False)
        spec = trial['evaluation']['retained']
        lost = [row['id'] for row, old, new in zip(before['retained'][0], before['retained'][1], after['retained'][1])
                if cohort_questions.correct(row, old['text'], release_scope=spec['release_scope'])
                and not cohort_questions.correct(row, new['text'], release_scope=spec['release_scope'])]
        decision['checks']['retained_answers'] = not lost
        decision['passed'] = all(decision['checks'].values())
        result = {'format': FORMAT+'/development', 'freeze': identity(freeze), 'checkpoint': current,
            'decision': decision, 'lost_correct': lost, 'final_opened': False,
            'native_activated': False, 'tokens_issued': 0, 'seconds': time.monotonic()-started}
        result = owners.exchange(result if rank == 0 else None)[0]
        save(output/'development.json', result)
        # Do not expose the final set until the controller has published the
        # exact terminal checkpoint decision. No more training is permitted.
        if decision['passed']:
            release = None
            while release is None:
                deadline()
                if rank == 0:
                    path = home/'open-final.json'
                    if path.exists():
                        release = json.loads(path.read_bytes())
                release = owners.exchange(release)[0]
                if release is None:
                    time.sleep(1)
            if release != {'freeze': identity(freeze), 'checkpoint': current['checkpoint'], 'development': identity(result)}:
                raise ValueError('The published final decision differs from the completed trial')
            # Paired final comparison reloads the initial accepted expert, not
            # a checkpoint chosen after examining final outputs.
            if rank == 3:
                cohort_state.initialize_tail(shard, parent, home/'objects', plan['split'], seed)
                shard.eval().requires_grad_(False)
            values, old = evaluate('test', 'before')
            if rank == 3:
                incremental_state.load(output/'checkpoints'/current['checkpoint'], shard, None,
                    expert_checkpoint.unpack(parent, current),
                    parent, binding['job'], plan['training'], restore_optimizer=False)
                shard.eval().requires_grad_(False)
            _, new = evaluate('test', 'after')
            final = cohort_questions.decision(values, old, new, trial['gates'], release_scope=False)
            final['checks']['retained_answers'] = not lost
            final['passed'] = all(final['checks'].values())
            save(output/'final.json', {'freeze': identity(freeze), 'checkpoint': current['checkpoint'],
                'decision': final, 'seconds': time.monotonic()-started, 'tokens_issued': 0, 'native_activated': False})
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.home, arguments.source)
