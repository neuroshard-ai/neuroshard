"""Produce or audit native expert work from available numerical state.

This is a numerical executor, never a consensus transition. Its profile,
training plan and installed source are trusted local configuration. Claims may
not choose them. Prefix production must already have its separate native audit;
checking a feature bank's bytes does not certify how those features were made.

Each invocation loads fresh weights and Adam state, executes at most four
updates, and persists a resumable window boundary only after agreement. It
never substitutes an archived report or a cached output for execution. Missing
inputs, storage errors and timeouts raise instead of producing a valid verdict.
The calling audit worker must also impose a process timeout.
"""
import errno
import json
import math
import os
from pathlib import Path
import tempfile
import time

from transformers import LlamaConfig

from .. import cohort_experiment, expert_checkpoint, expert_window, expert_work, reference
from ..reference_data import identity, save
from ..schema import integer, root
from . import checkpoint, feature_bank, features, incremental, incremental_state
from .expert_commitment import snapshot
from .expert_replay import batch_identity
from .feature_probe import load_head
from .model import Partition


def training_job(plan, prepared):
    """Preserve the original experiment's job domain across native adoption."""
    if plan['format'] not in (cohort_experiment.FORMAT, 'neuroshard-interpreted-cohort-v1'):
        raise ValueError('Unsupported original expert training contract')
    return identity({'format': plan['format'], 'plan': identity(plan), 'prepared': identity(prepared)})


def _job_context(before, profile, plan, prepared):
    """Validate prescribed inputs without requiring a future output checkpoint."""
    expert_work.validate_profile(profile)
    parent, initial = profile['parent'], profile['checkpoint']
    expert_checkpoint.unpack(parent, initial)
    expert_checkpoint.unpack(parent, before)
    job = training_job(plan, prepared)
    if (initial['step'] != 0 or initial['job'] != job or plan['parent'] != identity(parent)
            or plan['training'] != initial['recipe'] or plan['split'] != initial['split']
            or plan['expert_layout'] != initial['boundaries']
            or any(before[key] != initial[key] for key in ('parent', 'job', 'split', 'recipe', 'boundaries'))
            or identity(prepared) != profile['prepared']
            or prepared['schedule'] != profile['schedule']
            or len(profile['schedule']) != plan['training']['steps']):
        raise ValueError('Claim changed the configured parent, job, data, recipe or numerical profile')
    return parent, before, job


def _context(claim, profile, plan, prepared):
    """Reject substitutions before allocating a model or executing an update."""
    if profile['format'] == expert_work.PROSPECTIVE:
        if claim['kind'] != 'expert_features':
            raise ValueError('Training requires its accepted prefix execution profile')
        profile = expert_work.resolve_prefix(profile, claim['feature_root'], claim['batch_roots'])
    parent, before, job = _job_context(claim['input_checkpoint'], profile, plan, prepared)
    initial = profile['checkpoint']
    if (claim['kind'] not in expert_work.KINDS or claim['parent_checkpoint'] != parent
            or claim['prepared'] != profile['prepared'] or claim['feature_root'] != profile['feature_root']
            or claim['numerical_profile'] != profile['numerical_profile']):
        raise ValueError('Claim changed the configured parent, job, data, recipe or numerical profile')
    root(claim['id'])
    root(claim['record_root'])
    if claim['kind'] == 'expert_features':
        if (before != initial or type(claim['stages']) is not int
                or claim['stages'] != profile['feature_stages']):
            raise ValueError('Prefix production must cover the initial expert and every declared stage')
        return parent, before, job
    root(claim['feature_claim'])
    start, count = before['step'], claim['stages']
    if (type(start) is not int or type(count) is not int or not 1 <= count <= 4
            or not 0 <= start < start + count <= len(profile['schedule'])):
        raise ValueError('Execute only a bounded prescribed training window')
    batches = profile['batch_roots']
    if (not isinstance(batches, list) or len(batches) != len(prepared['batches'])
            or any(type(index) is not int or not 0 <= index < len(batches) for index in profile['schedule'])):
        raise ValueError('Invalid fixed feature batch schedule')
    expected = [batches[index] for index in profile['schedule'][start:start + count]]
    result = expert_window.validate(parent, before, claim['window'], claim['intermediates'],
                                    expected, profile['numerical_profile'])
    if (result['record_root'] != claim['record_root'] or result['steps'] != count
            or claim['output_checkpoint'] != claim['window']['output']
            or claim['work_ids'] != result['work_ids']):
        raise ValueError('Claim summary differs from its bounded execution record')
    return parent, before, job


def _persist(store, shard, optimizer, parent, sources, value):
    """Publish a complete boundary atomically; retries verify existing bytes."""
    common = expert_checkpoint.unpack(parent, value)
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    destination = store / value['checkpoint']
    if not destination.exists():
        with tempfile.TemporaryDirectory(prefix='.writing-', dir=store) as temporary:
            pending = Path(temporary)
            meta = incremental_state.write(pending, shard, optimizer, parent, sources,
                value['job'], value['step'], value['recipe'], 'tail-control', value['split'])
            if identity(meta) != common['shards'][shard.rank]:
                raise ValueError('Persisted tensor bytes differ from the computed checkpoint')
            save(pending / 'checkpoint.json', value)
            checkpoint.sync_directory(pending)
            try:
                os.rename(pending, destination)
            except OSError as error:
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
            checkpoint.sync_directory(store)
    # A preexisting directory is not evidence of either execution or durability.
    # Re-read every owned object, including original parent provenance and Adam.
    if json.loads((destination / 'checkpoint.json').read_bytes()) != value:
        raise ValueError('Existing checkpoint metadata differs from the computed boundary')
    incremental_state.load(destination, shard, optimizer, common, parent, value['job'], value['recipe'])
    return destination


def _train(before, count, profile, plan, prepared, *, inputs, objects, bank_home,
           checkpoint_store, max_seconds=300, expected=None):
    """Shared numerical path for producing work and independently replaying it."""
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 14400:
        raise ValueError('Declare a finite bounded execution deadline')
    started = time.monotonic()

    def deadline():
        if time.monotonic() - started >= max_seconds:
            raise TimeoutError('Bounded expert execution deadline expired')

    parent, before, job = _job_context(before, profile, plan, prepared)
    if profile['format'] != expert_work.FORMAT:
        raise ValueError('Training requires its accepted prefix execution profile')
    integer(count, 1, 4)
    if not 0 <= before['step'] < before['step'] + count <= len(profile['schedule']):
        raise ValueError('Produce only a bounded prescribed training window')
    runtime = reference.configure(plan['runtime']['device'], plan['threads'])
    runtime['allocator'] = os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
    if not plan['runtime'] or any(runtime.get(key) != value for key, value in plan['runtime'].items()):
        raise ValueError('Executor differs from the prescribed numerical runtime')
    records = cohort_experiment.rows(prepared, inputs, 'train', max_length=plan['max_length'])
    config = LlamaConfig(**parent['config'])
    config._attn_implementation = 'sdpa'
    binding = {'plan': identity(plan), 'prepared': identity(prepared), 'job': job,
        'previous_graph': plan['previous_graph'], 'retention_cache': prepared['retention_cache'],
        'cut': plan['split'], 'batches': identity(prepared['batches']), 'runtime': plan['runtime']}
    bank = feature_bank.Reader(bank_home, profile['feature_root'], binding, config,
                               plan['microbatch'], len(prepared['batches']))
    if [batch_identity(batch) for batch in bank.manifest['batches']] != profile['batch_roots']:
        raise ValueError('Feature bank changed the prescribed numerical batches')
    device = runtime['device']
    shard = Partition(config, before['boundaries'], len(before['boundaries']) - 2,
                      device, plan['parameter_limit'])
    optimizer = incremental.configure(shard, before['split'], before['recipe'])
    if before['step'] == 0:
        sources = incremental_state.initialize(shard, parent, objects, 'tail-control', before['split'])
    else:
        source = Path(checkpoint_store) / before['checkpoint']
        if json.loads((source / 'checkpoint.json').read_bytes()) != before:
            raise ValueError('Available checkpoint differs from the claimed input')
        sources = incremental_state.load(source, shard, optimizer,
            expert_checkpoint.unpack(parent, before), parent, job, before['recipe'])
    current = snapshot(shard, optimizer, parent, job, before['step'], before['recipe'])
    if current != before:
        raise ValueError('Actual input weights or Adam differ from the claimed checkpoint')
    head = load_head(parent, objects, device)
    if shard.resident_parameters + sum(p.numel() for p in head.parameters()) > plan['parameter_limit']:
        raise ValueError('Expert and read-only head exceed the configured ownership limit')
    stages, steps, intermediates = [], [], [current]
    for offset in range(count):
        deadline()
        index = before['step'] + offset
        batch_index = prepared['schedule'][index]
        rows = [records[i] for i in prepared['batches'][batch_index]]
        packets = bank.batch(batch_index, rows, device)
        measured = features.train_step(shard, head, optimizer, packets, rows, before['recipe'],
                                       index, bank.microbatch, **plan['objective'])
        del packets
        after = snapshot(shard, optimizer, parent, job, index + 1, before['recipe'])
        batch = profile['batch_roots'][batch_index]
        steps.append({'index': index, 'input_checkpoint': current['checkpoint'],
            'output_checkpoint': after['checkpoint'], 'batch': batch,
            'work_identity': expert_checkpoint.work_identity(parent, current, batch, profile['numerical_profile']),
            'metrics': {key: measured[key] for key in expert_window.METRICS}})
        valid = (expected is None or (current == expected['intermediates'][offset]
                 and after == expected['intermediates'][offset + 1]
                 and steps[-1] == expected['window']['steps'][offset]))
        stages.append({'stage': offset, 'valid': valid})
        intermediates.append(after)
        current = after
    deadline()
    window = {'format': expert_window.FORMAT, 'input': before, 'output': current,
              'steps': steps, 'numerical_profile': profile['numerical_profile']}
    expert_window.validate(parent, before, window, intermediates,
        [profile['batch_roots'][i] for i in profile['schedule'][before['step']:before['step'] + count]],
        profile['numerical_profile'])
    if all(stage['valid'] for stage in stages):
        _persist(checkpoint_store, shard, optimizer, parent, sources, current)
    deadline()
    return {'window': window, 'intermediates': intermediates, 'stages': stages}


def produce_training(before, count, profile, plan, prepared, **execution):
    """Compute a new claim from prescribed inputs, with no expected trajectory.

    The returned window uses the existing native claim format and is published
    only after its complete terminal weights and optimizer state are persisted
    and read back. This is a producer result, never an audit or quality verdict.
    """
    result = _train(before, count, profile, plan, prepared, **execution)
    return {'window': result['window'], 'intermediates': result['intermediates']}


def execute_training(claim, profile, plan, prepared, **execution):
    """Replay a submitted claim from actual input weights and Adam state.

    Paths and configuration come from the auditor. A nonzero cursor requires
    its available boundary, never an unbounded replay from the initial model.
    """
    parent, before, _ = _context(claim, profile, plan, prepared)
    if claim['kind'] != 'expert_training':
        raise ValueError('The training executor requires a training claim')
    actual = _train(before, claim['stages'], profile, plan, prepared, expected=claim, **execution)
    report = {'format': expert_work.FORMAT + '/replay', 'claim_id': claim['id'],
        'record_root': claim['record_root'], 'binding': {
            'parent': identity(parent), 'prepared': profile['prepared'],
            'input_checkpoint': before['checkpoint'], 'output_root': claim['output_checkpoint']['checkpoint'],
            'feature_root': profile['feature_root'], 'numerical_profile': profile['numerical_profile'],
            'feature_claim': claim['feature_claim']}, 'stages': actual['stages']}
    expert_work.replay_report(claim, report)
    return report


def main():
    """Private execution backend for audit_worker --execution-backend."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    producing = parser.add_mutually_exclusive_group()
    producing.add_argument('--produce', action='store_true',
                           help='Produce a new window from an input checkpoint and step count')
    producing.add_argument('--produce-features', action='store_true',
                           help='Produce the prescribed prefix without a known output hash')
    args = parser.parse_args()
    config = json.loads(args.config.read_bytes())
    fields = {'format', 'profile', 'plan', 'prepared', 'paths', 'max_seconds'}
    paths = {'inputs', 'objects', 'bank_home', 'checkpoint_store'}
    if (set(config) != fields or config['format'] != 'neuroshard-expert-executor-v1'
            or set(config['paths']) != paths
            or any(not isinstance(path, str) or not Path(path).is_absolute() for path in config['paths'].values())):
        raise ValueError('Require a locally configured executor and absolute artifact paths')
    raw = sys.stdin.buffer.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError('Expert claim exceeds the execution request limit')
    claim = json.loads(raw)
    if args.produce_features:
        if claim != {}:
            raise ValueError('Prefix production takes its inputs only from local job configuration')
        from .prefix_execution import produce_features
        result = produce_features(config['profile'], config['plan'], config['prepared'],
            **config['paths'], max_seconds=config['max_seconds'])
        print(json.dumps(result, separators=(',', ':'), allow_nan=False))
        return
    if args.produce:
        if not isinstance(claim, dict) or set(claim) != {'input_checkpoint', 'steps'}:
            raise ValueError('Production accepts the current checkpoint and bounded step count only')
        result = produce_training(claim['input_checkpoint'], claim['steps'], config['profile'],
            config['plan'], config['prepared'], **config['paths'], max_seconds=config['max_seconds'])
        print(json.dumps(result, separators=(',', ':'), allow_nan=False))
        return
    if claim['kind'] == 'expert_features':
        from .prefix_execution import execute_features
        executor = execute_features
    else:
        executor = execute_training
    report = executor(claim, config['profile'], config['plan'], config['prepared'],
                      **config['paths'], max_seconds=config['max_seconds'])
    print(json.dumps(report, separators=(',', ':'), allow_nan=False))


if __name__ == '__main__':
    main()
