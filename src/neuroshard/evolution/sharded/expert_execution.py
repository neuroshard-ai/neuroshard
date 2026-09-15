"""Execute one native expert training claim from available numerical state.

This is an auditor-side executor, never a consensus transition. Its profile,
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
from ..schema import root
from . import checkpoint, feature_bank, features, incremental, incremental_state
from .expert_commitment import snapshot
from .expert_replay import batch_identity
from .feature_probe import load_head
from .model import Partition


def _context(claim, profile, plan, prepared):
    """Reject substitutions before allocating a model or executing an update."""
    if (set(profile) != expert_work.PROFILE_FIELDS or profile['format'] != expert_work.FORMAT
            or claim['kind'] != 'expert_training'):
        raise ValueError('Require the configured native expert training profile')
    parent, initial = profile['parent'], profile['checkpoint']
    expert_checkpoint.unpack(parent, initial)
    before = claim['input_checkpoint']
    job = cohort_experiment.job(plan, prepared)
    if (initial['step'] != 0 or initial['job'] != job or plan['parent'] != identity(parent)
            or plan['training'] != initial['recipe'] or plan['split'] != initial['split']
            or plan['expert_layout'] != initial['boundaries']
            or any(before[key] != initial[key] for key in ('parent', 'job', 'split', 'recipe', 'boundaries'))
            or claim['parent_checkpoint'] != parent or claim['prepared'] != identity(prepared)
            or claim['prepared'] != profile['prepared'] or claim['feature_root'] != profile['feature_root']
            or claim['numerical_profile'] != profile['numerical_profile']
            or prepared['schedule'] != profile['schedule']
            or len(profile['schedule']) != plan['training']['steps']):
        raise ValueError('Claim changed the configured parent, job, data, recipe or numerical profile')
    root(claim['id'])
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


def execute_training(claim, profile, plan, prepared, *, inputs, objects, bank_home,
                     checkpoint_store, max_seconds=300):
    """Return the native replay report only after actual bounded execution.

    All paths and configuration come from the auditor, not the transaction.
    ``checkpoint_store/<checkpoint root>`` holds the existing incremental owner
    format plus its compact ``checkpoint.json``. A nonzero input cursor requires
    those actual payloads; it never triggers an unbounded replay from step zero.
    """
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 14400:
        raise ValueError('Declare a finite bounded execution deadline')
    started = time.monotonic()

    def deadline():
        if time.monotonic() - started >= max_seconds:
            raise TimeoutError('Bounded expert execution deadline expired')

    parent, before, job = _context(claim, profile, plan, prepared)
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
    stages = []
    for offset, expected in enumerate(claim['window']['steps']):
        deadline()
        index = before['step'] + offset
        batch_index = prepared['schedule'][index]
        rows = [records[i] for i in prepared['batches'][batch_index]]
        packets = bank.batch(batch_index, rows, device)
        measured = features.train_step(shard, head, optimizer, packets, rows, before['recipe'],
                                       index, bank.microbatch, **plan['objective'])
        del packets
        after = snapshot(shard, optimizer, parent, job, index + 1, before['recipe'])
        valid = (current == claim['intermediates'][offset]
                 and after == claim['intermediates'][offset + 1]
                 and all(measured[key] == expected['metrics'][key] for key in expert_window.METRICS))
        stages.append({'stage': offset, 'valid': valid})
        current = after
    deadline()
    if all(stage['valid'] for stage in stages):
        _persist(checkpoint_store, shard, optimizer, parent, sources, current)
    deadline()
    report = {'format': expert_work.FORMAT + '/replay', 'claim_id': claim['id'],
        'record_root': claim['record_root'], 'binding': {
            'parent': identity(parent), 'prepared': profile['prepared'],
            'input_checkpoint': before['checkpoint'], 'output_root': claim['output_checkpoint']['checkpoint'],
            'feature_root': profile['feature_root'], 'numerical_profile': profile['numerical_profile'],
            'feature_claim': claim['feature_claim']}, 'stages': stages}
    expert_work.replay_report(claim, report)
    return report


def main():
    """Private execution backend for audit_worker --execution-backend."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
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
    report = execute_training(json.loads(raw), config['profile'], config['plan'], config['prepared'],
                              **config['paths'], max_seconds=config['max_seconds'])
    print(json.dumps(report, separators=(',', ':'), allow_nan=False))


if __name__ == '__main__':
    main()
