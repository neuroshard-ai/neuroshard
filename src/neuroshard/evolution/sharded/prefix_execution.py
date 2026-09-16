"""Native prefix verdicts produced by actual sequential shard execution.

One invocation recomputes all parent partitions and retains their numerical
outputs. It never treats a submitted report chain as execution. The production
record excludes timing so separately executed honest audits can agree on it.
"""
import contextlib
import errno
import gc
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import torch
from transformers import LlamaConfig

from .. import cohort_experiment, expert_work, reference
from ..reference_data import identity
from . import checkpoint, expert_execution, feature_bank, prefix_audit
from .expert_replay import batch_identity
from .model import Partition


def production_record(reports):
    """Canonical commitment to locally produced stages; not an execution proof."""
    if not isinstance(reports, list) or len(reports) != 3:
        raise ValueError('Require the three complete production stages')
    context = reports[0]['context']
    incoming = identity({'records': context['records'], 'batches': context['batches']})
    partitions = []
    for rank, report in enumerate(reports):
        if (report['format'] != prefix_audit.FORMAT or report['rank'] != rank
                or report['context'] != context or report['completed'] is not True
                or report['input_root'] != incoming
                or type(report['microbatches']) is not int or report['microbatches'] <= 0
                or report['microbatches'] != reports[0]['microbatches']):
            raise ValueError('Production stages must cover the same connected numerical inputs')
        partitions.append({key: report[key] for key in ('rank', 'input_root', 'output_root', 'microbatches')})
        incoming = report['output_root']
    return {'format': expert_work.FORMAT + '/production', 'context': context, 'partitions': partitions}


def _verify_saved(home, reports, config, records, batches, microbatch, binding):
    """Check the retained numerical payloads, including a preexisting result."""
    for rank, report in enumerate(reports):
        saved = json.loads((home / f'rank-{rank}' / 'result.json').read_bytes())
        if any(saved[key] != report[key] for key in ('format', 'rank', 'context', 'input_root',
                                                    'output_root', 'completed', 'microbatches')):
            raise ValueError('Retained prefix report differs from the actual production')
        expected_binding = (binding if rank == 2 else
                            prefix_audit.stage_binding(report['context'], rank, report['input_root']))
        reader = feature_bank.Reader(home / f'rank-{rank}' / 'features', report['output_root'],
                                     expected_binding, config, microbatch, len(batches))
        for index, indices in enumerate(batches):
            reader.batch(index, [records[i] for i in indices], 'cpu')


def execute_features(claim, profile, plan, prepared, *, inputs, objects, bank_home,
                     checkpoint_store, max_seconds=900):
    """Recompute and retain a complete production claim before attesting.

    ``bank_home`` is the training cache path in the shared backend configuration;
    production is reconstructed independently. Valid outputs are retained under
    ``checkpoint_store/prefix/<record_root>/rank-N/features``. A false aggregate
    production commitment refutes the complete claim; these verdicts do not
    identify a dishonest individual partition owner.
    """
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 14400:
        raise ValueError('Declare a finite bounded prefix execution deadline')
    started = time.monotonic()

    def remaining():
        seconds = max_seconds - (time.monotonic() - started)
        if seconds <= 0:
            raise TimeoutError('Native prefix execution deadline expired')
        return min(3600, seconds)

    parent, before, job = expert_execution._context(claim, profile, plan, prepared)
    if claim['kind'] != 'expert_features' or plan['parent_layout'] != parent['boundaries']:
        raise ValueError('Require the configured prefix-production claim and ownership')
    runtime = reference.configure(plan['runtime']['device'], plan['threads'])
    runtime['allocator'] = os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
    if not plan['runtime'] or any(runtime.get(key) != value for key, value in plan['runtime'].items()):
        raise ValueError('Prefix executor differs from the prescribed numerical runtime')
    records = cohort_experiment.rows(prepared, inputs, 'train', max_length=plan['max_length'])
    microbatches = sum(math.ceil(len(batch) / plan['microbatch']) for batch in prepared['batches'])
    if claim['stages'] != 3 * microbatches:
        raise ValueError('Native prefix coverage must include every executed microbatch')
    config = LlamaConfig(**parent['config'])
    config._attn_implementation = 'sdpa'
    binding = {'plan': identity(plan), 'prepared': identity(prepared), 'job': job,
        'previous_graph': plan['previous_graph'], 'retention_cache': prepared['retention_cache'],
        'cut': plan['split'], 'batches': identity(prepared['batches']), 'runtime': plan['runtime']}
    store = Path(checkpoint_store) / 'prefix'
    store.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.writing-', dir=store) as temporary:
        pending = Path(temporary)
        reports, incoming = [], None
        for rank in range(3):
            remaining()
            shard = Partition(config, parent['boundaries'], rank, runtime['device'], plan['parameter_limit'])
            home = pending / f'rank-{rank}'
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    report = prefix_audit.replay_stage(shard, parent, Path(objects), records,
                        prepared['batches'], binding, plan['split'], plan['microbatch'],
                        profile['feature_root'], home, incoming, remaining())
            except ValueError:
                # The numerical kernel writes its completed result before
                # rejecting a differing final root. Only that observed mismatch
                # is a negative verdict; other failures remain unavailable.
                path = home / 'result.json'
                if rank != 2 or not path.is_file():
                    raise
                report = json.loads(path.read_bytes())
                if report.get('completed') is not True or report.get('valid') is not False:
                    raise
            finally:
                del shard
                gc.collect()
                if runtime['device'] == 'cuda':
                    torch.cuda.empty_cache()
            reports.append(report)
            incoming = (home / 'features', report)
        record = production_record(reports)
        actual_bank = json.loads((pending / 'rank-2/features/index.json').read_bytes())
        valid = (reports[-1]['valid'] is True and identity(record) == claim['record_root']
                 and [batch_identity(batch) for batch in actual_bank['batches']] == profile['batch_roots'])
        remaining()
        if valid:
            prefix_audit.complete(reports, profile['feature_root'])
            destination = store / claim['record_root']
            for rank in range(3):
                for path in (pending / f'rank-{rank}' / 'features').glob('*.safetensors'):
                    with path.open('rb') as payload:
                        os.fsync(payload.fileno())
                checkpoint.sync_directory(pending / f'rank-{rank}' / 'features')
                checkpoint.sync_directory(pending / f'rank-{rank}')
            checkpoint.sync_directory(pending)
            try:
                os.rename(pending, destination)
            except OSError as error:
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
            checkpoint.sync_directory(store)
            _verify_saved(destination, reports, config, records, prepared['batches'], plan['microbatch'], binding)
        remaining()
    report = {'format': expert_work.FORMAT + '/replay', 'claim_id': claim['id'],
        'record_root': claim['record_root'], 'binding': {'parent': identity(parent),
            'prepared': profile['prepared'], 'input_checkpoint': before['checkpoint'],
            'output_root': profile['feature_root'], 'feature_root': profile['feature_root'],
            'numerical_profile': profile['numerical_profile'], 'feature_claim': None},
        'stages': [{'stage': stage, 'valid': valid} for stage in range(claim['stages'])]}
    expert_work.replay_report(claim, report)
    return report
