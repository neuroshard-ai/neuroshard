"""A4 second execution: the addition arm trained across Granite owners against a single-host control.

Three owners fetch only their tensors, as in the first shard execution; owner 0
drives the declared schedule and the last owner holds the arm. A separate
reference host loads the complete checkpoint and runs the unchanged single-host
trainer on the same sequences. The owners' final tensors, losses and preference
margins must equal the reference bit for bit, uninterrupted and after the arm's
owner is lost mid-schedule. It measures training across owners, not quality.
"""

import json
import os
from pathlib import Path
import resource
import time

from neuroshard.evolution import granite_shard_execution as shard
from neuroshard.evolution.modular_reference_execution import ROOT, read, save, sha256

PLAN = 'config/experiments/granite-shard-training.json'
SCRIPT = 'scripts/run_granite_shard_training.py'
PROFILE = 'granite-shard-training'
PHASES = ('fetch', 'train', 'outage', 'resume')
REFERENCE_PHASES = ('download', 'reference')


def committed_sources(root=ROOT):
    return shard.committed_sources(root, PLAN)


def configure():
    shard.configure()


def freeze():
    return shard.freeze(PLAN)


def sequences(plan, root=ROOT):
    pinned = plan['sequences']
    if sha256(root / pinned['path']) != pinned['sha256']:
        raise ValueError('training sequences changed')
    value = read(root / pinned['path'])
    return value['experience'], value['replay'], value['pairs']


def job(plan, phase, rank, home, store):
    """Owner job for one phase; the arm's checkpoints live in the store, never in evidence."""
    experience, replay, pairs = sequences(plan)
    world = len(plan['boundaries']) - 1
    base = {'arm': plan['arm'], 'spec': plan['spec'], 'experience': experience, 'replay': replay, 'pairs': pairs,
            'max_tokens': plan['max_boundary_tokens'], 'threads': read(ROOT / shard.CANONICAL)['resources']['threads'],
            'warm_up': plan.get('warm_up', False)}
    at = plan['outage']['at_step']
    if phase == 'train':
        return {**base, 'checkpoints': str(store / 'arm-train')}
    if phase == 'outage':
        return {**base, 'checkpoints': str(store / 'arm-outage'), 'fail': {'step': at}}
    if phase == 'resume':
        extra = {'start': at, 'checkpoints': str(store / 'arm-outage')}
        if rank == world - 1:
            extra['resume_from'] = str(store / 'arm-outage')
        if rank == 0:
            extra['references'] = read(home / 'outage' / 'references.json')
        return {**base, **extra}
    raise ValueError('unknown training phase')


def owner(rank, address, port, phase, home, store):
    """One owner process for one phase: fetch, or train in the ring."""
    configure()
    source = freeze()
    plan = read(ROOT / PLAN)
    home, store = Path(home), Path(store)
    world = len(plan['boundaries']) - 1
    if not 0 <= rank < world or phase not in PHASES:
        raise ValueError('unsupported training owner role')
    if phase == 'fetch':
        receipt = {'freeze': source, 'rank': rank, **shard.prepare(plan, rank, store), 'completed': True}
        save(home / 'fetch.json', receipt, exclusive=True)
        return receipt
    from neuroshard.evolution.sharded import granite_training

    os.environ.setdefault('GLOO_SOCKET_IFNAME', shard.default_interface())
    directory = home / phase
    directory.mkdir(parents=True, exist_ok=False)
    save(directory / 'job.json', job(plan, phase, rank, home, store), exclusive=True)
    save(directory / 'binding.json', {'freeze': source, 'plan_sha256': sha256(ROOT / PLAN), 'rank': rank,
                                      'phase': phase}, exclusive=True)
    return granite_training.run_owner(store / 'config', store / 'shard', rank, world, address, port,
                                      directory / 'job.json', directory / 'result.json',
                                      timeout=plan['peer_timeout_seconds'])


def warm_up(model, lengths=(1024, 1)):
    """The owners' discarded passes on the complete model: a fresh process's first pass can round differently."""
    import torch

    with torch.inference_mode():
        for length in lengths:
            model(torch.zeros((1, length), dtype=torch.long))


def reference(phase, home, store):
    """The control host: the complete pinned checkpoint and the unchanged single-host trainer."""
    configure()
    source = freeze()
    plan = read(ROOT / PLAN)
    home, store = Path(home), Path(store)
    artifacts = read(ROOT / plan['model']['artifacts'])['models']['baseline']
    directory = store / 'models' / 'baseline'
    if phase == 'download':
        from neuroshard.evolution.modular_reference_execution import verify_artifacts

        started = time.monotonic()
        state = verify_artifacts(directory, artifacts, download=True)
        receipt = {'freeze': source, 'files': sorted(state), 'seconds': time.monotonic() - started, 'completed': True}
        save(home / 'download.json', receipt, exclusive=True)
        return receipt
    if phase != 'reference':
        raise ValueError('unsupported reference role')
    import torch
    from transformers import AutoModelForCausalLM

    from neuroshard.evolution import assistant_experience_train as trainer
    from neuroshard.evolution.sharded.granite_training import digests

    torch.set_num_threads(read(ROOT / shard.CANONICAL)['resources']['threads'])
    experience, replay, pairs = sequences(plan)
    result = {'freeze': source, 'plan_sha256': sha256(ROOT / PLAN)}
    started = time.monotonic()
    try:
        model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.bfloat16, attn_implementation='eager',
                                                     local_files_only=True).eval()
        if sum(p.numel() for p in model.parameters()) != artifacts['parameters']:
            raise ValueError('reference checkpoint inventory differs')
        if plan.get('warm_up'):
            warm_up(model)
            result['warm_up'] = True
        trainable, receipt = trainer.train(model, plan['arm'], experience, replay, plan['spec'], pairs=pairs)
        receipt.pop('optimizer_state')
        result.update(receipt=receipt, trainable_sha256=digests(trainable), completed=True)
    except Exception as error:
        result.update(completed=False, error=f'{type(error).__name__}: {error}')
    finally:
        result.update(seconds=time.monotonic() - started,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        save(home / 'reference.json', result, exclusive=True)
    return result


def assess(plan, fetches, phases, control):
    """Declared checks from the owners' phases and the reference host's result."""
    world = len(plan['boundaries']) - 1
    holder, at = world - 1, plan['outage']['at_step']
    train, outage, resumed = phases['train'], phases['outage'], phases['resume']
    expected = control.get('trainable_sha256') if control else None
    receipt = (control or {}).get('receipt') or {}
    driven = (train[0] or {}).get('receipt') or {}
    resumed_receipt = (resumed[0] or {}).get('receipt') or {}
    steps = plan['spec']['steps']
    peaks = {rank: max((phases[p][rank] or {}).get('peak_rss_bytes', 0) for p in ('train', 'outage', 'resume'))
             for rank in range(world)}
    checks = {
        'complete': bool(control and control.get('completed')) and all(r and r.get('completed') for r in train),
        'fetched_only_owned': all(f and f.get('completed') for f in fetches),
        'tensors': bool(expected) and (train[holder] or {}).get('trainable_sha256') == expected,
        'losses': bool(receipt) and driven.get('losses') == receipt.get('losses')
        and driven.get('preference_margins') == receipt.get('preference_margins'),
        'outage_injected': outage[holder] is None and bool(outage[0]) and not outage[0].get('completed'),
        'recovery': all(r and r.get('completed') for r in resumed) and bool(expected)
        and (resumed[holder] or {}).get('trainable_sha256') == expected
        and resumed_receipt.get('losses') == receipt.get('losses', [])[at:],
        'memory': all(0 < peaks[r] < (control or {}).get('peak_rss_bytes', 0) for r in range(world)),
    }
    return {'passed': all(checks.values()), 'checks': checks, 'steps': steps, 'outage_step': at,
            'owner_peak_rss_bytes': peaks, 'reference_peak_rss_bytes': (control or {}).get('peak_rss_bytes'),
            'owner_train_seconds': {r: (train[r] or {}).get('seconds') for r in range(world)},
            'reference_seconds': (control or {}).get('seconds'),
            'sent_bytes': {r: (train[r] or {}).get('sent_bytes') for r in range(world)},
            'trainable_parameters': (train[holder] or {}).get('trainable_parameters'),
            'losses': receipt.get('losses'), 'checklist_credit': False, 'admission_evidence': False}
