"""A4 fourth execution: more throughput from the owner ring without changing a single token.

The learned assistant is served on three owner hosts twice: one episode at a
time, then several episodes in flight, each step computed exactly as if alone.
Both passes must reproduce the single-host round-4 development episodes token
for token, and the concurrent pass must finish them materially faster. A single
host can only gain this by batching, which changes the computation.
"""

import os
from pathlib import Path

from neuroshard.evolution import granite_shard_execution as shard
from neuroshard.evolution import granite_shard_serving as serving
from neuroshard.evolution.modular_reference_execution import ROOT, read, save, sha256

PLAN = 'config/experiments/granite-shard-throughput.json'
SCRIPT = 'scripts/run_granite_shard_throughput.py'
PROFILE = 'granite-shard-throughput'
PHASES = ('fetch', 'sequential', 'concurrent')
UPLOADED = serving.UPLOADED


def committed_sources(root=ROOT):
    return shard.committed_sources(root, PLAN)


def configure():
    shard.configure()


def freeze():
    return shard.freeze(PLAN)


def owner(rank, address, port, phase, home, store, index=0):
    configure()
    source = freeze()
    plan = read(ROOT / PLAN)
    home, store = Path(home), Path(store)
    world = len(plan['boundaries']) - 1
    if not 0 <= rank < world or phase not in PHASES:
        raise ValueError('unsupported throughput owner role')
    if phase == 'fetch':
        receipt = {'freeze': source, 'rank': rank, **serving.prepare(plan, rank, store), 'completed': True}
        save(home / 'fetch.json', receipt, exclusive=True)
        return receipt
    from neuroshard.evolution.sharded import granite_serving

    os.environ.setdefault('GLOO_SOCKET_IFNAME', shard.default_interface())
    directory = home / phase
    directory.mkdir(parents=True, exist_ok=False)
    threads = read(ROOT / shard.CANONICAL)['resources']['threads']
    value = (serving.job(plan) if rank in (0, world - 1)
             else {'max_tokens': plan['max_boundary_tokens'], 'threads': threads})
    value['streams'] = plan['streams'][phase]
    if rank == 0:
        value['tokenizer'] = str(store / 'config')
    save(directory / 'job.json', value, exclusive=True)
    save(directory / 'binding.json', {'freeze': source, 'plan_sha256': sha256(ROOT / PLAN), 'rank': rank,
                                      'phase': phase}, exclusive=True)
    return granite_serving.run_owner(store / 'config', store / 'shard', rank, world, address, port,
                                     directory / 'job.json', directory / 'result.json',
                                     timeout=plan['peer_timeout_seconds'])


def assess_phases(plan, fetches, phases):
    """Both passes must equal the single-host result; the concurrent pass must beat the sequential one."""
    reports = {name: serving.assess(plan, fetches, [], phases[name]) for name in ('sequential', 'concurrent')}
    seconds = {name: (phases[name][0] or {}).get('episodes_seconds') for name in reports}
    expected = serving.target(plan)
    single_host = sum(row['seconds'] + row['selection_seconds'] for row in expected)
    ratio = seconds['sequential'] / seconds['concurrent'] if all(seconds.values()) else None
    checks = {
        'sequential_agreement': reports['sequential']['passed'],
        'concurrent_agreement': reports['concurrent']['passed'],
        'overlap': ((phases['concurrent'][0] or {}).get('peak_in_flight') or 0) >= 2,
        'throughput': ratio is not None and ratio >= plan['minimum_throughput_ratio'],
    }
    return {'passed': all(checks.values()), 'checks': checks, 'episodes_seconds': seconds,
            'throughput_ratio': ratio, 'single_host_episode_seconds': single_host,
            'ratio_vs_single_host': single_host / seconds['concurrent'] if seconds['concurrent'] else None,
            'peak_in_flight': (phases['concurrent'][0] or {}).get('peak_in_flight'),
            'phases': {name: {k: v for k, v in report.items() if k != 'determinism'} for name, report in reports.items()},
            'checklist_credit': False, 'admission_evidence': False}
