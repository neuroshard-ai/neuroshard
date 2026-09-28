"""A4 first execution: pinned Granite split across owner hosts, none holding the complete backbone.

Each owner fetches only its tensors' byte ranges from the pinned revision, checks
every tensor against the committed inventory, and serves its layers in a gloo
ring under the canonical CPU runtime. Owner 0 replays every recorded generation
of the canonical re-baseline and must reproduce its tokens exactly; an injected
outage must resume to the same tokens. It measures sharding, not quality.
"""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from neuroshard.evolution.modular_reference_execution import ROOT, read, save, sha256

PLAN = 'config/experiments/granite-shard-execution.json'
RUNTIME = 'config/experiments/assistant-workflow-canonical-execution.json'
CANONICAL = 'config/experiments/assistant-workflow-canonical.json'
SCRIPT = 'scripts/run_granite_shards.py'
PROFILE = 'granite-shard-owner'
PHASES = ('fetch', 'agreement', 'outage', 'resume')
SERVING = PHASES[1:]


def committed_sources(root=ROOT):
    plan = read(root / PLAN)
    for name, digest in plan['contracts'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'changed shard contract: {name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in plan['sources']:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root) != (root / name).read_bytes():
            raise ValueError(f'uncommitted shard source: {name}')
        sources[name] = sha256(root / name)
    return {'commit': commit, 'sources': sources}


def configure():
    if 'torch' in sys.modules:
        raise ValueError('configure the shard runtime before importing torch')
    for key, value in read(ROOT / RUNTIME)['environment'].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def freeze():
    """The canonical parent's CPU runtime, byte for byte in packages and environment."""
    source = committed_sources()
    runtime = read(ROOT / RUNTIME)
    packages = {key: importlib.metadata.version(key) for key in runtime['packages']}
    if packages != runtime['packages'] or platform.python_version() != runtime['python']:
        raise ValueError('shard runtime differs from the canonical runtime')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('shard owners require Linux x86_64')
    cpu = Path('/proc/cpuinfo').read_text().split()
    if any(flag not in cpu for flag in runtime['required_cpu_flags']):
        raise ValueError('shard owner CPU lacks required instructions')
    if any(os.environ.get(key) != value for key, value in runtime['environment'].items()):
        raise ValueError('shard numerical environment differs')
    return {**source, 'packages': packages, 'python': platform.python_version()}


def workload(plan, root=ROOT):
    """Every recorded generation of the canonical re-baseline, with its tokens as the expected output."""
    pinned = plan['workload']
    if sha256(root / pinned['source']) != pinned['sha256']:
        raise ValueError('canonical result changed')
    primary = read(root / pinned['source'])['primary']
    requests = [{'id': f"{episode['id']}/{index}", 'input_ids': g['input_token_ids'], 'expected': g['token_ids']}
                for episode in primary['episodes']
                for index, g in enumerate(episode['generations']) if g.get('executed', True)]
    requests += [{'id': row['id'], 'input_ids': row['input_token_ids'], 'expected': row['token_ids']}
                 for row in primary['anchors']]
    if len(requests) != pinned['generations'] or len({r['id'] for r in requests}) != len(requests):
        raise ValueError('canonical workload differs from its declaration')
    return [{**r, 'max_new_tokens': pinned['max_new_tokens']} for r in requests]


def inventory(plan, root=ROOT):
    if sha256(root / plan['model']['tensor_inventory']) != plan['model']['inventory_sha256']:
        raise ValueError('tensor inventory changed')
    value = read(root / plan['model']['tensor_inventory'])
    artifacts = read(root / plan['model']['artifacts'])['models']['baseline']
    files = {name: spec['digest'] for name, spec in artifacts['files'].items() if name.endswith('.safetensors')}
    if (value['repo'], value['revision'], value['files']) != (artifacts['repo'], artifacts['revision'], files):
        raise ValueError('tensor inventory is not for the pinned checkpoint')
    return value, artifacts


def prepare(plan, rank, home, base=None):
    """Fetch this owner's byte ranges and the pinned config; nothing else of the checkpoint."""
    from neuroshard.evolution.sharded import granite

    tensors, artifacts = inventory(plan)
    base = base or f"https://huggingface.co/{artifacts['repo']}/resolve/{artifacts['revision']}"
    read_range = granite.http_ranges(base)
    config = home / 'config'
    config.mkdir(parents=True, exist_ok=True)
    import urllib.request
    with urllib.request.urlopen(f'{base}/config.json', timeout=120) as response:
        raw = response.read()
    spec = artifacts['files']['config.json']
    if hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest() != spec['digest']:
        raise ValueError('config.json differs from the pinned artifact')
    (config / 'config.json').write_bytes(raw)
    started = time.monotonic()
    manifest = granite.fetch(tensors, plan['boundaries'], rank, home / 'shard', read_range)
    return {**manifest, 'fetch_seconds': time.monotonic() - started}


def job(plan, phase, committed=None):
    requests = workload(plan)
    outage = plan['outage']
    if phase == 'agreement':
        chosen, fail = requests, None
    elif phase == 'outage':
        chosen, fail = [r for r in requests if r['id'] == outage['request']], {'rank': outage['owner'],
                                                                              'at': outage['at_step']}
    elif phase == 'resume':
        chosen = [{**r, 'committed': committed} for r in requests if r['id'] == outage['request']]
        fail = None
    else:
        raise ValueError('unknown shard phase')
    return {'requests': [{k: v for k, v in r.items() if k != 'expected'} for r in chosen],
            'eos_ids': plan['workload']['eos_ids'], 'max_tokens': plan['max_boundary_tokens'],
            'threads': read(ROOT / CANONICAL)['resources']['threads'], 'fail': fail}


def owner(rank, address, port, phase, home, store):
    """One owner process for one phase: fetch into ``store``, or serve; owner 0 also drives.

    ``home`` holds only evidence; the shard stays in ``store`` and is never collected.
    """
    configure()
    source = freeze()
    plan = read(ROOT / PLAN)
    home, store = Path(home), Path(store)
    world = len(plan['boundaries']) - 1
    if not 0 <= rank < world or phase not in PHASES:
        raise ValueError('unsupported shard owner role')
    if phase == 'fetch':
        receipt = {'freeze': source, 'rank': rank, **prepare(plan, rank, store), 'completed': True}
        save(home / 'fetch.json', receipt, exclusive=True)
        return receipt
    from neuroshard.evolution.sharded import granite_pipeline as pipeline

    os.environ.setdefault('GLOO_SOCKET_IFNAME', default_interface())
    committed = read(home / 'committed.json') if phase == 'resume' and rank == 0 else None
    directory = home / phase
    directory.mkdir(parents=True, exist_ok=False)
    save(directory / 'job.json', job(plan, phase, committed), exclusive=True)
    save(directory / 'binding.json', {'freeze': source, 'plan_sha256': sha256(ROOT / PLAN), 'rank': rank,
                                      'phase': phase}, exclusive=True)
    return pipeline.run_owner(store / 'config', store / 'shard', rank, world, address, port, directory / 'job.json',
                              directory / 'result.json', timeout=plan['peer_timeout_seconds'])


def default_interface(routes='/proc/net/route'):
    """The interface of the default route, so gloo never binds to a loopback address."""
    for line in Path(routes).read_text().splitlines()[1:]:
        fields = line.split()
        if len(fields) > 1 and fields[1] == '00000000':
            return fields[0]
    raise ValueError('no default route for the owner ring')


def first_difference(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))


def assess(plan, fetches, phases):
    """Declared checks from every owner's fetch receipt and phase results."""
    tensors, _ = inventory(plan)
    requests = {r['id']: r for r in workload(plan)}
    world = len(plan['boundaries']) - 1
    from neuroshard.evolution.sharded import granite

    owned = {rank: sorted(n for n in tensors['tensors'] if granite.owner(n, plan['boundaries']) == rank)
             for rank in range(world)}
    sizes = {rank: sum(tensors['tensors'][n]['end'] - tensors['tensors'][n]['begin'] for n in owned[rank])
             for rank in range(world)}
    total = sum(sizes.values())
    agreement = phases['agreement']
    outputs = agreement[0].get('outputs', []) if agreement[0] else []
    mismatches = [{'id': o['id'], 'first_difference': first_difference(o['token_ids'], requests[o['id']]['expected'])}
                  for o in outputs if o['token_ids'] != requests[o['id']]['expected']]
    outage, resumed = phases['outage'], phases['resume']
    request = requests[plan['outage']['request']]
    committed = (outage[0] or {}).get('inflight', {}).get('token_ids', [])
    resumed_tokens = (resumed[0] or {}).get('outputs', [{}])[0].get('token_ids') if resumed[0] else None
    peak = {rank: max((phases[p][rank] or {}).get('peak_rss_bytes', 0) for p in SERVING) for rank in range(world)}
    generated = sum(len(o['token_ids']) for o in outputs)
    primary = read(ROOT / plan['workload']['source'])['primary']
    canonical_seconds = (sum(g['seconds'] for e in primary['episodes'] for g in e['generations'] if g.get('executed', True))
                         + sum(row['seconds'] for row in primary['anchors']))
    checks = {
        'complete': all(r and r.get('completed') for r in agreement) and len(outputs) == len(requests),
        'agreement': len(outputs) == len(requests) and not mismatches,
        'ownership': all(fetches[r]['tensors'] == owned[r] and fetches[r]['fetched_bytes'] == sizes[r]
                         and sizes[r] < total for r in range(world)),
        'memory': all(0 < peak[r] <= plan['memory_limit_bytes'] and peak[r] < plan['canonical_peak_rss_bytes']
                      for r in range(world)),
        'outage_injected': outage[plan['outage']['owner']] is None and bool(outage[0])
        and not outage[0].get('completed') and 0 < len(committed) < len(request['expected']),
        'recovery': bool(resumed_tokens) and resumed_tokens == request['expected']
        and committed == request['expected'][:len(committed)] and all(r and r.get('completed') for r in resumed),
    }
    return {'passed': all(checks.values()), 'checks': checks, 'mismatches': mismatches,
            'generations': len(outputs), 'generated_tokens': generated,
            'owned_bytes': sizes, 'checkpoint_bytes': total, 'peak_rss_bytes': peak,
            'sent_bytes': {r: (agreement[r] or {}).get('sent_bytes') for r in range(world)},
            'fetch_seconds': {r: fetches[r]['fetch_seconds'] for r in range(world)},
            'fetch_requests': {r: fetches[r]['requests'] for r in range(world)},
            'generation_seconds': sum(o['seconds'] for o in outputs), 'canonical_generation_seconds': canonical_seconds,
            'outage': {'committed_before_loss': len(committed), 'resumed_tokens': len(resumed_tokens or [])},
            'checklist_credit': False, 'admission_evidence': False}
