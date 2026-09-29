"""A4 diagnostic: is a Granite training ring reproducible across fresh launches on the same owners?

The shard training execution matched a single-host reference bit for bit, yet
its outage run computed its very first preference reference differently. This
diagnostic launches the same training job (references, then one step) several
times on the same three owners. Every owner records a digest of every tensor it
sends, so any difference is located at its first owner and message.
"""

import os
from pathlib import Path

from neuroshard.evolution import granite_shard_execution as shard
from neuroshard.evolution import granite_shard_training as training
from neuroshard.evolution.modular_reference_execution import ROOT, read, save, sha256

PLAN = 'config/experiments/granite-shard-determinism.json'
SCRIPT = 'scripts/run_granite_shard_determinism.py'
PROFILE = 'granite-shard-determinism'
UPLOADED = '.arms'


def phases(plan):
    return ('fetch',) + tuple(f'ring-{index}' for index in range(plan['launches']))


PHASES = ('fetch',) + tuple(f'ring-{index}' for index in range(16))


def committed_sources(root=ROOT):
    return shard.committed_sources(root, PLAN)


def configure():
    shard.configure()


def freeze():
    return shard.freeze(PLAN)


def job(plan):
    source = read(ROOT / training.PLAN)
    experience, replay, pairs = training.sequences(source)
    return {'arm': source['arm'], 'spec': {**source['spec'], 'steps': plan['steps']}, 'experience': experience,
            'replay': replay, 'pairs': pairs, 'max_tokens': source['max_boundary_tokens'], 'trace': True,
            'warm_up': plan.get('warm_up', False), 'threads': read(ROOT / shard.CANONICAL)['resources']['threads']}


def owner(rank, address, port, phase, home, store, index=0):
    configure()
    source = freeze()
    plan = read(ROOT / PLAN)
    home, store = Path(home), Path(store)
    world = len(plan['boundaries']) - 1
    if not 0 <= rank < world or phase not in phases(plan):
        raise ValueError('unsupported determinism owner role')
    if phase == 'fetch':
        receipt = {'freeze': source, 'rank': rank, **shard.prepare(plan, rank, store), 'completed': True}
        save(home / 'fetch.json', receipt, exclusive=True)
        return receipt
    from neuroshard.evolution.sharded import granite_training

    os.environ.setdefault('GLOO_SOCKET_IFNAME', shard.default_interface())
    directory = home / phase
    directory.mkdir(parents=True, exist_ok=False)
    value = {**job(plan), 'checkpoints': str(store / phase)}
    save(directory / 'job.json', value, exclusive=True)
    save(directory / 'binding.json', {'freeze': source, 'plan_sha256': sha256(ROOT / PLAN), 'rank': rank,
                                      'phase': phase}, exclusive=True)
    return granite_training.run_owner(store / 'config', store / 'shard', rank, world, address, port,
                                      directory / 'job.json', directory / 'result.json',
                                      timeout=plan['peer_timeout_seconds'])


def first_divergence(traces):
    """Index of the first message whose digest differs across launches, or None."""
    length = min(len(trace) for trace in traces)
    for index in range(length):
        if len({trace[index] for trace in traces}) > 1:
            return index
    return None if len({len(trace) for trace in traces}) == 1 else length


def assess_phases(plan, fetches, collected):
    world = len(plan['boundaries']) - 1
    launches = [collected[name] for name in phases(plan)[1:]]
    complete = all(r and r.get('completed') for rows in launches for r in rows)
    per_owner = {}
    for rank in range(world):
        traces = [rows[rank].get('trace', []) for rows in launches if rows[rank]]
        per_owner[rank] = {'messages': len(traces[0]) if traces else 0,
                           'distinct_traces': len({tuple(trace) for trace in traces}),
                           'first_divergence': first_divergence(traces) if traces else None}
    references = [tuple(map(tuple, ((rows[0] or {}).get('receipt') or {}).get('references') or [])) for rows in launches]
    tensors = [tuple(sorted(((rows[world - 1] or {}).get('trainable_sha256') or {}).items())) for rows in launches]
    return {'complete': complete, 'launches': len(launches), 'owners': per_owner,
            'distinct_references': len(set(references)), 'distinct_trained_tensors': len(set(tensors)),
            'reproducible': complete and len(set(references)) == 1 and len(set(tensors)) == 1
            and all(value['distinct_traces'] == 1 for value in per_owner.values()),
            'checklist_credit': False, 'admission_evidence': False}
