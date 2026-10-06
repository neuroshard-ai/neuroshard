"""A4 third execution: the complete learned assistant served across Granite owners.

Three owners fetch only their tensors. Owner 2 also holds the round-4 addition
arm; owner 0 holds the tokenizer and the refitted gate, renders every request,
runs the workspace tools and selects the parent or the arm once per episode
from the parent's final-layer feature. Every owner keeps one prefix-reusing
episode cache. The served development episodes must reproduce the single-host
round-4 development result token for token. A diagnostic also records whether
a fresh process computes its first forward pass reproducibly.
"""

import hashlib
import os
from pathlib import Path
import resource
import time
import urllib.request

from neuroshard.evolution import granite_shard_execution as shard
from neuroshard.evolution.modular_reference_execution import ROOT, read, save, sha256

PLAN = 'config/experiments/granite-shard-serving.json'
SCRIPT = 'scripts/run_granite_shard_serving.py'
PROFILE = 'granite-shard-serving'
PHASES = ('fetch', 'determinism', 'serve')
UPLOADED = '.arms'
SMALL_FILES = ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
               'special_tokens_map.json', 'chat_template.jinja', 'vocab.json', 'merges.txt')


def committed_sources(root=ROOT):
    return shard.committed_sources(root, PLAN)


def configure():
    shard.configure()


def freeze():
    return shard.freeze(PLAN)


def target(plan, root=ROOT):
    """The single-host served episodes of the pinned round-4 development result."""
    pinned = plan['target']
    if sha256(root / pinned['path']) != pinned['sha256']:
        raise ValueError('development result changed')
    episodes = read(root / pinned['path'])['replies'][pinned['system']]['episodes']
    if len(episodes) != pinned['episodes']:
        raise ValueError('development result differs from its declaration')
    return episodes


def prepare(plan, rank, store):
    """This owner's verified byte ranges plus the pinned small files (config, tokenizer)."""
    receipt = shard.prepare(plan, rank, store)
    artifacts = read(ROOT / plan['model']['artifacts'])['models']['baseline']
    base = f"https://huggingface.co/{artifacts['repo']}/resolve/{artifacts['revision']}"
    for name in SMALL_FILES:
        with urllib.request.urlopen(f'{base}/{name}', timeout=120) as response:
            raw = response.read()
        spec = artifacts['files'][name]
        digest = (hashlib.sha256(raw).hexdigest() if spec['algorithm'] == 'sha256'
                  else hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest())
        if digest != spec['digest']:
            raise ValueError(f'{name} differs from the pinned artifact')
        (store / 'config' / name).write_bytes(raw)
    return receipt


def arm_files(plan, directory):
    """The uploaded arm and gate, checked against the digests pinned by the development execution.

    The plan names the arm's kind, an added module or an update; a plan that names none serves the addition.
    """
    directory = Path(directory)
    pinned = plan['arm']
    kind = pinned.get('kind', 'addition')
    checkpoint = directory / f'{kind}-checkpoint'
    manifest = read(checkpoint / 'manifest.json')
    if (manifest['arm'] != kind or manifest['trainable_sha256'] != pinned['trainable_sha256']
            or sha256(checkpoint / 'trainable.safetensors') != pinned['trainable_sha256']):
        raise ValueError('uploaded arm differs from the pinned development arm')
    if sha256(directory / 'integration.json') != pinned['integration_sha256']:
        raise ValueError('uploaded gate differs from the pinned development gate')
    return checkpoint, read(directory / 'integration.json')['arms'][kind]['gate']


def job(plan):
    learning = read(ROOT / plan['learning'])
    arm, gate = arm_files(plan, ROOT / UPLOADED)
    return {'spec': learning['training'], 'arm': str(arm), 'gate': gate, 'policy': read(ROOT / learning['policy']),
            'split': 'development', 'case_ids': [row['id'] for row in target(plan)],
            'eos_ids': plan['eos_ids'], 'max_tokens': plan['max_boundary_tokens'],
            'threads': read(ROOT / shard.CANONICAL)['resources']['threads']}


def determinism(plan, rank, store, home, index):
    """One fresh process: the same fixed input through this owner's layers three times, as digests."""
    import torch
    from transformers import DynamicCache

    from neuroshard.evolution.sharded import granite

    torch.set_num_threads(read(ROOT / shard.CANONICAL)['resources']['threads'])
    config = granite.load_config(store / 'config')
    partition, _ = granite.load_partition(config, store / 'shard', rank)
    generator = torch.Generator().manual_seed(plan['determinism']['seed'])
    length = plan['determinism']['tokens']
    ids = torch.randint(0, config.vocab_size, (1, length), generator=generator)
    hidden = torch.randn((1, length, config.hidden_size), generator=generator).to(torch.bfloat16)
    digests = []
    with torch.inference_mode():
        for _ in range(plan['determinism']['passes']):
            out = partition(partition.embed(ids) if rank == 0 else hidden, None, DynamicCache())
            digests.append(hashlib.sha256(out.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
    result = {'rank': rank, 'process': index, 'digests': digests, 'completed': True,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    save(home / 'determinism' / f'process-{index}.json', result, exclusive=True)
    return result


def owner(rank, address, port, phase, home, store, index=0, plan_path=PLAN):
    configure()
    source = shard.freeze(plan_path)
    plan = read(ROOT / plan_path)
    home, store = Path(home), Path(store)
    world = len(plan['boundaries']) - 1
    if not 0 <= rank < world or phase not in PHASES:
        raise ValueError('unsupported serving owner role')
    if phase == 'fetch':
        receipt = {'freeze': source, 'rank': rank, **prepare(plan, rank, store), 'completed': True}
        save(home / 'fetch.json', receipt, exclusive=True)
        return receipt
    if phase == 'determinism':
        return determinism(plan, rank, store, home, index)
    from neuroshard.evolution.sharded import granite_serving

    os.environ.setdefault('GLOO_SOCKET_IFNAME', shard.default_interface())
    directory = home / 'serve'
    directory.mkdir(parents=True, exist_ok=False)
    value = job(plan) if rank in (0, world - 1) else {'max_tokens': plan['max_boundary_tokens'],
                                                     'threads': read(ROOT / shard.CANONICAL)['resources']['threads']}
    if rank == 0:
        value['tokenizer'] = str(store / 'config')
    save(directory / 'job.json', value, exclusive=True)
    save(directory / 'binding.json', {'freeze': source, 'plan_sha256': sha256(ROOT / plan_path), 'rank': rank},
         exclusive=True)
    return granite_serving.run_owner(store / 'config', store / 'shard', rank, world, address, port,
                                     directory / 'job.json', directory / 'result.json',
                                     timeout=plan['peer_timeout_seconds'])


def assess_phases(plan, fetches, phases):
    return assess(plan, fetches, phases.get('determinism', []), phases['serve'])


def assess(plan, fetches, determinism_rows, served):
    """Declared checks: every served generation and selection equals the single-host development result."""
    from neuroshard.evolution import assistant_experience_gate as gate
    from neuroshard.evolution import assistant_workflow as workflow
    from neuroshard.evolution import assistant_workflow_data as data

    world = len(plan['boundaries']) - 1
    expected = {row['id']: row for row in target(plan)}
    rows = (served[0] or {}).get('episodes', [])
    by_id = {case['id']: case for case in data.cases('development')}
    policy = read(ROOT / read(ROOT / plan['learning'])['policy'])
    keys = ('input_token_ids', 'token_ids', 'text', 'terminated', 'prompt_sha256', 'reused_prefix_tokens')
    mismatches = []
    for row in rows:
        want = expected[row['id']]
        if workflow.score(by_id[row['id']], row, policy) != row['score']:
            raise ValueError('served outcome rescore differs')
        same = (row['selected'] == want['selected'] and row['score'] == want['score']
                and len(row['generations']) == len(want['generations'])
                and all(all(a[k] == b[k] for k in keys) for a, b in zip(row['generations'], want['generations'])))
        if not same:
            mismatches.append(row['id'])
    tokenizer = (served[0] or {}).get('tokenizer') or {}
    pinned = read(ROOT / 'config/experiments/assistant-workflow-canonical.json')['tokenizer']
    peaks = {r: (served[r] or {}).get('peak_rss_bytes', 0) for r in range(world)}
    first = {r: sorted({d['digests'][0] for d in determinism_rows if d and d['rank'] == r}) for r in range(world)}
    later = {r: sorted({x for d in determinism_rows if d and d['rank'] == r for x in d['digests'][1:]}) for r in range(world)}
    checks = {
        'complete': all(r and r.get('completed') for r in served) and len(rows) == len(expected),
        'tokenizer': tokenizer.get('pipeline_sha256') == pinned['pipeline_sha256']
        and tokenizer.get('fixture_sha256') == pinned['fixture_sha256'],
        'fetched_only_owned': all(f and f.get('completed') for f in fetches),
        'arm_on_its_owner': (served[world - 1] or {}).get('arm_sha256') == plan['arm']['trainable_sha256']
        and all((served[r] or {}).get('arm_sha256') is None for r in range(world - 1)),
        'agreement': len(rows) == len(expected) and not mismatches,
        'memory': all(0 < peaks[r] <= plan['memory_limit_bytes'] for r in range(world)),
    }
    correct = sum(row['score']['passed'] for row in rows)
    return {'passed': all(checks.values()), 'checks': checks, 'mismatches': mismatches, 'correct': correct,
            'expected_correct': sum(row['score']['passed'] for row in expected.values()),
            'selected_arm_episodes': sum(row['selected'] == 'arm' for row in rows),
            'p95_seconds': gate.p95(rows, routed=True) if rows else None,
            'single_host_p95_seconds': gate.p95(list(expected.values()), routed=True),
            'owner_peak_rss_bytes': peaks, 'sent_bytes': {r: (served[r] or {}).get('sent_bytes') for r in range(world)},
            'determinism': {'processes_per_owner': plan.get('determinism', {}).get('processes', 0),
                            'distinct_first_pass_digests': {r: len(first[r]) for r in range(world)},
                            'distinct_later_pass_digests': {r: len(later[r]) for r in range(world)},
                            'first_equals_later': {r: first[r] == later[r] for r in range(world)}},
            'checklist_credit': False, 'admission_evidence': False}
