#!/usr/bin/env python3
"""Probe the real native executor against a frozen archived GPU trajectory."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

from neuroshard.evolution import expert_work, reference_data as data

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_bytes())


def probe(plan_path, inputs, home):
    plan = read(plan_path)
    if plan['format'] != 'neuroshard-native-expert-execution-probe-v1':
        raise ValueError('Require the frozen native execution probe')
    for name, digest in plan['sources'].items():
        if data.sha256(ROOT / name) != digest:
            raise ValueError('Execution source changed after the plan was frozen')
    config = read(inputs / 'executor.json')
    if (data.identity(config['profile']) != plan['profile']
            or data.identity(config['plan']) != plan['training_plan']
            or data.identity(config['prepared']) != plan['prepared']
            or config['max_seconds'] != plan['prefix_max_seconds']):
        raise ValueError('The declared original job changed')
    claims = [read(inputs / name) for name in ('prefix.json', 'window-0.json', 'window-4.json')]
    if [data.identity(claim) for claim in claims] != plan['claims']:
        raise ValueError('The prescribed bounded audit claims changed')
    home.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    outcomes = []

    def execute(label, claim, settings, expected):
        config_path = home / (label + '-config.json')
        data.save(config_path, settings)
        at = time.monotonic()
        with (home / (label + '.stderr')).open('wb') as log:
            result = subprocess.run([sys.executable, '-m', 'neuroshard.evolution.sharded.expert_execution',
                '--config', str(config_path)], input=data.canonical(claim), stdout=subprocess.PIPE,
                stderr=log, timeout=settings['max_seconds'] + 90)
        elapsed = time.monotonic() - at
        if expected is None:
            if result.returncode == 0 or result.stdout.strip():
                raise ValueError('Unavailable input produced an execution report')
            row = {'label': label, 'unavailable': True, 'seconds': elapsed}
        else:
            if result.returncode != 0:
                raise ValueError('Executor failed: ' + label)
            report = json.loads(result.stdout)
            verdict = expert_work.replay_report(claim, report)
            data.save(home / (label + '-report.json'), report)
            if verdict['valid'] is not expected:
                raise ValueError('Unexpected numerical verdict: ' + label)
            row = {'label': label, 'valid': expected, 'stages': len(report['stages']), 'seconds': elapsed}
        outcomes.append(row)
        data.save(home / 'progress.json', {'outcomes': outcomes})
        print(json.dumps(row), flush=True)

    execute('prefix', claims[0], config, True)
    actual = copy.deepcopy(config)
    actual['max_seconds'] = plan['training_max_seconds']
    retained = Path(config['paths']['checkpoint_store']) / 'prefix' / claims[0]['record_root'] / 'rank-2/features'
    actual['paths']['bank_home'] = str(retained)
    # Each subprocess begins with a fresh model and optimizer. The second loads
    # only the persisted boundary; the archive supplies commitments, not tensors.
    execute('window-0', claims[1], actual, True)
    execute('window-4', claims[2], actual, True)
    forged = copy.deepcopy(claims[1])
    forged['window']['steps'][0]['metrics']['loss'] += 1
    forged['record_root'] = data.identity(forged['window'])
    execute('forged-measurement', forged, actual, False)
    missing = copy.deepcopy(actual)
    missing['paths']['checkpoint_store'] = str(home / 'missing-boundary')
    execute('missing-boundary', claims[2], missing, None)
    outputs = []
    for claim in claims[1:]:
        compact = claim['output_checkpoint']
        folder = Path(config['paths']['checkpoint_store']) / compact['checkpoint'] / f'shard-{compact["step"]:06d}'
        for spec in compact['tensors'].values():
            path = folder / (spec['sha256'] + '.safetensors')
            if path.stat().st_size != spec['bytes'] or data.sha256(path) != spec['sha256']:
                raise ValueError('Available boundary differs from the exact archived checkpoint')
        outputs.append({'step': compact['step'], 'checkpoint': compact['checkpoint'],
                        'tensor_bytes': sum(spec['bytes'] for spec in compact['tensors'].values())})
    result = {'passed': True, 'plan': data.identity(plan), 'outcomes': outcomes, 'boundaries': outputs,
        'seconds': time.monotonic() - started, 'training_updates': 8,
        'forged_claim_updates_recomputed': 4, 'tokens_issued': 0, 'native_activated': False}
    data.save(home / 'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--home', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(probe(args.plan, args.inputs, args.home)), flush=True)
