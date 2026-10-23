#!/usr/bin/env python3
"""Replay saved owner logs under the runtime installed here, exactly as the live auditor does.

This qualifies a candidate numerical runtime. Logs committed on the live network under the current runtime
are replayed with the same steps as ``assistant audit``:

1. follow the enacted version;
2. load the shard with ``fetch_stage`` and ``granite.load_partition``;
3. load its adapter and warm it up;
4. read each log with ``granite_audit.load`` and check it with ``granite_audit.replay``.

**Bit-identity.** A candidate runtime is bit-identical for this workload only if every log replays with
``valid`` true and no input mismatch.
- A mismatch means the candidate would compute different numbers from the owners that produced the logs.
- Under it, an auditor would accuse honest owners of fraud, and an owner would be wrongly slashed.

The installed runtime is recorded, not assumed:
- Python version and package versions;
- the CPU flags the runtime requires;
- the numerical environment variables.

Nothing is submitted to the chain; this is read-only.

    python scripts/replay_owner_logs.py --rank 1 --home ~/.neuroshard/assistant-owner \\
        --logs ~/.neuroshard/assistant-owner/logs --limit 40 --out replay.json
"""
import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path


def installed(names):
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--rank', type=int, required=True, help='The shard these logs are for')
    parser.add_argument('--home', type=Path, required=True, help='A home holding (or able to fetch) that stage')
    parser.add_argument('--logs', type=Path, required=True, help='Directory of <job_id>/{log.json,inputs.safetensors}')
    parser.add_argument('--network', type=Path, help='Network descriptor (default: the bundled one)')
    parser.add_argument('--limit', type=int, default=0, help='Replay at most this many logs, newest first (0: all)')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    from neuroshard.client import live_runtime as runtime
    runtime.configure()  # the pinned numerical environment, before torch loads
    import os
    import torch
    from neuroshard.assistant import network
    from neuroshard.client.assistant_cli import following
    from neuroshard.evolution.modular_reference_execution import ROOT, read
    from neuroshard.evolution.sharded import granite, granite_audit

    pinned = read(ROOT / runtime.LIVE_RUNTIME)
    torch.set_num_threads(8)
    environment = {key: os.environ.get(key) for key in pinned['environment']}
    record = {
        'runtime_file': runtime.LIVE_RUNTIME,
        'python': platform.python_version(),
        'packages_installed': installed(pinned['packages']),
        'packages_pinned_by_live_runtime': pinned['packages'],
        'environment': environment,
        'cpu_flags_present': {f: f in Path('/proc/cpuinfo').read_text().split() for f in pinned['required_cpu_flags']},
        'torch_threads': torch.get_num_threads(),
        'rank': args.rank, 'logs': [], 'started_unix': time.time(),
    }
    record['packages_differing_from_pin'] = {k: [v, record['packages_installed'][k]]
                                             for k, v in pinned['packages'].items() if record['packages_installed'][k] != v}

    descriptor = network.descriptor(args.network)
    chain = network.chain(descriptor, args.home)
    descriptor = following(descriptor, chain, args.home, network)
    record['model_root'] = network.model_root(descriptor)
    served, stage = network.fetch_stage(descriptor, args.rank, args.home)
    partition, _ = granite.load_partition(granite.load_config(stage['config']), stage['shard'], args.rank)
    adapter = network.load_adapter(partition, served, stage)
    network.warm(partition, adapter)

    directories = sorted((d for d in args.logs.iterdir() if (d / 'log.json').exists()),
                         key=lambda d: (d / 'log.json').stat().st_mtime, reverse=True)
    if args.limit:
        directories = directories[:args.limit]
    for directory in directories:
        began = time.monotonic()
        entry = {'job': directory.name}
        try:
            log, inputs = granite_audit.load(directory)
            if granite_audit.unattested(log):
                entry.update(outcome='unattested')
            else:
                report = granite_audit.replay(partition, log, inputs, adapter)
                entry.update(outcome='valid' if report['valid'] and report['input_mismatch'] is None else 'MISMATCH',
                             forwards_checked=report.get('forwards_checked'), first_mismatch=report.get('first_mismatch'),
                             input_mismatch=report.get('input_mismatch'))
        except Exception as error:  # noqa: BLE001 - recorded per log
            entry.update(outcome='error', error=f'{type(error).__name__}: {error}'[:300])
        entry['seconds'] = round(time.monotonic() - began, 2)
        record['logs'].append(entry)
        print(json.dumps(entry), flush=True)
        args.out.write_text(json.dumps(record, indent=1))
    counts = {}
    for entry in record['logs']:
        counts[entry['outcome']] = counts.get(entry['outcome'], 0) + 1
    record['summary'] = counts
    record['bit_identical'] = bool(record['logs']) and counts.get('valid', 0) + counts.get('unattested', 0) == len(record['logs'])
    args.out.write_text(json.dumps(record, indent=1))
    print(json.dumps({'summary': counts, 'bit_identical': record['bit_identical'],
                      'packages_differing_from_pin': record['packages_differing_from_pin']}), file=sys.stderr)
    sys.exit(0 if record['bit_identical'] else 1)


if __name__ == '__main__':
    main()
