#!/usr/bin/env python3
"""Execute one locally configured native learning operation on its owner.

The controller installs JSON prescriptions and available object inventories.
Network claims never supply executable code. Audits always run fresh numerical
work; only a publisher may recover its already persisted operation response.
"""
import argparse
import json
from pathlib import Path
import time

from neuroshard.demo.protocol import parse_json
from neuroshard.evolution.reference_data import identity, save


def read(path):
    with Path(path).open('rb') as handle:
        raw = handle.read(8*1024**2+1)
    if len(raw) > 8*1024**2:
        raise ValueError('Bound the installed numerical operation')
    return parse_json(raw)


def run(config):
    from neuroshard.evolution.sharded import expert_execution, prefix_execution
    action = config['action']
    plan, prepared = read(config['plan']), read(config['prepared'])
    paths = {key: Path(config['paths'][key]) for key in ('inputs', 'objects', 'checkpoint_store')}
    started = time.monotonic()
    if action == 'initialize':
        value = expert_execution.initialize(read(config['parent']), plan, prepared,
            objects=paths['objects'], checkpoint_store=paths['checkpoint_store'])
    else:
        profile = read(config['profile'])
        if action == 'prefix_stage':
            incoming = config.get('incoming')
            if incoming:
                incoming = (Path(incoming['features']), read(incoming['report']))
            value = prefix_execution.owned_stage(config['rank'], profile, plan, prepared,
                inputs=paths['inputs'], objects=paths['objects'], home=Path(config['home']),
                incoming=incoming, max_seconds=config['max_seconds'])
        else:
            paths['bank_home'] = Path(config['paths']['bank_home'])
            if action == 'training':
                value = expert_execution.produce_training(read(config['before']), config['updates'],
                    profile, plan, prepared, **paths, max_seconds=config['max_seconds'])
            elif action == 'training_audit':
                value = expert_execution.execute_training(read(config['claim']), profile, plan, prepared,
                    **paths, max_seconds=config['max_seconds'])
            else:
                raise ValueError('Unknown installed numerical action')
    result = {'operation': identity(config), 'action': action, 'seconds': time.monotonic()-started, 'result': value}
    save(Path(config['result']), result)
    print(json.dumps({'action': action, 'result_root': identity(value), 'seconds': result['seconds']}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    run(read(args.config))
