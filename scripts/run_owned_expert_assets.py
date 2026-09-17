#!/usr/bin/env python3
"""Restore or preserve bounded committed assets on an actual shard owner.

Upload capabilities arrive through stdin and never enter logs or checkpoints.
The controller supplies paths only inside its installed experiment home.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

from neuroshard.evolution.sharded.retained_objects import restore, transfer

HOME = Path('/home/ubuntu/native-expert-live')
PUBLIC = 'https://dwquwt9gkkeil.cloudfront.net/research/native-expert-live-20260916/objects/'


def run(request):
    if request['action'] not in ('fetch', 'publish') or not 1 <= len(request['objects']) <= 4096:
        raise ValueError('Require a bounded installed asset operation')
    started = time.monotonic()

    def one(item):
        key, spec = item
        path = Path(spec['path'])
        if path.is_symlink() or not path.resolve().is_relative_to(HOME):
            raise ValueError('The owner may transfer assets only inside its configured home')
        if request['action'] == 'fetch':
            urls = spec.get('urls', [PUBLIC+key])
            if any(not url.startswith((PUBLIC, 'https://github.com/neuroshard-ai/neuroshard/releases/download/'))
                   for url in urls):
                raise ValueError('Restore only the configured immutable public replicas')
            result = restore(key, spec['bytes'], urls, path, max_seconds=900)
        else:
            result = transfer(key, spec['bytes'], PUBLIC+key, source=path, put_url=spec['put'], max_seconds=900)
        return key, {'sha256': key, **result, 'url': PUBLIC+key}

    with ThreadPoolExecutor(max_workers=4) as pool:
        objects = dict(pool.map(one, sorted(request['objects'].items())))
    return {'action': request['action'], 'objects': objects, 'seconds': time.monotonic()-started,
            'bytes': sum(row['bytes'] for row in objects.values()), 'all_hashes_verified': True}


if __name__ == '__main__':
    try:
        print(json.dumps(run(json.load(sys.stdin))), flush=True)
    except Exception as error:
        # A library exception can contain an expiring signed capability.
        print('Asset operation failed: '+type(error).__name__, file=sys.stderr, flush=True)
        raise SystemExit(1)
