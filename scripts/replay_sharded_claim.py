#!/usr/bin/env python3
"""Replay a native portable claim using operator-staged files, one GPU shard at a time.

The catalog contains local paths, never commands supplied by a claim. It points
at the frozen numerical checkout, its GPU Python executable, prepared data,
rank-specific seeds, checkpoint paths and transcript witnesses. Network fetching
is deliberately outside this verifier; unavailable files cannot earn a report.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from neuroshard.evolution import portable_work
from neuroshard.evolution.reference_data import identity, save

# Catch only explicit mathematical/witness mismatches from the pinned auditor.
# OOM, process failure, missing files and corrupt downloads remain unavailable.
MISMATCHES = [
    'Replayed forward value or backward gradient differs',
    'Replayed scalar differs',
    'Replayed gradient inventory or declaration differs',
    'Replayed weights or Adam moments differ from the claimed output',
    'Replay requested an unrecorded operation',
    'Replay communication order differs',
    'Replay did not cover the complete communication window',
    'Replayed input endpoint or shape differs',
    'Invalid witness tensor inventory',
    'Invalid witness tensor',
]


def run(catalog, claim):
    if claim.get('kind') != 'portable_training':
        raise ValueError('Require a native portable training claim')
    prepared = json.loads(Path(catalog['prepared']).read_bytes())
    if identity(prepared) != claim['prepared']:
        raise ValueError('Prepared computation differs from native obligation')
    before, after = claim['input_checkpoint'], claim['output_checkpoint']
    count = len(before['boundaries'])-1
    source = catalog['checkpoints'][identity(before)]
    expected = catalog['checkpoints'][identity(after)]
    witnesses = catalog['transcripts'][claim['record_root']]
    rows = json.loads(Path(witnesses['manifest']).read_bytes())
    if identity(rows) != claim['record_root']:
        raise ValueError('Closed transcript commitment differs')
    if any(len(values) != count for values in [source, expected, catalog['seeds'], witnesses['ranks']]):
        raise ValueError('The catalog must cover every model partition')
    base = Path(catalog['home'])/claim['id']
    base.mkdir(parents=True, exist_ok=False)
    reports = []
    started = time.monotonic()
    for rank in range(count):
        for path, committed in [(source[rank], before), (expected[rank], after)]:
            if json.loads(Path(path).read_bytes()) != committed:
                raise ValueError('Local checkpoint differs from native input or output')
        home = base/f'rank-{rank}'
        args = [str(Path(catalog['repository'])/'scripts/run_sharded_training.py'), 'adaptive', 'audit',
            '--prepared', catalog['prepared'], '--seed', catalog['seeds'][rank], '--home', str(home),
            '--resume', source[rank], '--expected', expected[rank],
            '--transcripts', witnesses['manifest'], '--witness', witnesses['ranks'][rank]]
        if catalog.get('reference'):
            args += ['--reference', catalog['reference'][rank]]
        # Run the unchanged frozen entry point. The wrapper produces a negative
        # verdict only for its enumerated, typed replay mismatch exceptions.
        wrapper = """import json,runpy,sys
from pathlib import Path
from neuroshard.evolution.reference_data import save
spec=json.loads(sys.argv[1])
sys.argv=spec['args']
try:
 runpy.run_path(sys.argv[0],run_name='__main__')
except ValueError as exc:
 if str(exc) not in spec['mismatches']:raise
 rows=json.loads(Path(spec['transcripts']).read_bytes())
 save(Path(spec['home'])/'audit.json',{'valid':False,'rank':spec['rank'],
      'binding':rows[spec['rank']]['binding'],'transcript_root':spec['root'],'reason':str(exc)})
"""
        spec = {'args': args, 'mismatches': MISMATCHES, 'transcripts': witnesses['manifest'],
                'home': str(home), 'rank': rank, 'root': claim['record_root']}
        env = {**os.environ, 'RANK': str(rank), 'WORLD_SIZE': str(count),
               'PYTHONPATH': str(Path(catalog['repository'])/'src'),
               'ATEN_CPU_CAPABILITY': 'default', 'MKL_ENABLE_INSTRUCTIONS': 'SSE4_2'}
        with (base/f'rank-{rank}.log').open('wb') as log:
            result = subprocess.run([catalog['python'], '-c', wrapper, json.dumps(spec)],
                env=env, cwd=catalog['repository'], stdout=log, stderr=log,
                timeout=catalog.get('rank_timeout_seconds', 600))
        if result.returncode:
            raise RuntimeError('A partition is unavailable; inspect the private replay log')
        reports.append(json.loads((home/'audit.json').read_bytes()))
    result = portable_work.replay_report(claim, reports)
    save(base/'report.json', {**result, 'seconds': time.monotonic()-started})
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    args = parser.parse_args()
    raw = sys.stdin.buffer.read(2*1024**2+1)
    if len(raw) > 2*1024**2:
        raise ValueError('Native claim exceeds its bound')
    print(json.dumps(run(json.loads(args.catalog.read_bytes()), json.loads(raw))), flush=True)
