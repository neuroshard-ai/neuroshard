"""Ledger freshness checks; a responsive process need not be making progress."""
from datetime import datetime, timezone
import math
import re
import time


def assess(native, summary, *, now=None, maximum_age=180):
    now = time.time() if now is None else now
    if not math.isfinite(now) or not 0 < maximum_age <= 600:
        raise ValueError('Invalid network health clock or freshness limit')
    sync = native['sync_info']
    if native['node_info']['network'] != summary['chain_id']:
        raise ValueError('Consensus endpoint and application belong to different chains')
    timestamp = re.sub(r'\.(\d+)', lambda match: '.' + match[1][:6].ljust(6, '0'), sync['latest_block_time'])
    block_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    if block_time.tzinfo is None:
        raise ValueError('Block time requires a timezone')
    age = now - block_time.astimezone(timezone.utc).timestamp()
    height = int(sync['latest_block_height'])
    if type(summary['height']) is not int or type(sync['catching_up']) is not bool:
        raise ValueError('Malformed ledger progress')
    coherent = height > 0 and abs(summary['height'] - height) <= 1
    current = -5 <= age <= maximum_age
    ready = coherent and current and not sync['catching_up']
    reason = ('ready' if ready else 'catching up' if sync['catching_up'] else
              'inconsistent heights' if not coherent else 'future block time' if age < -5 else 'stalled ledger')
    return {'network_ready': ready, 'network_status': reason, 'stalled': age > maximum_age,
            'seconds_since_block': round(max(0, age), 1), 'latest_block_time': sync['latest_block_time'],
            'latest_block_height': height, 'catching_up': sync['catching_up']}


def require_ready(status):
    if status.get('network_ready') is not True:
        raise ValueError('Network is not ready: ' + status.get('network_status', 'freshness not established')
                         + '. No new payment or contribution was submitted.')
