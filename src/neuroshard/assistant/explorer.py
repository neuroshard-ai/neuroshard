"""A read-only explorer for the assistant network, as the website shows it.

``python -m neuroshard.assistant.explorer --port 39670`` serves, under ``/assistant/``:

- ``network``: chain height and validators (from CometBFT); served version, owners, governance, growth
  pool and queue, recent model history (from the ledger state); every RPC's height and whether its state
  root agrees with the others at that height.
- ``jobs``: the most recent settled, fraud and grown results, with who served, paid and contributed.

It holds no keys, accepts no input, and caches each reply for a few seconds so a busy page never
loads the chain. Several validators' RPCs are read and cross-checked by ``network.Chain``; a page
built from this API can say which validators answered and whether they agreed.
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import urllib.request

from neuroshard.assistant import network as client

CACHE_SECONDS = 5


def rpc(address, method, params=None, timeout=5):
    query = '&'.join(f'{k}={v}' for k, v in (params or {}).items())
    with urllib.request.urlopen(f"{address}/{method}{'?' + query if query else ''}", timeout=timeout) as response:
        return json.loads(response.read())['result']


def snapshot(descriptor):
    """Everything the network page shows, read once from the chain."""
    endpoints = client.endpoints(descriptor)
    chain = client.Chain(endpoints, descriptor['chain_id'])
    state = chain.state()
    rpcs = []
    for address in endpoints:
        try:
            status = rpc(address, 'status')['sync_info']
            rpcs.append({'rpc': address, 'height': int(status['latest_block_height']), 'catching_up': status['catching_up'],
                         'block_time': status['latest_block_time']})
        except (OSError, ValueError, KeyError):
            rpcs.append({'rpc': address, 'height': None})
    answering = next((r['rpc'] for r in rpcs if r.get('height')), endpoints[0])
    validators = rpc(answering, 'validators', {'per_page': '100'})['validators']
    commit = rpc(answering, 'commit')['signed_header']['commit']
    signed = {s.get('validator_address') for s in commit['signatures'] if s.get('signature')}
    served = client.version(descriptor)[0] if client.model_root(descriptor) == state['model_root'] else None
    governance = state.get('governance', {})
    owners = [{'log_key': key, 'shard': owner['shard'], 'status': owner['status'], 'bond': owner['amount'],
               'endpoint': owner.get('endpoint')} for key, owner in sorted(state['owners'].items(),
                                                                          key=lambda kv: (kv[1]['shard'], kv[0]))]
    history = [{'height': h['height'], 'model_root': h['model_root'], 'authority': 'owners' if h['authority'] == 'owners'
                else 'steward', 'approving_weight': h.get('approving_weight'), 'total_weight': h.get('total_weight')}
               for h in state.get('model_history', [])]
    proposals = [{'id': pid, 'kind': p.get('kind', 'model'), 'approving': client.ledger.tally(p)[0],
                  'total': client.ledger.tally(p)[1], 'deadline': p['deadline']}
                 for pid, p in sorted(governance.get('proposals', {}).items())]
    return {
        'chain_id': state['chain_id'], 'height': state['height'], 'model_root': state['model_root'],
        'version': served.name if served else None, 'shards': state['shards'] - 1,
        'validators': [{'address': v['address'], 'power': int(v['voting_power']), 'signed_last_block': v['address'] in signed}
                       for v in validators],
        'rpcs': rpcs, 'owners': owners,
        'governance': {'approval_ppm': governance.get('approval_ppm'), 'growth_levy_ppm': governance.get('growth_levy_ppm', 0),
                       'proposals': proposals, 'pending_model': bool(governance.get('pending_model'))},
        'growth': {'pool': state.get('growth_pool', 0), 'queued': len(state.get('growth_queue', [])),
                   'open_jobs': [{'job_id': j, 'segments_committed': len(x['commits']), 'steps': x['steps'],
                                  'segment_steps': x['segment_steps'], 'reward': x['reward']}
                                 for j, x in sorted(state.get('growth', {}).items())]},
        'model_history': history[-8:],
        'params': {k: state['params'][k] for k in ('fee', 'owner_bond_minimum', 'minimum_price', 'challenge_deposit',
                                                   'auditor_share_ppm')},
        'supply': {'burned': state.get('burned', 0)},
        'read_at': time.time(),
    }


def recent(descriptor, limit=30):
    state = client.Chain(client.endpoints(descriptor), descriptor['chain_id']).state()
    rows = []
    for job_id, result in state.get('results', {}).items():
        status = result.get('status')
        if status == 'settled':
            rows.append({'kind': 'conversation', 'id': job_id, 'owners': result['owners'], 'paid': result['paid'],
                         'pooled': result.get('pooled', 0), 'refunded': result['refunded'], 'positions': result['positions']})
        elif status == 'fraud':
            rows.append({'kind': 'fraud', 'id': job_id, 'guilty': result['guilty'], 'slashed': result['slashed'],
                         'height': result.get('height')})
        elif status == 'grown':
            rows.append({'kind': 'growth', 'id': job_id, 'contributors': result['contributors'], 'steps': result['steps'],
                         'final_root': result['final_root'], 'settled_unix': result.get('settled_unix')})
    return {'height': state['height'], 'results': rows[-limit:][::-1], 'open_conversations': len(state['jobs'])}


def following(descriptor, home):
    """The descriptor of the version the chain serves, rebuilt from committed growth (as every node does)."""
    from neuroshard.evolution import growth_runtime

    chain = client.Chain(client.endpoints(descriptor), descriptor['chain_id'])
    try:
        return growth_runtime.follow(descriptor, chain.state(), home, descriptor.get('proof_endpoints', []),
                                     lambda *_: None)
    except Exception:  # noqa: BLE001 - show the chain with the pinned descriptor until it can rebuild
        return descriptor


HEX = set('0123456789abcdef')


def lookup(descriptor, key):
    """What the chain records about one key: an owner log key, an account, or a validator's consensus key.

    Read-only: everything returned is already public on the ledger."""
    key = key.strip().lower()
    if not 40 <= len(key) <= 130 or set(key) - HEX:
        raise KeyError('enter a hex public key: an owner log key, an account key or a consensus key')
    state = client.Chain(client.endpoints(descriptor), descriptor['chain_id']).state()
    out = {'key': key, 'height': state['height'], 'found': []}
    owner = state['owners'].get(key)
    owned = [k for k, o in state['owners'].items() if o['account'] == key]
    for log_key in ([key] if owner else []) + owned:
        o = state['owners'][log_key]
        served = [(j, r) for j, r in state.get('results', {}).items() if r.get('status') == 'settled' and log_key in r['owners']]
        earned = sum(r['paid'][r['owners'].index(log_key)] for _, r in served)
        out['found'].append({'role': 'owner', 'log_key': log_key, 'shard': o['shard'], 'status': o['status'],
                             'bond': o['amount'], 'endpoint': o.get('endpoint'), 'account': o['account'],
                             'conversations_served': len(served), 'earned': earned,
                             'open_jobs': sum(1 for j in state['jobs'].values() if log_key in j['owners']),
                             'reachable': client.reachable(o['endpoint']) if o.get('endpoint') and o['status'] == 'active' else None})
    account = state['accounts'].get(key)
    if account is not None:
        out['found'].append({'role': 'account', 'balance': account['balance'], 'nonce': account['nonce']})
    power = state.get('validators', {}).get(key)
    if power is not None:
        out['found'].append({'role': 'validator', 'power': power,
                             'admitted': next((h['height'] for h in state.get('validator_history', [])
                                               if h['consensus_key'] == key), None)})
    grown = [r for r in state.get('results', {}).values() if r.get('status') == 'grown' and key in r.get('contributors', [])]
    if grown:
        out['found'].append({'role': 'contributor', 'jobs': len(grown), 'steps': sum(r['steps'] for r in grown)})
    return out


def serve(descriptor, host, port, home=None):
    lock, cache = threading.Lock(), {}
    current = {'descriptor': descriptor, 'at': 0.0}

    def latest():
        if home is not None and time.monotonic() - current['at'] > 300:
            current['descriptor'], current['at'] = following(current['descriptor'], home), time.monotonic()
        return current['descriptor']

    def cached(name, build):
        with lock:
            value, at = cache.get(name, (None, 0))
            if value is not None and time.monotonic() - at < CACHE_SECONDS:
                return value
        value = build()
        with lock:
            cache[name] = (value, time.monotonic())
        return value

    routes = {'/assistant/network': lambda: cached('network', lambda: snapshot(latest())),
              '/assistant/jobs': lambda: cached('jobs', lambda: recent(latest()))}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path, _, query = self.path.partition('?')
            route = routes.get(path)
            try:
                if path == '/assistant/lookup':
                    from urllib.parse import parse_qs

                    key = (parse_qs(query).get('key') or [''])[0]
                    payload, code = json.dumps(cached(f'lookup:{key}', lambda: lookup(latest(), key))).encode(), 200
                elif route is None:
                    raise KeyError('unknown path')
                else:
                    payload, code = json.dumps(route()).encode(), 200
            except KeyError as error:
                payload, code = json.dumps({'error': str(error)}).encode(), 404
            except Exception as error:  # noqa: BLE001 - the page shows that the chain could not be read
                payload, code = json.dumps({'error': f'{type(error).__name__}: {error}'}).encode(), 503
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.serve_forever()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--network', help='Network descriptor; defaults to the bundled assistant testnet')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=39670)
    parser.add_argument('--home', help='Where to keep rebuilt versions, to name the version the chain serves')
    args = parser.parse_args(argv)
    from pathlib import Path

    serve(client.descriptor(args.network), args.host, args.port, Path(args.home) if args.home else None)


if __name__ == '__main__':
    main()
