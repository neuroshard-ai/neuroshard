#!/usr/bin/env python3
"""Bounded alpha starter credits; one sponsored transfer per public key.

This is an operated faucet, not an identity or Sybil-resistance mechanism.
Requests contain only public wallet keys. The native ledger remains authoritative.
"""
import argparse
import fcntl
from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
import threading

from neuroshard.client import provider_wire, wire
from neuroshard.client.local_node import LocalNode
from neuroshard.demo import protocol
from neuroshard.evolution.provider_transport import Server, certificate
from neuroshard.evolution.reference_data import save
from neuroshard.evolution.transactions import Outbox
from neuroshard.lab.state import public_key


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_args):
        pass

    def do_POST(self):
        status, value = 400, {'status': 'invalid_request'}
        try:
            if self.path != '/credits' or self.headers.get('Transfer-Encoding') is not None:
                raise ValueError('Use one bounded credit request')
            length = int(self.headers.get('Content-Length', '0'))
            if not 1 <= length <= 1024:
                raise ValueError('Public key request exceeds its bound')
            request = wire.parse(self.rfile.read(length))
            if set(request) != {'public_key', 'chain_id'} or request['chain_id'] != self.server.config['chain_id']:
                raise ValueError('Credit request belongs to another network')
            recipient = public_key(request['public_key'])
            if not self.server.control.acquire(blocking=False):
                status, value = 503, {'status': 'busy; retry the same public key'}
            else:
                try:
                    status, value = self.grant(recipient)
                finally:
                    self.server.control.release()
        except (OSError, ValueError, KeyError) as error:
            value = {'status': 'unavailable', 'error': type(error).__name__}
        raw = wire.canonical(self.server.owner.sign(value))
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(raw)
        self.close_connection = True

    def grant(self, recipient):
        config, owner = self.server.config, self.server.owner
        box = Outbox(Path(config['home'])/'credits.sqlite', config['node_rpc'], config['chain_id'], owner)
        try:
            operation = 'grant:'+recipient
            if box.pending() and box.pending() != operation:
                try:
                    box.confirm(box.pending(), timeout=3)
                except (OSError, ValueError):
                    return 503, {'status': 'earlier transfer pending; retry later'}
            if not box.recorded(operation):
                count = box.db.execute('SELECT count(*) FROM operations').fetchone()[0]
                if (count+1)*config['grant_atoms'] > config['ceiling_atoms']:
                    return 409, {'status': 'starter_credit_budget_exhausted'}
            try:
                box.send(operation, 'transfer', to=recipient, amount=config['grant_atoms'], timeout=3)
            except (OSError, ValueError):
                return 202, {'status': 'pending; retry the same public key', 'recipient': recipient}
            return 200, {'status': 'granted', 'recipient': recipient, 'amount_atoms': config['grant_atoms'],
                'chain_id': config['chain_id'], 'transaction_id': box.logical_id(operation),
                'notice': 'Trial service credits have no demonstrated market value. Conversations are public.'}
        finally:
            box.close()


def serve(config):
    home = Path(config['home'])
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (home/'credits.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bound = home/'bound.json'
        if bound.exists() and (json.loads(bound.read_bytes()) != config or not (home/'credits.sqlite').exists()):
            raise ValueError('Restore the original funded faucet configuration and transaction journal')
        if (type(config['grant_atoms']) is not int or type(config['ceiling_atoms']) is not int
                or not 1 <= config['grant_atoms'] <= config['ceiling_atoms'] <= 10_000_000_000):
            raise ValueError('Require a finite declared starter-credit budget')
        owner = protocol.Identity.load_or_create(home/'identity')
        LocalNode(config['node_rpc'], config['chain_id'], config['manifest_root'])
        box = Outbox(home/'credits.sqlite', config['node_rpc'], config['chain_id'], owner)
        box.close()
        save(bound, config)
        tls, fingerprint = certificate(home, owner)
        server = Server((config['bind'], config['port']), tls, None, max_connections=8)
        server.RequestHandlerClass = Handler
        server.config, server.owner, server.control = config, owner, threading.Lock()
        print(json.dumps({'owner': owner.public_key, 'certificate': fingerprint}), flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Local server configuration')
    parser.add_argument('--descriptor', type=Path, help='Reviewed local alpha access.json for a customer request')
    parser.add_argument('--wallet-home', type=Path)
    args = parser.parse_args()
    if args.config:
        if args.descriptor or args.wallet_home:
            parser.error('Serve local credits or request credits, using separate invocations')
        serve(json.loads(args.config.read_bytes()))
        return
    if not args.descriptor or not args.wallet_home:
        parser.error('Request credits with --descriptor and --wallet-home')
    descriptor = json.loads(args.descriptor.read_bytes())
    wallet = wire.Wallet(args.wallet_home/'account.key')
    faucet = descriptor['starter_credits']
    request = wire.canonical({'chain_id': descriptor['chain_id'], 'public_key': wallet.public_key})
    connection = provider_wire.PinnedConnection(faucet['endpoint'], faucet['certificate'], timeout=45)
    try:
        connection.request('POST', '/credits', body=request, headers={'Content-Type': 'application/json'})
        response = connection.getresponse()
        raw = response.read(8193)
        if len(raw) > 8192:
            raise ValueError('Starter-credit response exceeds its bound')
        value, signer = wire.verify(wire.parse(raw))
        if signer != faucet['owner']:
            raise ValueError('The starter-credit response has another signer')
        if value.get('status') == 'granted' and (value['recipient'] != wallet.public_key
                or value['chain_id'] != descriptor['chain_id'] or value['amount_atoms'] != faucet['grant_atoms']):
            raise ValueError('The credit receipt changed its network, recipient or amount')
        print(json.dumps(value, indent=2))
    finally:
        connection.close()


if __name__ == '__main__':
    main()
