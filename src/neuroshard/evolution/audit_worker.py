"""Complete replay and durable audit reporting for the funded candidate profile.

The daemon serves explicitly configured sponsors. It does not implement an
independent-operator registry or prove that an auditor used separate hardware.
"""
import argparse
import base64
import fcntl
import json
import os
import secrets
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

import requests

from neuroshard.dataflow.store import canonical
from neuroshard.demo import protocol, client as wire
from . import auditing, forward, lifecycle
from .objects import Objects, MAX_OBJECT_BYTES, digest
from .settlement import CHUNK_BYTES
from .transactions import Outbox
from .verification import Metadata, audit, dependencies


class MissingArtifact(FileNotFoundError):
    def __init__(self, key):
        self.key = key
        super().__init__('Required audit artifact is unavailable: '+key)


def required_objects(claim):
    metadata = Metadata(claim['metadata'])
    record = metadata.json(claim['record_root'])
    needed = set()
    for stage in range(auditing.stages(claim)):
        if claim.get('kind') in ('score', 'inference'):
            inputs, outputs = lifecycle.dispute(metadata, claim['record_root'], stage)
            needed.update(inputs + outputs)
        elif claim.get('kind') == 'growth':
            parent = metadata.json(record['parent'])
            needed.add(parent['components'][f'block_{parent["config"]["num_hidden_layers"]-1:03}']['root'])
        else:
            needed.update(dependencies(metadata, record['traces'][stage]))
    needed.update(c['root'] for c in metadata.json(claim['model_root'])['components'].values())
    return sorted(needed)


def replay(store, claim):
    """Read all declared objects and replay every stage before attesting.

    Missing/corrupt objects raise; neither is converted into a matching report.
    The invalid stage result is suitable for the existing native fraud path.
    """
    started = time.monotonic()
    metadata = Metadata(claim['metadata'])
    for key, value in metadata.values.items():
        if store.put_json(value) != key:
            raise ValueError('Audit metadata checksum mismatch')
    size = 0
    objects = required_objects(claim)
    for key in objects:
        try:
            size += len(store.get(key))
        except FileNotFoundError as exc:
            raise MissingArtifact(key) from exc
    reports = []
    for stage in range(auditing.stages(claim)):
        at = time.monotonic()
        verdict = audit(store, metadata, claim['record_root'], stage)
        reports.append({'stage': stage, 'seconds': time.monotonic()-at, **verdict})
        if not verdict['valid'] and claim.get('audit_profile') != auditing.QUORUM_FORMAT:
            break
    valid = len(reports) == auditing.stages(claim) and all(r['valid'] for r in reports)
    return {'record_root': claim['record_root'], 'valid': valid, 'stages': reports,
            'coverage_root': auditing.coverage(claim) if valid else None,
            'object_count': len(objects), 'object_bytes': size, 'seconds': time.monotonic()-started}


class Worker:
    def __init__(self, home, url, genesis_hash, key, sponsors, store, max_stages=64, portable_backend=None):
        self.home, self.url, self.store = Path(home), url, store
        self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = (self.home/'worker.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.owner = protocol.Identity.load_or_create(key)
        self.sponsors = set(sponsors)
        self.max_stages = max_stages
        self.portable_backend = portable_backend
        genesis = wire.rpc(url, 'genesis')['genesis']
        if digest(canonical(genesis)) != genesis_hash:
            raise ValueError('Audit worker genesis differs from configured commitment')
        self.chain_id = genesis['chain_id']
        self.native = genesis['app_state']['manifest'].get('auditing', {}).get('format') == auditing.QUORUM_FORMAT
        self.portable = any(key in genesis['app_state']['manifest'] for key in ('portable_work', 'expert_work'))
        from .app import code_hash
        from .runtime import check
        check()
        if genesis['app_state']['manifest']['code_hash'] != code_hash():
            raise ValueError('Audit worker source differs from candidate genesis')
        self.outbox = Outbox(self.home/'outbox.sqlite', url, self.chain_id, self.owner)
        self.db = sqlite3.connect(self.home/'reports.sqlite')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS reports (id TEXT PRIMARY KEY, salt TEXT, report BLOB)')

    def send(self, operation, kind, **fields):
        return self.outbox.send(operation, kind, **fields)

    def tick(self):
        if self.portable and not self.portable_backend:
            return {'phase': 'portable_replay_backend_required'}
        pending = self.outbox.pending()
        if pending:
            self.outbox.confirm(pending)
        service = wire.query(self.url, '/auditing')
        if service is None:
            raise ValueError('Candidate has no funded audit profile')
        owner = self.owner.public_key
        active = [b for b in service['budgets'].values() if b['auditors'].get(owner, {}).get('bond')]
        for key, budget in sorted(service['budgets'].items()):
            selected = budget['auditors'].get(owner)
            if (selected is not None and not selected['bond'] and not active
                    and budget['reservation'] is None and (self.native or budget['sponsor'] in self.sponsors)
                    and budget['stage_limit'] <= self.max_stages):
                self.send('accept:'+key, 'accept_audit', budget_id=key)
                return {'phase': 'audit_offer_accepted', 'budget': key}
        claim = wire.query(self.url, '/candidate')
        if not claim:
            return {'phase': 'idle'}
        budget = service['budgets'].get(claim['audit_budget'])
        if (not budget or owner not in budget['auditors'] or not budget['auditors'][owner]['bond']
                or budget['auditors'][owner]['revealed']):
            return {'phase': 'idle'}
        if claim['challenge']:
            if claim['challenge']['kind'] == 'fraud' and claim['challenge']['owner'] == owner:
                return self.dispute(claim)
        row = self.db.execute('SELECT salt, report FROM reports WHERE id=?', (claim['id'],)).fetchone()
        if row is None:
            if shutil.disk_usage(self.store.root).free < 2*1024**3:
                raise OSError('Audit artifact store has less than 2 GiB free')
            try:
                if claim.get('kind') in ('portable_training', 'portable_quality', 'portable_inference',
                                         'expert_features', 'expert_training'):
                    if not self.portable_backend:
                        return {'phase': 'portable_replay_backend_required', 'claim': claim['id']}
                    backend = self.portable_backend
                    if (not isinstance(backend['argv'], list) or not 1 <= len(backend['argv']) <= 64
                            or any(not isinstance(a, str) or not a for a in backend['argv'])
                            or type(backend['timeout_seconds']) is not int
                            or not 1 <= backend['timeout_seconds'] <= 14400):
                        raise ValueError('Invalid operator-configured GPU replay backend')
                    # The command comes exclusively from local configuration.
                    # Claim contents are JSON on stdin, never executable text.
                    completed = subprocess.run(backend['argv'], input=canonical(claim),
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=backend['timeout_seconds'])
                    if completed.returncode or len(completed.stdout) > 8*1024**2:
                        return {'phase': 'portable_replay_unavailable', 'claim': claim['id'],
                                'returncode': completed.returncode}
                    if claim['kind'] in ('expert_features', 'expert_training'):
                        from .expert_work import replay_report
                    else:
                        from .portable_work import replay_report
                    report = replay_report(claim, protocol.parse_json(completed.stdout))
                else:
                    report = replay(self.store, claim)
            except MissingArtifact as exc:
                if claim['challenge']:
                    return {'phase': 'waiting_for_dispute', 'claim': claim['id']}
                stage = self.availability_stage(claim, exc.key)
                self.send('availability:'+claim['id']+':'+exc.key, 'challenge', claim_id=claim['id'],
                          stage=stage, challenge_kind='availability', object_root=exc.key)
                return {'phase': 'availability_requested', 'claim': claim['id'], 'object_root': exc.key}
            salt = secrets.token_hex(32)
            with self.db:
                self.db.execute('INSERT INTO reports VALUES (?, ?, ?)', (claim['id'], salt, canonical(report)))
        else:
            salt, raw = row
            report = protocol.parse_json(raw)
        native = 'voting_snapshot' in budget
        if not report['valid'] and not native:
            if claim['challenge']:
                return {'phase': 'waiting_for_dispute', 'claim': claim['id']}
            bad = next(r['stage'] for r in report['stages'] if not r['valid'])
            self.send('challenge:'+claim['id'], 'challenge', claim_id=claim['id'], stage=bad,
                      challenge_kind='fraud', object_root=None)
            return {'phase': 'fraud_challenged', 'claim': claim['id'], 'stage': bad}
        selected = budget['auditors'][owner]
        if selected['commitment'] is None:
            value = (auditing.verdict_commitment(self.chain_id, claim['id'], owner, auditing.coverage(claim),
                     salt, report['valid']) if native else
                     auditing.commitment(self.chain_id, claim['id'], owner, report['coverage_root'], salt))
            self.send('commit:'+claim['id'], 'audit_commit', claim_id=claim['id'], commitment=value)
            return {'phase': 'full_audit_committed', 'claim': claim['id'], 'report': report}
        height = wire.query(self.url)['height']
        if claim['audit_commit_end'] < height <= claim['audit_reveal_end']:
            if native:
                self.send('reveal:'+claim['id'], 'audit_verdict', claim_id=claim['id'], salt=salt,
                          coverage_root=auditing.coverage(claim), valid=report['valid'])
            else:
                self.send('reveal:'+claim['id'], 'audit_reveal', claim_id=claim['id'], salt=salt,
                          coverage_root=report['coverage_root'])
            return {'phase': 'full_audit_revealed', 'claim': claim['id']}
        return {'phase': 'waiting_for_reveal', 'claim': claim['id']}

    @staticmethod
    def availability_stage(claim, key):
        metadata = Metadata(claim['metadata'])
        record = metadata.json(claim['record_root'])
        for stage in range(auditing.stages(claim)):
            if claim.get('kind') in ('score', 'inference'):
                needed, outputs = lifecycle.dispute(metadata, claim['record_root'], stage)
            elif claim.get('kind') == 'growth':
                return 0
            else:
                needed = dependencies(metadata, record['traces'][stage])
                outputs = [c['root'] for c in metadata.json(claim['model_root'])['components'].values()]
            if key in needed + outputs:
                return stage
        raise ValueError('Unavailable object is outside the native claim dependencies')

    def dispute(self, claim):
        challenge, claim_id = claim['challenge'], claim['id']
        for key in challenge['needed']:
            if key in challenge['sealed']:
                continue
            raw = self.store.get(key)
            first = len(challenge['uploads'].get(key, []))
            for index, start in enumerate(range(0, len(raw), CHUNK_BYTES)):
                if index < first:
                    continue
                self.send(f'upload:{claim_id}:{key}:{index}', 'upload', claim_id=claim_id,
                          object_root=key, index=index, data=base64.b64encode(raw[start:start+CHUNK_BYTES]).decode())
            self.send(f'seal:{claim_id}:{key}', 'seal', claim_id=claim_id, object_root=key)
        self.send('resolve:'+claim_id, 'resolve', claim_id=claim_id)
        return {'phase': 'fraud_resolved', 'claim': claim_id}

    def close(self):
        self.db.close()
        self.outbox.close()
        self.lock.close()


def http_source(url):
    session = requests.Session()
    session.trust_env = False
    def fetch(key):
        with session.get(url.rstrip('/')+'/'+key[:2]+'/'+key, timeout=120, stream=True) as response:
            if response.status_code == 404:
                return None
            response.raise_for_status()
            raw = bytearray()
            for chunk in response.iter_content(1024*1024):
                if len(raw)+len(chunk) > MAX_OBJECT_BYTES:
                    raise ValueError('Audit source returned an oversized object')
                raw.extend(chunk)
            return bytes(raw)
    return fetch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--rpc', required=True, help='Trusted local full-node RPC, or SSH tunnel to one')
    parser.add_argument('--genesis-sha256', required=True)
    parser.add_argument('--key', type=Path, required=True)
    parser.add_argument('--sponsor', action='append', default=[])
    parser.add_argument('--portable-backend', '--execution-backend', type=Path,
                        help='Local JSON command configuration for full GPU shard replay; not a remote report service')
    parser.add_argument('--objects', type=Path, required=True)
    parser.add_argument('--source', action='append', default=[], help='Read-only content-addressed artifact mirror')
    parser.add_argument('--max-stages', type=int, default=64)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    store = Objects(args.objects)
    store.fetchers.extend(http_source(url) for url in args.source)
    backend = json.loads(args.portable_backend.read_bytes()) if args.portable_backend else None
    worker = Worker(args.home, args.rpc, args.genesis_sha256, args.key, args.sponsor, store, args.max_stages, backend)
    try:
        while True:
            try:
                result = worker.tick()
            except (OSError, requests.RequestException, TimeoutError, subprocess.TimeoutExpired) as exc:
                # A missing object or unavailable node is never a valid audit.
                result = {'phase': 'waiting_for_inputs_or_rpc', 'error': str(exc)}
            print(json.dumps(result), flush=True)
            if args.once:
                break
            time.sleep(1)
    finally:
        worker.close()


if __name__ == '__main__':
    main()
