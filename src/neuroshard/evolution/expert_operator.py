"""Reconcile repeated expert cohorts against committed native chain state.

The operator signs only its publisher account. Curators vote independently and
the existing audit workers accept and execute funded obligations. A configured
numerical backend returns real worker receipts and preserved execution evidence.
Neither backend success nor this journal can approve work or promote a model.
"""
import copy
import fcntl
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile

from neuroshard.dataflow.store import canonical
from neuroshard.demo import protocol
from . import auditing, expert_admission, expert_lifecycle, expert_work
from .reference_data import identity
from .schema import integer

FORMAT = 'neuroshard-native-expert-operator-v1'


def command(config, request):
    """Run an explicitly installed backend; network data is JSON, never code."""
    if (not isinstance(config, dict) or set(config) != {'argv', 'timeout_seconds'}
            or not isinstance(config['argv'], list) or not 1 <= len(config['argv']) <= 64
            or any(not isinstance(arg, str) or not arg for arg in config['argv'])):
        raise ValueError('Invalid locally configured execution command')
    integer(config['timeout_seconds'], 1, 14400)
    # A misbehaving child cannot fill the parent's RAM through captured output.
    # The OS still owns process/disk limits for locally installed executors.
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as error:
        result = subprocess.run(config['argv'], input=canonical(request), stdout=output, stderr=error,
                                timeout=config['timeout_seconds'])
        if result.returncode or output.tell() > 8*1024**2:
            raise OSError('Configured execution backend failed or exceeded its response bound')
        output.seek(0)
        value = protocol.parse_json(output.read())
    if not isinstance(value, dict):
        raise ValueError('Execution backend must return a JSON object')
    return value


class Operator:
    def __init__(self, home, outbox, state_reader, backend, next_job, workers, *, max_jobs=3,
                 max_steps=4096, max_attempts=2, window_updates=4):
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = (self.home/'operator.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.outbox, self.read_state = outbox, state_reader
        self.backend, self.next_job = backend, next_job
        self.workers = copy.deepcopy(workers)
        if (set(workers) != {'prefix', 'training'} or len(workers['prefix']) != 3
                or len(set(workers['prefix'])) != 3):
            raise ValueError('Assign three prefix owners and the separate training owner')
        from neuroshard.lab.state import public_key
        for key in [*workers['prefix'], workers['training']]:
            public_key(key)
        self.limits = {'jobs': integer(max_jobs, 1, 100000), 'steps': integer(max_steps, 1, 65536),
                       'attempts': integer(max_attempts, 1, 8), 'window': integer(window_updates, 1, 4)}
        self.db = sqlite3.connect(self.home/'operator.sqlite')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS configuration (id INTEGER PRIMARY KEY, value BLOB);
            CREATE TABLE IF NOT EXISTS cohorts (id TEXT PRIMARY KEY, job BLOB, outcome BLOB);
            CREATE TABLE IF NOT EXISTS execution (id TEXT PRIMARY KEY, request BLOB, response BLOB);
            CREATE TABLE IF NOT EXISTS attempts (id TEXT PRIMARY KEY, count INTEGER NOT NULL);
        ''')
        config = canonical({'format': FORMAT, 'chain_id': outbox.chain_id,
                            'owner': outbox.owner.public_key, 'workers': workers, 'limits': self.limits})
        previous = self.db.execute('SELECT value FROM configuration WHERE id=1').fetchone()
        if previous is not None and previous[0] != config:
            raise ValueError('Operator journal belongs to another account, network or prescription')
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO configuration VALUES (1,?)', (config,))

    def close(self):
        self.db.close()
        self.lock.close()

    def finish(self, key, **outcome):
        value = {'cohort': key, **outcome}
        with self.db:
            self.db.execute('UPDATE cohorts SET outcome=? WHERE id=?', (canonical(value), key))
        return value

    def enqueue(self, state, job, *, initial=False):
        if len(job['work']['schedule']) > self.limits['steps']:
            raise ValueError('The proposed cohort exceeds the operator step budget')
        if not initial:
            expert_admission.validate_job(state, job)
        key = identity(job)
        existing = self.db.execute('SELECT outcome FROM cohorts WHERE id=?', (key,)).fetchone()
        if existing is not None:
            raise ValueError('The feed repeated an already attempted cohort')
        with self.db:
            self.db.execute('INSERT INTO cohorts VALUES (?,?,NULL)', (key, canonical(job)))
        return key

    def reject_data(self, result):
        """Quarantine a reviewed feed entry before funding or signing a job.

        The preparation command's durable history includes this outcome, so it
        can select the next immutable entry instead of retrying rejected data.
        A transient fetch/execution exception is not a rejection.
        """
        from .schema import root
        if (set(result) != {'entry', 'review'} or not isinstance(result['review'], dict)
                or result['review'].get('mechanical_checks_passed') is not False):
            raise ValueError('Rejected feed data needs its identity and failed review evidence')
        key = 'data/'+root(result['entry'])
        outcome = {'phase': 'data_rejected', 'cohort': key, 'review': result['review']}
        with self.db:
            self.db.execute('INSERT INTO cohorts VALUES (?,?,?)',
                            (key, canonical({'rejected_data': result}), canonical(outcome)))
        return outcome

    def execute(self, scope, request):
        """Persist the exact result before any claim can be signed.

        The request identity is also sent to the backend: after a process dies
        during execution, the backend must restore or reproduce that operation,
        never select a different checkpoint or run extra training steps.
        """
        raw = canonical(request)
        old = self.db.execute('SELECT request,response FROM execution WHERE id=?', (scope,)).fetchone()
        if old is not None and old[0] != raw:
            raise ValueError('A numerical operation changed its committed request')
        if old is not None and old[1] is not None:
            return protocol.parse_json(old[1])
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO execution VALUES (?,?,NULL)', (scope, raw))
        value = self.backend({'operation': scope, 'request_root': identity(request), **copy.deepcopy(request)})
        if not isinstance(value, dict) or len(canonical(value)) > 8*1024**2:
            raise ValueError('Require bounded complete numerical execution evidence')
        with self.db:
            self.db.execute('UPDATE execution SET response=? WHERE id=?', (canonical(value), scope))
        return value

    def send(self, operation, kind, **fields):
        self.outbox.send(operation, kind, **fields)
        return {'phase': 'submitted', 'kind': kind, 'transaction': self.outbox.logical_id(operation)}

    def tick(self):
        # An unknown outcome always has priority over fresh reads and new work.
        pending = self.outbox.pending()
        if pending is not None:
            self.outbox.confirm(pending)
            return {'phase': 'transaction_recovered', 'operation': pending}
        state = self.read_state()
        if state['chain_id'] != self.outbox.chain_id:
            raise ValueError('Committed node state belongs to another network')
        admission = expert_admission.bookkeeping(state)
        if admission is None or not auditing.native(state):
            raise ValueError('Require native repeated admission and funded execution quorums')
        life = state['expert_lifecycle']
        pending = self.db.execute('SELECT id,job FROM cohorts WHERE outcome IS NULL').fetchall()
        if len(pending) > 1:
            raise ValueError('Only one cohort may be reconciled at a time')
        if not pending:
            if self.db.execute('SELECT count(*) FROM cohorts').fetchone()[0] >= self.limits['jobs']:
                return {'phase': 'cohort_budget_complete'}
            if state['assignment'] or state['candidate'] or admission['proposal']:
                return {'phase': 'waiting_for_native_work'}
            if not life['quality_closed']:
                # A fresh genesis may already prescribe the first complete job.
                # An active job proposed by another account is never adopted.
                if admission['active'] is not None:
                    return {'phase': 'waiting_for_active_cohort'}
                job = {'work': state['manifest']['expert_work'],
                       'lifecycle': state['manifest']['expert_lifecycle'], 'data': admission['data']}
                key = self.enqueue(state, job, initial=True)
            else:
                job = self.next_job(copy.deepcopy(state))
                if job is None:
                    return {'phase': 'no_new_cohort'}
                if set(job) == {'rejected_data'}:
                    return self.reject_data(job['rejected_data'])
                from .expert_preparation import snapshot
                if snapshot(self.read_state()) != snapshot(state):
                    return {'phase': 'prepared_snapshot_changed'}
                key = self.enqueue(state, job)
            return {'phase': 'cohort_prepared', 'cohort': key}
        key, raw = pending[0]
        job = protocol.parse_json(raw)
        numerical_job = job['work']['checkpoint']['job']
        admitted = admission['seen_jobs'].get(numerical_job)
        proposal = admission['proposal']
        if admitted is None:
            operation = key+'/propose'
            previous = next((row for row in life['history']
                             if row.get('kind') == 'activation' and row.get('job') == key), None)
            if previous is not None:
                return self.finish(key, phase='admission_rejected', evidence=previous)
            if proposal is not None or state['assignment'] or state['candidate'] or not life['quality_closed']:
                return {'phase': 'waiting_for_admission', 'cohort': key}
            expert_admission.validate_job(state, job)
            return self.send(operation, 'propose_expert_job', job=job)
        active_work = expert_work.prescription(state)
        if active_work['checkpoint']['job'] != numerical_job:
            raise ValueError('Recover the earlier cohort outcome before following a different active job')
        if life['quality_closed']:
            outcome = next((row for row in reversed(life['history']) if row.get('kind') == 'quality'
                            and row['report']['prepared'] == job['lifecycle']['quality']['prepared']), None)
            if outcome is not None:
                return self.finish(key, phase='promoted' if outcome['promoted'] else 'quality_rejected',
                                   evidence=outcome, serving_root=state['serving_root'])
            expired = next((row for row in reversed(life['history'])
                            if row.get('kind') == 'job_expired' and row['id'] == admitted), None)
            if expired is not None:
                return self.finish(key, phase='cohort_expired', evidence=expired, serving_root=state['serving_root'])
            raise ValueError('Closed cohort lacks its committed native outcome')
        candidate = state['candidate']
        if candidate is not None:
            return {'phase': 'waiting_for_audit', 'claim': candidate['id']}
        current = state['expert_work']
        if current['feature_claim'] is None:
            phase, stages = 'prefix', job['work']['feature_stages']
        elif current['checkpoint']['step'] < len(job['work']['schedule']):
            phase = 'training'
            stages = min(self.limits['window'], len(job['work']['schedule']) - current['checkpoint']['step'])
        else:
            phase, stages = 'quality', job['lifecycle']['quality']['stages']
        base = key+'/'+phase+'/'+current['checkpoint']['checkpoint']
        attempt = self.db.execute('SELECT count FROM attempts WHERE id=?', (base,)).fetchone()
        attempt = attempt[0] if attempt else 0
        if attempt >= self.limits['attempts']:
            raise ValueError('Native work exhausted its bounded retry allowance')
        scope = base+'/'+str(attempt)
        # Funding is durable before a reservation, and only the audit workers
        # accept its obligations. The publisher never signs their transactions.
        budget_id = None
        try:
            budget_id = self.outbox.logical_id(scope+'/fund')
        except ValueError:
            return self.send(scope+'/fund', 'fund_audit', publisher=self.outbox.owner.public_key,
                             auditors=[], stage_limit=stages, expires_in=100000)
        budget = state['auditing']['budgets'].get(budget_id)
        if budget is None:
            # A rejected claim or expired reservation consumes this attempt.
            # Retain its evidence and acquire a new budget under a fresh ID.
            completed = state['auditing']['history']
            if not any(row.get('id') == budget_id for row in completed):
                return {'phase': 'waiting_for_funding', 'budget': budget_id}
            with self.db:
                self.db.execute('INSERT INTO attempts VALUES (?,?) ON CONFLICT(id) DO UPDATE SET count=excluded.count',
                                (base, attempt+1))
            return {'phase': 'native_attempt_closed', 'budget': budget_id}
        if not auditing.enough(budget, lambda row: bool(row['bond'])):
            return {'phase': 'waiting_for_auditors', 'budget': budget_id}
        assignment = state['assignment']
        if phase != 'quality' and assignment is None:
            if phase == 'prefix':
                return self.send(scope+'/reserve', 'reserve_expert_inputs',
                                 workers=self.workers['prefix'], audit_budget=budget_id)
            return self.send(scope+'/reserve', 'reserve_expert', input_checkpoint=current['checkpoint']['checkpoint'],
                             worker=self.workers['training'], audit_budget=budget_id)
        if assignment is not None and (assignment['owner'] != self.outbox.owner.public_key
                                       or assignment['audit_budget'] != budget_id):
            return {'phase': 'waiting_for_reservation', 'assignment': assignment['id']}
        request = {'phase': phase, 'job': job, 'work': current, 'assignment': assignment,
                   'stages': stages, 'chain_id': state['chain_id']}
        fields = self.execute(scope+'/execute', request)
        kind = {'prefix': 'claim_expert_prefix' if job['work']['format'] == expert_work.PROSPECTIVE
                else 'claim_expert_inputs', 'training': 'claim_expert', 'quality': 'quality_expert'}[phase]
        if phase == 'quality':
            fields = {**fields, 'audit_budget': budget_id}
            # A correctly measured failed gate is submitted and audited too.
            # Native settlement then closes this job without replacing serving.
            if type(fields.get('report', {}).get('passed')) is not bool:
                raise ValueError('The backend must report the measured quality decision')
        return self.send(scope+'/claim', kind, **fields)


def main():
    import argparse
    import os
    import time
    from .committed_state import read
    from .transactions import Outbox
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--native-home', type=Path, required=True)
    parser.add_argument('--genesis-sha256', required=True, help='SHA-256 of the local genesis file bytes')
    parser.add_argument('--rpc', required=True, help='RPC of the same operator-owned full node')
    parser.add_argument('--key', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True,
                        help='Pinned worker keys, numerical command, preparation command and finite limits')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    config = protocol.parse_json(args.config.read_bytes())
    state = read(args.native_home, args.genesis_sha256)
    from neuroshard.demo import client as wire
    # Match the RPC genesis too: a different chain with the same name is unsafe.
    remote = wire.rpc(args.rpc, 'genesis')['genesis']
    local = protocol.parse_json((args.native_home/'config/genesis.json').read_bytes())
    if identity(remote) != identity(local):
        raise ValueError('RPC and local committed state belong to different genesis files')
    args.home.mkdir(parents=True, exist_ok=True, mode=0o700)
    frozen = args.home/'configuration.json'
    expected = canonical({'configuration': config, 'genesis_sha256': args.genesis_sha256})
    if frozen.exists():
        if frozen.read_bytes() != expected:
            raise ValueError('Restart cannot change the frozen operator configuration')
    else:
        with frozen.open('xb') as stream:
            stream.write(expected)
            stream.flush()
            os.fsync(stream.fileno())
    owner = protocol.Identity.load_or_create(args.key)
    outbox = Outbox(args.home/'outbox.sqlite', args.rpc, state['chain_id'], owner)
    operator = None
    try:
        def prepare(current):
            history = [protocol.parse_json(row[0]) for row in operator.db.execute(
                'SELECT outcome FROM cohorts WHERE outcome IS NOT NULL ORDER BY rowid')]
            from .expert_preparation import snapshot
            scope = 'prepare/'+identity({'snapshot': snapshot(current), 'history': history})
            request = {'phase': 'prepare', 'state': current, 'history': history}
            # Height/time may advance while source selection and its history do
            # not. The backend sees the first durable snapshot on every retry.
            old = operator.db.execute('SELECT request FROM execution WHERE id=?', (scope,)).fetchone()
            if old:
                request = protocol.parse_json(old[0])
            result = operator.execute(scope, request)
            if set(result) == {'rejected_data'}:
                return result
            if set(result) != {'job'}:
                raise ValueError('Preparation must return a sealed job, job:null, or a failed data review')
            if result['job'] is None:
                # No selection or numerical work happened. Permit polling for
                # later immutable feed entries without changing a signed intent.
                with operator.db:
                    operator.db.execute('DELETE FROM execution WHERE id=?', (scope,))
            return result['job']

        def execute(request):
            key = 'preparation' if request['phase'] == 'prepare' else 'execution'
            return command(config[key], request)

        operator = Operator(args.home, outbox, lambda: read(args.native_home, args.genesis_sha256),
            execute, prepare, config['workers'], **config['limits'])
        while True:
            try:
                result = operator.tick()
            except (OSError, TimeoutError, subprocess.TimeoutExpired) as error:
                result = {'phase': 'waiting_for_backend_or_node', 'error': str(error)}
            print(json.dumps(result), flush=True)
            if args.once or result['phase'] == 'cohort_budget_complete':
                break
            time.sleep(1)
    finally:
        if operator is not None:
            operator.close()
        outbox.close()


if __name__ == '__main__':
    main()
