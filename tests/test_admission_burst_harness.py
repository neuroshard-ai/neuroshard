"""The live admission burst harness (scripts/admission_burst.py, measurement 3).

Negative controls from the reviews of PR #128 (8ccb1aa, 7bb2124) first, then the corrections."""
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import admission_burst as burst  # noqa: E402
from neuroshard.inference import optimistic as ledger_rules  # noqa: E402

A0, A1, A2, A3 = 'acct0', 'acct1', 'acct2', 'acct3'
ACCOUNTS = {'0': A0, '1': A1, '2': A2, '3': A3}
BUSY = 'Shards 1, 2 busy; nothing funded, waiting\nNeuroShard: the network stayed busy until the deadline; nothing was funded\n'


def envelope(account, nonce, price=100):
    return {'body': {'kind': 'serve_open', 'nonce': nonce, 'price': price, 'chain_id': 'c'}, 'public_key': account,
            'signature': 'x'}


def job_of(account, nonce, price=100):
    return ledger_rules.transaction_id(envelope(account, nonce, price))


def record(account, nonce, state='served', superseded=False, turn=None, price=100):
    """An operation record exactly as ``funding.Operations`` writes it."""
    env = envelope(account, nonce, price)
    return {'turn_id': turn or f'{account}-{nonce}', 'envelope': env, 'nonce': nonce,
            'job_id': ledger_rules.transaction_id(env), 'state': state, 'superseded': superseded, 'exposed': True}


def summarized(rec):
    """The harness's own summary of a record (``all_operations``)."""
    return {'turn_id': rec['turn_id'], 'job_id': rec['job_id'], 'nonce': rec['nonce'], 'account': rec['envelope']['public_key'],
            'price': rec['envelope']['body']['price'], 'state': rec['state'], 'exposed': True,
            'superseded': rec['superseded'], 'bound': True}


class Chain:
    def __init__(self):
        self.s = {'height': 100, 'jobs': {}, 'results': {}, 'accounts': {}, 'params': {'job_blocks': 3}}
        self.fail_reads = 0

    def state(self):
        if self.fail_reads:
            self.fail_reads -= 1
            raise OSError('verified RPC read failed')
        return json.loads(json.dumps(self.s))

    def open(self, job, user, price=100):
        self.s['jobs'][job] = {'user': user, 'opened': self.s['height'], 'price': price, 'owners': ['o1', 'o2']}
        self.s['accounts'].setdefault(user, {'nonce': 0})['nonce'] += 1


@pytest.fixture
def ctx(tmp_path):
    burst.CONTEXT.clear()
    burst.CONTEXT.update(run='t', measurement=3, release='test')
    burst.SHELL[:] = []
    burst.QUARANTINED.clear()
    yield


def write_records(home, *records):
    (home / 'operations').mkdir(parents=True, exist_ok=True)
    for rec in records:
        (home / 'operations' / f"{rec['turn_id']}.json").write_text(json.dumps(rec))


# -- a run's events, built the way the harness records them ------------------------------------------------------
class Run:
    """An event log for ``judge``: every attempt carries its own operation records; jobs, settlements and nonces
    follow from them unless a test changes them."""

    def __init__(self, phases=('baseline',), repeats=1, hold_seconds=240):
        self.plan = burst.instance_plan(list(phases), repeats)
        self.hold_seconds = hold_seconds
        self.events, self.nonce = [], {a: 0 for a in ACCOUNTS.values()}
        self.ops = {slot: [] for slot in ACCOUNTS}

    def attempt(self, instance, slot, outcome='served', funded=True, settle='settled', role=None, **extra):
        account, slot = ACCOUNTS[str(slot)], str(slot)
        role = role or dict(next(p for p in self.plan if p['instance'] == instance)['slots'])[slot]
        records, opened = [], []
        if funded:
            rec = record(account, self.nonce[account], turn=f'{instance}-{slot}')
            self.nonce[account] += 1
            records.append(summarized(rec))
            self.ops[slot].append(summarized(rec))
            opened = [rec['job_id']]
            self.events.append({'kind': 'job-seen', 'job': rec['job_id'], 'account': account, 'price': 100})
            if settle:
                self.events.append({'kind': 'settlement', 'job': rec['job_id'], 'status': settle,
                                    **({'paid': [40, 40], 'pooled': 0, 'refunded': 20} if settle == 'settled' else {})})
        now = 1000.0
        self.events.append({'kind': 'attempt', 'instance': instance, 'slot': int(slot), 'role': role,
                            'attempt': f'{instance}-{slot}', 'account': account, 'outcome': outcome,
                            'operations': records, 'opened': opened, 'claim_reconciled': True,
                            'unresolved_operations': [], 'client_stopped': True, 'started_unix': now + 10, **extra})
        return opened

    def holder(self, instance, held=240, exit_early=False, ready=1000.0):
        self.attempt(instance, 0, role='holder', held_from_unix=ready,
                     release_commanded_unix=None if exit_early else ready + held,
                     exited_unix=ready + (0.1 if exit_early else held + 1), early_exit=exit_early, started_unix=ready - 30)

    def events_(self, end=True):
        start = {'kind': 'start', 'plan': self.plan, 'accounts': ACCOUNTS, 'hold_seconds': self.hold_seconds,
                 'nonces': {a: 0 for a in ACCOUNTS.values()}, 'operations_before': {s: [] for s in ACCOUNTS}}
        tail = [{'kind': 'workload-end'}, {'kind': 'drained', 'nonces': dict(self.nonce), 'open': []},
                {'kind': 'operations-final', 'homes': self.ops}, {'kind': 'end'}] if end else []
        return [{'run': 't', **e} for e in [start] + self.events + tail]

    def judge(self, end=True):
        return burst.judge(self.events_(end))


def fill_burst_owner(run, instance='busy-owner#0', **holder):
    run.holder(instance, **holder)
    run.attempt(instance, 1)
    run.attempt(instance, 2, outcome='busy-unfunded', funded=False)


# -- baselines: a clean run passes; nothing passes by default --------------------------------------------------------
def test_a_clean_run_passes():
    run = Run(('baseline', 'burst-2'))
    run.attempt('baseline#0', 0)
    run.attempt('burst-2#0', 0)
    run.attempt('burst-2#0', 1, outcome='busy-unfunded', funded=False)
    verdict = run.judge()
    assert verdict['verdict'] == 'pass', verdict


def test_a_clean_busy_owner_run_passes():
    run = Run(('busy-owner',))
    fill_burst_owner(run)
    verdict = run.judge()
    assert verdict['verdict'] == 'pass', verdict


def test_a_truly_empty_event_file_is_not_a_pass(tmp_path, capsys):
    (tmp_path / 'events.jsonl').write_text('')
    assert burst.verify(SimpleNamespace(out=str(tmp_path), run=None)) != 0
    assert burst.verify(SimpleNamespace(out=str(tmp_path / 'missing'), run=None)) != 0


def test_an_empty_or_interrupted_run_is_never_a_pass():
    run = Run(('burst-2',))
    assert run.judge(end=False)['verdict'] == 'inconclusive'
    assert run.judge()['verdict'] != 'pass', 'planned attempts with no results'
    assert burst.judge([{'run': 't', 'kind': 'start'}])['verdict'] == 'inconclusive', 'no persisted plan'


# -- outcomes ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('reply', ['', 'assistant (1 s): (no reply)\n'])
def test_no_reply_is_not_served(reply):
    job = job_of(A0, 0)
    text = f'Job {job[:16]}: x\n{reply}Check it with: neuroshard assistant status --job {job}\n'
    assert burst.classify(text, 0, [job], job) == 'funded-failed'


def test_outcome_classes():
    job, other = job_of(A0, 0), job_of(A0, 1)
    served = f'assistant (u1): 9\nCheck it with: neuroshard assistant status --job {job}\n'
    assert burst.classify(served, 0, [], job) == 'unfunded-error', 'a printed job is a claim, not chain evidence'
    assert burst.classify(served, 0, [job], job) == 'served'
    assert burst.classify(BUSY, 1, [], None) == 'busy-unfunded'
    assert burst.classify(BUSY, 1, [job], None) == 'violation'
    assert burst.classify(served, 0, [job, other], job) == 'violation'


@pytest.mark.parametrize('outcome', ['funded-failed', 'unfunded-error'])
def test_every_request_is_served_or_busy_before_funding(outcome):
    run = Run(('burst-2',))
    run.attempt('burst-2#0', 0)
    run.attempt('burst-2#0', 1, outcome=outcome, funded=outcome == 'funded-failed')
    verdict = run.judge()
    assert verdict['verdict'] == 'fail' and any('neither served nor busy' in r for r in verdict['reasons']), verdict


# -- attribution by signed record, not by timing (follow-up finding 1) ---------------------------------------------
def test_a_late_job_belongs_to_the_attempt_whose_record_signed_it():
    """A delayed transaction of attempt A executing after attempt B began is A's, and A reported busy: a failure.
    B is not charged with it."""
    run = Run(('burst-2',), repeats=2)
    late = record(A0, 0, turn='burst-2#0-0')
    run.events.append({'kind': 'attempt', 'instance': 'burst-2#0', 'slot': 0, 'role': 'contender',
                       'attempt': 'burst-2#0-0', 'account': A0, 'outcome': 'busy-unfunded', 'opened': [],
                       'operations': [summarized(late)], 'claim_reconciled': True, 'unresolved_operations': [],
                       'client_stopped': True})
    run.ops['0'].append(summarized(late))
    run.nonce[A0] = 1
    run.attempt('burst-2#0', 1)
    run.attempt('burst-2#1', 0)  # B: its own job at the next nonce
    run.attempt('burst-2#1', 1, outcome='busy-unfunded', funded=False)
    run.events.append({'kind': 'job-seen', 'job': late['job_id'], 'account': A0, 'price': 100})  # visible late
    run.events.append({'kind': 'settlement', 'job': late['job_id'], 'status': 'expired'})
    verdict = run.judge()
    reasons = ' '.join(verdict['reasons'])
    assert verdict['verdict'] == 'fail' and 'burst-2#0-0 reported busy' in reasons, verdict
    assert 'burst-2#1-0 funded 2' not in reasons, 'the next attempt is not charged with it'


def test_a_job_no_attempt_signed_fails_the_run():
    run = Run(('baseline',))
    run.attempt('baseline#0', 0)
    run.events.append({'kind': 'job-seen', 'job': job_of(A0, 7), 'account': A0, 'price': 100})
    verdict = run.judge()
    assert verdict['verdict'] == 'fail' and any('no operation record' in r for r in verdict['reasons'])


def test_records_bind_jobs_by_their_signed_body(tmp_path, ctx):
    home = tmp_path / 'h'
    good, forged = record(A0, 0), record(A0, 1)
    forged['job_id'] = job_of(A0, 9)  # a record claiming another job than its envelope signs
    write_records(home, good, forged)
    by_turn = {r['turn_id']: r for r in burst.all_operations(home)}
    assert by_turn[good['turn_id']]['bound'] and not by_turn[forged['turn_id']]['bound']


def test_an_attempt_records_the_operations_it_created(tmp_path, ctx):
    home, chain = tmp_path / 'h', Chain()
    old = record(A0, 0, turn='earlier')
    write_records(home, old)
    ledger = burst.Ledger(tmp_path, chain, {'0': A0})
    ledger.begin()
    new = record(A0, 1, turn='this-attempt')
    client = [sys.executable, '-u', '-c',
              f"import json,sys,pathlib; p=pathlib.Path({str(home)!r})/'operations'/'this-attempt.json'; "
              f"p.write_text({json.dumps(json.dumps(new))}); print('assistant (u1): 9', flush=True); sys.stdin.readline()"]
    chain.open(new['job_id'], A0)
    result = burst.attempt(tmp_path, ledger, 'baseline#0', 0, 'contender', 0, client, home, A0, 10)
    assert [r['turn_id'] for r in result['operations']] == ['this-attempt']
    assert result['signed_jobs'] == [new['job_id']] == result['opened'] and result['outcome'] == 'served'


# -- the busy-owner hold, observed (follow-up finding 2) ------------------------------------------------------------
def test_a_holder_that_exits_early_does_not_prove_the_hold():
    run = Run(('busy-owner',))
    fill_burst_owner(run, exit_early=True)
    verdict = run.judge()
    assert verdict['verdict'] != 'pass' and 'exited before' in verdict['progress']['busy-owner#0']['hold'], verdict


def test_a_hold_shorter_than_planned_does_not_count():
    run = Run(('busy-owner',))
    fill_burst_owner(run, held=0.1)
    assert run.judge()['verdict'] != 'pass'


def test_contenders_must_start_while_the_owners_are_held():
    run = Run(('busy-owner',))
    run.holder('busy-owner#0')
    run.attempt('busy-owner#0', 1, started_unix=2000.0)  # after the release
    run.attempt('busy-owner#0', 2, outcome='busy-unfunded', funded=False)
    assert run.judge()['verdict'] != 'pass'


def test_a_real_holder_that_replies_then_exits_is_an_early_exit(tmp_path, ctx):
    chain = Chain()
    ledger = burst.Ledger(tmp_path, chain, {'0': A0})
    ledger.begin()
    client = [sys.executable, '-u', '-c', "import sys; sys.stdin.readline(); print('assistant (1 s): A real reply', flush=True)"]
    hold, release = threading.Event(), threading.Event()
    result = burst.attempt(tmp_path, ledger, 'busy-owner#0', 0, 'holder', 0, client, tmp_path / 'h', A0, 5,
                           hold=hold, release=release)
    assert hold.is_set() and not release.is_set()
    assert result['early_exit'] and result['release_commanded_unix'] is None
    assert burst.hold_check(result, [], 240) is not None


def test_a_real_holder_that_stays_until_release_is_a_hold(tmp_path, ctx):
    chain = Chain()
    ledger = burst.Ledger(tmp_path, chain, {'0': A0})
    ledger.begin()
    client = [sys.executable, '-u', '-c',
              "import sys; sys.stdin.readline(); print('assistant (1 s): A real reply', flush=True); sys.stdin.readline()"]
    hold, release = threading.Event(), threading.Event()
    threading.Timer(1.5, release.set).start()
    result = burst.attempt(tmp_path, ledger, 'busy-owner#0', 0, 'holder', 0, client, tmp_path / 'h', A0, 10,
                           hold=hold, release=release)
    assert not result['early_exit'] and result['release_commanded_unix'] is not None
    assert result['exited_unix'] >= result['release_commanded_unix']
    contender = {'started_unix': result['held_from_unix'] + 0.5}
    assert burst.hold_check(result, [contender], 1) is None


# -- nonces: one outcome per nonce (follow-up finding 3) -------------------------------------------------------------
def test_a_superseded_flag_with_a_funded_job_is_one_nonce():
    """The funding won the race against its superseding transfer: one job, one nonce, a valid run."""
    run = Run(('baseline',))
    run.attempt('baseline#0', 0)
    run.ops['0'][0]['superseded'] = True
    assert run.judge()['verdict'] == 'pass'


def test_a_superseding_transfer_that_won_is_one_nonce():
    run = Run(('burst-2',))
    run.attempt('burst-2#0', 0)
    lost = summarized(record(A1, 0, state='unfunded', superseded=True, turn='burst-2#0-1'))
    run.events.append({'kind': 'attempt', 'instance': 'burst-2#0', 'slot': 1, 'role': 'contender',
                       'attempt': 'burst-2#0-1', 'account': A1, 'outcome': 'busy-unfunded', 'opened': [],
                       'operations': [lost], 'claim_reconciled': True, 'unresolved_operations': [], 'client_stopped': True})
    run.ops['1'].append(lost)
    run.nonce[A1] = 1
    assert run.judge()['verdict'] == 'pass'


def test_an_unexplained_nonce_fails():
    run = Run(('baseline',))
    run.attempt('baseline#0', 0)
    run.nonce[A0] += 1  # a transaction of this account the run did not make
    verdict = run.judge()
    assert verdict['verdict'] == 'fail' and any('nonce 1' in r for r in verdict['reasons'])


def test_old_records_in_a_reused_home_are_not_counted():
    run = Run(('baseline',))
    run.attempt('baseline#0', 0)
    stale = summarized(record(A0, 0, turn='before-the-run', superseded=True))
    run.ops['0'].insert(0, stale)
    events = run.events_()
    events[0]['operations_before'] = {'0': [stale], '1': [], '2': [], '3': []}
    assert burst.judge(events)['verdict'] == 'pass'


# -- completeness ----------------------------------------------------------------------------------------------------
def test_worker_failures_become_explicit_error_results(tmp_path, ctx, monkeypatch):
    monkeypatch.setattr(burst, 'attempt', lambda *a, **k: (_ for _ in ()).throw(OSError('verified read failed')))
    item = burst.instance_plan(['burst-2'], 1)[0]
    args = SimpleNamespace(client='unused', wait=1, attempt_margin=1, hold_seconds=0,
                           client_home=str(tmp_path / 'h{slot}'))
    burst.phase(tmp_path, None, item, args, ACCOUNTS, 0)
    results = [json.loads(l) for l in (tmp_path / 'events.jsonl').read_text().splitlines()]
    assert [r['outcome'] for r in results if r['kind'] == 'attempt'] == ['error', 'error']


def test_a_silent_client_is_killed_at_its_deadline_and_output_kept(tmp_path, ctx):
    chain = Chain()
    ledger = burst.Ledger(tmp_path, chain, {'0': A0})
    ledger.begin()
    started = time.monotonic()
    noisy = [sys.executable, '-u', '-c', 'import time; print("Job 0123456789abcdef: x", flush=True); time.sleep(60)']
    result = burst.attempt(tmp_path, ledger, 'burst-2#0', 0, 'contender', 0, noisy, tmp_path / 'h', A0, 2)
    assert time.monotonic() - started < 40 and result['outcome'] == 'error' and 'deadline' in result['error']
    assert 'Job 0123456789abcdef' in Path(result['output_file']).read_text()
    assert result['client_stopped']


def test_a_client_not_confirmed_stopped_quarantines_its_home(tmp_path, ctx, monkeypatch):
    monkeypatch.setattr(burst, 'client_gone', lambda home, attempts=6: False)
    monkeypatch.setattr(burst, 'kill_remote', lambda home: False)
    chain = Chain()
    ledger = burst.Ledger(tmp_path, chain, {'0': A0})
    ledger.begin()
    first = burst.attempt(tmp_path, ledger, 'b#0', 0, 'contender', 0, [sys.executable, '-c', ''], tmp_path / 'h', A0, 5)
    second = burst.attempt(tmp_path, ledger, 'b#1', 0, 'contender', 1, [sys.executable, '-c', ''], tmp_path / 'h', A0, 5)
    assert first['outcome'] == 'error' and not first['client_stopped']
    assert second['outcome'] == 'error' and 'not started' in second['error']


def test_one_successful_repeat_does_not_hide_a_failed_one():
    run = Run(('burst-2',), repeats=2)
    run.attempt('burst-2#0', 0)
    run.attempt('burst-2#0', 1, outcome='busy-unfunded', funded=False)
    run.attempt('burst-2#1', 0, outcome='busy-unfunded', funded=False)
    run.attempt('burst-2#1', 1, outcome='busy-unfunded', funded=False)
    verdict = run.judge()
    assert verdict['verdict'] == 'fail' and any(r.startswith('burst-2#1') for r in verdict['reasons'])


def test_the_holder_alone_never_satisfies_busy_owner_progress():
    run = Run(('busy-owner',))
    run.holder('busy-owner#0')
    run.attempt('busy-owner#0', 1, outcome='busy-unfunded', funded=False)
    run.attempt('busy-owner#0', 2, outcome='busy-unfunded', funded=False)
    verdict = run.judge()
    assert verdict['verdict'] == 'fail' and verdict['progress']['busy-owner#0']['contenders_served_and_terminal'] == 0


# -- settlement --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('settlement, verdict', [
    ({'status': 'unknown', 'paid': None, 'refunded': None}, 'fail'),
    ({'status': 'fraud'}, 'fail'),
    ({'status': 'voided'}, 'fail'),
    ({'status': 'settled', 'paid': [40, 40], 'pooled': 0, 'refunded': 30}, 'fail'),   # 110 != 100
    ({'status': 'settled', 'paid': [40, 40], 'pooled': 5, 'refunded': 15}, 'pass'),   # growth share counted
    ({'status': 'expired'}, 'pass'),                                                  # full refund by rule
])
def test_terminal_outcomes_are_checked_against_the_escrow(settlement, verdict):
    run = Run(('baseline',))
    job, = run.attempt('baseline#0', 0, settle=None)
    run.events.append({'kind': 'settlement', 'job': job, **settlement})
    assert run.judge()['verdict'] == verdict


def test_a_job_without_its_opening_record_is_not_certified():
    assert 'not known' in burst.settlement_check({'status': 'settled', 'paid': [1], 'refunded': 0}, None)
