"""The live admission burst harness (scripts/admission_burst.py, measurement 2).

Negative controls first: each case from the review of PR #128 that must not pass, then the corrections."""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import admission_burst as burst  # noqa: E402

JOB, JOB2 = 'ab' * 32, 'cd' * 32
A0, A1, A2 = 'acct0', 'acct1', 'acct2'
SERVED = f'Job {JOB[:16]}: shards served by a, b.\nassistant (u1): The interval is 9.\n' \
         f'Check it with: neuroshard assistant status --job {JOB}\n'
BUSY = 'Shards 1, 2 busy; nothing funded, waiting\nNeuroShard: the network stayed busy until the deadline; nothing was funded\n'


class Chain:
    """Verified state as the harness reads it: jobs open, results closed, nonces, height."""

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

    def close(self, job, **result):
        job_ = self.s['jobs'].pop(job)
        self.s['results'][job] = {'user': job_['user'], 'height': self.s['height'], **result}


@pytest.fixture
def ctx():
    burst.CONTEXT.clear()
    burst.CONTEXT.update(run='t', measurement=2, release='test')
    burst.SHELL[:] = []
    yield


def plan_start(plan, accounts=None, nonces=None):
    return {'run': 't', 'kind': 'start', 'plan': plan, 'accounts': accounts or {'0': A0, '1': A1, '2': A2},
            'nonces': nonces if nonces is not None else {}}


def judge(events):
    return burst.judge([{'run': 't', **e} for e in events])


def complete(events, nonces=None):
    """The closing records of a run that finished normally."""
    return events + [{'kind': 'workload-end'}, {'kind': 'drained', 'nonces': nonces or {}},
                     {'kind': 'operations-final', 'homes': {}}, {'kind': 'end'}]


# -- 1. no reply is never a completed conversation ----------------------------------------------------------------
@pytest.mark.parametrize('reply', ['', 'assistant (1 s): (no reply)\n'])
def test_no_reply_is_not_served(reply):
    text = f'Job {JOB[:16]}: shards served by a,b.\n{reply}Check it with: neuroshard assistant status --job {JOB}\n'
    assert burst.classify(text, 0, [JOB], JOB) == 'funded-failed'


def test_a_printed_job_is_a_claim_not_chain_evidence():
    assert burst.classify(SERVED, 0, [], JOB) == 'unfunded-error', 'nothing on chain: not served'
    assert burst.classify(SERVED, 0, [JOB], JOB) == 'served'


def test_outcome_classes():
    assert burst.classify(BUSY, 1, [], None) == 'busy-unfunded'
    assert burst.classify(BUSY, 1, [JOB], None) == 'violation'
    assert burst.classify(SERVED, 0, [JOB, JOB2], JOB) == 'violation'
    assert burst.classify('NeuroShard: no route\n', 1, [], None) == 'unfunded-error'


# -- 2. worker failures and incomplete runs -----------------------------------------------------------------------
def test_worker_failures_become_explicit_error_results(tmp_path, ctx, monkeypatch):
    monkeypatch.setattr(burst, 'attempt', lambda *a, **k: (_ for _ in ()).throw(OSError('verified read failed')))
    item = burst.instance_plan(['burst-2'], 1)[0]
    args = SimpleNamespace(client='unused', wait=1, attempt_margin=1, hold_seconds=0, client_home=str(tmp_path / 'h{slot}'))
    burst.phase(tmp_path, None, item, args, {'0': A0, '1': A1}, 0)
    events = [json.loads(l) for l in (tmp_path / 'events.jsonl').read_text().splitlines()]
    results = [e for e in events if e['kind'] == 'attempt']
    assert [r['outcome'] for r in results] == ['error', 'error']
    verdict = judge(complete([plan_start([item])] + [{k: v for k, v in r.items() if k != 'run'} for r in results]))
    assert verdict['verdict'] != 'pass', verdict


def test_an_empty_or_interrupted_run_is_never_a_pass():
    plan = burst.instance_plan(['burst-2'], 1)
    assert judge([plan_start(plan)])['verdict'] == 'inconclusive', 'no attempts, no end'
    assert judge(complete([plan_start(plan)]))['verdict'] != 'pass', 'planned attempts with no results'
    assert judge([{'kind': 'start'}])['verdict'] == 'inconclusive', 'no persisted plan'


def test_verify_exits_nonzero_unless_the_run_passes(tmp_path, capsys):
    (tmp_path / 'events.jsonl').write_text(json.dumps({'run': 't', 'kind': 'start', 'plan': []}) + '\n')
    assert burst.verify(SimpleNamespace(out=str(tmp_path), run=None)) != 0


def test_a_silent_client_is_killed_at_the_attempt_deadline(tmp_path, ctx):
    chain = Chain()
    ledger = burst.Ledger(tmp_path, chain, {'0': A0})
    started = time.monotonic()
    silent = [sys.executable, '-c', 'import time; time.sleep(60)']
    record = burst.attempt(tmp_path, ledger, 'burst-2#0', 0, 'contender', 0, silent, tmp_path / 'h', A0, 2)
    assert time.monotonic() - started < 15 and record['outcome'] == 'error' and 'deadline' in record['error']


def test_output_is_kept_even_when_the_attempt_fails(tmp_path, ctx):
    chain = Chain()
    ledger = burst.Ledger(tmp_path, chain, {'0': A0})
    noisy = [sys.executable, '-u', '-c', 'import time; print("Job 0123456789abcdef: x", flush=True); time.sleep(60)']
    record = burst.attempt(tmp_path, ledger, 'burst-2#0', 0, 'contender', 0, noisy, tmp_path / 'h', A0, 2)
    assert 'Job 0123456789abcdef' in Path(record['output_file']).read_text()


def test_every_attempt_is_planned_before_it_runs(tmp_path, ctx):
    chain = Chain()
    chain.fail_reads = 1  # the very first verified read fails
    ledger = burst.Ledger(tmp_path, chain, {'0': A0})
    record = burst.attempt(tmp_path, ledger, 'burst-2#0', 0, 'contender', 0, [sys.executable, '-c', ''],
                           tmp_path / 'h', A0, 5)
    kinds = [json.loads(l)['kind'] for l in (tmp_path / 'events.jsonl').read_text().splitlines()]
    assert kinds[0] == 'planned-attempt' and record['outcome'] == 'error'


# -- 3. funding visible after the client exits --------------------------------------------------------------------
def test_funding_visible_after_exit_turns_a_busy_attempt_into_a_failure(tmp_path, ctx):
    chain = Chain()
    ledger = burst.Ledger(tmp_path, chain, {'0': A0})
    say_busy = [sys.executable, '-u', '-c', f'print({BUSY.splitlines()[1]!r}, flush=True)']
    record = burst.attempt(tmp_path, ledger, 'burst-2#0', 0, 'contender', 0, say_busy, tmp_path / 'h', A0, 10)
    assert record['outcome'] == 'busy-unfunded'  # provisional: nothing visible yet
    chain.s['height'] += 1
    chain.open(JOB, A0)  # the "unfunded" attempt's job becomes visible later
    ledger.scan()
    events = [json.loads(l) for l in (tmp_path / 'events.jsonl').read_text().splitlines()]
    plan = [{'instance': 'burst-2#0', 'phase': 'burst-2', 'repeat': 0, 'slots': {'0': 'contender'}}]
    verdict = judge(complete([plan_start(plan, {'0': A0})] + [{k: v for k, v in e.items() if k != 'run'}
                                                              for e in events if e['kind'] != 'start']))
    assert verdict['verdict'] == 'fail' and any('reported busy' in r for r in verdict['reasons']), verdict


def test_jobs_are_attributed_by_account_window_and_followed_from_first_sight(tmp_path, ctx):
    chain = Chain()
    ledger = burst.Ledger(tmp_path, chain, {'0': A0})
    ledger.begin(A0, 'first')
    chain.open(JOB, A0)
    chain.s['height'] = 110
    ledger.begin(A0, 'second')
    chain.open(JOB2, A0)
    ledger.scan()
    assert ledger.opened_by('first') == [JOB] and ledger.opened_by('second') == [JOB2]
    chain.close(JOB, status='settled', paid=[40, 40], refunded=20)
    ledger.scan()
    assert ledger.open_jobs() == [JOB2]


def test_unresolved_client_operations_keep_the_run_inconclusive():
    plan = [{'instance': 'baseline#0', 'phase': 'baseline', 'repeat': 0, 'slots': {'0': 'contender'}}]
    events = complete([plan_start(plan, {'0': A0}),
                       {'kind': 'attempt', 'instance': 'baseline#0', 'slot': 0, 'attempt': 'a', 'outcome': 'busy-unfunded',
                        'opened': [], 'unresolved_operations': [{'state': 'broadcast', 'job_id': JOB}]}])
    assert judge(events)['verdict'] != 'pass'


def test_nonces_must_be_explained_by_the_run():
    plan = [{'instance': 'baseline#0', 'phase': 'baseline', 'repeat': 0, 'slots': {'0': 'contender'}}]
    events = complete([plan_start(plan, {'0': A0}, nonces={A0: 0}),
                       {'kind': 'attempt', 'instance': 'baseline#0', 'slot': 0, 'attempt': 'a', 'outcome': 'busy-unfunded',
                        'opened': []}], nonces={A0: 1})
    verdict = judge(events)
    assert verdict['verdict'] == 'fail' and any('nonces' in r for r in verdict['reasons'])


# -- 4. progress ----------------------------------------------------------------------------------------------------
def served_attempt(instance, slot, job, role='contender', **extra):
    return {'kind': 'attempt', 'instance': instance, 'slot': slot, 'role': role, 'attempt': f'{instance}-{slot}',
            'outcome': 'served', 'opened': [job], **extra}


def busy_attempt(instance, slot):
    return {'kind': 'attempt', 'instance': instance, 'slot': slot, 'role': 'contender', 'attempt': f'{instance}-{slot}',
            'outcome': 'busy-unfunded', 'opened': []}


def funded(job, attempt, price=100):
    return [{'kind': 'job-seen', 'job': job, 'account': A0, 'attempt': attempt, 'price': price},
            {'kind': 'settlement', 'job': job, 'status': 'settled', 'paid': [40, 40], 'pooled': 0, 'refunded': 20}]


def test_one_successful_repeat_does_not_hide_a_failed_one():
    plan = burst.instance_plan(['burst-2'], 2)
    events = [plan_start(plan), served_attempt('burst-2#0', 0, JOB), busy_attempt('burst-2#0', 1),
              busy_attempt('burst-2#1', 0), busy_attempt('burst-2#1', 1)] + funded(JOB, 'burst-2#0-0')
    verdict = judge(complete(events))
    assert verdict['verdict'] == 'fail' and any(r.startswith('burst-2#1') for r in verdict['reasons']), verdict
    assert verdict['progress']['burst-2#0']['contenders_served_and_terminal'] == 1


def test_the_holder_alone_never_satisfies_busy_owner_progress():
    plan = burst.instance_plan(['busy-owner'], 1)
    events = [plan_start(plan), served_attempt('busy-owner#0', 0, JOB, role='holder', held_from_s=50.0),
              busy_attempt('busy-owner#0', 1), busy_attempt('busy-owner#0', 2)] + funded(JOB, 'busy-owner#0-0')
    verdict = judge(complete(events))
    assert verdict['verdict'] == 'fail' and verdict['progress']['busy-owner#0']['contenders_served_and_terminal'] == 0


def test_a_holder_that_never_held_invalidates_the_phase():
    plan = burst.instance_plan(['busy-owner'], 1)
    events = [plan_start(plan), served_attempt('busy-owner#0', 0, JOB, role='holder'),
              served_attempt('busy-owner#0', 1, JOB2), busy_attempt('busy-owner#0', 2)] + \
        funded(JOB, 'busy-owner#0-0') + funded(JOB2, 'busy-owner#0-1')
    assert judge(complete(events))['verdict'] != 'pass'


def test_a_clean_run_passes():
    plan = burst.instance_plan(['burst-2'], 1)
    events = [plan_start(plan, nonces={A0: 0, A1: 0, A2: 0}), served_attempt('burst-2#0', 0, JOB),
              busy_attempt('burst-2#0', 1)] + funded(JOB, 'burst-2#0-0')
    verdict = judge(complete(events, nonces={A0: 1, A1: 0, A2: 0}))
    assert verdict['verdict'] == 'pass', verdict


@pytest.mark.parametrize('outcome', ['funded-failed', 'unfunded-error'])
def test_every_request_is_served_or_busy_before_funding(outcome):
    """DESIGN_ADMISSION test 10: a funded attempt with no reply, or an error, fails even if its job settled."""
    plan = burst.instance_plan(['burst-2'], 1)
    other = {'kind': 'attempt', 'instance': 'burst-2#0', 'slot': 1, 'role': 'contender', 'attempt': 'burst-2#0-1',
             'outcome': outcome, 'opened': [JOB2] if outcome == 'funded-failed' else []}
    events = [plan_start(plan, nonces={A0: 0, A1: 0}), served_attempt('burst-2#0', 0, JOB), other] + \
        funded(JOB, 'burst-2#0-0') + ([{**e, 'account': A1} for e in funded(JOB2, 'burst-2#0-1')]
                                      if outcome == 'funded-failed' else [])
    nonces = {A0: 1, A1: 1 if outcome == 'funded-failed' else 0}
    verdict = judge(complete(events, nonces=nonces))
    assert verdict['verdict'] == 'fail' and any('neither served nor busy' in r for r in verdict['reasons']), verdict


# -- 5. settlement --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('settlement, kind', [
    ({'status': 'unknown', 'paid': None, 'refunded': None}, 'fail'),
    ({'status': 'fraud'}, 'fail'),
    ({'status': 'voided'}, 'fail'),
    ({'status': 'settled', 'paid': [40, 40], 'pooled': 0, 'refunded': 30}, 'fail'),   # 110 != 100
    ({'status': 'settled', 'paid': [40, 40], 'pooled': 5, 'refunded': 15}, 'pass'),   # growth share counted
    ({'status': 'expired', 'paid': None, 'refunded': None}, 'pass'),                  # full refund by rule
])
def test_terminal_outcomes_are_checked_against_the_escrow(settlement, kind):
    plan = burst.instance_plan(['baseline'], 1)
    events = [plan_start(plan, nonces={A0: 0}), served_attempt('baseline#0', 0, JOB),
              {'kind': 'job-seen', 'job': JOB, 'account': A0, 'attempt': 'baseline#0-0', 'price': 100},
              {'kind': 'settlement', 'job': JOB, **settlement}]
    verdict = judge(complete(events, nonces={A0: 1}))
    if settlement['status'] == 'expired':  # the contender was not served with a settled job: progress decides
        assert not any('job ' in r and 'escrow' in r for r in verdict['reasons'])
    else:
        assert verdict['verdict'] == kind, verdict


def test_a_job_without_its_opening_record_is_not_certified():
    assert 'not known' in burst.settlement_check({'status': 'settled', 'paid': [1], 'refunded': 0}, None)
