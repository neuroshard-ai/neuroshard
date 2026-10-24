"""The gate's pass/fail rules (scripts/gate_rules.py) and the harness's pure parts, including the reviewer's
negative controls: an unexpected honest-contributor penalty, an undecided scenario, a recovery that is behind its
peers or disagrees with them, and unreadable blocks must never pass."""
import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import gate_rules as rules  # noqa: E402

spec = importlib.util.spec_from_file_location('gate', SCRIPTS / 'growth_quorum_gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

RC2, P1 = {'flavour': 'rc2', 'model': True}, {'flavour': 'pilot1', 'model': True}
BARE_RC2 = {'flavour': 'rc2', 'model': False}


def test_stages_upgrade_model_holders_first_and_never_add_a_model_to_an_old_validator():
    order = ['today', 'step1', 'step2', 'step3', 'all-rc2', 'all-equipped']
    for before, after in zip(order, order[1:]):
        b, a = gate.STAGES[before], gate.STAGES[after]
        assert set(b['rc2']) <= set(a['rc2']) and set(b['model']) <= set(a['model'])
        assert set(a['model']) - set(b['model']) <= set(a['rc2'])
    legacy_holders = [len(set(gate.STAGES[s]['model']) - set(gate.STAGES[s]['rc2'])) for s in order]
    assert legacy_holders == sorted(legacy_holders, reverse=True) and max(legacy_holders) < rules.QUORUM
    assert len(set(gate.STAGES['forbidden-4-legacy-holders']['model'])) == rules.QUORUM


def test_verdicts_are_read_from_the_cached_worker_log_lines():
    assert gate.log_verdict('x\n2026 INFO neuroshard.settlement verification ab12 verdict True in 53.1 s\n') == 'True'
    assert gate.log_verdict('2026 INFO neuroshard.settlement verification ab12 verdict False in 9 s') == 'False'
    no_model = ('2026 ERROR neuroshard.settlement verification failed for ab12\nTraceback (most recent call last):\n'
                "FileNotFoundError: [Errno 2] No such file or directory: '/x/baseline/chat_template.jinja'\n")
    assert gate.log_verdict(no_model) == 'no-verdict (FileNotFoundError)'
    assert gate.log_verdict('NeuroShard settlement ABCI listening on 29652') == 'not-judged'


def test_expected_verdicts_and_penalties_follow_versions_models_and_availability():
    equipped = [RC2] * 5
    assert rules.expected_verdicts(equipped, 'honest') == {i: 'False' for i in range(5)}
    assert rules.expect_penalty(rules.expected_verdicts(equipped, 'legacy8'))
    assert rules.expect_penalty(rules.expected_verdicts(equipped, 'forged', offline={4}))
    # Two unavailable (one offline, one without the bundle): three finders cannot reach quorum.
    assert not rules.expect_penalty(rules.expected_verdicts(equipped, 'forged', offline={4}, missing_store={3}))
    today = [P1, P1, P1, {'flavour': 'pilot1', 'model': False}, {'flavour': 'pilot1', 'model': False}]
    assert rules.expected_verdicts(today, 'honest') == {0: 'True', 1: 'True', 2: 'True', 3: 'no-verdict', 4: 'no-verdict'}
    assert not rules.expect_penalty(rules.expected_verdicts(today, 'honest'))
    forbidden = [P1, P1, P1, P1, BARE_RC2]
    assert rules.expect_penalty(rules.expected_verdicts(forbidden, 'honest'))  # the hazard is the expected outcome


def observation(case='honest', outcome='lapsed-and-settled', validators=None, offline=(), **over):
    validators = validators or [RC2] * 5
    expected = rules.expected_verdicts(validators, case, offline=set(offline))
    penalty = outcome == 'penalty-committed'
    nonces = {'sponsor': 1, 'contributor': 1, 'challenger': 2 + (1 if penalty else 0), 'bystander': 7}
    balances, burned = rules.expected_balances(penalty, nonces)
    obs = {'expected_verdicts': expected, 'verdicts': dict(expected), 'outcome': outcome,
           'fault_height': 30 if penalty else None, 'deadline': 60, 'balances': balances, 'burned': burned,
           'nonces': nonces, 'blocks_complete': True}
    obs.update(over)
    return obs


def test_correct_outcomes_pass():
    assert rules.judge_scenario(observation())[0] == 'pass'
    assert rules.judge_scenario(observation('forged', 'penalty-committed'))[0] == 'pass'
    assert rules.judge_scenario(observation('legacy8', 'penalty-committed', offline=(4,)))[0] == 'pass'


def test_an_unexpected_honest_contributor_penalty_fails():
    obs = observation('honest', 'penalty-committed')
    obs['expected_verdicts'] = rules.expected_verdicts([RC2] * 5, 'honest')  # honest: no penalty is expected
    verdict, reasons = rules.judge_scenario(obs)
    assert verdict == 'fail' and any('penalty was not expected' in r for r in reasons)


def test_a_missed_forgery_penalty_fails():
    obs = observation('forged', 'lapsed-and-settled')
    verdict, reasons = rules.judge_scenario(obs)
    assert verdict == 'fail' and any('penalty was expected' in r for r in reasons)


def test_an_undecided_scenario_is_never_a_pass():
    obs = observation('honest', 'undecided')
    assert rules.judge_scenario(obs)[0] == 'inconclusive'


def test_a_penalty_after_the_deadline_fails():
    obs = observation('forged', 'penalty-committed', fault_height=61)
    assert rules.judge_scenario(obs)[0] == 'fail'


def test_wrong_verdicts_or_balances_fail_and_missing_verdicts_are_inconclusive():
    obs = observation('forged', 'penalty-committed')
    obs['verdicts'][2] = 'False'
    assert rules.judge_scenario(obs)[0] == 'fail'
    obs = observation('forged', 'penalty-committed')
    obs['balances']['contributor'] += 1
    assert rules.judge_scenario(obs)[0] == 'fail'
    obs = observation()
    obs['verdicts'][1] = 'not-judged'
    assert rules.judge_scenario(obs)[0] == 'inconclusive'


def test_unreadable_blocks_make_the_evidence_incomplete():
    assert rules.judge_scenario(observation(blocks_complete=False))[0] == 'inconclusive'


def sample(mine_height=150, peer_heights=(150, 151, 150, 150), common_mine=('b', 'a'), common_peers=None,
           catching_up=False, signed=range(100, 110), address='V4'):
    common_height = min((mine_height,) + tuple(peer_heights)) - 1
    return {'mine': {'height': mine_height, 'catching_up': catching_up},
            'peers': [{'height': h} for h in peer_heights],
            'common': {'height': common_height, 'mine': common_mine,
                       'peers': list(common_peers or [common_mine] * len(peer_heights))},
            'signatures': {h: ([address, 'X'] if h in signed else ['X']) for h in range(100, 110)}}


def test_recovery_requires_current_peers_matching_hashes_and_fresh_signatures():
    assert rules.recovery_decision(sample(), 'V4')[0] == 'recovered'
    # The reviewer's negative control: behind current peers (150 vs 500) is not recovered, whatever it reads of itself.
    assert rules.recovery_decision(sample(peer_heights=(500, 500, 500, 500)), 'V4')[0] == 'pending'
    assert rules.recovery_decision(sample(common_peers=[('b', 'a'), ('b', 'other'), ('b', 'a'), ('b', 'a')]),
                                   'V4')[0] == 'disagreement'
    assert rules.recovery_decision(sample(catching_up=True), 'V4')[0] == 'pending'
    assert rules.recovery_decision(sample(signed=range(100, 103)), 'V4')[0] == 'pending'


def test_a_recovery_that_times_out_or_disagrees_does_not_pass_the_scenario():
    obs = observation('forged', 'penalty-committed', offline=(4,), recovery={'result': 'timeout', 'detail': 'x'})
    assert rules.judge_scenario(obs)[0] == 'inconclusive'
    obs = observation('forged', 'penalty-committed', offline=(4,), recovery={'result': 'disagreement', 'detail': 'x'})
    assert rules.judge_scenario(obs)[0] == 'fail'
    obs = observation('forged', 'penalty-committed', offline=(4,), recovery={'result': 'recovered', 'detail': 'x'})
    assert rules.judge_scenario(obs)[0] == 'pass'


def test_block_gaps_report_the_longest_pauses():
    gaps = rules.block_gaps({1: 0.0, 2: 1.0, 3: 53.5, 4: 54.5})
    assert gaps['max_seconds'] == 52.5 and gaps['largest'][0] == {'seconds': 52.5, 'from': 2, 'to': 3}


def test_a_late_store_may_or_may_not_judge_but_the_outcome_must_follow_the_verdicts():
    expected = rules.expected_verdicts([RC2] * 5, 'forged', late_store={3})
    assert expected[3] == 'True|no-verdict' and rules.expect_penalty(expected) is True  # four others suffice
    two_late = rules.expected_verdicts([RC2] * 5, 'forged', offline={4}, late_store={3})
    assert rules.expect_penalty(two_late) is None  # three certain plus one late: timing decides
    obs = observation('forged', 'penalty-committed', offline=(4,))
    obs['expected_verdicts'] = two_late
    obs['verdicts'][3] = 'no-verdict (ProofUnavailable)'  # then only three found fraud: a penalty cannot be right
    assert rules.judge_scenario(obs)[0] == 'fail'
    obs['verdicts'][3] = 'True'
    assert rules.judge_scenario(obs)[0] == 'pass'
