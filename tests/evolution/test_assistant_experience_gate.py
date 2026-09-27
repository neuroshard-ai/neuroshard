import pytest

from neuroshard.evolution import assistant_experience_gate as gate
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution.modular_reference_execution import ROOT, read

PLAN = read(ROOT / 'config/experiments/assistant-experience-learning.json')


def rows(cases, successes, seconds=60):
    return [{'id': c['id'], 'score': {'passed': c['id'] in successes}, 'seconds': seconds} for c in cases]


def test_development_needs_parent_gain_no_protected_loss_and_update_parity():
    cases = data.cases('development')
    ids = [c['id'] for c in cases]
    parent = set(ids[:8])
    addition = set(ids[:20])
    update = set(ids[:21])
    report = gate.development(PLAN, cases, rows(cases, parent), rows(cases, update), rows(cases, addition), parent)
    assert report['passed'] and report['versus_parent']['net'] == 12 and report['versus_update']['net'] == -1
    worse = gate.development(PLAN, cases, rows(cases, parent), rows(cases, set(ids[:22])),
                             rows(cases, addition), parent)
    assert not worse['passed'] and not worse['checks']['net_vs_update']
    lost = gate.development(PLAN, cases, rows(cases, parent), rows(cases, update),
                            rows(cases, addition - {ids[0]} | {ids[20]}), parent)
    assert not lost['passed'] and lost['lost_protected'] == [ids[0]] and not lost['checks']['lost_parent_successes']
    slow = gate.development(PLAN, cases, rows(cases, parent), rows(cases, update, 50),
                            rows(cases, addition, 150), parent)
    assert slow['checks']['p95'] and not slow['checks']['p95_ratio']
    with pytest.raises(ValueError, match='every gated episode'):
        gate.development(PLAN, cases, rows(cases, parent)[:-1], rows(cases, update), rows(cases, addition), parent)


def test_confirmation_bootstraps_over_families_and_requires_each_family():
    cases = data.cases('confirmation')
    ids = [c['id'] for c in cases]
    by_family = {f: [c['id'] for c in cases if c['family'] == f] for f in data.FAMILIES}
    parent = {k for f in data.FAMILIES for k in by_family[f][:4]}
    addition = {k for f in data.FAMILIES for k in by_family[f][:11]}
    update = {k for f in data.FAMILIES for k in by_family[f][:11]}
    report = gate.confirmation(PLAN, cases, rows(cases, parent), rows(cases, update), rows(cases, addition), parent)
    assert report['correct']['addition'] == 88 and report['lower_95_gain_vs_parent'] > 0
    assert report['lower_95_gain_vs_update'] == 0 and report['passed']
    thin = addition - set(by_family['scope'][4:])
    weak = gate.confirmation(PLAN, cases, rows(cases, parent), rows(cases, update), rows(cases, thin), parent)
    assert not weak['checks']['per_family'] and not weak['passed']
    assert len(ids) == 96
