"""The planner turns a sealed probe into recipes for exactly the weak families, from fresh verified cases."""

from neuroshard.evolution import growth_planner as planner


REPORT = {'round': 1, 'cases_sha256': 'aa' * 32, 'route': 'documents', 'families': {
    'claim-deadline': {'passed': 5, 'cases': 10, 'rate': 0.5},
    'invoice-sum': {'passed': 8, 'cases': 10, 'rate': 0.8},
    'vendor-city': {'passed': 10, 'cases': 10, 'rate': 1.0}}}


def test_weakest_is_deterministic_and_skips_strong_families():
    assert planner.weakest(REPORT) == ['claim-deadline', 'invoice-sum']
    assert planner.weakest(REPORT, target=0.6) == ['claim-deadline']
    assert planner.weakest({**REPORT, 'families': {'vendor-city': REPORT['families']['vendor-city']}}) == []


def test_probe_counts_per_family():
    cases = {'round': 3, 'sha256': 'bb' * 32, 'sets': {'documents': [
        {'family': 'a', 'id': 1}, {'family': 'a', 'id': 2}, {'family': 'b', 'id': 3}]}}
    report = planner.probe(lambda case: case['id'] != 2, cases, 'documents')
    assert report['families'] == {'a': {'passed': 1, 'cases': 2, 'rate': 0.5}, 'b': {'passed': 1, 'cases': 1, 'rate': 1.0}}


def test_demonstrations_are_fresh_and_depend_on_the_randomness():
    families = [name for name in planner.weakest(REPORT)
                if name in planner.sealed.templates('documents')]
    if not families:
        families = sorted(planner.sealed.templates('documents'))[:1]
    one = planner.demonstrations(families, '11' * 32, per_family=3)
    again = planner.demonstrations(families, '11' * 32, per_family=3)
    other = planner.demonstrations(families, '22' * 32, per_family=3)
    assert one == again and one != other
    assert {case['family'] for case in one} == set(families) and all(c['split'] == 'generated-train' for c in one)


def test_recipe_is_built_from_verified_demonstrations_of_only_the_weak_families():
    import json
    from pathlib import Path

    policy = json.loads((Path(__file__).resolve().parents[2]
                         / 'config/experiments/assistant-general-policy-documents5.json').read_text())
    families = sorted(planner.sealed.templates('documents'))
    report = {**REPORT, 'families': {name: {'passed': 9, 'cases': 10, 'rate': 0.9 if i else 0.4}
                                     for i, name in enumerate(families)}}
    built = planner.recipe(report, '33' * 32, policy, {'seed': 0}, 'ab' * 32,
                           encode=lambda item: item['case_id'], replay=['r'], per_family=4)
    assert built['families'] == [families[0]] and len(built['sources']['experience']) == 4
    assert planner.recipe({**report, 'families': {families[1]: report['families'][families[1]]}}, '33' * 32,
                          policy, {}, 'ab' * 32, encode=str, replay=[]) is None
