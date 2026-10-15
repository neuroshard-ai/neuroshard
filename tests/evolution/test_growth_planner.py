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
                           encode=lambda item: item['case_id'], per_family=4, replay_per_family=1)
    assert built['families'] == [families[0]] and len(built['sources']['experience']) == 4
    assert built['others'] == families[1:] and len(built['sources']['replay']) == len(families) - 1
    assert not set(built['sources']['experience']) & set(built['sources']['replay'])
    assert planner.recipe({**report, 'families': {families[1]: report['families'][families[1]]}}, '33' * 32,
                          policy, {}, 'ab' * 32, encode=str) is None


def chat_policy():
    import json
    from pathlib import Path

    return json.loads((Path(__file__).resolve().parents[2]
                       / 'config/experiments/assistant-general-policy-chat3.json').read_text())


def solver(case):
    """A stand-in for the model that answers lowercase and arithmetic cases correctly, the latter with a tool."""
    import json
    import re

    constraints = case['turns'][0]['expected']['constraints']
    state = {'turn': 0}

    def respond(messages, tools):
        state['turn'] += 1
        kinds = {c['kind'] for c in constraints}
        if 'number' in kinds:
            value = next(c['value'] for c in constraints if c['kind'] == 'number')
            if state['turn'] == 1:
                call = {'name': 'calculate', 'arguments': {'operation': 'add', 'left': int(value) - 1, 'right': 1}}
                text = f'<tool_call>{json.dumps(call)}</tool_call>'
            else:
                result = json.loads(messages[-1]['content'])
                text = f"The total is {result.get('value', result.get('result'))}"
        elif kinds == {'lowercase'}:
            text = 'four calm lines\nabout the topic\nall lowercase\nnothing more'
        else:
            text = 'I cannot.'
        return {'text': text, 'terminated': True}
    del re
    return respond


def encode(item):
    import hashlib
    import json

    return hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()


CHAT_REPORT = {'round': 2, 'cases_sha256': 'cc' * 32, 'route': 'chat', 'families': {
    'arithmetic': {'passed': 2, 'cases': 12, 'rate': 2 / 12}, 'lowercase': {'passed': 5, 'cases': 12, 'rate': 5 / 12},
    'bullets': {'passed': 12, 'cases': 12, 'rate': 1.0}}}


def test_chat_recipe_keeps_only_checked_replies_and_owners_reverify_every_one():
    import copy

    policy = chat_policy()
    built = planner.chat_recipe(CHAT_REPORT, '44' * 32, policy, {'seed': 0}, 'ab' * 32, encode, [solver],
                                per_family=6, replay_per_family=2, minimum_items=4)
    assert built['families'] == ['arithmetic', 'lowercase'] and built['others'] == ['bullets']
    assert built['evidence']['replay'] == []       # the stand-in cannot write bullets: no unverified replay is kept
    by_family = {}
    for item in built['evidence']['experience']:
        by_family[item['family']] = by_family.get(item['family'], 0) + 1
    assert by_family.get('arithmetic', 0) >= 1 and sum(by_family.values()) == len(built['sources']['experience'])
    assert any(m['role'] == 'tool' for item in built['evidence']['experience'] for m in item['messages'])
    assert planner.verify_chat(built, policy, encode)

    def tampered(change):
        copied = copy.deepcopy(built)
        change(copied)
        return planner.verify_chat(copied, policy, encode)

    items = built['evidence']['experience']
    tool_item = next(i for i, e in enumerate(items) if any(m['role'] == 'tool' for m in e['messages']))
    assert not tampered(lambda r: r['evidence']['experience'][0]['messages'][-1].update(content='WRONG ANSWER 0'))
    assert not tampered(lambda r: next(m for m in r['evidence']['experience'][tool_item]['messages']
                                       if m['role'] == 'tool').update(content='{"value": 1}'))  # a forged tool result
    assert not tampered(lambda r: r['sources']['experience'].__setitem__(0, 'ff' * 32))  # trains on other bytes
    assert not tampered(lambda r: r['sources']['replay'].append('ff' * 32))              # unverified replay
    assert not tampered(lambda r: r['evidence']['experience'][0]['messages'][1].update(content='a different request'))
    assert not tampered(lambda r: r.update(randomness='55' * 32))            # cases not of this beacon
    assert not tampered(lambda r: r['evidence']['experience'].append(copy.deepcopy(items[0])))


def test_chat_recipe_waits_when_too_few_replies_verify():
    def silent(case):
        return lambda messages, tools: {'text': 'no', 'terminated': True}

    assert planner.chat_recipe(CHAT_REPORT, '44' * 32, chat_policy(), {}, 'ab' * 32, encode, [silent],
                               per_family=3) is None


def test_documents_recipe_reverifies_from_the_beacon():
    import json
    from pathlib import Path

    policy = json.loads((Path(__file__).resolve().parents[2]
                         / 'config/experiments/assistant-general-policy-documents5.json').read_text())
    families = sorted(planner.sealed.templates('documents'))
    report = {**REPORT, 'families': {families[0]: {'passed': 1, 'cases': 10, 'rate': 0.1}}}
    report['families'][families[1]] = {'passed': 10, 'cases': 10, 'rate': 1.0}
    built = planner.recipe(report, '33' * 32, policy, {}, 'ab' * 32, encode, per_family=3, replay_per_family=2)
    assert planner.verify_documents(built, policy, encode)
    assert not planner.verify_documents({**built, 'randomness': '34' * 32}, policy, encode)
    assert not planner.verify_documents({**built, 'sources': {**built['sources'], 'replay': []}}, policy, encode)


def test_a_growth_round_can_continue_from_the_unit_it_replaces(tmp_path):
    import hashlib

    from safetensors.torch import save as save_tensors

    from neuroshard.evolution import growth_replay
    from test_network_growth import RECIPE, tiny

    model = tiny()
    fresh, optimizer = growth_replay.start(model, RECIPE)
    trained = {name: value.detach().clone() + 0.25 for name, value in fresh.items()}
    payload = save_tensors(trained)
    digest = hashlib.sha256(payload).hexdigest()
    (tmp_path / 'L9').mkdir()
    (tmp_path / 'L9' / 'trainable.safetensors').write_bytes(payload)
    growth_replay.UNIT_STORE = tmp_path
    try:
        continued = {**RECIPE, 'continue_from': digest}
        trainable, _ = growth_replay.start(tiny(), continued)
        assert all(torch_equal(trainable[n], trained[n]) for n in trained)
        assert growth_replay.recipe_root(continued, 'ab' * 32) != growth_replay.recipe_root(RECIPE, 'ab' * 32)
        import pytest
        with pytest.raises(FileNotFoundError):
            growth_replay.start(tiny(), {**RECIPE, 'continue_from': 'ee' * 32})
    finally:
        growth_replay.UNIT_STORE = None


def torch_equal(a, b):
    import torch

    return torch.equal(a, b.to(a.dtype))


def test_the_agent_command_line_parses_every_role():
    import argparse

    from neuroshard.client import assistant_cli

    parser = argparse.ArgumentParser()
    assistant_cli.register(parser.add_subparsers(dest='command'), __import__('pathlib').Path('.'))
    args = parser.parse_args(['assistant', 'grow', 'agent', '--plan', '--evaluate', '--vote-shards', '1',
                              '--no-contribute', '--segment-steps', '8', '--per-family', '24', '--minimum', '0.85',
                              '--reward-per-step', '500', '--growth-steps', '32', '--probe-every', '60'])
    assert args.plan and args.evaluate and args.vote_shards == [1] and args.no_contribute and args.segment_steps == 8
