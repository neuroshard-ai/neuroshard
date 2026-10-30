"""Step 0's driver logic with scripted models: reproduction plumbing, stopping, routing and the host contract."""

import gzip
import importlib.util
import json
import random
import subprocess
import sys
import types

import pytest

from neuroshard.evolution.general_eval import development_run as step0
from neuroshard.evolution.general_eval import items, served
from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256

POLICIES, NETWORK = served.a4_policies()


def accepted_generations(route, case_id):
    episodes = json.loads(gzip.decompress((ROOT / step0.EPISODES).read_bytes()))['episodes']
    row = next(r for r in episodes[step0.ACCEPTED_SETS[route]] if r['id'] == case_id)
    return [g for g in row['generations'] if g.get('executed', True)]


def replaying(route_of):
    """A system whose route-``r`` model replays the accepted generations of the conversation being run."""
    state = {}

    def route(text, opening):
        return route_of[opening]

    def factory(model, tokenizer, policy):
        queue = iter(state['queue'])
        return lambda messages, tools: dict(next(queue))

    system = step0.served_system(POLICIES, route, {'documents': 'L9', 'chat': 'C1'}, None, factory)
    return system, state


def test_choose_route_matches_the_network_function():
    from neuroshard.assistant import network
    from neuroshard.evolution import assistant_turn_router as turn_router

    names = served.route_names(NETWORK)
    general = read(ROOT / served.GENERAL_ROUTER)
    rng = random.Random(7)
    dimension = len(turn_router.verify(general)['weights'][0]) // 2 if 'weights' in general else None

    class Served:
        route_names = names

    a3 = {'rule': 'constant-parent', 'counts': {}}
    tried = 0
    for _ in range(40):
        size = dimension or 2560
        feature, opening = [rng.gauss(0, 1) for _ in range(size)], [rng.gauss(0, 1) for _ in range(size)]
        try:
            expected = network.choose_route(a3, Served, feature, opening, general)
        except (ValueError, KeyError, IndexError):
            continue
        assert step0_route(a3, names, feature, opening, general) == expected
        tried += 1
    assert tried >= 20
    assert served.choose_route(a3, names, [0.0], [0.0]) == network.choose_route(a3, Served, [0.0], [0.0])


def step0_route(a3, names, feature, opening, general):
    return served.choose_route(a3, names, feature, opening, general)


def test_replayed_accepted_generations_reproduce_all_ten_through_the_served_session(tmp_path):
    work = step0.reproduction_work(5)
    assert [r for r, _, _ in work] == ['documents'] * 5 + ['chat'] * 5
    route_of = {case['turns'][0]['user']: route for route, case, _ in work}
    system, state = replaying(route_of)
    rows = []
    for route, case, expected in work:
        state['queue'] = accepted_generations(route, case['id'])
        rows += step0.reproduce(system, [(route, case, expected)])
    assert [r['matched'] for r in rows] == [True] * 10, [r['first_difference'] for r in rows]


def test_a_single_changed_token_stops_the_run_before_any_development_item(tmp_path):
    work = step0.reproduction_work(1)
    route_of = {case['turns'][0]['user']: route for route, case, _ in work}
    system, state = replaying(route_of)
    queues = {case['id']: accepted_generations(route, case['id']) for route, case, _ in work}
    tampered = json.loads(json.dumps(queues[work[1][1]['id']]))
    tampered[-1]['token_ids'] = tampered[-1]['token_ids'][:-2] + [1] + tampered[-1]['token_ids'][-1:]
    queues[work[1][1]['id']] = tampered
    original = system.converse

    def converse(world, users):
        state['queue'] = queues[next(case['id'] for _, case, _ in work if case['turns'][0]['user'] == users[0])]
        return original(world, users)

    system.converse = converse
    ran = []
    system.episode = lambda item: ran.append(item) or {}
    reply = step0.execute(system, tmp_path, work=work, items=items.load_split()[:3], split_digest='d', cost={},
                          deadline=float('inf'))
    assert not reply['execution_completed'] and reply['reproduction']['matched'] == 1
    assert reply['reproduction']['first_mismatch']['first_difference'] == f'generation {len(tampered) - 1} token_ids'
    assert not ran and not (tmp_path / 'general-eval-development-report.json').exists()
    assert read(tmp_path / 'reproduction.json')['planned'] == 2


def test_development_runs_round_robin_writes_transcripts_and_stops_at_its_deadline(tmp_path):
    split = items.load_split()
    ordered = step0.interleave(split)
    assert [i['area'] for i in ordered[:6]] == list(items.AREAS) and len(ordered) == 120
    clock = iter(range(1000))

    def factory(model, tokenizer, policy):
        return lambda messages, tools: {'text': 'Answer: 1', 'terminated': True, 'token_ids': [1, 2]}

    system = step0.served_system(POLICIES, lambda t, o: 'chat', {'chat': 'C1', 'documents': 'L9'}, None, factory)
    reply = step0.execute(system, tmp_path, work=[], items=split, split_digest='d', cost={'usd_per_hour': 0.0},
                          deadline=8, clock=lambda: next(clock))
    assert not reply['execution_completed'] and 'time budget' in reply['error'] and reply['items_completed'] == 8
    report = read(tmp_path / 'general-eval-development-report.json')
    assert report['label'] == 'development, not a baseline claim' and report['stopped_early']
    assert set(report['areas']) == set(items.AREAS) and report['route_histogram'] == {'None': 1, 'chat': 7}
    assert len(list((tmp_path / 'transcripts').glob('*.json'))) == 8
    full = step0.execute(step0.served_system(POLICIES, lambda t, o: 'chat', {'chat': 'C1'}, None, factory),
                         tmp_path / 'full', work=[], items=split, split_digest='d', cost={}, deadline=float('inf'))
    assert full['execution_completed'] and full['items_completed'] == 120
    assert full['report']['areas']['tool_use']['unsupported'] == 14


def test_a3_routes_fail_like_serving_and_are_never_silently_answered():
    def factory(model, tokenizer, policy):
        raise AssertionError('no responder may be built for a route that is not loaded')

    system = step0.served_system(POLICIES, lambda t, o: 'drafting', {'chat': 'C1'}, None, factory)
    item = next(i for i in items.load_split() if i['area'] == 'planning')
    outcome = system.episode(item)
    assert outcome['failure'].startswith('routed to drafting without a workspace')
    documents = next(i for i in items.load_split() if i['scoring'].get('toolset') == 'documents/2')
    with pytest.raises((step0.RouteNotLoaded, ValueError)):
        system.episode(documents)


def test_routers_and_policies_are_checked_against_the_a4_pins(tmp_path):
    fake = tmp_path / 'router.json'
    fake.write_text(json.dumps({'gate': {'rule': 'constant-parent'}}))
    with pytest.raises(ValueError, match='router.json differs'):
        served.routers(NETWORK, fake)
    assert sha256(ROOT / served.GENERAL_ROUTER) == NETWORK['files']['general-router.json']['sha256']
    assert served.route_names(NETWORK) == {'drafting': 1, 'scheduling': 2, 'documents': 3, 'chat': 4}


def cloud_module():
    try:
        import boto3  # noqa: F401 - CI installs it; this development box may not
        import botocore.exceptions  # noqa: F401
    except ImportError:
        for name in ('boto3', 'botocore', 'botocore.exceptions'):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules['botocore.exceptions'].ClientError = type('ClientError', (Exception,), {})
    spec = importlib.util.spec_from_file_location('step0_cloud', ROOT / 'scripts/modular_reference_cloud.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_host_contract_is_one_canonical_cpu_host_under_the_approved_cap():
    from neuroshard.evolution import assistant_workflow_canonical as canonical

    execution, declaration = read(ROOT / step0.EXECUTION), read(ROOT / step0.DECLARATION)
    pinned = read(ROOT / canonical.EXECUTION)
    assert all(execution[k] == pinned[k] for k in ('packages', 'python', 'required_cpu_flags', 'environment'))
    assert execution['contracts'] == {step0.DECLARATION: sha256(ROOT / step0.DECLARATION)}
    assert declaration['development_split_sha256'] == items.split_digest(items.load_split())
    assert declaration['modules']['L9']['trainable_sha256'] == execution['units']['L9']['trainable_sha256']
    cloud = cloud_module()
    resources = cloud.resources(step0.PROFILE)
    assert resources['instance_type'] == 'r7i.4xlarge' and not resources['gpu'] and resources['attempts'] == 1
    assert resources['hours'] * resources['price']['usd_per_hour'] <= 10 - 4  # instance expiry bounds compute at $5.29
    assert (execution['prepare_seconds'] + execution['worker_seconds'] + resources['setup_seconds']
            + resources['copy_seconds'] + 600 <= resources['hours'] * 3600)
    assert set(resources['upload']['files']) == {f'{u}/{f}' for u in ('L9', 'C1')
                                                 for f in ('manifest.json', 'trainable.safetensors')} | {'router.json'}
    assert cloud.UPLOAD_PROFILES[step0.PROFILE] + '/router.json' == step0.A3_ROUTER
    assert cloud.GRANITE_PROFILES[step0.PROFILE][0] == 'general_eval.development_run'
    command = cloud.remote_command(step0.PROFILE)
    assert command[1].endswith(step0.SCRIPT) and command[-2:] == ['--profile', step0.PROFILE]
    assert step0.SCRIPT in execution['sources'] and not execution['training_authorized']


def test_every_file_the_worker_touches_is_in_the_checked_out_sources():
    probe = subprocess.check_output([sys.executable, str(ROOT / 'scripts/general_eval_step0_probe.py')], cwd=ROOT,
                                    text=True, env={'PYTHONPATH': str(ROOT / 'src'), 'PATH': '/usr/bin:/bin'})
    needed = {name for name in json.loads(probe) if name not in ('transcripts', 'scripts/general_eval_step0_probe.py')}
    sources = set(read(ROOT / step0.EXECUTION)['sources'])
    assert needed <= sources, sorted(needed - sources)
