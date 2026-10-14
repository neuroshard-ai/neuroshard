import json

import pytest

from neuroshard.assistant import interfaces, network
from neuroshard.assistant.public import Session, bind
from neuroshard.evolution import assistant_turn_router as turn_router
from neuroshard.evolution.modular_reference_execution import ROOT, read

POLICIES = {name: read(ROOT / f'config/experiments/{path}.json') for name, path in (
    ('drafting', 'assistant-workflow-policy'), ('scheduling', 'assistant-workflow-policy-calendar-slots'),
    ('documents', 'assistant-general-policy-documents3'), ('chat', 'assistant-general-policy-chat3'))}


def world():
    from neuroshard.evolution import assistant_workflow_data as data

    return data.cases('development')[0]['world']


def call(name, **arguments):
    return '<tool_call>\n' + json.dumps({'name': name, 'arguments': arguments}) + '\n</tool_call>'


def scripted(replies):
    replies = iter(replies)
    return lambda messages, tools: {'text': next(replies), 'terminated': True}


def test_every_route_interface_resolves_and_a3_bindings_are_unchanged():
    groups = {name: interfaces.interface(policy).group for name, policy in POLICIES.items()}
    assert groups == {'drafting': 'calendar', 'scheduling': 'calendar', 'documents': 'documents', 'chat': 'chat'}
    a3 = {name: POLICIES[name] for name in ('drafting', 'scheduling')}
    from neuroshard.evolution import assistant_workflow as workflow
    for policy in a3.values():
        assert interfaces.interface(policy).TOOLS == workflow.interface(policy).TOOLS
    assert bind(POLICIES['drafting'], route_policies=a3)['version_sha256'] == bind(
        POLICIES['drafting'], route_policies=a3)['version_sha256']


def test_a_session_keeps_one_workspace_per_route_group_and_switches_by_route():
    files = world()
    first = files['documents'][0]
    session = Session(POLICIES['drafting'], files, name='a4-general', modules=('U1', 'L2', 'L3', 'L7', 'C1'),
                      route_policies=POLICIES)
    assert session.worlds == {}  # each route group's workspace is built when the route is first used
    result = session.turn('What does my first document say?', scripted([
        call('read_document', document_id=first['id']), f"It is {first['title']} ({first['id']})."]),
        route='documents')
    assert result['completed'] and result['snapshot']['read_ids'] == [first['id']]
    assert set(session.worlds) == {'documents'}
    result = session.turn('What is 6 times 7?', scripted([call('calculate', operation='multiply', left=6, right=7),
                                                           'It is 42.']), route='chat')
    assert result['final_text'] == 'It is 42.' and result['snapshot']['events'][0]['result'] == {'value': 42}
    assert session.calls[-1]['result'] == {'value': 42}
    projected = interfaces.documents_world(files)['documents'][0]
    assert set(projected) == {'id', 'title', 'content'} and projected['id'] == first['id']


def test_k_route_choice_names_a_served_route_or_refuses():
    class Served:
        route_names = {'drafting': 1, 'scheduling': 2, 'documents': 3, 'chat': 4}

    rows = {f'{route}-{i}': (route, 1.0) for route in Served.route_names for i in range(3)}
    basis = {'drafting': [1, 0, 0, 0], 'scheduling': [0, 1, 0, 0], 'documents': [0, 0, 1, 0], 'chat': [0, 0, 0, 1]}
    features = {key: turn_router.context_feature(basis[route], basis[route]) for key, (route, _) in rows.items()}
    router = turn_router.fit(features, rows, steps=200, fallback='chat')
    assert network.choose_route(router, Served, basis['documents'], basis['documents']) == ('documents', 3)
    assert network.choose_route(router, Served, basis['chat'], basis['drafting'])[0] in Served.route_names

    class Fewer:
        route_names = {'drafting': 1, 'scheduling': 2}

    with pytest.raises(ValueError, match='does not serve'):
        network.choose_route(router, Fewer, basis['documents'], basis['documents'])


def test_route_tables_reject_bad_rows_and_keep_a3():
    table = network.route_table({'drafting': {'route': 1, 'base': 'U1', 'addition': 'L3'},
                                 'documents': {'route': 3, 'base': 'parent', 'addition': 'L7'},
                                 'selector': 'router.json'})
    assert table == {1: ('L3', 'U1'), 3: ('L7', 'parent')}
    for bad in ({'a': {'route': 1, 'addition': 'L3'}, 'b': {'route': 1, 'addition': 'L2'}},
                {'a': {'route': 9, 'addition': 'L3'}}, {'a': {'route': 1, 'addition': '../x'}},
                {'a': {'route': 1, 'addition': 'L3', 'base': 'other'}}):
        with pytest.raises(ValueError):
            network.route_table(bad)
    served, _ = network.version(network.descriptor())
    assert served.routes == {1: ('L3', 'U1'), 2: ('L2', 'U1')} and served.route_names == {'drafting': 1,
                                                                                          'scheduling': 2}


def test_chat_results_render_as_the_accepted_chat_runner_did_and_documents_alone_need_no_drafting_schema():
    from neuroshard.evolution import assistant_chat_data as chat
    from neuroshard.evolution import assistant_documents_data as docs
    from neuroshard.evolution import assistant_general as general

    rule = 'ends_with:\u201cKeep at it\u201d'
    case = chat.cases('confirmation7')[0]
    reply = call('check_text', text='Keep at it.', rules=[rule])
    recorded = general.chat_episode(case, scripted([reply, 'Keep at it.']), POLICIES['chat'])
    session = Session(POLICIES['drafting'], {'documents': []}, route_policies=POLICIES)
    served = session.turn(case['turns'][0]['user'], scripted([reply, 'Keep at it.']), route='chat')
    tool_messages = [m['content'] for m in session.messages if m['role'] == 'tool']
    assert tool_messages and all(message.isascii() for message in tool_messages)
    assert served['final_text'] == recorded['final_text'] and session.calls == recorded['calls']
    documents_case = docs.cases('confirmation7')[0]
    session = Session(POLICIES['drafting'], documents_case['world'], route_policies=POLICIES)
    first = documents_case['world']['documents'][0]['id']
    out = session.turn('x', scripted([call('read_document', document_id=first), f'ok ({first})']), route='documents')
    assert out['completed'] and set(session.worlds) == {'documents'}


def test_the_general_router_sends_a3_turns_to_a3s_unchanged_router():
    class Served:
        route_names = {'drafting': 1, 'scheduling': 2, 'documents': 3, 'chat': 4}

    basis = {'a3': [1, 0, 0], 'documents': [0, 1, 0], 'chat': [0, 0, 1]}
    rows = {f'{route}-{i}': (route, 1.0) for route in basis for i in range(3)}
    features = {key: turn_router.context_feature(basis[route], basis[route]) for key, (route, _) in rows.items()}
    general = turn_router.fit(features, rows, steps=200)
    a3_router = {'rule': 'constant-arm', 'counts': {}}
    assert network.choose_route(a3_router, Served, basis['documents'], basis['documents'], general) == ('documents', 3)
    assert network.choose_route(a3_router, Served, basis['a3'], basis['a3'], general) == ('scheduling', 2)
    assert network.choose_route({'rule': 'constant-parent', 'counts': {}}, Served, basis['a3'], basis['a3'],
                                general) == ('drafting', 1)


def test_the_general_release_contract_pins_its_sources_workload_and_model():
    from neuroshard.assistant import release_check_general as general
    from neuroshard.evolution.modular_reference_execution import sha256

    plan = read(ROOT / general.CONTRACT)
    for name, digest in {**plan['sources'], **plan['contracts']}.items():
        assert sha256(ROOT / name) == digest, name
    descriptor = network.descriptor(ROOT / 'src/neuroshard/client/networks/assistant-release-general.json')
    assert network.model_root(descriptor) == plan['model_root']
    served, _ = network.version(descriptor)
    assert served.routes == {1: ('L3', 'U1'), 2: ('L2', 'U1'), 3: ('L9', 'parent'), 4: ('C1', 'parent')}
    work = general.checked_workload(plan)
    assert len(work) == 324 and {name for name, _, _ in work[:10]} == {'documents', 'chat', 'a3-cross', 'a3-drafting',
                                                                        'a3-scheduling'}
    assert not plan['training_authorized'] and not plan['new_final_authorized'] and plan['budget_usd'] <= 40
    router = read(ROOT / 'config/experiments/assistant-general-router-a4.json')
    report = read(ROOT / 'config/experiments/assistant-general-router-a4-report.json')
    assert router['routes'] == ['a3', 'chat', 'documents'] and report['router_sha256'] == router['sha256']
    assert report['development']['misrouted_count'] == 0 and report['workload']['misrouted_count'] == 0
