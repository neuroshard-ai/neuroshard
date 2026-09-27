import copy
import importlib.util
import json

import pytest

from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution import assistant_workflow_baseline as study
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution import assistant_workspace as workspace
from neuroshard.evolution.granite_partition_plan import estimate


def policy():
    return study.read(study.ROOT / 'config/experiments/assistant-workflow-policy.json')


def reply(text, terminated=True, executed=True):
    return {'text': text, 'terminated': terminated, 'executed': executed,
            'input_token_ids': [1], 'token_ids': list(text.encode()) if executed else [],
            'prompt_sha256': 'test-only', 'seconds': .1}


def envelope(name, args):
    return '<tool_call>' + json.dumps({'name': name, 'arguments': args}) + '</tool_call>'


def reference_texts(case):
    # Goal-directed fixture ONLY validates the environment, never a model result
    # or a source of training trajectories from an evaluation partition.
    outputs = []
    for turn in case['turns']:
        goal = turn['expected']
        outputs.append(envelope('list_documents', {'project': goal['project']}))
        outputs.append('\n'.join(envelope('read_document', {'document_id': key}) for key in goal['source_ids']))
        outputs.append(envelope('save_draft', goal))
        outputs.append('Draft saved locally.')
    return outputs


def execute_fixture(case):
    texts = iter(reference_texts(case))
    return workflow.execute(case, lambda messages, tools: reply(next(texts)), policy())


def test_complete_conversation_updates_state_and_replays_from_actual_responses():
    case = data.make_case('development', 'reschedule', 0)
    result = execute_fixture(case)
    assert result['score']['passed'] and result['score']['round_successes'] == [True, True]
    first = result['rounds'][0]['snapshot']['drafts']
    last = result['rounds'][1]['snapshot']['drafts']
    project = case['turns'][0]['expected']['project']
    assert first[project]['due_date'] != last[project]['due_date']
    assert first[project]['recipient'] != last[project]['recipient']
    assert first[project]['total'] == last[project]['total']
    assert workflow.replay_matches(result, execute_fixture(case))
    forged = copy.deepcopy(result)
    forged['calls'][-1]['call']['arguments']['total'] += 1
    with pytest.raises(ValueError, match='tool actions'):
        workflow.score(case, forged, policy())
    forged = copy.deepcopy(result)
    forged['rounds'][0]['snapshot']['drafts'][project]['total'] += 1
    with pytest.raises(ValueError, match='workspace rounds'):
        workflow.score(case, forged, policy())


def test_wrong_valid_draft_is_executable_but_does_not_pass_outcome_gate():
    case = data.make_case('development', 'copy', 0)
    world = workspace.Workspace(case['world'])
    goal = case['turns'][0]['expected']
    for key in goal['source_ids']:
        world.execute({'name': 'read_document', 'arguments': {'document_id': key}})
    incorrect = {**goal, 'total': goal['total'] + 1}
    assert world.execute({'name': 'save_draft', 'arguments': incorrect})['saved']
    assert not workspace.score_round(world.snapshot(), goal, True, 'Done')
    world.execute({'name': 'save_draft', 'arguments': goal})
    assert workspace.score_round(world.snapshot(), goal, True, 'Done')
    assert not workspace.score_round(world.snapshot(), goal, False, 'Done')
    with pytest.raises(ValueError, match='public documents'):
        workspace.Workspace({**case['world'], 'expected': goal})


@pytest.mark.parametrize('text', [
    '<tool_call>{"name":"read_document","name":"save_draft","arguments":{}}</tool_call>',
    envelope('calculate', {'operation': 'add', 'left': True, 'right': 2}),
    envelope('shell', {'cmd': 'echo bad'}),
    'I will do this ' + envelope('list_documents', {'project': 'x'}),
    '<tool_call>{"name":"read_document","arguments": NaN}</tool_call>',
    '<tool_call>__import__("os").system("true")</tool_call>',
])
def test_tool_parser_refuses_ambiguous_unsafe_or_invalid_calls(text):
    with pytest.raises((ValueError, TypeError)):
        workspace.parse_calls(text)


def test_tools_check_sources_arithmetic_dates_and_have_no_external_effects():
    case = data.make_case('development', 'copy', 0)
    world = workspace.Workspace(case['world'])
    assert 'error' in world.execute({'name': 'save_draft', 'arguments': case['turns'][0]['expected']})
    assert not world.drafts
    assert world.execute({'name': 'shift_date', 'arguments': {'start_date': '2028-02-28', 'days': 2}}) == {'date': '2028-03-01'}
    assert world.execute({'name': 'calculate', 'arguments': {'operation': 'multiply', 'left': 7, 'right': 8}}) == {'value': 56}
    assert 'error' in world.execute({'name': 'shift_date', 'arguments': {'start_date': '2027-02-29', 'days': 1}})
    assert 'error' in world.execute({'name': 'send_email', 'arguments': {'to': 'x'}})


def test_responder_never_receives_goals_or_future_user_turns():
    case = data.make_case('development', 'recipient', 0)
    original = execute_fixture(case)
    changed = copy.deepcopy(case)
    changed['turns'][1]['expected']['total'] += 10
    texts = iter(reference_texts(case))
    received = []
    def respond(messages, tools):
        received.append(messages)
        return reply(next(texts))
    altered = workflow.execute(changed, respond, policy())
    assert original['messages'] == altered['messages']
    assert not altered['score']['passed']
    assert received[0] == original['messages'][:2]
    assert case['turns'][1]['user'] not in json.dumps(received[0])


def test_first_turn_failure_cannot_be_repaired_by_a_correct_second_turn():
    case = data.make_case('development', 'recipient', 0)
    texts = reference_texts(case)
    texts[2] = envelope('save_draft', {**case['turns'][0]['expected'], 'total': 0})
    iterator = iter(texts)
    result = workflow.execute(case, lambda m, t: reply(next(iterator)), policy())
    assert result['score']['round_successes'] == [False, True]
    assert not result['score']['passed']


def test_nonterminated_or_overbudget_generation_fails_and_counts_only_actual_work():
    case = data.make_case('development', 'copy', 0)
    result = workflow.execute(case, lambda m, t: reply('', False, False), policy())
    assert not result['score']['passed']
    assert result['model_calls'] == result['input_tokens'] == result['output_tokens'] == 0
    assert result['model_attempts'] == 1
    broken = workflow.execute(case, lambda m, t: reply('<tool_call>broken</tool_call>'), policy())
    assert broken['model_calls'] == 6 and not broken['score']['passed']
    assert broken['rounds'][0]['failure'] == 'model-turn budget exhausted'


def test_frozen_splits_disjoint_manifest_consistent_and_baseline_development_only():
    manifest = study.read(study.ROOT / 'config/experiments/assistant-workflow-data.json')
    seen = set()
    for split, binding in manifest['splits'].items():
        rows = data.cases(split)
        assert len(rows) == binding['count'] and study.identity(rows) == binding['sha256']
        projects = {d['project'] for c in rows for d in c['world']['documents']}
        assert not projects & seen
        seen |= projects
    plan = study.read(study.ROOT / study.PLAN)
    assert len(study.load_cases(plan)) == 24
    for split in ('train', 'integration', 'confirmation'):
        with pytest.raises(ValueError, match='may not access'):
            study.load_cases({**plan, 'split': split})
    # A goal-directed fixture checks all development environment outcomes, not
    # model quality. Do not evaluate confirmation using an assistant here.
    assert all(execute_fixture(c)['score']['passed'] for c in data.cases('development'))


def test_partition_estimate_preserves_granite_shapes_and_counts_tied_head_once():
    document = study.read(study.ROOT / 'config/experiments/assistant-granite-partition.json')
    report = estimate(document['config'])
    assert report == document['estimate']
    assert report['parameters'] == 3402836480
    assert report['owner_weight_bytes'] == [3659735040, 3145937920]
    assert report['owner_kv_bytes'] == [251658240] * 2
    assert report['last_eight_q_v_lora_rank16_parameters'] == 1048576
    assert report['required_native_semantics']['embedding_multiplier'] == 12
    assert not report['measured_peak_memory']


def test_execution_inventory_pins_entire_runtime_and_cloud_remains_bounded():
    execution = study.read(study.ROOT / study.EXECUTION)
    for name, digest in execution['contracts'].items():
        assert study.sha256(study.ROOT / name) == digest
    required = {study.PLAN, study.EXECUTION, study.SCRIPT,
                'src/neuroshard/evolution/assistant_workflow_data.py',
                'src/neuroshard/evolution/assistant_workspace.py',
                'src/neuroshard/evolution/assistant_workflow.py',
                'src/neuroshard/evolution/modular_tools.py',
                'src/neuroshard/evolution/granite_reference.py'}
    assert required <= set(execution['sources'])
    spec = importlib.util.spec_from_file_location('workflow_cloud', study.ROOT / 'scripts/modular_reference_cloud.py')
    cloud = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cloud)
    resources = cloud.resources('assistant-workflow-baseline')
    assert resources['hours'] == 2 and resources['planning_cap_usd'] == 6 and not resources['gpu']
    assert cloud.remote_command('assistant-workflow-baseline')[1].endswith(study.SCRIPT)
    assert not execution['training_authorized'] and not execution['gpu_launch_authorized']


def test_baseline_qualification_requires_complete_outcomes_and_each_protected_answer():
    plan = study.read(study.ROOT / study.PLAN)
    anchors = []
    for task in study.read(study.ROOT / study.reference.PLAN)['tasks']:
        text = task['accept'][0] if task['kind'] == 'exact' else json.dumps(task['expected'])
        if task['kind'] == 'tool':
            text = '<tool_call>' + text + '</tool_call>'
        anchors.append({'id': task['id'], 'text': text, 'terminated': True, 'passed': True})
    primary = {'execution_completed': True, 'peak_rss_bytes': 1024, 'anchors': anchors,
               'episodes': [execute_fixture(case) for case in data.cases('development')]}
    report = study.assess(plan, primary)
    assert report['baseline_qualified'] and report['correct'] == 24
    assert report['insufficient_development_headroom']
    assert not report['training_authorized'] and not report['checklist_credit']
    missing = {**primary, 'episodes': primary['episodes'][:-1]}
    assert not study.assess(plan, missing)['baseline_qualified']
    bad = copy.deepcopy(primary)
    protected = next(row for row in bad['anchors'] if row['id'] in plan['protected_anchor_ids'])
    protected.update(text='incorrect', passed=False)
    assert not study.assess(plan, bad)['baseline_qualified']
    bad = copy.deepcopy(primary)
    bad['episodes'][0]['score']['passed'] = False
    with pytest.raises(ValueError, match='outcome rescore'):
        study.assess(plan, bad)


def test_native_input_cap_stops_without_calling_a_model():
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return 'too large'
        def __call__(self, *args, **kwargs):
            return {'input_ids': [1] * 6145}
    respond = study.native_responder(None, Tokenizer(), policy())
    value = respond([{'role': 'user', 'content': 'hello'}], workspace.TOOLS)
    assert not value['executed'] and not value['terminated'] and value['token_ids'] == []
