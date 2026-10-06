import copy

import pytest

from neuroshard.evolution import assistant_calendar as calendar
from neuroshard.evolution import assistant_routing as routing
from neuroshard.evolution import assistant_schedule_data as schedule
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution import assistant_workspace as drafting
from neuroshard.evolution.modular_reference_execution import ROOT, read

from test_assistant_calendar import POLICY as CALENDAR, schedule_texts
from test_assistant_workflow import execute_fixture, reference_texts, reply

DRAFTING = read(ROOT / 'config/experiments/assistant-workflow-policy.json')
POLICIES = {'drafting': DRAFTING, 'scheduling': CALENDAR}


def scripted(texts, seen):
    texts = iter(texts)

    def respond(messages, tools):
        seen.append((messages[0]['content'], [t['function']['name'] for t in tools]))
        return reply(next(texts))
    return respond


def run(case, texts, chosen):
    seen = []
    respond = scripted(texts, seen)
    routes = {name: (respond, policy) for name, policy in POLICIES.items()}
    return routing.execute(case, routes, lambda turn, user: chosen[turn]), seen


def test_a_conversation_switches_from_drafting_to_scheduling_at_its_follow_up():
    case = schedule.make_case('cross-development', 'handoff', 0)
    texts = schedule_texts(case)
    result, seen = run(case, texts, ['drafting', 'scheduling'])
    assert result['score']['passed'] and result['score']['routes'] == ['drafting', 'scheduling']
    first = result['rounds'][0]['generation_count']
    assert all(instruction == DRAFTING['system_instruction'] and tools == [t['function']['name'] for t in drafting.TOOLS]
               for instruction, tools in seen[:first])
    assert all(instruction == CALENDAR['system_instruction'] and 'save_meeting' in tools for instruction, tools in seen[first:])
    # The scheduling turn sees the drafting turn's messages and tool results.
    assert any(m['role'] == 'tool' and '"saved": true' in m['content'] for m in result['messages'][:12])
    assert result['selection_seconds'] >= 0 and all('selection_seconds' in r for r in result['rounds'])


def test_a_drafting_route_cannot_book_and_a_scheduling_route_can_still_draft():
    case = schedule.make_case('development', 'slot', 0)
    result, _ = run(case, schedule_texts(case), ['drafting'])
    assert not result['score']['passed']
    assert any('invalid tool-call syntax' in m['content'] for m in result['messages'] if m['role'] == 'tool')
    review = schedule.make_case('cross-development', 'review', 1)
    assert run(review, schedule_texts(review), ['scheduling'])[0]['score']['passed']


def test_drafting_routed_alone_scores_as_the_single_route_scorer_does():
    for case in data.cases('development')[:8]:
        result, _ = run(case, reference_texts(case), ['drafting'] * len(case['turns']))
        assert result['score']['round_successes'] == execute_fixture(case)['score']['round_successes']
    single = routing.single(case, scripted(reference_texts(case), []), DRAFTING)
    assert single['score']['passed'] == execute_fixture(case)['score']['passed']


def test_the_scorer_replays_routes_and_rejects_forged_transcripts():
    case = schedule.make_case('cross-development', 'handoff', 1)
    result, _ = run(case, schedule_texts(case), ['drafting', 'scheduling'])
    assert routing.score(case, result, POLICIES) == result['score']
    forged = copy.deepcopy(result)
    forged['rounds'][1]['route'] = 'drafting'
    with pytest.raises(ValueError):
        routing.score(case, forged, POLICIES)
    forged = copy.deepcopy(result)
    forged['calls'][-1]['call']['arguments']['start_time'] = '16:00'
    with pytest.raises(ValueError, match='tool actions'):
        routing.score(case, forged, POLICIES)
    with pytest.raises(ValueError, match='binding'):
        routing.score(case, result, {**POLICIES, 'scheduling': {**CALENDAR, 'limits': {**CALENDAR['limits']}, 'x': 1}})


def test_turn_targets_follow_success_rates_and_ties_keep_the_drafting_route():
    drafting_runs = {'a': [[True, False], [True, False]], 'b': [[False], [False]], 'c': [[True], [True]]}
    scheduling_runs = {'a': [[True, True], [False, True]], 'b': [[True], [False]], 'c': [[True], [True]]}
    rows = routing.turn_targets(drafting_runs, scheduling_runs, {'a': 2, 'b': 1, 'c': 1})
    assert rows == {'a#0': (0.0, 0.5), 'a#1': (1.0, 1.0), 'b#0': (1.0, 0.5), 'c#0': (0.0, 0.5)}
    with pytest.raises(ValueError, match='no integration outcome'):
        routing.turn_targets({'a': []}, scheduling_runs, {'a': 1})
    failed = {'d': [[False], [False]]}
    assert routing.turn_targets(failed, failed, {'d': 1}) == {'d#0': (0.0, 0.5)}
    assert routing.turn_targets({**drafting_runs, **failed}, {**scheduling_runs, **failed},
                                {'a': 2, 'b': 1, 'c': 1, 'd': 1}, failed_ties=False) == rows


def test_a_message_feature_reuses_the_shared_prefix_and_matches_a_full_forward(tmp_path):
    torch = pytest.importorskip('torch')
    from test_assistant_experience import TEMPLATE, tiny_model
    from test_granite_tokenizer import granite_like, load_tiny

    directory = granite_like(tmp_path / 'granite')
    (directory / 'chat_template.jinja').write_text(TEMPLATE)
    tokenizer = load_tiny(directory)[0]
    model = tiny_model(tokenizer)
    prefix = routing.message_prefix(model, tokenizer, DRAFTING, 'cpu')
    features = []
    for user in ('Schedule a 45-minute review with the design team.', 'Change the recipient to the finance team.'):
        ids = routing.turn_ids(tokenizer, DRAFTING, user)
        assert ids[:len(prefix['ids'])] == prefix['ids'] and len(ids) > len(prefix['ids'])
        with torch.no_grad():
            full = model.model(input_ids=torch.tensor([ids])).last_hidden_state[0, len(prefix['ids']):].mean(dim=0)
        features.append(routing.message_feature(model, tokenizer, DRAFTING, user, 'cpu', prefix))
        assert torch.allclose(torch.tensor(features[-1]), full, atol=1e-5)
        assert prefix['cache'].get_seq_length() == len(prefix['ids'])
    assert features[0] != features[1]


def test_a_turn_feature_renders_one_user_message_after_the_policy_instruction_and_tools():
    captured = {}

    class Tokenizer:
        def apply_chat_template(self, messages, tools, add_generation_prompt, tokenize):
            captured.update(messages=messages, tools=tools)
            return 'prompt'

        def __call__(self, text, add_special_tokens):
            return {'input_ids': [1, 2, 3]}

    assert routing.turn_ids(Tokenizer(), DRAFTING, 'Schedule a meeting.') == [1, 2, 3]
    assert captured['messages'] == [{'role': 'system', 'content': DRAFTING['system_instruction']},
                                    {'role': 'user', 'content': 'Schedule a meeting.'}]
    assert captured['tools'] == drafting.TOOLS
    routing.turn_ids(Tokenizer(), CALENDAR, 'x')
    assert captured['tools'] == calendar.TOOLS
