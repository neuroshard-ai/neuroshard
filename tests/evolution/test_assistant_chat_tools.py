import json

import pytest

from neuroshard.evolution import assistant_chat_data as chat
from neuroshard.evolution import assistant_general as general
from neuroshard.evolution import assistant_general_l4 as l4
from neuroshard.evolution import assistant_phrasings as phrasings
from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256

POLICY = read(ROOT / 'config/experiments/assistant-general-policy-chat3.json')


def check(text, rules):
    return general.ChatTools().execute({'name': 'check_text', 'arguments': {'text': text, 'rules': rules}})


def test_check_text_agrees_with_the_scorer_for_every_rule_kind():
    assert check('Tango is lively.', ['exclude:and'])['passes']
    assert 'appears 1 time' in check('Tango and salsa.', ['exclude:and'])['violations'][0]
    assert check('so steady', ['include:steady', 'lowercase'])['passes']
    assert check('Keep going. **Keep at it**.', ['ends_with:Keep at it'])['passes']
    assert not check('one two three four', ['max_words:3'])['passes']
    assert check('- a\n- b', ['bullets:2'])['passes']
    assert check('{"a": 1, "b": 2}', ['json_keys:a, b'])['passes']
    assert 'error' in check('x', ['exclude:'])
    assert 'error' in check('x', ['rhyme:moon'])


def test_check_text_never_sees_the_hidden_constraints_and_the_scorer_is_unchanged():
    case = chat.cases('generated-development4')[0]
    seen = []

    def respond(messages, tools):
        seen.append((messages, tools))
        return {'text': 'Final.', 'terminated': True}

    row = general.chat_episode(case, respond, POLICY)
    flat = json.dumps(seen)
    assert 'constraints' not in flat and {t['function']['name'] for t in seen[0][1]} == {'calculate', 'check_text'}
    assert row['passed'] == chat.score(case['turns'][0]['expected'], 'Final.', True)


def test_check_text_calls_are_recorded_and_rescore():
    case = next(c for c in chat.cases('generated-development4') if c['family'] == 'exclude')
    word = case['turns'][0]['expected']['constraints'][0]['word']
    replies = iter(['<tool_call>\n' + json.dumps({'name': 'check_text', 'arguments': {
        'text': f'Some {word} text.', 'rules': [f'exclude:{word}']}}) + '\n</tool_call>', 'Some text here.'])
    row = general.chat_episode(case, lambda m, t: {'text': next(replies), 'terminated': True}, POLICY)
    assert row['calls'][0]['result']['passes'] is False and row['passed']
    assert general.rescore([case], [row], 'chat') == 1


def test_c1_declares_a_chat_unit_with_pinned_contract():
    profile = 'assistant-general-c1-gpu'
    declaration_path, execution_path = l4.PROFILES[profile]
    declaration, execution = read(ROOT / declaration_path), read(ROOT / execution_path)
    assert declaration['route'] == 'chat' and declaration['integrate'] == ['chat']
    assert l4.splits(declaration)[0] == ['generated-train', 'generated-train-styles2']
    assert all(sha256(ROOT / name) == digest for name, digest in execution['contracts'].items())
    for name in (phrasings.FILE4, 'config/experiments/assistant-general-policy-chat3.json',
                 'src/neuroshard/evolution/assistant_phrasings.py'):
        assert name in execution['sources']
    assert all((ROOT / name).exists() for name in execution['sources'])


@pytest.mark.parametrize('split,n', [('generated-train-styles2', 240), ('generated-development4', 48),
                                     ('confirmation6', 96)])
def test_round_four_chat_splits_are_distinct_and_quoted(split, n):
    import re

    cases = chat.cases(split)
    assert len({c['turns'][0]['user'] for c in cases}) == n
    for case in cases:
        for c in case['turns'][0]['expected']['constraints']:
            value = c.get('word') or c.get('phrase')
            if value:
                assert re.search(r'["“\'‘]' + re.escape(value) + r'[.!]?["”\'’]', case['turns'][0]['user'])


def test_confirmation6_is_new_to_every_set():
    norm = lambda t: ' '.join(t.lower().split())  # noqa: E731
    earlier = [read(ROOT / f)['sets'] for f in (phrasings.FILE, phrasings.FILE2, phrasings.FILE3)]
    known = {norm(t) for sets in earlier for by in sets.values() for rows in by.values() for ts in rows.values()
             for t in ts}
    fresh = read(ROOT / phrasings.FILE4)['sets']
    assert all(norm(t) not in known for by in fresh.values() for rows in by.values() for ts in rows.values()
               for t in ts)
