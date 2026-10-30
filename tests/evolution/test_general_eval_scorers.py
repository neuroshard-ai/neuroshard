"""Program scorers for every area of the general-capability evaluation, on hand-written replies."""

import json

import pytest

from neuroshard.evolution.general_eval import items, scorers


def answer(kind, **fields):
    return {'kind': 'answer', 'answer': {'type': kind, **fields}}


def test_numeric_answers_use_the_answer_line_then_the_last_number():
    spec = answer('number', value=215, tolerance=0)
    assert scorers.score_answer(spec, 'It is 3 h 35 min.\nAnswer: 215 minutes')['passed']
    assert scorers.score_answer(spec, 'So 13:15 minus 09:40 gives 215')['passed']
    assert not scorers.score_answer(spec, 'Answer: 215 or maybe 216')['passed']  # the last number on the line
    assert not scorers.score_answer(spec, 'Answer: 205')['passed']
    assert scorers.score_answer(answer('number', value=450), '**Answer:** $4,50')['passed'] is False
    assert scorers.score_answer(answer('number', value=1450), 'Answer: $1,450.')['passed']
    third = answer('number', value=0.111111, tolerance=0.0002)
    assert scorers.score_answer(third, 'Answer: 1/9')['passed'] and scorers.score_answer(third, 'Answer: 0.111')['passed']
    assert not scorers.score_answer(third, 'Answer: 1/6')['passed']


def test_choice_and_text_answers_need_the_answer_line():
    assert scorers.score_answer(answer('choice', value='D'), 'None must follow.\nAnswer: D) None of these')['passed']
    assert not scorers.score_answer(answer('choice', value='D'), 'The answer is D')['passed']
    text = answer('text', accepted=['Thursday'])
    assert scorers.score_answer(text, 'Answer: **Thursday**')['passed']
    assert not scorers.score_answer(text, 'Answer: Thursday or Friday')['passed']


def test_code_extraction_takes_the_last_block_defining_the_entry_point():
    reply = '```python\ndef helper():\n    pass\n```\nand\n```python\ndef f(x):\n    return x\n```'
    assert scorers.extract_code(reply, 'f').startswith('def f')
    assert scorers.extract_code('```python\nprint(1)\n```', 'f') is None
    assert scorers.extract_code('def f(x):\n    return 1', 'f') is not None
    assert scorers.extract_code('```python\nclass LRUCache:\n    pass\n```', 'LRUCache') is not None


def test_code_scoring_runs_the_hidden_tests():
    spec = {'entry_point': 'f', 'tests': "f = solution['f']\nassert same(f(2), 4)\n", 'timeout': 5}
    assert scorers.score_code(spec, '```python\ndef f(x):\n    return x * 2\n```')['passed']
    assert not scorers.score_code(spec, '```python\ndef f(x):\n    return x + 3\n```')['passed']
    assert not scorers.score_code(spec, 'I would use multiplication.')['passed']


def test_writing_constraints():
    check = scorers.check_constraint
    assert check({'kind': 'paragraphs', 'n': 2}, 'One.\n\nTwo.') and not check({'kind': 'paragraphs', 'n': 2}, 'One.')
    assert check({'kind': 'lines', 'n': 2}, 'a\n\nb') and not check({'kind': 'lines', 'n': 2}, 'a\nb\nc')
    assert check({'kind': 'starts_with', 'phrase': 'Heads up'}, '**Heads up**, team')
    assert check({'kind': 'min_words', 'n': 3}, 'one two three') and not check({'kind': 'min_words', 'n': 4}, 'a b c')
    assert check({'kind': 'include_all', 'words': ['roof', 'March']}, 'The roof, in March.')
    assert not check({'kind': 'exclude_all', 'words': ['luck', 'journey']}, 'Good luck!')
    assert check({'kind': 'no_char', 'char': ','}, 'No commas here.') and not check({'kind': 'no_char', 'char': ','}, 'a, b')
    assert check({'kind': 'title_line', 'max_words': 8}, '# Cooking at Forty\n\nBody text.')
    assert not check({'kind': 'title_line', 'max_words': 8}, 'This is a sentence that is far too long to be a title.\n')
    assert check({'kind': 'max_sentence_words', 'n': 4}, 'Short one. Also short!')
    assert not check({'kind': 'max_sentence_words', 'n': 4}, 'This sentence has six words. Ok.')
    # the existing chat constraints are reused unchanged, including ending markup stripping
    assert check({'kind': 'ends_with', 'phrase': 'Thanks again, Priya', 'markup': 'strip'}, 'Hi.\n**Thanks again, Priya**!')
    spec = {'kind': 'constraints', 'constraints': [{'kind': 'bullets', 'n': 2}, {'kind': 'include', 'word': 'tap'}]}
    assert scorers.score_constraints(spec, '- fix the tap\n- soon')['passed']
    assert not scorers.score_constraints(spec, '- fix it\n- soon')['passed']
    assert not scorers.score_constraints(spec, '')['passed']


def test_short_answers_accept_aliases_and_refuse_hedges():
    spec = {'kind': 'short_answer', 'accepted': ['Canberra'], 'rejected': ['Sydney']}
    assert scorers.score_short_answer(spec, 'Answer: Canberra.')['passed']
    assert not scorers.score_short_answer(spec, 'Answer: Sydney or Canberra')['passed']
    assert not scorers.score_short_answer(spec, 'Answer: Canberrafield')['passed']
    numeric = {'kind': 'short_answer', 'accepted': ['273.15']}
    assert scorers.score_short_answer(numeric, 'Answer: 273.150 K')['passed']
    assert not scorers.score_short_answer(numeric, 'Answer: 1273.15')['passed']


def plan_reply(value):
    return 'Here is the plan:\n```json\n' + json.dumps(value) + '\n```'


def dev(item_id):
    return next(item for item in items.load_split() if item['id'] == item_id)


def test_every_development_reference_plan_validates_and_mutations_fail():
    planning = [item for item in items.load_split() if item['area'] == 'planning']
    assert len(planning) == 20
    for item in planning:
        spec = item['scoring']
        assert scorers.score_plan(spec, plan_reply(spec['reference']))['passed'], item['id']
    saturday = dev('planning-dev-001')['scoring']
    late_gym = {**saturday['reference'], 'gym': '11:00'}  # ends 12:30, after the gym closes and into lunch
    assert scorers.score_plan(saturday, plan_reply(late_gym))['detail']['problems']
    assert not scorers.score_plan(saturday, plan_reply({'gym': '09:00'}))['passed']  # tasks missing
    assert not scorers.score_plan(saturday, 'I would go to the gym first.')['passed']


def test_optimal_plans_must_be_optimal():
    office = dev('planning-dev-004')['scoring']
    assert scorers.score_plan(office, plan_reply(['chair', 'monitor', 'headset']))['passed']
    feasible_not_best = scorers.score_plan(office, plan_reply(['chair', 'monitor', 'lamp', 'plant']))
    assert not feasible_not_best['passed'] and 'more total value' in feasible_not_best['detail']['problems'][0]
    errands = dev('planning-dev-005')['scoring']
    assert scorers.score_plan(errands, plan_reply(['home', 'post', 'pharmacy', 'bakery', 'home']))['passed']
    assert not scorers.score_plan(errands, plan_reply(['home', 'pharmacy', 'bakery', 'post', 'home']))['passed']


def test_json_extraction_prefers_fenced_then_last_balanced_value():
    assert scorers.extract_json('x {"a": 1} then {"b": [2]}') == {'b': [2]}
    assert scorers.extract_json('```json\n["a"]\n```\n{"later": 1}') == ['a']
    assert scorers.extract_json('no json') is None


def test_spec_checks_reject_unsound_items():
    good = dev('coding-dev-001')
    broken = {**good, 'scoring': {k: v for k, v in good['scoring'].items() if k != 'reference'}}
    with pytest.raises(ValueError):
        items.validate(broken)
    with pytest.raises(ValueError):
        items.validate({**good, 'area': 'poetry'})
    with pytest.raises(ValueError):
        items.validate({**good, 'turns': [{'user': 'x', 'expected': 'leak'}]})
