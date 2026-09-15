"""A new final must keep the known facts and change its wording and identity."""
import pytest
from neuroshard.evolution import preserved_interpreter as contract


def examples():
    task = {'entity': 4, 'attribute': 'city', 'name': 'Robin Finch', 'expected': 'Sofia'}
    return [{'id': str(i), 'task': dict(task), 'messages': [{'role': 'user', 'content': 'old-' + str(i)}]}
            for i in range(2)]


def test_new_final_preserves_fact_but_never_reuses_an_exposed_question(monkeypatch):
    monkeypatch.setattr(contract.data, 'conversation', lambda *args: {'targets': 1})
    rows = examples()
    forms = {'format': 'test-v1', 'final': {'city': ['Residence of {name}?', 'Which city for {name}?']}, 'suffix': ''}
    actual = contract.final_questions(rows, forms, None, 1024)
    assert [r['task'] for r in actual] == [r['task'] for r in rows]
    assert {r['id'] for r in actual}.isdisjoint(r['id'] for r in rows)
    assert len({r['id'] for r in actual}) == 2
    forms['final']['city'][0] = 'old-0'
    with pytest.raises(ValueError, match='repeat an exposed question'):
        contract.final_questions(rows, forms, None, 1024)


def test_new_final_refuses_incomplete_or_extra_entity_forms(monkeypatch):
    monkeypatch.setattr(contract.data, 'conversation', lambda *args: {'targets': 1})
    forms = {'format': 'test-v1', 'final': {'city': ['a {name}', 'b {name}']}, 'suffix': ''}
    with pytest.raises(ValueError, match='Incomplete'):
        contract.final_questions(examples()[:1], forms, None, 1024)
    with pytest.raises(ValueError, match='exactly two'):
        contract.final_questions(examples() + examples()[:1], forms, None, 1024)
