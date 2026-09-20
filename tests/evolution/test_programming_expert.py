import copy
import importlib.util
import os
from pathlib import Path
import shutil

import pytest

from neuroshard.evolution import programming_expert as experiment


def sandbox():
    path = Path(__file__).resolve().parents[2] / 'scripts/programming_sandbox.py'
    spec = importlib.util.spec_from_file_location('programming_sandbox_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prompt_does_not_include_solution_or_withheld_tests():
    row = {'text': 'Return the sum.', 'code': 'SECRET_SOLUTION',
           'test_list': ['assert add(1, 2) == 3', 'SECRET_TEST']}
    prompt = experiment.routing_text(experiment.code_prompt(row))
    assert 'assert add(1, 2) == 3' in prompt
    assert 'SECRET' not in prompt
    with pytest.raises(ValueError):
        experiment.routing_text([{'role': 'assistant', 'content': 'label: code'}])


def test_code_extraction_never_repairs_or_joins_multiple_candidates():
    assert experiment.extract_code('```python\ndef f(): return 1\n```') == 'def f(): return 1\n'
    for bad in ('', 'def f(:', '```python\na=1\n```\n```python\na=2\n```'):
        with pytest.raises((ValueError, SyntaxError)):
            experiment.extract_code(bad)


@pytest.mark.skipif(not shutil.which('bwrap') or not Path(f'/run/user/{os.getuid()}/bus').exists(),
                    reason='requires Linux bubblewrap and systemd user scopes')
def test_sandbox_checks_actual_results_and_blocks_early_exit_runaway_and_host_read(tmp_path):
    check = sandbox().check
    assert check('def f(): return 7', '', ['assert f() == 7'])['passed']
    assert not check('def f(): return 8', '', ['assert f() == 7'])['passed']
    assert not check('import os; os._exit(0)', '', ['assert False'])['passed']
    assert not check('while True: pass', '', ['assert True'], seconds=.3)['passed']
    secret = tmp_path / 'host-canary'
    secret.write_text('private')
    assert not check(f'x=open({str(secret)!r}).read()', '', ['assert True'])['passed']
    assert not check("import socket\nsocket.create_connection(('1.1.1.1', 53), .2)", '', ['assert True'])['passed']


def case():
    plan = {'seed': 10, 'bootstrap_samples': 100,
            'gate': {'minimum_code_gain': .05, 'maximum_p95_ratio': 1.5, 'maximum_p95_seconds': 90}}
    rows = [{'id': str(i), 'kind': 'code', 'setup': '', 'tests': ['assert True']} for i in range(8)]
    rows += [{'id': 'general', 'kind': 'general'}]
    outputs = []
    for row in rows:
        for arm in ('base', 'automatic', 'replacement', 'ablated'):
            improved = row['kind'] == 'code' and arm in ('automatic', 'replacement')
            outputs.append({'id': row['id'], 'arm': arm, 'ids': [2 if improved else 1],
                            'text': 'good = True' if improved else 'bad = True',
                            'seconds': 1., 'route': 'code' if improved else 'parent'})
    return plan, rows, outputs


def test_scoring_requires_actual_automatic_gain_and_preservation():
    plan, rows, outputs = case()
    check = lambda code, setup, tests: {'passed': code.startswith('good')}
    assert experiment.score(rows, outputs, plan, check)['passed']
    bad = copy.deepcopy(outputs)
    next(x for x in bad if x['id'] == 'general' and x['arm'] == 'automatic')['ids'] = [99]
    result = experiment.score(rows, bad, plan, check)
    assert not result['passed'] and not result['gates']['retention']
    for out in outputs:
        if out['arm'] == 'automatic':
            out['text'] = 'bad = True'
    assert not experiment.score(rows, outputs, plan, check)['gates']['gain']


def test_scoring_fails_on_missing_repeated_or_invalid_observations():
    plan, rows, outputs = case()
    check = lambda *args: {'passed': True}
    for incomplete in (outputs[:-1], outputs + outputs[:1]):
        with pytest.raises(ValueError, match='coverage'):
            experiment.score(rows, incomplete, plan, check)
    outputs[0]['seconds'] = float('nan')
    with pytest.raises(ValueError, match='measurements'):
        experiment.score(rows, outputs, plan, check)


def test_scoring_rejects_an_ablation_that_changed_the_base():
    plan, rows, outputs = case()
    next(x for x in outputs if x['arm'] == 'ablated')['ids'] = [99]
    with pytest.raises(ValueError, match='restore'):
        experiment.score(rows, outputs, plan, lambda *args: {'passed': True})
