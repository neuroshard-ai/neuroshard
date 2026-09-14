"""Frozen calculation-step experiment, using the existing sharded trainer.

The interpreter generates training labels only. Inference extracts model-written
JSON and never executes, calculates or repairs the predicted answer.
"""
import json

from . import grounded_tasks as tasks
from .reference_data import identity

FORMAT = 'neuroshard-reasoned-learning-v1'
PLAN_ROOT = 'd4ecf3612a47e516351fa697b4483d85752fd02c2de161eca8a10cc532a29c2f'


def enabled(plan):
    return plan.get('format') == FORMAT


def validate(plan):
    from .continued import STATUSES
    if (not enabled(plan) or plan.get('status') not in STATUSES
            or identity({k: v for k, v in plan.items() if k != 'status'}) != PLAN_ROOT):
        raise ValueError('Reasoned-learning constants differ from the frozen plan')
    return plan


def messages(case, use_reasoning):
    if not use_reasoning or case['family'] != 'total':
        return [{'role': 'user', 'content': tasks.prompt(case)},
                {'role': 'assistant', 'content': json.dumps(tasks.expected(case), separators=(',', ':'))}]
    visible = [{key: row[key] for key in ('id', 'units', 'price')} for row in case['rows']]
    prompt = ('Use only the records below. Calculate the invoice total by multiplying units by price '
              'for every record and adding all products. Write your calculation inside <work> and '
              '</work>: one line per record with its id, multiplication result and running total. '
              'Then return exactly one JSON object with the integer key "total".\nRecords:\n'
              + json.dumps(visible, separators=(',', ':')))
    total, lines = 0, []
    for row in visible:
        product = row['units'] * row['price']
        total += product
        lines.append(f'{row["id"]}: {row["units"]} * {row["price"]} = {product}; running total = {total}')
    answer = '<work>\n' + '\n'.join(lines) + '\n</work>\n' + json.dumps({'total': total}, separators=(',', ':'))
    return [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}]


def final_json(text):
    """Extract a bounded, explicit work block; a direct JSON answer is also valid."""
    if not isinstance(text, str) or len(text.encode()) > 32768:
        raise ValueError('Invalid generated response size')
    value = text.strip()
    if not value.startswith('<work>'):
        tasks.parse_answer(value)
        return value, False
    if value.count('<work>') != 1 or value.count('</work>') != 1:
        raise ValueError('Require exactly one complete work block')
    work, answer = value[len('<work>'):].split('</work>')
    if not work.strip():
        raise ValueError('Empty work block')
    tasks.parse_answer(answer.strip())
    return answer.strip(), True


def check_answer(plan, case, text, role):
    if not enabled(plan) or role not in plan['method']['reasoning_roles'] or case['family'] != 'total':
        return tasks.check_answer(case, text)
    try:
        answer, used = final_json(text)
    except (ValueError, TypeError):
        return {'correct': False, 'valid_json_object': False, 'reason': 'invalid_reasoned_response'}
    return {**tasks.check_answer(case, answer), 'reasoning_prefix': used}


def previous_prepared(plan):
    from . import continued
    path = continued.repo_root() / plan['previous_continuation']['prepared_path']
    raw = path.read_bytes()
    previous = json.loads(raw)
    if (continued.git_bytes('HEAD', path) != raw
            or identity(previous) != plan['previous_continuation']['prepared']):
        raise ValueError('Previous continuation exclusion is not the committed artifact')
    return previous


def generation_summary(plan, prepared, generation):
    """Recheck every development answer against its frozen, generated task."""
    result = {}
    if set(generation) != {'dev-new', 'dev-prior'}:
        raise ValueError('Development must cover both declared answer roles')
    for role, pair in generation.items():
        seed = plan['task_seed'] if role == 'dev-new' else plan['prior_probe_seed']
        cases = [tasks.make_case(seed, 'dev', i) for i in range(plan['development_cases'])]
        expected = [tasks.task_identity(case) for case in cases]
        if expected != prepared['roles'][role]['ids'] or set(pair) != {'before', 'after'}:
            raise ValueError('Development task identities differ from the prepared freeze')
        checked = {}
        for name, answers in pair.items():
            if [row['id'] for row in answers] != expected:
                raise ValueError('Development answers are missing, reordered or duplicated')
            checked[name] = []
            for case, answer in zip(cases, answers):
                check = check_answer(plan, case, answer['text'], role)
                if check != answer['check']:
                    raise ValueError('Development verdict differs from the generated answer')
                checked[name].append({'id': answer['id'], 'correct': check['correct']})
        summary = tasks.paired_accuracy(checked['before'], checked['after'])
        families = {}
        for family in tasks.FAMILIES:
            indices = [i for i, case in enumerate(cases) if case['family'] == family]
            families[family] = tasks.paired_accuracy(
                [checked['before'][i] for i in indices], [checked['after'][i] for i in indices])
        summary['families'] = families
        result[role] = summary
    return result


def development(plan, prepared, step, retention, generation):
    if step not in plan['checkpoints']:
        raise ValueError('Undeclared development checkpoint')
    summary = generation_summary(plan, prepared, generation)
    gate = plan['quality_gate']
    primary = summary['dev-new']
    checks = {
        'retention': retention['passed'],
        'arithmetic_progress': (step < gate['development_futility_step']
                                or primary['families']['total']['candidate_correct']
                                >= gate['development_min_correct_totals']),
        'final_gain': (step != plan['final_step'] or primary['wins'] - primary['losses']
                       >= gate['development_final_min_net_gain']),
        'final_floors': (step != plan['final_step'] or all(
            family['candidate_correct'] >= family['baseline_correct']
            for role in summary.values() for family in role['families'].values())),
    }
    return {**retention, 'step': step, 'generation': generation, 'generation_summary': summary,
            'checks': checks, 'passed': all(checks.values())}
