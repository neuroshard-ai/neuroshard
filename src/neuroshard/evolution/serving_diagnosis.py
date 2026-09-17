"""Classify complete-serving failures without training or opening a new final.

Labels are exclusive for the primary recommendation: knowledge, selection,
decomposition, then assembly. Gold standalone expert questions, when present,
decide knowledge before any planner judgement. Answer tables stay in scoring;
they must not appear in the user messages sent to the model.
"""
import json
import re
import unicodedata
from pathlib import Path

from .cohort_questions import normalized
from .reference_data import identity, sha256
from .schema import integer


def conversation(messages):
    if not isinstance(messages, list) or not 1 <= len(messages) <= 31 or len(messages) % 2 != 1:
        raise ValueError('Require a bounded conversation ending with a user turn')
    size = 0
    for index, message in enumerate(messages):
        if (not isinstance(message, dict) or set(message) != {'role', 'content'}
                or message['role'] != ('user' if index % 2 == 0 else 'assistant')
                or not isinstance(message['content'], str) or not message['content'].strip()):
            raise ValueError('Require alternating nonempty user and assistant turns')
        size += len(message['content'].encode())
    if size > 32768:
        raise ValueError('Conversation exceeds its byte bound')
    return messages

FORMAT = 'neuroshard-ordinary-serving-diagnostic-v1'
MODES = ('knowledge', 'selection', 'decomposition', 'assembly')
GRAMMAR = ('First:', 'Second:')
SOURCES = (
    'src/neuroshard/evolution/serving_diagnosis.py',
    'config/experiments/continual-expert-facts.json',
    'scripts/score_serving_diagnosis.py',
    'scripts/run_ordinary_serving_diagnostic.py',
)


def folded(text):
    if not isinstance(text, str):
        raise ValueError('Score actual generated text')
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def contains_terms(text, terms):
    value = folded(text)
    return all(folded(term) in value for term in terms)


def atomic_correct(text, atom):
    if 'contains' in atom:
        return all(term.casefold() in text.casefold() for term in atom['contains'])
    if 'answer' not in atom:
        raise ValueError('An atom needs a scoring answer or contains list')
    expected = normalized(atom['answer'], '')
    actual = normalized(text.split(';')[0] if ';' not in atom['answer'] else text, '')
    if actual == expected:
        return True
    pattern = r'(?<!\d)' + re.escape(expected) + r'(?!\d)'
    return re.search(pattern, actual) is not None


def final_contains(text, atom):
    if 'contains' in atom:
        return contains_terms(text, atom['contains'])
    return atomic_correct(text, atom)


def forbidden_finals(curation):
    questions = []
    for cohort in curation['cohorts']:
        for fact in cohort['facts']:
            questions.append(fact['test_question'])
    return questions


def validate_case(case, finals):
    conversation(case['messages'])
    integer(case.get('max_tokens', 64), 1, 256)
    if case['stratum'] not in ('retained', 'new'):
        raise ValueError('Cases must declare retained A/B knowledge or new isolated-expert facts')
    atoms = case['atoms']
    if not isinstance(atoms, list) or not 1 <= len(atoms) <= 2:
        raise ValueError('Require one or two scored atoms')
    text = ' '.join(message['content'] for message in case['messages'] if message['role'] == 'user')
    if any(marker in text for marker in GRAMMAR):
        raise ValueError('Ordinary diagnostic cases cannot use the explicit two-question grammar')
    compact = folded(text)
    for question in finals:
        if folded(question) in compact:
            raise ValueError('Semantic final wording cannot enter this diagnostic')
    seen = set()
    for atom in atoms:
        if not {'expert', 'terms', 'gold_question'} <= set(atom) or not {'answer', 'contains'} & set(atom):
            raise ValueError('Atoms bind an expert, subject terms, a gold question and a score')
        key = identity([atom['expert'], atom['gold_question']])
        if (key in seen or not isinstance(atom['terms'], list)
                or not 1 <= len(atom['terms']) <= 8
                or any(not isinstance(term, str) or not term.strip() for term in atom['terms'])
                or not isinstance(atom['gold_question'], str) or not atom['gold_question'].strip()
                or any(marker in atom['gold_question'] for marker in GRAMMAR)
                or any(folded(question) in folded(atom['gold_question']) for question in finals)):
            raise ValueError('Gold questions must be standalone, unique and outside the opened final')
        seen.add(key)
        if atom.get('answer') is not None and atom['answer'] in text:
            raise ValueError('Do not put scoring answers in the user request')
    return case


def validate(plan, curation, source_home=None):
    if plan.get('format') != FORMAT or plan.get('no_training') is not True:
        raise ValueError('This diagnostic is inference-only')
    if plan.get('serving') != 'planned-graph-service' or plan.get('final_opened') is not False:
        raise ValueError('Evaluate the planned serving path and keep a new final closed')
    integer(plan['resources']['gpus'], 4, 6)
    integer(plan['resources']['compute_share_usd_cap'], 1, 25)
    finals = forbidden_finals(curation)
    if len(plan['cases']) < 8 or len(plan['cases']) > 24:
        raise ValueError('Keep the ordinary development screen bounded')
    ids = []
    for case in plan['cases']:
        validate_case(case, finals)
        ids.append(case['id'])
    if len(set(ids)) != len(ids):
        raise ValueError('Diagnostic case identities must be unique')
    if not any(case['stratum'] == 'retained' for case in plan['cases']):
        raise ValueError('The retained A/B stratum is required')
    if not any(case['stratum'] == 'new' for case in plan['cases']):
        raise ValueError('The new isolated-expert stratum is required')
    if source_home is not None:
        for name in SOURCES:
            digest = sha256(Path(source_home) / name)
            if plan['sources'][name] != digest:
                raise ValueError('Frozen diagnostic source changed')
    return plan


def assign(plan, atoms):
    used, matched = set(), []
    for atom in atoms:
        hit = next((index for index, question in enumerate(plan)
                     if index not in used and contains_terms(question, atom['terms'])), None)
        if hit is not None:
            used.add(hit)
        matched.append(hit)
    return matched


def routes_of(response):
    rows = []
    for row in response.get('routing') or []:
        decision = row.get('decision') if isinstance(row, dict) else None
        rows.append(decision.get('route') if isinstance(decision, dict) else None)
    if len(rows) == len(response.get('answers') or []) and rows:
        return rows
    return [row.get('expert') for row in response.get('answers') or []]


def classify(case, response):
    """Label one ordinary serving response. Gold is applied by diagnose()."""
    plan = response.get('plan') or []
    error = response.get('error')
    if response.get('status') != 'completed' or error in (
            'invalid_neural_plan', 'invalid_directory_arguments') or not plan:
        return 'decomposition'
    if error == 'invalid_source_value_or_output_bound':
        return 'assembly'
    if len(plan) != len(case['atoms']):
        return 'decomposition'
    matched = assign(plan, case['atoms'])
    if any(index is None for index in matched):
        return 'decomposition'
    routes = routes_of(response)
    answers = response.get('answers') or []
    if len(routes) != len(plan) or len(answers) != len(plan):
        return 'decomposition'
    for atom, index in zip(case['atoms'], matched):
        if routes[index] != atom['expert']:
            return 'selection'
    for atom, index in zip(case['atoms'], matched):
        if not atomic_correct(answers[index]['text'], atom):
            return 'knowledge'
    if any(not final_contains(response.get('text') or '', atom) for atom in case['atoms']):
        return 'assembly'
    return None


def gold_case(atom, max_tokens):
    return {'id': 'gold', 'stratum': 'new', 'max_tokens': max_tokens,
            'messages': [{'role': 'user', 'content': atom['gold_question']}],
            'atoms': [atom]}


def diagnose(case, response, gold_responses=None):
    ordinary = classify(case, response)
    gold = []
    for atom, gold_response in zip(case['atoms'], gold_responses or []):
        gold.append(classify(gold_case(atom, case.get('max_tokens', 64)), gold_response))
    primary = 'knowledge' if any(mode == 'knowledge' for mode in gold) else ordinary
    return {'id': case['id'], 'stratum': case['stratum'], 'passed': ordinary is None and not any(gold),
            'mode': primary, 'ordinary': ordinary, 'gold': gold,
            'plan': list(response.get('plan') or []),
            'routes': routes_of(response),
            'answers': [row.get('text') for row in response.get('answers') or []],
            'text': response.get('text') or '',
            'status': response.get('status'),
            'error': response.get('error')}


def summarize(rows):
    totals = {mode: 0 for mode in ('passed',) + MODES}
    strata = {}
    for row in rows:
        mode = 'passed' if row['passed'] else row['mode']
        if mode not in totals:
            raise ValueError('Unclassified serving failure')
        totals[mode] += 1
        bucket = strata.setdefault(row['stratum'], {key: 0 for key in totals})
        bucket[mode] += 1
    failed = [mode for mode in MODES if totals[mode]]
    return {'format': FORMAT + '/summary', 'count': len(rows), 'totals': totals,
            'strata': strata, 'dominant': failed[0] if len(failed) == 1 else (failed or [None])[0],
            'mixed': len(failed) > 1, 'classified': len(rows)}


def score_traces(plan, traces):
    """Score recorded planned-graph or simplified traces without neural execution."""
    by_id = {row['id']: row for row in traces}
    if set(by_id) != {case['id'] for case in plan['cases']}:
        raise ValueError('Recorded traces must cover the frozen diagnostic cases exactly once')
    rows = []
    for case in plan['cases']:
        trace = by_id[case['id']]
        response = trace['response']
        gold = trace.get('gold_responses')
        rows.append(diagnose(case, response, gold))
    return {'format': FORMAT + '/result', 'plan': identity(plan), 'rows': rows,
            'summary': summarize(rows)}
