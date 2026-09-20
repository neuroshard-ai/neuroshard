"""A bounded, operated test of useful expert growth; no ledger transitions.

Preparation uses public benchmark splits. It never treats these old public
questions as evidence of absence from the seed model's pretraining corpus.
"""
import ast
import json
import math
from pathlib import Path
import random
import re

from .reference_data import conversation, identity, save, sha256, tokenizer_identity

FORMAT = 'neuroshard-programming-expert-v1'
MBPP_SHA = 'ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f'
GENERAL_SHA = 'be6773dcce145f3918ff14237b1f765affa427b0b13f6a02d397e665ac908b9a'
GENERAL_GROUPS = {
    'conversation': ['everyday-conversations'],
    'constraints': ['smol-contraints'],
    'summary': ['smol-summarize-20k', 'smol-summarize-5k'],
    'rewrite': ['smollm-rewrite-30k', 'explore-instruct-rewrite'],
}


def routing_text(messages):
    """Route the actual prompt, including system instructions, without labels."""
    if (not messages or messages[-1]['role'] != 'user'
            or any(m['role'] not in ('system', 'user', 'assistant') for m in messages)):
        raise ValueError('A conversation ending in a user message is required')
    return '\n'.join(m['role'] + ': ' + m['content'] for m in messages)


def code_prompt(row):
    # A single public example defines the callable interface. The other tests
    # are withheld from the model, including during teacher-forced training.
    return [{'role': 'user', 'content': row['text'].strip()
             + '\n\nUse this callable interface and behavior:\n' + row['test_list'][0]
             + '\n\nReturn only the complete Python code, including needed imports.'}]


def extract_code(text):
    """Accept plain code or one Python fence; never repair a model's answer."""
    if not isinstance(text, str) or len(text.encode()) > 32768:
        raise ValueError('Invalid generated code size')
    if '```' in text:
        matches = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, re.S)
        if len(matches) != 1 or text.count('```') != 2:
            raise ValueError('Require exactly one code block')
        text = matches[0]
    ast.parse(text)
    if not text.strip():
        raise ValueError('Empty program')
    return text


def _words(text):
    return set(re.findall(r'[a-z0-9]+', text.lower()))


def _similar(a, b):
    left, right = _words(a), _words(b)
    return len(left & right) / max(1, len(left | right)) >= .8


def prepare(plan, mbpp, general, tokenizer, home, gold_check):
    """Freeze task IDs, complete training tokens and exact schedules before GPUs."""
    import pyarrow.parquet as pq
    if plan['format'] != FORMAT or sha256(mbpp) != MBPP_SHA or sha256(general) != GENERAL_SHA:
        raise ValueError('Source differs from the pinned experiment')
    home = Path(home)
    home.mkdir(parents=True, exist_ok=False)
    raw = [json.loads(line) for line in Path(mbpp).read_text().splitlines()]
    if sorted(row['task_id'] for row in raw) != list(range(1, 975)):
        raise ValueError('Incomplete public benchmark')
    pools = {'train': [], 'dev': [], 'final': []}
    excluded = []
    heldout = [r for r in raw if 11 <= r['task_id'] <= 600]
    for row in raw:
        number = row['task_id']
        if number < 11:
            continue
        role = 'train' if number >= 601 else 'dev' if number >= 511 else 'final'
        if role == 'train' and any(_similar(row['text'], other['text']) for other in heldout):
            excluded.append({'task': number, 'reason': 'near-duplicate-heldout-prompt'})
            continue
        messages = code_prompt(row)
        try:
            tokenized = conversation(tokenizer, messages + [{'role': 'assistant', 'content': row['code']}],
                                     plan['max_length'])
        except OverflowError:
            excluded.append({'task': number, 'reason': 'complete-reference-exceeds-context'})
            continue
        verdict = gold_check(row['code'], row['test_setup_code'], row['test_list'])
        if not verdict['passed']:
            excluded.append({'task': number, 'reason': 'reference-' + verdict['status']})
            continue
        pools[role].append({'id': identity({'dataset': MBPP_SHA, 'task': number}), 'task_id': number,
                            'kind': 'code', 'messages': messages, 'reference': row['code'],
                            'setup': row['test_setup_code'], 'tests': row['test_list'], **tokenized})
    rng = random.Random(plan['seed'])
    for role in ('dev', 'final'):
        pools[role].sort(key=lambda r: identity([plan['seed'], r['id']]))
        count = plan['counts'][role + '_code']
        if len(pools[role]) < count:
            raise ValueError('Insufficient eligible benchmark tasks')
        pools[role] = pools[role][:count]
    general_rows = pq.read_table(general).to_pylist()
    candidates = {group: [] for group in GENERAL_GROUPS}
    seen = set()
    for index, row in enumerate(general_rows):
        group = next((g for g, names in GENERAL_GROUPS.items() if row['source'] in names), None)
        if group is None:
            continue
        # Retain supplied conversation history up to the last user turn. Taking
        # only the greeting collapses many different conversations to one row.
        user_turns = [i for i, m in enumerate(row['messages']) if m['role'] == 'user']
        messages = row['messages'][:user_turns[-1] + 1] if user_turns else []
        if not messages or messages[-1]['role'] != 'user':
            continue
        text = routing_text(messages)
        key = identity(text)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if key in seen or not 8 <= len(ids) <= plan['max_prompt_tokens']:
            continue
        seen.add(key)
        candidates[group].append({'id': identity({'dataset': GENERAL_SHA, 'row': index}),
                                  'kind': 'general', 'group': group, 'row': index, 'messages': messages})
    for group, rows in candidates.items():
        rows.sort(key=lambda r: identity([plan['seed'], r['id']]))
        cursor = 0
        for role in ('train', 'dev', 'final'):
            count = plan['counts'][role + '_general_per_group']
            selected = rows[cursor:cursor + count]
            if len(selected) != count:
                raise ValueError(f'Insufficient general prompts in {group}: {len(rows)} eligible')
            pools[role] += selected
            cursor += count
    manifests = {}
    for role, rows in pools.items():
        path = home / (role + '.jsonl')
        path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
        manifests[role] = {'file': path.name, 'sha256': sha256(path), 'ids': [r['id'] for r in rows],
                           'code': sum(r['kind'] == 'code' for r in rows),
                           'general': sum(r['kind'] == 'general' for r in rows)}
    # Fixed batches are cached once and replayed over the prescribed epochs.
    indices = [i for i, r in enumerate(pools['train']) if r['kind'] == 'code']
    rng.shuffle(indices)
    batches = [indices[i:i + plan['batch_size']] for i in range(0, len(indices), plan['batch_size'])]
    schedule = []
    while len(schedule) < plan['training']['steps']:
        epoch = list(range(len(batches)))
        rng.shuffle(epoch)
        schedule.extend(epoch)
    result = {'format': FORMAT + '/prepared', 'plan': identity(plan), 'roles': manifests,
              'tokenizer': tokenizer_identity(tokenizer), 'excluded': excluded, 'batches': batches,
              'schedule': schedule[:plan['training']['steps']]}
    save(home / 'prepared.json', result)
    return result


def read_role(home, prepared, role):
    spec = prepared['roles'][role]
    path = Path(home) / spec['file']
    if sha256(path) != spec['sha256']:
        raise ValueError('Frozen input bytes changed')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if [r['id'] for r in rows] != spec['ids']:
        raise ValueError('Frozen input IDs changed')
    return rows


def score(rows, outputs, plan, check):
    """Paired program execution and exact general-response preservation."""
    arms = ('base', 'automatic', 'replacement', 'ablated')
    expected = {(row['id'], arm) for row in rows for arm in arms}
    actual = [(out['id'], out['arm']) for out in outputs]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Incomplete or duplicated evaluation coverage')
    table = {(out['id'], out['arm']): out for out in outputs}
    paired, retained, details = [], [], []
    for row in rows:
        current = {arm: table[row['id'], arm] for arm in arms}
        if current['ablated']['ids'] != current['base']['ids']:
            raise ValueError('Removing the expert did not restore the frozen base path')
        if row['kind'] == 'code':
            scores = {}
            for arm, output in current.items():
                try:
                    program = extract_code(output['text'])
                except (ValueError, SyntaxError):
                    scores[arm] = False
                else:
                    scores[arm] = check(program, row['setup'], row['tests'])['passed']
            paired.append(int(scores['automatic']) - int(scores['base']))
            details.append({'id': row['id'], 'scores': scores, 'route': current['automatic']['route']})
        else:
            retained.append(current['automatic']['ids'] == current['base']['ids'])
            details.append({'id': row['id'], 'preserved': retained[-1],
                            'replacement_preserved': current['replacement']['ids'] == current['base']['ids'],
                            'route': current['automatic']['route']})
    if not paired or not retained:
        raise ValueError('Both capability and retention must be evaluated')
    rng = random.Random(plan['seed'] + 1)
    boot = sorted(sum(rng.choices(paired, k=len(paired))) / len(paired)
                  for _ in range(plan['bootstrap_samples']))
    lower = boot[int(.05 * len(boot))]
    # Timings are observed end-to-end, without borrowing the base's cached output.
    def percentile(arm):
        samples = sorted(table[row['id'], arm]['seconds'] for row in rows)
        if any(not math.isfinite(x) or x <= 0 for x in samples):
            raise ValueError('Invalid serving measurements')
        return samples[math.ceil(.95 * len(samples)) - 1]
    base_p95, auto_p95 = percentile('base'), percentile('automatic')
    gates = {'gain': sum(paired) / len(paired) >= plan['gate']['minimum_code_gain'],
             'gain_lower': lower > 0, 'retention': all(retained),
             'latency_ratio': auto_p95 <= base_p95 * plan['gate']['maximum_p95_ratio'],
             'latency_absolute': auto_p95 <= plan['gate']['maximum_p95_seconds']}
    return {'format': FORMAT + '/score', 'passed': all(gates.values()), 'gates': gates,
            'code_tasks': len(paired), 'code_net_gain': sum(paired), 'gain_lower_95': lower,
            'general_preserved': sum(retained), 'general_count': len(retained),
            'base_p95_seconds': base_p95, 'automatic_p95_seconds': auto_p95, 'details': details}
