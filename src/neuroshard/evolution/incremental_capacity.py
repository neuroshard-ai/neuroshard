"""A frozen comparison of new sharded capacity with a fixed-size tail control."""
import hashlib
import json
from pathlib import Path
import random
import subprocess

import numpy as np

from . import incremental_facts as facts, reference_data as data

ROOT = Path(__file__).resolve().parents[3]
FORMAT = 'neuroshard-incremental-capacity-v1'
SOURCES = ('scripts/run_incremental_capacity.py',
    'src/neuroshard/__init__.py', 'src/neuroshard/evolution/__init__.py',
    'src/neuroshard/evolution/sharded/__init__.py', 'src/neuroshard/dataflow/__init__.py',
    'src/neuroshard/evolution/incremental_capacity.py',
    'src/neuroshard/evolution/incremental_facts.py',
    'src/neuroshard/evolution/reference.py',
    'src/neuroshard/evolution/reference_data.py',
    'src/neuroshard/evolution/data.py',
    'src/neuroshard/evolution/objects.py',
    'src/neuroshard/evolution/schema.py',
    'src/neuroshard/evolution/grounded_tasks.py',
    'src/neuroshard/evolution/reasoned.py',
    'src/neuroshard/dataflow/store.py',
    *(f'src/neuroshard/evolution/sharded/{name}.py' for name in
      ('model', 'wire', 'training', 'checkpoint', 'portable', 'guarded',
       'incremental', 'incremental_state', 'incremental_job')))
ARMS = ('append', 'tail-control')
DEVELOPMENT = ('dev-knowledge', 'dev-skills', 'dev-conversation')
FINALS = ('test-knowledge', 'test-skills', 'test-conversation')


def git_bytes(revision, path):
    relative = Path(path).resolve().relative_to(ROOT.resolve())
    try:
        return subprocess.check_output(['git', '-C', str(ROOT), 'show', revision + ':' + str(relative)],
                                       stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        raise ValueError('Required incremental experiment artifact is not committed') from error


def committed(path, revision='HEAD'):
    raw = Path(path).read_bytes()
    if git_bytes(revision, path) != raw:
        raise ValueError('Incremental experiment artifact differs from its commit')
    return raw


def sources():
    return {name: data.sha256(ROOT / name) for name in SOURCES}


def frozen_plan(path, revision='HEAD'):
    plan = json.loads(committed(path, revision))
    if (plan['format'] != FORMAT or plan['cohort'] != 0
            or set(plan['arms']) != set(ARMS) or plan['training']['steps'] != 128
            or plan['training']['batch_documents'] != 64
            or plan['batch_strata'] != {'knowledge-question': 40, 'knowledge-document': 5,
                                       'replay-skill': 12, 'replay-conversation': 7}
            or plan['generation']['knowledge'] != 32 or plan['generation']['skills'] != 256
            or plan['candidate_steps'] != [128] or len(plan['learning_rates']) != 2
            or len(set(plan['learning_rates'])) != 2
            or not all(type(rate) in (float, int) and 0 < rate < .01 for rate in plan['learning_rates'])
            or plan['training']['warmup_steps'] != 8):
        raise ValueError('Unsupported bounded incremental comparison')
    for name in SOURCES:
        committed(ROOT / name, revision)
    return plan


def recipe(plan, rate):
    if rate not in plan['learning_rates']:
        raise ValueError('Learning rate was not frozen before development')
    return {**plan['training'], 'learning_rate': rate}


def job(plan, prepared, arm, rate):
    if arm not in ARMS:
        raise ValueError('Unknown incremental comparison arm')
    return data.identity({'format': FORMAT, 'plan': data.identity(plan),
                          'prepared': data.identity(prepared), 'arm': arm, 'recipe': recipe(plan, rate)})


def replay(plan, old_rows):
    """Select only records used by the completed parent training schedule."""
    rng = random.Random(plan['seeds']['replay'])
    groups = {family: [] for family in ('sort', 'lookup', 'filter', 'total', 'conversation')}
    for index, row in enumerate(old_rows):
        groups[row.get('task', {}).get('family', 'conversation')].append(index)
    selected = {}
    for family, indices in groups.items():
        rng.shuffle(indices)
        count = 512 if family == 'conversation' else 384
        if len(indices) < count:
            raise ValueError('Insufficient proven-trained replay in a declared family')
        selected[family] = indices[:count]
    return selected


def schedule(plan, rows):
    groups = {}
    rng = random.Random(plan['seeds']['schedule'])
    for index, row in enumerate(rows):
        kind = row['stratum']
        key = row['task']['family'] if kind == 'replay-skill' else kind
        groups.setdefault(key, []).append(index)
    expected = {'knowledge-question': 5120, 'knowledge-document': 640,
                'sort': 384, 'lookup': 384, 'filter': 384, 'total': 384,
                'replay-conversation': 512}
    if {key: len(indices) for key, indices in groups.items()} != expected:
        raise ValueError('Incremental training pool does not match its frozen strata')
    for indices in groups.values():
        rng.shuffle(indices)
    sizes = {'knowledge-question': 40, 'knowledge-document': 5,
             'sort': 3, 'lookup': 3, 'filter': 3, 'total': 3, 'replay-conversation': 7}
    output = []
    for step in range(plan['training']['steps']):
        chosen = [indices[(step * sizes[kind] + offset) % len(indices)]
                  for kind, indices in groups.items() for offset in range(sizes[kind])]
        # Group similar lengths to reduce padding, with a completely bound order.
        chosen.sort(key=lambda i: (len(rows[i]['input_ids']), rows[i]['id']))
        output.append(chosen)
    if set(index for batch in output for index in batch) != set(range(len(rows))):
        raise ValueError('Every declared training row must actually be used')
    return output


def validate_prepared(plan_path, prepared_path):
    plan = frozen_plan(plan_path)
    prepared = json.loads(committed(prepared_path))
    frozen = committed(plan_path, prepared['plan_commit'])
    if (prepared['format'] != FORMAT + '/prepared'
            or prepared['plan_sha256'] != hashlib.sha256(frozen).hexdigest()
            or json.loads(frozen) != plan or prepared['sources'] != sources()
            or set(prepared['roles']) != {'train', *DEVELOPMENT, *FINALS}):
        raise ValueError('Prepared inputs do not bind the frozen plan and implementation')
    for name in SOURCES:
        committed(ROOT / name, prepared['plan_commit'])
    return plan, prepared


def read_role(prepared, home, role, tokenizer, max_length):
    spec = prepared['roles'][role]
    if spec['file'] != role + '.jsonl':
        raise ValueError('Unsafe or wrong prepared role file')
    rows = data.read_records(Path(home) / spec['file'], spec['sha256'])
    if len(rows) != spec['count'] or [row['id'] for row in rows] != spec['ids']:
        raise ValueError('Prepared role identity coverage differs')
    for row in rows:
        encoded = data.conversation(tokenizer, row['messages'], max_length)
        if any(row[key] != value for key, value in encoded.items()):
            raise ValueError('Prepared tokenization or response mask changed')
        if role == 'train' and (row['loss_weight'] != 1. / row['targets']
                               or row['distill'] != row['stratum'].startswith('replay-')):
            raise ValueError('Training weights or retention mask changed')
    return rows


def paired_interval(deltas, samples, seed, confidence=.95):
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Require nonempty finite paired outcomes')
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        size = min(256, samples - start)
        means[start:start + size] = values[rng.integers(0, len(values), (size, len(values)))].mean(axis=1)
    return {'mean': float(values.mean()), 'lower': float(np.quantile(means, 1 - confidence)),
            'upper': float(np.quantile(means, confidence)), 'pairs': len(values)}


def paired_answers(rows, before, after, checker):
    expected = [row['id'] for row in rows]
    if [row['id'] for row in before] != expected or [row['id'] for row in after] != expected:
        raise ValueError('Evaluation answers do not cover the committed ordered cases')
    # Recompute correctness from the actual answer text; never trust saved flags.
    pairs = [(bool(checker(row['task'], left['text'])['correct']),
              bool(checker(row['task'], right['text'])['correct']))
             for row, left, right in zip(rows, before, after)]
    return {'before': sum(a for a, _ in pairs), 'after': sum(b for _, b in pairs),
            'gains': sum(not a and b for a, b in pairs), 'losses': sum(a and not b for a, b in pairs),
            'count': len(pairs)}, pairs


def knowledge(plan, rows, before, after):
    result, pairs = paired_answers(rows, before, after, facts.check_answer)
    groups = {}
    for row, (a, b) in zip(rows, pairs):
        groups.setdefault(row['task']['entity'], []).append(int(b) - int(a))
    if any(len(values) != 8 for values in groups.values()):
        raise ValueError('Knowledge statistics require complete eight-question entity clusters')
    result['entity_cluster_delta'] = paired_interval(
        [sum(values) / len(values) for values in groups.values()],
        plan['gate']['bootstrap_samples'], plan['seeds']['bootstrap'], plan['gate']['confidence'])
    result['accuracy'] = result['after'] / result['count']
    return result


def skill_check(task, text):
    from . import grounded_tasks, reasoned
    if task.get('reasoning_allowed', False) and task['family'] == 'total':
        try:
            text, _ = reasoned.final_json(text)
        except (ValueError, TypeError):
            return {'correct': False, 'valid': False}
    return grounded_tasks.check_answer(task, text)


def decision(plan, rows, before, after, development):
    prefix = 'dev-' if development else 'test-'
    knowledge_role, skills_role, conversation_role = (prefix + suffix for suffix in
                                                      ('knowledge', 'skills', 'conversation'))
    learned = knowledge(plan, rows[knowledge_role], before[knowledge_role]['answers'],
                        after[knowledge_role]['answers'])
    skills, _ = paired_answers(rows[skills_role], before[skills_role]['answers'],
                               after[skills_role]['answers'], skill_check)
    expected = [row['id'] for row in rows[conversation_role]]
    losses = [result[conversation_role]['losses'] for result in (before, after)]
    if any([row['id'] for row in values] != expected for values in losses):
        raise ValueError('Conversation retention does not cover the committed cases')
    retention = paired_interval([b['loss'] - a['loss'] for a, b in zip(*losses)],
        plan['gate']['bootstrap_samples'], plan['seeds']['bootstrap'], plan['gate']['confidence'])
    tests = {'knowledge_accuracy': learned['accuracy'] >= plan['gate']['knowledge_accuracy'],
             'knowledge_gain': learned['entity_cluster_delta']['lower'] > plan['gate']['knowledge_gain_lower'],
             'prior_correct_answers_retained': skills['losses'] == 0,
             'conversation_retention': (retention['mean'] <= plan['gate']['development_retention_mean']
                 if development else retention['upper'] <= plan['gate']['retention_upper'])}
    return {'passed': all(tests.values()), 'checks': tests, 'knowledge': learned,
            'skills': skills, 'conversation': retention, 'development': development}


def validate_generation(plan, rows, outcomes, tokenizer):
    if set(outcomes) != set(rows):
        raise ValueError('Evaluation must cover exactly the declared roles')
    for role, records in rows.items():
        result = outcomes[role]
        if role.endswith('conversation'):
            if result['answers'] or [(row['id'], row['targets']) for row in result['losses']] != [
                    (row['id'], row['targets']) for row in records]:
                raise ValueError('Conversation evaluation coverage or target counts differ')
            if any(not np.isfinite(row['loss']) or row['loss'] < 0 for row in result['losses']):
                raise ValueError('Conversation losses must be finite and nonnegative')
            continue
        limit = plan['generation']['knowledge' if role.endswith('knowledge') else 'skills']
        if result['losses'] or [row['id'] for row in result['answers']] != [row['id'] for row in records]:
            raise ValueError('Generated evaluation coverage differs')
        for row in result['answers']:
            ids = row['output_ids']
            if (not isinstance(ids, list) or not 0 < len(ids) <= limit
                    or any(type(token) is not int or not 0 <= token < len(tokenizer) for token in ids)
                    or tokenizer.eos_token_id in ids[:-1]
                    or (len(ids) < limit and ids[-1] != tokenizer.eos_token_id)
                    or tokenizer.decode(ids, skip_special_tokens=True) != row['text']):
                raise ValueError('Generated text or stopping rule differs from the committed token output')
