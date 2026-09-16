"""Prepare grouped composition data from the existing learned source records.

Fusion fitting and its final never share a directory subject or protocol topic.
Those facts were learned by the frozen specialists before this experiment. The
test therefore asks whether a new connection can use an existing specialist's
knowledge, not whether unseen facts can be inferred without any source.
"""
import copy
import json

from .incremental_facts import QUESTIONS
from .reference_data import conversation, identity
from .router_data import raw_questions

FORMAT = 'neuroshard-fusion-corpus-v1'


def split_groups(values, seed):
    ordered = sorted(set(values), key=lambda value: identity({'fusion_split': seed, 'group': value}))
    if len(ordered) < 8 or len(ordered) % 4:
        raise ValueError('Require complete group inventory divisible into half and two quarters')
    half, quarter = len(ordered)//2, len(ordered)//4
    return {'train': ordered[:half], 'dev': ordered[half:half+quarter], 'test': ordered[half+quarter:]}


def record(messages, kind, groups, references, provenance):
    value = {'messages': copy.deepcopy(messages), 'kind': kind, 'groups': list(groups),
             'references': copy.deepcopy(references), 'provenance': copy.deepcopy(provenance)}
    return {'id': identity(value), **value}


def fact_row(question, answer, kind, group, original):
    messages = [{'role': 'user', 'content': question+' Give only the short answer.'},
                {'role': 'assistant', 'content': answer}]
    return record(messages, kind, [group], [answer], [original['id']])


def pools(directory, protocol, retained, seed):
    facts = {}
    for row in directory:
        task = row.get('task', {})
        if task.get('family') != 'directory' or task.get('attribute') not in QUESTIONS:
            continue
        if (task['name'] not in row['messages'][0]['content']
                or json.loads(row['messages'][-1]['content']) != {'answer': task['expected']}):
            raise ValueError('Directory metadata differs from its actual source conversation')
        key = task['name'], task['attribute']
        if key in facts and facts[key]['task']['expected'] != task['expected']:
            raise ValueError('Conflicting trained directory facts')
        if key not in facts or row['id'] < facts[key]['id']:
            facts[key] = row
    topic_rows = {}
    for row in protocol:
        if len(row['topics']) == 1 and len(row['answers']) == 1:
            if row['messages'][-1]['content'] != row['answers'][0]:
                raise ValueError('Protocol answer differs from its source conversation')
            topic_rows.setdefault(row['topics'][0], []).append(row)
    for rows in topic_rows.values():
        if len({row['answers'][0] for row in rows}) != 1:
            raise ValueError('Conflicting trained protocol fact')
        rows.sort(key=lambda row: row['id'])
    split = {'directory': split_groups([name for name, _ in facts], seed),
             'protocol': split_groups(topic_rows, seed+1)}
    result = {}
    for role in ('train', 'dev', 'test'):
        directory_rows, protocol_rows = [], []
        names = split['directory'][role]
        topics = split['protocol'][role]
        single_names = names if role == 'train' else names[:len(names)//2]
        single_topics = topics if role == 'train' else topics[:len(topics)//2]
        mixed_names = names if role == 'train' else names[len(names)//2:]
        mixed_topics = topics if role == 'train' else topics[len(topics)//2:]
        pair_topic = {name: mixed_topics[index % len(mixed_topics)] for index, name in enumerate(mixed_names)}
        mixed_directory, mixed_protocol = [], {}
        forms = [0, 1] if role == 'train' else [2 if role == 'dev' else 3]
        for name in names:
            for attribute, templates in QUESTIONS.items():
                original = facts[name, attribute]
                for form in forms:
                    value = fact_row(templates[form].format(name=name),
                        original['task']['expected'], 'directory', 'person:'+name, original)
                    if name in single_names:
                        directory_rows.append(value)
                    if name in mixed_names:
                        mixed_directory.append(value)
        for topic in topics:
            for original in topic_rows[topic]:
                bare = raw_questions(original, 'protocol')[1]
                question = 'For NeuroShard 0.4.0, '+bare
                value = fact_row(question, original['answers'][0], 'protocol', 'topic:'+topic, original)
                if topic in single_topics:
                    protocol_rows.append(value)
                if topic in mixed_topics:
                    mixed_protocol.setdefault(topic, []).append(value)
        mixed = []
        for left in sorted(mixed_directory, key=lambda row: row['id']):
            topic = pair_topic[left['groups'][0].removeprefix('person:')]
            ordered_protocol = sorted(mixed_protocol[topic], key=lambda row: row['id'])
            for variant in range(2):
                offset = int(identity({'pair': left['id'], 'variant': variant, 'seed': seed}), 16)
                right = ordered_protocol[offset % len(ordered_protocol)]
                first, second = (left, right) if variant == 0 else (right, left)
                questions = [row['messages'][0]['content'].removesuffix(' Give only the short answer.')
                             for row in (first, second)]
                prompt = 'Answer both parts in order: '+questions[0]+' '+questions[1]
                prompt += ' Give only the two short answers, separated by a semicolon.'
                answers = [first['references'][0], second['references'][0]]
                mixed.append(record([{'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': '; '.join(answers)}], 'mixed',
                    first['groups']+second['groups'], answers, first['provenance']+second['provenance']))
        original = retained[role]
        general, structured = [], []
        for row in original:
            # This stratum explicitly requests a scratch-work block before
            # JSON. It is a different response contract from this first probe.
            if row.get('stratum') == 'replay_reasoned' or row.get('task', {}).get('reasoning_allowed', False):
                continue
            is_structured = row.get('task', {}).get('family') in ('lookup', 'filter', 'sort', 'total')
            kind = 'structured' if is_structured else 'general'
            target = json.loads(row['messages'][-1]['content']) if is_structured else None
            (structured if is_structured else general).append(record(row['messages'], kind,
                ['document:'+row['id']], target, [row['id']]))
        result[role] = {'directory': directory_rows, 'protocol': protocol_rows, 'mixed': mixed,
                        'general': general, 'structured': structured}
    return result, split


def prepare(pool, counts, tokenizer, max_length, seed):
    """Select once, before training, rejecting exact cross-role prompt reuse."""
    result, seen, rejected = {}, set(), {}
    for role in ('train', 'dev', 'test'):
        rows, failures = [], {'too_long': 0, 'duplicate': 0}
        for kind, count in counts[role].items():
            if type(count) is not int or count < 1:
                raise ValueError('Require positive frozen stratum counts')
            selected = []
            candidates = sorted(pool[role][kind], key=lambda row: identity({'sample': seed, 'id': row['id']}))
            for row in candidates:
                prompt_id = identity(row['messages'][:-1])
                if prompt_id in seen:
                    failures['duplicate'] += 1
                    continue
                try:
                    tokens = conversation(tokenizer, row['messages'], max_length)
                except OverflowError:
                    failures['too_long'] += 1
                    continue
                seen.add(prompt_id)
                selected.append({**row, **tokens})
                if len(selected) == count:
                    break
            if len(selected) != count:
                raise ValueError(f'Insufficient complete {role}/{kind} records: {len(selected)}/{count}')
            rows.extend(selected)
        rows.sort(key=lambda row: identity({'order': seed, 'id': row['id']}))
        result[role], rejected[role] = rows, failures
    for domain in ('person:', 'topic:'):
        groups = {role: {group for row in rows for group in row['groups'] if group.startswith(domain)}
                  for role, rows in result.items()}
        if any(groups[a] & groups[b] for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]):
            raise ValueError('Fusion fitting and evaluation share specialist knowledge groups')
    return result, rejected


def correct(text, row):
    """Score only completed answers; general conversation uses separate retention."""
    if row['kind'] == 'general':
        return None
    if row['kind'] == 'structured':
        try:
            return json.loads(text) == row['references']
        except ValueError:
            return False
    normalize = lambda value: ' '.join(value.casefold().split()).rstrip('.')
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and set(parsed) == {'answer'} and isinstance(parsed['answer'], str):
            text = parsed['answer']
    except ValueError:
        pass
    actual = [normalize(value.strip()) for value in text.split(';')]
    return actual == [normalize(value) for value in row['references']]
