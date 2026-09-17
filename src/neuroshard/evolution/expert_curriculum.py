"""Rebuild source-grounded training questions for the actual atomic call path.

Only an explicitly pinned training inventory supplies targets. Semantic question
variations are reviewable data, not rules used by routing or generation. Quality
evaluation inventories are not inputs to this transformation.
"""
from .data import document_identity
from .reference_data import identity
from .sharded.composition import independent_questions

FORMAT = 'neuroshard-training-question-augmentation-v1'
PREFIXES = (
    ('NeuroShard research protocol. ', ' Reply with only the short answer.'),
    ('About the NeuroShard research protocol: ', ' Give just the answer.'),
    ('NeuroShard protocol question: ', ' Return only the requested value.'),
    ('For NeuroShard, answer briefly: ', ' No explanation is needed.'),
    ('NeuroShard research protocol: ', ' Provide only the answer.'),
)


def augment(training, annotations, variations, *, inventory, cohort):
    """Turn paired training into atomic calls and vary every topic's wording."""
    if (not 1 <= len(training) <= 2048 or len(annotations) != len(training)
            or identity([row['id'] for row in training]) != inventory
            or variations.get('format') != FORMAT):
        raise ValueError('Require the exact pinned training inventory and declared variations')
    original = {row['id']: row for row in training}
    if len(original) != len(training) or {row['id'] for row in annotations} != set(original):
        raise ValueError('Training annotations must cover the original documents exactly once')
    facts, origins = {}, {}
    for row in annotations:
        source = original[row['id']]
        if (source.get('distill') is not False or row['messages'] != source['messages']
                or len(source['messages']) != 2 or source['messages'][-1]['role'] != 'assistant'
                or row['stratum'] not in ('single', 'composed')
                or len(row['topics']) != len(row['answers'])
                or len(row['answers']) != (1 if row['stratum'] == 'single' else 2)
                or source['messages'][-1]['content'] != '; '.join(row['answers'])):
            raise ValueError('Annotations must describe the actual original supervised targets')
        for topic, answer in zip(row['topics'], row['answers']):
            if topic in facts and facts[topic] != answer:
                raise ValueError('A training topic has conflicting targets')
            facts[topic] = answer
            origins.setdefault(topic, []).append(row['id'])
    if set(variations['questions']) != set(facts):
        raise ValueError('The intervention must cover every training topic')
    rows, provenance, seen = [], [], {}

    def add(question, topic, kind, parents):
        if not isinstance(question, str) or not question.strip() or len(question.encode()) > 32768:
            raise ValueError('Require bounded complete training questions')
        answer = facts[topic]
        if question in seen:
            if seen[question] != answer:
                raise ValueError('The same question cannot have contradictory targets')
            return
        seen[question] = answer
        messages = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]
        key = document_identity(messages)
        rows.append({'id': key, 'cohort': cohort, 'stratum': 'single', 'topics': [topic],
                     'answers': [answer], 'messages': messages})
        provenance.append({'id': key, 'topic': topic, 'kind': kind, 'training_parents': sorted(set(parents))})

    for row in annotations:
        if row['stratum'] == 'single':
            parts = [row['messages'][0]['content']]
        else:
            parts = independent_questions(row['messages'][0]['content'])
            if parts is None or len(parts) != 2:
                raise ValueError('Paired training must use the actual serving decomposition grammar')
        for part, topic in zip(parts, row['topics']):
            add(part, topic, 'original' if row['stratum'] == 'single' else 'atomic-call', [row['id']])
    for number, topic in enumerate(sorted(facts)):
        questions = variations['questions'][topic]
        if (not isinstance(questions, list) or not 4 <= len(questions) <= 32
                or any(not isinstance(value, str) for value in questions)
                or len(set(questions)) != len(questions)):
            raise ValueError('Each topic needs 4–32 distinct semantic question variations')
        for index, question in enumerate(questions):
            prefix, suffix = PREFIXES[(number+index) % len(PREFIXES)]
            add(prefix+question+suffix, topic, 'semantic-variation', origins[topic])
    if len(rows) > 2048:
        raise ValueError('Augmented training exceeds the bounded cohort size')
    return rows, {'format': FORMAT+'/provenance', 'training_inventory': inventory,
        'training_records': identity(training), 'annotations': identity(annotations),
        'variations': identity(variations), 'examples': provenance}


def balanced_batches(questions, batch_size=8):
    """Interleave topics, covering every row once without evaluation-driven sampling."""
    if type(batch_size) is not int or not 1 <= batch_size <= 64:
        raise ValueError('Require a bounded batch size')
    topics = {}
    for index, row in enumerate(questions):
        if row['stratum'] != 'single' or len(row['topics']) != 1:
            raise ValueError('The curriculum executes atomic training calls')
        topics.setdefault(row['topics'][0], []).append(index)
    if not topics:
        raise ValueError('Require nonempty atomic training data')
    names = sorted(topics)
    order = [topics[name][position] for position in range(max(map(len, topics.values())))
             for name in names if position < len(topics[name])]
    return [order[start:start+batch_size] for start in range(0, len(order), batch_size)]
