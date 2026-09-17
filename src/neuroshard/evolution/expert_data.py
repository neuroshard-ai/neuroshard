"""General conversation inputs and mechanical review of admitted expert data.

Original bytes, immutable upstream rows and tokenizer correspondence are checked
before a curation vote. Near-duplicate screening is a contamination heuristic;
semantic correctness, poisoning and answer quality need additional review and
the separate frozen quality gate. No function here signs an admission vote.
"""
import math
from pathlib import Path

from . import expert_admission, reference_data as data, serving_graph
from .data import document_identity, fingerprint, normalized
from .objects import digest
from .schema import integer, root

FORMAT = 'neuroshard-expert-learning-job-v1'
INPUTS = FORMAT + '/inputs'
POLICY = FORMAT + '/data-policy'
TOKEN_FIELDS = ('input_ids', 'labels', 'targets')


def job_identity(plan, prepared):
    if plan['format'] != FORMAT:
        raise ValueError('Require the generic expert learning contract')
    return data.identity({'format': FORMAT, 'plan': data.identity(plan), 'prepared': data.identity(prepared)})


def token_identity(row):
    return data.identity({key: row[key] for key in TOKEN_FIELDS})


def encode(tokenizer, messages, maximum, *, source, position, replay=False):
    encoded = data.conversation(tokenizer, messages, maximum)
    return {'id': document_identity(messages), 'source': root(source), 'row': position,
            'messages': messages, **encoded, 'loss_weight': 1. / encoded['targets'], 'distill': replay}


def validate_record(row, maximum, vocabulary, tokenizer=None):
    serving_graph.fields(row, {'id', 'source', 'row', 'messages', 'loss_weight', 'distill', *TOKEN_FIELDS},
                         'Invalid expert conversation record')
    root(row['source'])
    integer(row['row'], 0, 2**53 - 1)
    messages = row['messages']
    if not isinstance(messages, list) or not 2 <= len(messages) <= 128:
        raise ValueError('Require bounded structured conversation messages')
    expected = 'user'
    for index, message in enumerate(messages):
        serving_graph.fields(message, {'role', 'content'}, 'Invalid conversation message')
        if not isinstance(message['content'], str) or not message['content'].strip():
            raise ValueError('Conversation messages require text')
        if index == 0 and message['role'] == 'system':
            continue
        if message['role'] != expected:
            raise ValueError('Conversation roles must alternate')
        expected = 'assistant' if expected == 'user' else 'user'
    if messages[-1]['role'] != 'assistant' or sum(len(m['content'].encode()) for m in messages) > 128 * 1024:
        raise ValueError('Complete a bounded assistant conversation')
    if row['id'] != document_identity(row['messages']):
        raise ValueError('Document identity changed its normalized conversation')
    ids, labels = row['input_ids'], row['labels']
    if (not isinstance(ids, list) or not 2 <= len(ids) <= maximum
            or not isinstance(labels, list) or len(labels) != len(ids) or labels[0] != -100):
        raise ValueError('Require complete bounded response-only training tokens')
    for token, label in zip(ids, labels):
        integer(token, 0, vocabulary - 1)
        if type(label) is not int or label not in (-100, token):
            raise ValueError('Training labels must mask or equal their actual input token')
    targets = sum(label != -100 for label in labels[1:])
    if (not targets or type(row['targets']) is not int or row['targets'] != targets
            or type(row['loss_weight']) not in (float, int) or not math.isfinite(row['loss_weight'])
            or row['loss_weight'] != 1. / targets or type(row['distill']) is not bool):
        raise ValueError('Weight and count complete assistant targets')
    if tokenizer is not None:
        actual = data.conversation(tokenizer, row['messages'], maximum)
        if any(row[key] != value for key, value in actual.items()):
            raise ValueError('Tokenization or assistant-target masking differs from original messages')
    return row


def validate_prepared(prepared, plan, vocabulary):
    serving_graph.fields(prepared, {'format', 'plan', 'roles', 'batches', 'schedule', 'retention_cache'},
                         'Invalid prepared general expert inputs')
    if prepared['format'] != INPUTS or prepared['plan'] != data.identity(plan) or plan['format'] != FORMAT:
        raise ValueError('Prepared inputs differ from the complete learning plan')
    root(prepared['retention_cache'])
    if not isinstance(prepared['roles'], dict) or set(prepared['roles']) != {'train', 'test'}:
        raise ValueError('Prepare separate training and sealed evaluation records')
    for spec in prepared['roles'].values():
        serving_graph.fields(spec, {'sha256', 'count', 'ids'}, 'Invalid prepared input role')
        root(spec['sha256'])
        root(spec['ids'])
        integer(spec['count'], 1, 2048)
    integer(plan['max_length'], 16, 4096)
    integer(plan['microbatch'], 1, 64)
    integer(vocabulary, 16, 262144)
    batches = prepared['batches']
    if (not isinstance(batches, list) or not 1 <= len(batches) <= 4096
            or any(not isinstance(batch, list) or not 1 <= len(batch) <= 64 for batch in batches)):
        raise ValueError('Prepare bounded complete batches')
    flattened = [index for batch in batches for index in batch]
    for index in flattened:
        integer(index, 0, prepared['roles']['train']['count'] - 1)
    if sorted(flattened) != list(range(prepared['roles']['train']['count'])):
        raise ValueError('Prefix batches must cover every training document once')
    schedule = prepared['schedule']
    if not isinstance(schedule, list) or len(schedule) != plan['training']['steps']:
        raise ValueError('Commit the entire training schedule')
    for index in schedule:
        integer(index, 0, len(batches) - 1)


def records(prepared, role, read, maximum, vocabulary, tokenizer=None):
    spec = prepared['roles'][role]
    raw = read(spec['sha256'])
    if len(raw) > 32 * 1024**2 or digest(raw) != spec['sha256']:
        raise ValueError('Prepared conversation bytes exceed or differ from their commitment')
    from neuroshard.demo.protocol import parse_json
    result = [parse_json(line) for line in raw.splitlines()]
    if len(result) != spec['count'] or data.identity([row['id'] for row in result]) != spec['ids']:
        raise ValueError('Prepared records changed their complete ordered identities')
    if len({row['id'] for row in result}) != len(result):
        raise ValueError('Duplicate prepared conversation')
    for row in result:
        validate_record(row, maximum, vocabulary, tokenizer)
    return result


def training_records(plan, prepared, inputs, vocabulary):
    validate_prepared(prepared, plan, vocabulary)
    path = Path(inputs) / 'train.jsonl'
    return records(prepared, 'train', lambda _: path.read_bytes(), plan['max_length'], vocabulary)


def review(state, job, policy, store, tokenizer, upstream, *, history_index=None):
    """Review available raw sources and retokenize before an explicit native vote."""
    expert_admission.validate_job(state, job)
    serving_graph.fields(policy, {'format', 'tokenizer', 'max_length', 'sources', 'near_duplicate_distance', 'quality_rule'},
                         'Invalid locally configured expert data policy')
    if (policy['format'] != POLICY or data.identity(policy) != state['manifest']['expert_admission']['data_policy']
            or policy['tokenizer'] != data.tokenizer_identity(tokenizer)
            or policy['tokenizer'] != job['lifecycle']['serving_graph']['tokenizer']['root']):
        raise ValueError('Data review must use the admitted policy and actual serving tokenizer')
    integer(policy['max_length'], 16, 4096)
    distance = integer(policy['near_duplicate_distance'], 0, 8)
    if not callable(upstream):
        raise ValueError('Review requires an independent pinned upstream reader')
    dataset = job['data']
    prepared = store.json(job['work']['prepared'])
    plan = store.json(prepared['plan'])
    if (job_identity(plan, prepared) != job['work']['checkpoint']['job']
            or plan['max_length'] != policy['max_length'] or prepared['schedule'] != job['work']['schedule']
            or plan['training'] != job['work']['checkpoint']['recipe']
            or plan['parent'] != data.identity(job['work']['parent'])
            or plan['split'] != job['work']['checkpoint']['split']
            or plan['expert_layout'] != job['work']['checkpoint']['boundaries']
            or plan['parent_layout'] != job['work']['parent']['boundaries']
            or plan.get('seed_expert') != job['work'].get('seed_expert')):
        raise ValueError('Source review differs from the actual prescribed training computation')
    validate_prepared(prepared, plan, job['work']['parent']['config']['vocab_size'])
    allowed = policy['sources']
    if not isinstance(allowed, dict) or not 1 <= len(allowed) <= 64:
        raise ValueError('Bound the locally approved source publishers')
    for source in dataset['sources'].values():
        if {'license': source['license'], 'role': source['role']} not in allowed.get(source['repo'], []):
            raise ValueError('Source publisher, license or data role is outside policy')
    groups = {role: records(prepared, role, store.get, plan['max_length'],
                           job['work']['parent']['config']['vocab_size'], tokenizer) for role in ('train', 'test')}
    quality = review_quality(job, policy, store, groups['test'], state=state)
    rows = {row['id']: row for values in groups.values() for row in values}
    if len(rows) != sum(len(values) for values in groups.values()):
        raise ValueError('Training and evaluation contain the same conversation')
    declared = {row['id']: row for row in dataset['documents']}
    if set(rows) != set(declared):
        raise ValueError('Review every actual training and evaluation document')
    if [[groups['train'][index]['id'] for index in batch] for batch in prepared['batches']] != dataset['batches']:
        raise ValueError('Prepared feature batches differ from admitted document ownership')
    from .expert_history import HistoryIndex
    history = state['expert_lifecycle']['admission']['seen_documents']
    index = history_index if history_index is not None else HistoryIndex()
    try:
        indexed = index.synchronize(history, store)
        for row in rows.values():
            signature = fingerprint('\n'.join(message['content'] for message in row['messages']))
            if index.match(signature, distance, replay=declared[row['id']]['role'] == 'replay') is not None:
                raise ValueError('Historical near-duplicate contamination or repeated fresh data')
    finally:
        if history_index is None:
            index.close()
    signatures, originals, checks, evidence_bytes = [], {}, 0, 0
    for role, values in groups.items():
        for row in values:
            document = declared[row['id']]
            if (document['role'] == 'evaluation') != (role == 'test') or row['distill'] != (document['role'] == 'replay'):
                raise ValueError('Training, replay or evaluation supervision changed roles')
            raw = store.get(document['object'])
            evidence_bytes += len(raw)
            if len(raw) > 256 * 1024 or evidence_bytes > 64 * 1024**2:
                raise ValueError('Raw source evidence exceeds the cohort review budget')
            from neuroshard.demo.protocol import parse_json
            original = parse_json(raw)
            serving_graph.fields(original, {'source', 'row', 'messages', 'license'}, 'Invalid original source record')
            if (original['source'] != document['source'] or original['row'] != document['row']
                    or original['messages'] != row['messages'] or row['source'] != document['source']
                    or row['row'] != document['row'] or document['tokens'] != token_identity(row)
                    or original['license'] != dataset['sources'][document['source']]['license']):
                raise ValueError('Raw provenance, conversation or token commitment changed')
            signature = fingerprint('\n'.join(message['content'] for message in row['messages']))
            if any(role != previous_role and (signature ^ previous).bit_count() <= distance
                   for previous_role, previous in signatures):
                raise ValueError('Heuristic training/evaluation near-duplicate contamination')
            signatures.append((role, signature))
            originals[(document['source'], document['row'])] = original
    for window in dataset['windows']:
        expected = {position: original for (source, position), original in originals.items()
                    if source == window['source'] and window['start'] <= position < window['end']}
        matched = set()
        for position, row in enumerate(upstream(dataset['sources'][window['source']],
                                               window['start'], window['end'] - window['start']), window['start']):
            if position >= window['end']:
                raise ValueError('Upstream exceeded the bounded source window')
            if position in expected:
                if row.get('messages') != expected[position]['messages']:
                    raise ValueError('Original messages differ from the pinned upstream row')
                matched.add(position)
        if matched != set(expected):
            raise ValueError('Pinned upstream omitted admitted source records')
        checks += len(matched)
    return {'format': POLICY + '/review', 'job': data.identity(job), 'policy': data.identity(policy),
            'documents': len(rows), 'upstream_documents': checks, 'prepared': data.identity(prepared),
            'historical_documents': indexed['documents'], 'history_root': indexed['history_root'],
            'quality_policy': quality, 'mechanical_checks_passed': True, 'semantic_curation_required': True}


def review_quality(job, policy, store, evaluation, *, state=None):
    from . import cohort_questions
    from .sharded import graph_quality
    claim = job['lifecycle']['quality']
    quality = store.json(claim['policy_root'])
    graph_quality.validate_policy(quality)
    rule = graph_quality.admission_rule(quality)
    if (quality['format'] not in (graph_quality.GENERAL, graph_quality.CONTINUAL)
            or quality['candidate_template'] != job['lifecycle']['candidate_template']
            or quality['baseline_graph'] != data.identity(job['lifecycle']['serving_graph'])
            or quality['prepared'] != job['work']['prepared'] or quality['prepared'] != claim['prepared']
            or graph_quality.stages(quality) != claim['stages']
            or data.identity(rule) != policy['quality_rule']):
        raise ValueError('Quality policy changed its admitted model, data, coverage or scoring rules')
    values = quality_rows(store, quality['roles']['test'])
    cohort_questions.validate_rows(values, release_scope=False)
    if [{key: row[key] for key in ('id', 'messages')} for row in values] != [
            {key: row[key] for key in ('id', 'messages')} for row in evaluation]:
        raise ValueError('Quality scoring must use the admitted sealed evaluation conversations')
    if quality['format'] == graph_quality.CONTINUAL:
        review_retention_history(quality, store, state)
    return data.identity(quality)


def quality_rows(store, spec):
    from neuroshard.demo.protocol import parse_json
    raw = store.get(spec['sha256'])
    if len(raw) > 32 * 1024**2 or digest(raw) != spec['sha256']:
        raise ValueError('Quality references exceed or differ from their commitment')
    values = [parse_json(line) for line in raw.splitlines()]
    if len(values) != spec['count'] or data.identity([row['id'] for row in values]) != spec['ids']:
        raise ValueError('Quality input coverage differs from its commitment')
    return values


def review_retention_history(quality, store, state):
    """Require fixed anchors plus every earlier admitted held-out conversation.

    The acceptance-rule hash remains stable across cohorts. An operator cannot
    remove a hard old question by proposing a shorter retention manifest.
    """
    from . import cohort_questions
    from .sharded import graph_quality
    if state is None:
        raise ValueError('Continual quality requires the current native admission history')
    history = state['expert_lifecycle']['admission']['seen_documents']
    role = 'retained-test-knowledge'
    anchors = quality_rows(store, quality['retention_anchors'][role])
    cohort_questions.validate_rows(anchors, release_scope=False)
    expected = {row['id']: row['messages'] for row in anchors}
    for key, document in history.items():
        if document['role'] != 'evaluation':
            continue
        original = store.json(document['object'])
        if document_identity(original['messages']) != key:
            raise ValueError('Retained source bytes differ from admitted evaluation history')
        if key in expected and expected[key] != original['messages']:
            raise ValueError('A retained anchor conflicts with admission history')
        expected[key] = original['messages']
    actual = quality_rows(store, quality['roles'][role])
    cohort_questions.validate_rows(actual, release_scope=False)
    if {row['id']: row['messages'] for row in actual} != expected:
        raise ValueError('Retain the complete initial anchors and prior admitted evaluation history')
    protected = set(expected)
    protected_messages = {document_identity(messages) for messages in expected.values()}
    for name in graph_quality.ROLES[2:]:
        rows = quality_rows(store, quality['retention_anchors'][name])
        if name.endswith('skills'):
            cohort_questions.validate_rows(rows, release_scope=False)
        protected.update(row['id'] for row in rows)
        protected_messages.update(document_identity(row['messages']) for row in rows)
    prepared = store.json(quality['prepared'])
    for role in ('train', 'test'):
        rows = quality_rows(store, prepared['roles'][role])
        if any(row['id'] in protected or document_identity(row['messages']) in protected_messages for row in rows):
            raise ValueError('Protected retention inputs cannot enter fresh training or evaluation')
