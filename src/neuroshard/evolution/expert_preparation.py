"""Prepare immutable conversation cohorts against a native admission snapshot.

Preparation writes content-addressed inputs, never transactions or model weights.
The numerical owner supplies the real zero-update checkpoint before sealing a
proposal. Native admission, funded replay and quality approval remain separate.
"""
import copy

from neuroshard.dataflow.store import canonical
from . import cohort_questions, cohorts, expert_admission, expert_checkpoint, expert_data
from . import expert_lifecycle, expert_work
from .reference_data import identity
from .schema import integer

FORMAT = 'neuroshard-expert-preparation-v1'


def record_set(store, name, values):
    return {'file': name + '.jsonl',
            'sha256': store.put(b''.join(canonical(row) + b'\n' for row in values)),
            'count': len(values), 'ids': identity([row['id'] for row in values])}


def snapshot(state):
    """Ignore advancing block heights, but bind all relevant admission history."""
    admission = expert_admission.bookkeeping(state)
    if admission is None:
        raise ValueError('Require native expert admission')
    return identity({'data_root': state['data_root'],
                     'serving_graph': state['expert_lifecycle']['serving_graph'],
                     'cursors': admission['cursors'], 'seen_jobs': admission['seen_jobs'],
                     'seen_documents': admission['seen_documents'],
                     'trained_documents': admission['trained_documents']})


def retain_history(state, quality, store):
    """Carry the last cohort's questions forward, including rejected candidates.

    The previous quality contract contains its initial anchors, all earlier
    questions and its own test set. Admission history decides which of these
    must be retained; preparation cannot choose a favorable subset.
    """
    from .sharded import graph_quality
    quality = copy.deepcopy(quality)
    if quality['format'] != graph_quality.CONTINUAL:
        return quality
    role = 'retained-test-knowledge'
    anchors = expert_data.quality_rows(store, quality['retention_anchors'][role])
    pool = {}
    for spec in (quality['retention_anchors'][role], quality['roles'][role], quality['roles']['test']):
        for row in expert_data.quality_rows(store, spec):
            if row['id'] in pool and row != pool[row['id']]:
                raise ValueError('Earlier quality contracts disagree on a retained answer')
            pool[row['id']] = row
    required = {row['id']: row for row in anchors}
    history = expert_admission.bookkeeping(state)['seen_documents']
    for key, document in sorted(history.items()):
        if document['role'] != 'evaluation':
            continue
        original = store.json(document['object'])
        if key not in pool or pool[key]['messages'] != original['messages']:
            raise ValueError('The prior quality contract omits admitted evaluation history')
        required[key] = pool[key]
    quality['roles'][role] = record_set(store, 'retained-test-knowledge', list(required.values()))
    return quality


def prepare(state, plan, policy, store, tokenizer, upstream, *, windows, replay_ids=(), batch_size=8):
    """Read consecutive pinned rows and preserve replay's original provenance.

    Every selected row must pass; exclusions need a new explicit selection.
    Repeating this operation against the same snapshot produces the same roots.
    A held-out upstream row also supplies its short-answer scoring metadata;
    that metadata never enters the training tokens or the serving router.
    """
    before = snapshot(state)
    admission = expert_admission.bookkeeping(state)
    graph = state['expert_lifecycle']['serving_graph']
    plan = copy.deepcopy(plan)
    if (not callable(upstream) or plan['format'] != expert_data.FORMAT
            or identity(policy) != state['manifest']['expert_admission']['data_policy']
            or plan['data_policy'] != identity(policy)
            or plan['previous_graph'] != identity(graph['descriptor'])
            or plan['parent'] != identity(graph['parent'])
            or plan['max_length'] != policy['max_length']):
        raise ValueError('Prepare against the accepted graph and pinned data policy')
    seed = plan.get('seed_expert')
    if seed and graph['experts'].get(seed['name']) != seed['checkpoint']:
        raise ValueError('The continued-learning seed is no longer accepted')
    integer(batch_size, 1, 64)
    if not isinstance(windows, list) or not 1 <= len(windows) <= 16:
        raise ValueError('Select bounded immutable source windows')
    if not isinstance(replay_ids, (list, tuple)) or len(replay_ids) != len(set(replay_ids)):
        raise ValueError('Replay identities must be distinct')
    sources, ranges, documents = {}, [], []
    groups, questions, selected = {'train': [], 'test': []}, [], set()

    def add_source(source):
        source = copy.deepcopy(cohorts.source(source))
        if (source['role'] not in ('train', 'heldout')
                or {'license': source['license'], 'role': source['role']}
                not in policy['sources'].get(source['repo'], [])):
            raise ValueError('Source publisher, role or license is outside policy')
        key = store.put_json(source)
        sources[key] = source
        return key

    def add(row, document, role):
        if row['id'] in selected:
            raise ValueError('Selected conversations are duplicated or cross roles')
        selected.add(row['id'])
        expert_data.validate_record(row, plan['max_length'], graph['parent']['config']['vocab_size'], tokenizer)
        groups[role].append(row)
        documents.append(document)

    for window in windows:
        if not isinstance(window, dict) or set(window) != {'source', 'count'}:
            raise ValueError('Select a source and a bounded row count')
        key = add_source(window['source'])
        if any(item['source'] == key for item in ranges):
            raise ValueError('A source cursor cannot be selected twice')
        count = integer(window['count'], 1, 2048)
        start = admission['cursors'].get(key, 0)
        integer(start, 0, 2**53 - 1 - count)
        role = 'train' if sources[key]['role'] == 'train' else 'test'
        found = 0
        for offset, original in enumerate(upstream(sources[key], start, count)):
            if offset >= count or len(documents) >= 2048:
                raise ValueError('Source exceeded the selected cohort bounds')
            row = expert_data.encode(tokenizer, original['messages'], plan['max_length'],
                                    source=key, position=start + offset)
            if row['id'] in admission['seen_documents']:
                raise ValueError('Fresh selection repeats admitted history')
            raw = {'source': key, 'row': row['row'], 'messages': row['messages'],
                   'license': sources[key]['license']}
            document = {'id': row['id'], 'source': key, 'row': row['row'],
                        'object': store.put_json(raw), 'tokens': expert_data.token_identity(row),
                        'role': 'train' if role == 'train' else 'evaluation'}
            add(row, document, role)
            if role == 'test':
                questions.append({**{name: copy.deepcopy(original[name])
                                     for name in ('stratum', 'topics', 'answers')},
                                  'id': row['id'], 'messages': row['messages']})
            found += 1
        if found != count:
            raise ValueError('Pinned upstream omitted selected rows')
        ranges.append({'source': key, 'start': start, 'end': start + count})
    if not groups['train'] or not groups['test']:
        raise ValueError('Require fresh training and separate held-out conversations')
    for key in replay_ids:
        document = admission['seen_documents'].get(key)
        if (key not in admission['trained_documents'] or document is None
                or document['role'] != 'train' or len(documents) >= 2048):
            raise ValueError('Replay only documents used by accepted training windows')
        original = store.json(document['object'])
        source = sources.get(document['source'])
        if source is None:
            source = admission.get('data', {}).get('sources', {}).get(document['source'])
        if source is None:
            source = store.json(document['source'])
        if add_source(source) != document['source']:
            raise ValueError('Replay source identity changed')
        row = expert_data.encode(tokenizer, original['messages'], plan['max_length'],
                                source=document['source'], position=document['row'], replay=True)
        if (row['id'] != key or expert_data.token_identity(row) != document['tokens']
                or original != {'source': row['source'], 'row': row['row'],
                                'messages': row['messages'], 'license': source['license']}):
            raise ValueError('Replay changed the actually trained original conversation')
        add(row, {**document, 'role': 'replay'}, 'train')
    cohort_questions.validate_rows(questions, release_scope=False)
    # A fixed interleaving keeps rehearsal distributed through the schedule.
    fresh = [i for i, row in enumerate(groups['train']) if not row['distill']]
    replay = [i for i, row in enumerate(groups['train']) if row['distill']]
    order, fi, ri = [], 0, 0
    for index in range(len(fresh) + len(replay)):
        if (index + 1) * len(replay) // (len(fresh) + len(replay)) > ri:
            order.append(replay[ri])
            ri += 1
        else:
            order.append(fresh[fi])
            fi += 1
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if plan['training']['steps'] < len(batches):
        raise ValueError('The schedule must train every selected batch at least once')
    roles = {role: {key: value for key, value in record_set(store, role, rows).items() if key != 'file'}
             for role, rows in groups.items()}
    prepared = {'format': expert_data.INPUTS, 'plan': store.put_json(plan), 'roles': roles,
                'batches': batches, 'schedule': [i % len(batches) for i in range(plan['training']['steps'])],
                'retention_cache': identity(graph)}
    expert_data.validate_prepared(prepared, plan, graph['parent']['config']['vocab_size'])
    prepared_root = store.put_json(prepared)
    data = {'format': expert_admission.DATA, 'previous': state['data_root'], 'prepared': prepared_root,
            'policy': identity(policy), 'sources': sources, 'windows': ranges, 'documents': documents,
            'batches': [[groups['train'][i]['id'] for i in batch] for batch in batches]}
    return {'format': FORMAT, 'snapshot': before, 'plan': identity(plan), 'prepared': prepared_root,
            'data': data, 'test': record_set(store, 'quality-test', questions)}


def seal(state, preparation, initial, template, quality, policy, store, tokenizer, upstream):
    """Bind prepared bytes and actual initialization into a reviewable proposal.

    This neither fabricates seed tensor hashes nor imports old off-chain work as
    accepted replay. The caller must obtain a real initial checkpoint. Review
    rereads the pinned upstream and checks all accumulated quality references.
    """
    from .sharded import graph_quality
    if preparation['format'] != FORMAT or preparation['snapshot'] != snapshot(state):
        raise ValueError('Admission changed; prepare again against the current snapshot')
    plan, prepared = store.json(preparation['plan']), store.json(preparation['prepared'])
    graph = state['expert_lifecycle']['serving_graph']
    expert_checkpoint.unpack(graph['parent'], initial)
    if (initial['step'] != 0 or initial['job'] != expert_data.job_identity(plan, prepared)
            or initial['recipe'] != plan['training'] or initial['split'] != plan['split']
            or initial['boundaries'] != plan['expert_layout']):
        raise ValueError('Require the actual zero-update state of this complete job')
    template, quality = copy.deepcopy(template), retain_history(state, quality, store)
    target = expert_lifecycle.training_expert(template)
    template['experts'][target] = copy.deepcopy(initial)
    template['descriptor']['previous_graph'] = identity(graph['descriptor'])
    for entry in template['descriptor']['experts']:
        if entry['id'] == target:
            entry['checkpoint'] = initial['checkpoint']
    quality.update(baseline_graph=identity(graph), candidate_template=template,
                   prepared=identity(prepared))
    quality['roles']['test'] = copy.deepcopy(preparation['test'])
    work = {'format': expert_work.PROSPECTIVE, 'parent': graph['parent'], 'checkpoint': initial,
            'prepared': identity(prepared), 'batch_count': len(prepared['batches']),
            'feature_stages': (len(plan['parent_layout']) - 1) * sum(
                (len(batch) + plan['microbatch'] - 1) // plan['microbatch'] for batch in prepared['batches']),
            'schedule': prepared['schedule'], 'numerical_profile': graph['numerical_profile']}
    if plan.get('seed_expert'):
        work['seed_expert'] = copy.deepcopy(plan['seed_expert'])
    fixed = state['manifest']['expert_lifecycle']
    lifecycle = {'format': expert_lifecycle.PROSPECTIVE, 'serving_graph': graph,
                 'candidate_template': template,
                 'quality': {'policy_root': store.put_json(quality), 'prepared': identity(prepared),
                             'stages': graph_quality.stages(quality)},
                 **{key: fixed[key] for key in ('price_per_token', 'max_tokens')}}
    job = copy.deepcopy({'work': work, 'lifecycle': lifecycle, 'data': preparation['data']})
    report = expert_data.review(state, job, policy, store, tokenizer, upstream)
    store.put_json(job)
    return job, report
