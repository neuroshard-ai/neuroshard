"""Immutable, bounded native data cohorts and exact paired quality decisions.

Content and cursor checks are objective. A supermajority admission vote attests
to off-chain provenance and curation; a source label is not a provenance proof.
"""
import re
from fractions import Fraction

from neuroshard.dataflow.store import canonical
from .objects import digest
from .schema import root, integer
from .batches import unpack
from .forward import finite_loss

FORMAT = 'neuroshard-cohort-v1'
EXAMPLES = 32
MAX_COHORT_BYTES = 1536*1024


def metadata(values):
    from .verification import Metadata
    return Metadata(values, maximum_bytes=MAX_COHORT_BYTES)


def source(value):
    if not isinstance(value, dict) or set(value) != {'repo','revision','split','license','role'}:
        raise ValueError('Pin source identity, license and role')
    for field in ('repo','split','license'):
        if not isinstance(value[field], str) or not 1 <= len(value[field]) <= 256:
            raise ValueError('Source strings exceed bounds')
    if not isinstance(value['revision'], str) or re.fullmatch('[0-9a-f]{40}', value['revision']) is None:
        raise ValueError('Source revision must be immutable')
    if value['role'] not in ('train','retention','fresh','heldout'):
        raise ValueError('Unknown source role')
    return value


def validate(metadata, key, state):
    value = metadata.json(root(key))
    if set(value) != {'format','previous','tokenizer_root','windows','documents'} or value['format'] != FORMAT:
        raise ValueError('Invalid data cohort')
    life = state['lifecycle']
    profile = state['manifest']['lifecycle']
    if value['previous'] != state['data_root'] or value['tokenizer_root'] != profile['tokenizer_root']:
        raise ValueError('Stale dataset parent or tokenizer mismatch')
    if not isinstance(value['windows'], list) or not 1 <= len(value['windows']) <= 8:
        raise ValueError('A cohort requires bounded source windows')
    windows = {}
    for window in value['windows']:
        if set(window) != {'source','start','end'}:
            raise ValueError('Invalid cursor window')
        spec = source(metadata.json(root(window['source'])))
        start = integer(window['start'], 0, 2**53-1)
        end = integer(window['end'], start+1, min(start+4096, 2**53-1))
        if window['source'] in windows or start != life['cursors'].get(window['source'], 0):
            raise ValueError('Source cursor is repeated or nonconsecutive')
        windows[window['source']] = (spec, start, end)
    if not isinstance(value['documents'], list) or not 2*EXAMPLES+1 <= len(value['documents']) <= 2*EXAMPLES+128:
        raise ValueError('Cohort document count exceeds bounds')
    roles = {role:[] for role in ('train','retention','fresh')}
    documents, batches, positions = set(), set(), set()
    for document in value['documents']:
        if set(document) != {'id','source','row','object','batches','role','omitted_targets'}:
            raise ValueError('Invalid document provenance commitment')
        identity = root(document['id'])
        root(document['object'])
        role = document['role']
        spec, start, end = windows[document['source']]
        integer(document['row'], start, end-1)
        if role not in roles or spec['role'] not in (role, 'heldout' if role != 'train' else 'train'):
            raise ValueError('Source role cannot be relabeled for training')
        omitted = integer(document['omitted_targets'], 0, 32768)
        if role != 'train' and omitted:
            raise ValueError('Evaluation must score complete document targets')
        position = (document['source'], document['row'])
        if identity in documents or identity in life['seen_documents'] or position in positions:
            raise ValueError('Document or source row has already been consumed')
        if not isinstance(document['batches'],list) or not 1 <= len(document['batches']) <= 4:
            raise ValueError('Each document requires 1–4 bounded response windows')
        for key in document['batches']:
            root(key)
            if key in batches or key in life['seen_batches']:
                raise ValueError('Fresh data cannot repeat an existing token batch')
            batch = metadata.json(key)
            ids, labels = unpack(batch, profile['vocabulary'])
            if labels is None or len(ids) != 1 or labels[0][0] != -100:
                raise ValueError('Each window needs one response-labeled row with its first label ignored')
            batches.add(key)
        documents.add(identity)
        positions.add(position)
        roles[role].append(document)
    if any(len(roles[role]) != EXAMPLES for role in ('retention','fresh')):
        raise ValueError('Exactly 32 distinct documents are required per evaluation group')
    replay = life['active']['train'] if life['active'] else []
    steps = profile['steps_per_cohort']
    replay_count = steps//4 if replay else 0
    if sum(len(doc['batches']) for doc in roles['train']) < steps-replay_count:
        raise ValueError('Insufficient unique fresh training documents for the epoch budget')
    # Evaluation transactions score up to four document rows together. Fix the
    # row length within each group so the batch size remains mechanically bound.
    for role in ('retention','fresh'):
        lengths = {len(metadata.json(key)['input_ids'][0]) for d in roles[role] for key in d['batches']}
        if len(lengths) != 1 or next(iter(lengths)) > 128:
            raise ValueError('Evaluation rows require a common length at most 128')
    return value, roles


def schedule(key, fresh, replay, steps):
    def ranked(documents,domain):
        return sorted((batch for doc in documents for batch in doc['batches']),
                      key=lambda batch:digest(canonical([key,domain,batch])))
    fresh, replay = ranked(fresh,'fresh'),ranked(replay,'replay')
    count = steps//4 if replay else 0
    result, new_index, old_index = [], 0, 0
    for index in range(steps):
        if (index+1)*count//steps > index*count//steps:
            result.append(replay[old_index % len(replay)])
            old_index += 1
        else:
            result.append(fresh[new_index])
            new_index += 1
    return result


def evaluation_rows(active,role):
    if role not in ('retention','fresh'):
        raise ValueError('Unknown evaluation group')
    return [{'document':doc['id'],'batch':key} for doc in active[role] for key in doc['batches']]


def evaluation_batch(active, role, offset):
    all_rows = evaluation_rows(active,role)
    integer(offset, 0, len(all_rows)-1)
    if offset % 4:
        raise ValueError('Evaluation offsets must align to four documents')
    documents = all_rows[offset:offset+4]
    rows = [active['batches'][doc['batch']] for doc in documents]
    batch = {field:[row[field][0] for row in rows] for field in ('input_ids','labels')}
    return documents, batch


def comparison(before, after, margin_micros):
    """Exact integer decision; display floats do not determine consensus.

Losses are quantized down to micronats. A two-micronat conservative allowance
    covers quantization of both sides. The normal-interval model is approximate;
    public tests, selection and distribution shift still limit its interpretation.
    """
    if len(before) != EXAMPLES or len(after) != EXAMPLES:
        raise ValueError('Quality decision requires the entire fixed cohort')
    def quantize(value):
        numerator, denominator = finite_loss(value).as_integer_ratio()
        return numerator*1_000_000//denominator
    delta = [quantize(b)-quantize(a) for a,b in zip(before, after)]
    count, total = len(delta), sum(delta)
    squares = count*sum(d*d for d in delta)-total*total
    gap = (margin_micros-2)*count-total
    passes = gap > 0 and gap*gap*1_000_000*(count-1) > 2576*2576*squares
    return {'passes':passes, 'documents':count, 'sum_delta_micros':total,
            'variance_numerator_micros_squared':squares, 'margin_micros':margin_micros,
            'quantization_allowance_micros':2, 'z_numerator':2576, 'z_denominator':1000}


def document_losses(active,role,losses):
    """One target-weighted observation per complete document, never per chunk."""
    if len(losses) != len(evaluation_rows(active,role)):
        raise ValueError('Missing evaluation windows')
    values, offset = [], 0
    for document in active[role]:
        total, targets = Fraction(0), 0
        for key in document['batches']:
            count = sum(label != -100 for label in active['batches'][key]['labels'][0][1:])
            total += Fraction.from_float(finite_loss(losses[offset]))*count
            targets += count
            offset += 1
        values.append(float(total/targets).hex())
    return values


def decision(measurements, active=None):
    if active is not None:
        measurements = {side:{role:document_losses(active,role,values) for role,values in groups.items()}
                        for side,groups in measurements.items()}
    results = {role:comparison(measurements['baseline'][role], measurements['candidate'][role], margin)
               for role, margin in (('retention',20000),('fresh',-1000))}
    return {'promote':all(value['passes'] for value in results.values()), **results,
            'scope':'one fixed public cohort, paired response loss; not a general capability or safety guarantee'}
