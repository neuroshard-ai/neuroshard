"""Frozen answer-balanced continuation and independent endpoint selection."""
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess

from . import consolidation, continued, grounded_tasks as tasks, reasoned, reference_data as data

FORMAT = 'neuroshard-balanced-continuation-v1'
PREPARED = FORMAT + '/prepared'
SELECTION = FORMAT + '/selection'
PLAN_ROOT = 'fbfd04a2989b7228fea36549e53bc1d2ede55e8bbc4ce006d638aa9ed03845e3'
ROOT = Path(__file__).resolve().parents[3]
PLAN = 'config/experiments/balanced-continuation.json'
SOURCES = (*consolidation.SOURCES, 'src/neuroshard/evolution/balanced.py',
           'src/neuroshard/evolution/sharded/balanced_job.py')
DEVELOPMENT = consolidation.DEVELOPMENT
FINALS = consolidation.FINALS
STRATA = ('new_sort', 'new_lookup', 'new_filter', 'replay_reasoned', 'replay_total', 'replay_conversation')


def validate(plan):
    if (plan.get('format') != FORMAT
            or plan.get('status') not in ('plan-frozen', 'prepared-committed', 'selection-committed')
            or data.identity({k: v for k, v in plan.items() if k != 'status'}) != PLAN_ROOT):
        raise ValueError('Balanced-continuation constants differ from the frozen plan')
    return plan


def load(path=None):
    return validate(json.loads(Path(path or ROOT / PLAN).read_bytes()))


def git_bytes(revision, filename):
    relative = Path(filename).resolve().relative_to(ROOT.resolve())
    try:
        return subprocess.check_output(['git', '-C', str(ROOT), 'show', revision + ':' + str(relative)],
                                       stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        raise ValueError('Required balanced-continuation artifact is not committed') from error


def sources():
    return {name: data.sha256(ROOT / name) for name in SOURCES}


def path(plan, kind):
    return ROOT / plan['open_source'][kind + '_path']


def replay_source(plan):
    filename = ROOT / plan['replay']['prepared_path']
    raw = filename.read_bytes()
    old = json.loads(raw)
    if git_bytes('HEAD', filename) != raw or data.identity(old) != plan['replay']['prepared']:
        raise ValueError('Replay source is not the committed completed training artifact')
    schedule = old['schedule']
    used = [index for entry in schedule for index in entry['indices']]
    count = old['roles']['train']['count']
    if (len(schedule) != plan['replay']['completed_step'] - old['plan']['parent_step']
            or any(entry['role'] != 'train' for entry in schedule)
            or set(used) != set(range(count))):
        raise ValueError('Replay source does not prove completed training coverage')
    return old


def schedule(plan):
    groups, offset = {}, 0
    rng = random.Random(plan['seeds']['schedule'])
    for kind in STRATA:
        count = plan['strata'][kind]
        groups[kind] = list(range(offset, offset + count))
        rng.shuffle(groups[kind])
        offset += count
    result = []
    for step in range(plan['training']['steps']):
        indices = []
        for kind in STRATA:
            size = plan['batch_strata'][kind]
            indices.extend(groups[kind][step * size:(step + 1) * size])
        rng.shuffle(indices)
        if len(indices) != plan['training']['batch_documents']:
            raise ValueError('Incomplete balanced batch')
        result.append({'role': 'train', 'indices': indices})
    if sorted(i for row in result for i in row['indices']) != list(range(offset)):
        raise ValueError('Balanced schedule must use every document once')
    return result


def case(plan, role, index, kind=None):
    if role == 'train':
        if kind not in ('new_sort', 'new_lookup', 'new_filter'):
            raise ValueError('Only declared fresh families have generated training cases')
        value = tasks.make_case(plan['seeds']['train_new'], 'balanced-training', index,
                               family=kind.removeprefix('new_'))
        value['variant'] = index % 4
    else:
        seed = plan['seeds']['new_tasks' if role.endswith('new') else 'prior_tasks']
        value = tasks.make_case(seed, 'dev' if role.startswith('dev-') else 'test', index)
        value['variant'] = (index // len(tasks.FAMILIES)) % 4
    return value


def choose_replay(plan, records):
    eligible = {'replay_reasoned': [], 'replay_total': [], 'replay_conversation': []}
    for index, row in enumerate(records):
        family = row.get('task', {}).get('family')
        if family is None:
            eligible['replay_conversation'].append(index)
        elif family == 'total':
            eligible['replay_total' if row.get('distill', False) else 'replay_reasoned'].append(index)
    rng = random.Random(plan['seeds']['replay'])
    selected = {}
    for kind, indices in eligible.items():
        rng.shuffle(indices)
        selected[kind] = indices[:plan['strata'][kind]]
        if len(selected[kind]) != plan['strata'][kind]:
            raise ValueError('Insufficient proven-trained replay for the declared stratum')
    return selected


def committed_prepared(plan, prepared):
    validate(plan)
    if plan['status'] not in ('prepared-committed', 'selection-committed'):
        raise ValueError('Balanced execution requires committed prepared inputs')
    for filename, value in ((ROOT / PLAN, plan), (path(plan, 'prepared'), prepared)):
        raw = filename.read_bytes()
        if json.loads(raw) != value or git_bytes('HEAD', filename) != raw:
            raise ValueError('Balanced execution requires unchanged committed artifacts')
    revision = prepared['plan_commit']
    if re.fullmatch(r'[0-9a-f]{40}', revision) is None:
        raise ValueError('Pin the complete immutable source commit')
    frozen = git_bytes(revision, ROOT / PLAN)
    if (prepared['format'] != PREPARED
            or hashlib.sha256(frozen).hexdigest() != prepared['plan_digest']
            or json.loads(frozen) != {**plan, 'status': 'plan-frozen'}
            or prepared['sources'] != sources()):
        raise ValueError('Prepared inputs do not bind frozen balanced sources and plan')
    for filename, digest in prepared['sources'].items():
        raw = (ROOT / filename).read_bytes()
        if (hashlib.sha256(raw).hexdigest() != digest or git_bytes('HEAD', ROOT / filename) != raw
                or git_bytes(revision, ROOT / filename) != raw):
            raise ValueError('Balanced numerical source differs from its frozen commit')
    if set(prepared['roles']) != set(plan['roles']) or prepared['schedule'] != schedule(plan):
        raise ValueError('Balanced role coverage or schedule differs')
    all_ids = []
    for role, count in plan['roles'].items():
        spec = prepared['roles'][role]
        if (set(spec) != {'file', 'sha256', 'count', 'ids'} or spec['file'] != role + '.jsonl'
                or spec['count'] != count or len(spec['ids']) != count):
            raise ValueError('Balanced role count or file differs')
        continued.digest(spec['sha256'])
        for identifier in spec['ids']:
            continued.digest(identifier)
        all_ids.extend(spec['ids'])
    if len(set(all_ids)) != len(all_ids):
        raise ValueError('Balanced roles overlap')
    old = replay_source(plan)
    if prepared['replay_source'] != {'file': 'trained-source.jsonl', 'sha256': old['roles']['train']['sha256'],
                                     'prepared': data.identity(old)}:
        raise ValueError('Replay bytes differ from the completed source')
    replay_ids = set()
    if set(prepared['replay_indices']) != set(STRATA[3:]):
        raise ValueError('Replay must contain exactly the declared strata')
    for kind in ('replay_reasoned', 'replay_total', 'replay_conversation'):
        indices = prepared['replay_indices'][kind]
        if len(indices) != plan['strata'][kind] or len(set(indices)) != len(indices):
            raise ValueError('Incomplete or repeated replay indices')
        for index in indices:
            if type(index) is not int or not 0 <= index < old['roles']['train']['count']:
                raise ValueError('Replay index is outside trained coverage')
            replay_ids.add(old['roles']['train']['ids'][index])
    if len(replay_ids) != sum(plan['strata'][k] for k in prepared['replay_indices']):
        raise ValueError('Replay strata overlap')
    if not replay_ids <= set(prepared['roles']['train']['ids']):
        raise ValueError('Prepared training omits declared replay')
    fresh_ids = set(all_ids) - replay_ids
    expected_exclusions = {}
    for label, name in plan['exclusions'].items():
        if label == 'policy':
            continue
        raw = (ROOT / name).read_bytes()
        prior = json.loads(raw)
        if git_bytes('HEAD', ROOT / name) != raw:
            raise ValueError('A preceding exclusion artifact changed')
        expected_exclusions[label] = data.identity(prior)
        if fresh_ids & {key for spec in prior['roles'].values() for key in spec['ids']}:
            raise ValueError('Fresh balanced data overlaps earlier exposed data')
    if prepared['excluded_prepared'] != expected_exclusions:
        raise ValueError('Balanced exclusions do not cover every earlier experiment')
    return True


def read_role(plan, directory, prepared, role, tokenizer):
    directory = Path(directory)
    spec = prepared['roles'][role]
    rows = data.read_records(directory / spec['file'], spec['sha256'])
    if [row['id'] for row in rows] != spec['ids']:
        raise ValueError('Balanced role identities differ from the freeze')
    old_rows = None
    if role == 'train':
        old = replay_source(plan)
        old_rows = data.read_records(directory / prepared['replay_source']['file'], prepared['replay_source']['sha256'])
        if [r['id'] for r in old_rows] != old['roles']['train']['ids']:
            raise ValueError('Replay source records differ from the committed training manifest')
        if choose_replay(plan, old_rows) != prepared['replay_indices']:
            raise ValueError('Replay selection changed after the freeze')
    seen = {kind: 0 for kind in plan['strata']}
    ordered_kinds = [kind for kind in STRATA for _ in range(plan['strata'][kind])]
    for index, row in enumerate(rows):
        if role == 'train':
            kind = ordered_kinds[index]
            local_index = seen[kind]
            seen[kind] += 1
            if row['stratum'] != kind or row['distill'] is not True or row['loss_weight'] != 1.0 / row['targets']:
                raise ValueError('Balanced answer weight, stratum or reference mask differs')
            if kind.startswith('new_'):
                value = case(plan, role, local_index, kind)
                messages = reasoned.messages(value, False)
                if row['task'] != value or row['id'] != tasks.task_identity(value) or row['messages'] != messages:
                    raise ValueError('Fresh balanced training differs from its generated case')
            else:
                source_index = prepared['replay_indices'][kind][local_index]
                source = old_rows[source_index]
                expected = {**source, 'stratum': kind, 'source_index': source_index,
                            'loss_weight': 1.0 / source['targets'], 'distill': True}
                if row != expected:
                    raise ValueError('Replay is not the unchanged proven-trained example')
        elif role in ('dev-new', 'dev-prior', 'test-new', 'test-prior'):
            value = case(plan, role, index)
            if (row['task'] != value or row['id'] != tasks.task_identity(value)
                    or row['messages'] != reasoned.messages(value, role in plan['method']['reasoning_roles'])):
                raise ValueError('Balanced evaluation differs from its frozen prompt or target')
        elif row['id'] != data.identity({k: v for k, v in row.items() if k != 'id'}):
            raise ValueError('Conversation identity does not bind its full content')
        encoded = data.conversation(tokenizer, row['messages'], plan['max_length'])
        if any(row[key] != value for key, value in encoded.items()):
            raise ValueError('Balanced tokenization or response target mask differs')
    return rows


def generation(plan, prepared, role, before, after, tokenizer=None):
    cases = [case(plan, role, i) for i in range(plan['roles'][role])]
    ids = [tasks.task_identity(value) for value in cases]
    if ids != prepared['roles'][role]['ids']:
        raise ValueError('Balanced cases differ from committed identities')
    checked = []
    for answers in (before, after):
        if [answer['id'] for answer in answers] != ids:
            raise ValueError('Balanced answers are incomplete or reordered')
        current = []
        for value, answer in zip(cases, answers):
            if tokenizer is not None and tokenizer.decode(answer['output_ids'], skip_special_tokens=True) != answer['text']:
                raise ValueError('Balanced text differs from generated tokens')
            check = consolidation.check_answer(plan, value, answer['text'], role)
            if check != answer['check']:
                raise ValueError('Balanced verdict differs from the actual answer')
            current.append({'id': answer['id'], 'correct': check['correct']})
        checked.append(current)
    result = tasks.paired_accuracy(*checked)
    result['families'] = {family: tasks.paired_accuracy(
        *[[row for row, value in zip(values, cases) if value['family'] == family] for values in checked])
        for family in tasks.FAMILIES}
    return result


def development(plan, prepared, baseline, training_parent_new, candidate, tokenizer=None):
    prior = generation(plan, prepared, 'dev-prior', baseline['dev-prior']['answers'], candidate['dev-prior']['answers'], tokenizer)
    new = generation(plan, prepared, 'dev-new', baseline['dev-new']['answers'], candidate['dev-new']['answers'], tokenizer)
    parent_new = generation(plan, prepared, 'dev-new', training_parent_new['answers'], candidate['dev-new']['answers'], tokenizer)
    retention = consolidation.retention(plan, prepared, 'dev-retention',
                                       baseline['dev-retention']['losses'], candidate['dev-retention']['losses'])
    checks = {'prior_preserved': prior['losses'] <= plan['development_gate']['prior_losses_at_most'],
              'new_gain': new['wins'] - new['losses'] >= plan['development_gate']['new_net_gain_at_least'],
              'new_family_floors': all(v['candidate_correct'] >= v['baseline_correct'] for v in new['families'].values()),
              'learned_skill_retained': all(v['candidate_correct'] >= v['baseline_correct'] for v in parent_new['families'].values()),
              'retention': retention['mean_delta'] <= plan['development_gate']['retention_mean_delta_at_most']}
    return {'prior': prior, 'new': new, 'training_parent_new': parent_new, 'retention': retention,
            'checks': checks, 'passed': all(checks.values())}


def job(prepared):
    return data.identity({'domain': FORMAT, 'prepared': data.identity(prepared)})


def validate_selection(plan, prepared, selected, tokenizer=None):
    from .sharded import portable
    if (selected['format'] != SELECTION or selected['prepared'] != data.identity(prepared)
            or selected['parent'] != plan['parent']['checkpoint'] or selected['baseline'] != plan['baseline']['checkpoint']):
        raise ValueError('Balanced selection belongs to another experiment')
    parent = selected['input_checkpoint']
    portable.validate(parent)
    if data.identity(parent) != selected['parent']:
        raise ValueError('Balanced selection changed its complete starting state')
    checkpoints = selected['checkpoints']
    if [value['step'] for value in checkpoints] != plan['checkpoints']:
        raise ValueError('Balanced selection must include every declared checkpoint')
    previous = parent
    semantics = lambda values: [{k: v for k, v in row.items() if k != 'lr'} for row in values]
    for value in checkpoints:
        portable.validate(value)
        if (value['job'] != job(prepared) or value['parent'] != data.identity(previous)
                or value['config'] != parent['config'] or value['boundaries'] != plan['boundaries']
                or value['transition'] is not None or len(value['shards']) != len(plan['boundaries']) - 1
                or semantics(value['optimizer']) != semantics(parent['optimizer'])
                or any(any(value['tensors'][name][key] != spec[key] for key in ('shape', 'born', 'group'))
                       for name, spec in parent['tensors'].items())):
            raise ValueError('Balanced checkpoint changed its job, architecture, ages or optimizer')
        previous = value
    if (previous['step'] != plan['selectable_checkpoint'] or selected['candidate'] != data.identity(previous)
            or selected['decision'] != development(plan, prepared, selected['baseline_development'],
                selected['training_parent_new'], selected['candidate_development'], tokenizer)
            or not selected['decision']['passed']):
        raise ValueError('Balanced selection requires the actual passing terminal checkpoint')
    return True


def committed_selection(plan, prepared, selected):
    filename = path(plan, 'selection')
    raw = filename.read_bytes()
    if plan['status'] != 'selection-committed' or git_bytes('HEAD', filename) != raw or json.loads(raw) != selected:
        raise ValueError('Balanced finals require the actual committed selection')
    return validate_selection(plan, prepared, selected)
