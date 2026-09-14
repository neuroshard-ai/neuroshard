"""Frozen, non-issuing consolidation of an existing sharded learning update.

One set of interpolated weights is selected on development data. The optimizer
remains the fast checkpoint's complete Adam state. Final evidence is opened only
after the actual selected checkpoint and all attempted screens are committed.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess

from . import continued, grounded_tasks as tasks, reasoned, reference_data as data

FORMAT = 'neuroshard-consolidated-learning-v1'
PREPARED = FORMAT + '/prepared'
SELECTION = FORMAT + '/selection'
PLAN_ROOT = '6265719980b80bc6b95eb04468aaf67cca22bcc425c335aa8e6a681afa80cd84'
ROOT = Path(__file__).resolve().parents[3]
PLAN = 'config/experiments/consolidated-learning.json'
SOURCES = (*continued.SOURCE_PATHS,
           'src/neuroshard/evolution/consolidation.py',
           'src/neuroshard/evolution/sharded/consolidation_job.py')
DEVELOPMENT = ('dev-prior', 'dev-new', 'dev-retention')
FINALS = ('test-new', 'test-prior', 'retention')


def validate(plan):
    if (plan.get('format') != FORMAT
            or plan.get('status') not in ('plan-frozen', 'prepared-committed', 'selection-committed')
            or data.identity({k: v for k, v in plan.items() if k != 'status'}) != PLAN_ROOT):
        raise ValueError('Consolidation constants differ from the frozen plan')
    return plan


def load(path=None):
    return validate(json.loads(Path(path or ROOT / PLAN).read_bytes()))


def git_bytes(revision, path):
    relative = Path(path).resolve().relative_to(ROOT.resolve())
    try:
        return subprocess.check_output(['git', '-C', str(ROOT), 'show', revision + ':' + str(relative)],
                                       stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        raise ValueError('Required consolidation artifact is not Git-committed') from error


def sources():
    return {name: data.sha256(ROOT / name) for name in SOURCES}


def path(plan, kind):
    return ROOT / plan['open_source'][kind + '_path']


def committed_prepared(plan, prepared):
    validate(plan)
    if plan['status'] not in ('prepared-committed', 'selection-committed'):
        raise ValueError('Consolidation requires committed prepared inputs')
    for filename, value in ((ROOT / PLAN, plan), (path(plan, 'prepared'), prepared)):
        raw = filename.read_bytes()
        if json.loads(raw) != value or git_bytes('HEAD', filename) != raw:
            raise ValueError('Consolidation requires unchanged Git-committed artifacts')
    if re.fullmatch(r'[0-9a-f]{40}', prepared['plan_commit']) is None:
        raise ValueError('Pin the complete immutable preparation commit')
    frozen = git_bytes(prepared['plan_commit'], ROOT / PLAN)
    if (prepared['format'] != PREPARED
            or hashlib.sha256(frozen).hexdigest() != prepared['plan_digest']
            or json.loads(frozen) != {**plan, 'status': 'plan-frozen'}
            or prepared['sources'] != sources()):
        raise ValueError('Prepared consolidation does not bind frozen sources and plan')
    for filename, digest in prepared['sources'].items():
        local = (ROOT / filename).read_bytes()
        if (hashlib.sha256(local).hexdigest() != digest
                or git_bytes('HEAD', ROOT / filename) != local
                or git_bytes(prepared['plan_commit'], ROOT / filename) != local):
            raise ValueError('Consolidation numerical source is not the frozen committed source')
    if set(prepared['roles']) != set(plan['roles']):
        raise ValueError('Consolidation role coverage differs')
    all_ids = []
    for role, count in plan['roles'].items():
        spec = prepared['roles'][role]
        if (set(spec) != {'file', 'sha256', 'count', 'ids'}
                or spec['count'] != count or len(spec['ids']) != count
                or spec['file'] != role + '.jsonl'):
            raise ValueError('Prepared role count or filename differs')
        continued.digest(spec['sha256'])
        for identifier in spec['ids']:
            continued.digest(identifier)
        all_ids.extend(spec['ids'])
    if len(set(all_ids)) != len(all_ids):
        raise ValueError('Consolidation roles overlap')
    expected_exclusions = {}
    for label, filename in plan['exclusions'].items():
        if label == 'policy':
            continue
        raw = (ROOT / filename).read_bytes()
        if git_bytes('HEAD', ROOT / filename) != raw:
            raise ValueError('Prior exclusion artifact was modified')
        old = json.loads(raw)
        expected_exclusions[label] = data.identity(old)
        prior_ids = {key for spec in old['roles'].values() for key in spec['ids']}
        if set(all_ids) & prior_ids:
            raise ValueError('Consolidation evaluation overlaps prior exposed data')
    if prepared['excluded_prepared'] != expected_exclusions:
        raise ValueError('Prepared exclusions do not bind all previous experiments')
    if expected_exclusions['reasoned'] != plan['fast']['prepared']:
        raise ValueError('The fast checkpoint training artifact changed')
    return True


def job(prepared, alpha):
    return data.identity({'domain': FORMAT, 'prepared': data.identity(prepared), 'alpha': alpha})


def read_role(plan, directory, prepared, role, tokenizer=None):
    spec = prepared['roles'][role]
    records = data.read_records(Path(directory) / spec['file'], spec['sha256'])
    if [record['id'] for record in records] != spec['ids']:
        raise ValueError('Role records differ from committed identities')
    for index, record in enumerate(records):
        if role in ('dev-new', 'dev-prior', 'test-new', 'test-prior'):
            seed = plan['seeds']['new_tasks' if role.endswith('new') else 'prior_tasks']
            case = tasks.make_case(seed, 'dev' if role.startswith('dev-') else 'test', index)
            messages = reasoned.messages(case, role in plan['method']['reasoning_roles'])
            if (record['id'] != tasks.task_identity(case) or record['task'] != case
                    or record['messages'] != messages):
                raise ValueError('Generated input differs from its frozen task and prompt')
        elif record['id'] != data.identity({k: v for k, v in record.items() if k != 'id'}):
            raise ValueError('Conversation identity does not bind its complete content')
        if tokenizer is not None:
            encoded = data.conversation(tokenizer, record['messages'], plan['max_length'])
            if any(record[key] != value for key, value in encoded.items()):
                raise ValueError('Prepared tokenization or target mask differs')
    return records


def check_answer(plan, case, text, role):
    if role in plan['method']['reasoning_roles'] and case['family'] == 'total':
        try:
            answer, used = reasoned.final_json(text)
        except (ValueError, TypeError):
            return {'correct': False, 'valid_json_object': False, 'reason': 'invalid_reasoned_response'}
        return {**tasks.check_answer(case, answer), 'reasoning_prefix': used}
    return tasks.check_answer(case, text)


def generation(plan, prepared, role, before, after, tokenizer=None):
    count = plan['roles'][role]
    seed = plan['seeds']['new_tasks' if role.endswith('new') else 'prior_tasks']
    split = 'dev' if role.startswith('dev-') else 'test'
    cases = [tasks.make_case(seed, split, index) for index in range(count)]
    expected_ids = [tasks.task_identity(case) for case in cases]
    if expected_ids != prepared['roles'][role]['ids']:
        raise ValueError('Prepared generated tasks differ from frozen seeds')
    checked = []
    for answers in (before, after):
        if [row['id'] for row in answers] != expected_ids:
            raise ValueError('Generated answers are incomplete, reordered or repeated')
        rows = []
        for case, answer in zip(cases, answers):
            if tokenizer is not None and tokenizer.decode(answer['output_ids'], skip_special_tokens=True) != answer['text']:
                raise ValueError('Generated text differs from output tokens')
            verdict = check_answer(plan, case, answer['text'], role)
            if verdict != answer['check']:
                raise ValueError('Reported verdict differs from the actual generated answer')
            rows.append({'id': answer['id'], 'correct': verdict['correct']})
        checked.append(rows)
    summary = tasks.paired_accuracy(*checked)
    summary['families'] = {family: tasks.paired_accuracy(
        *[[row for row, case in zip(values, cases) if case['family'] == family] for values in checked])
        for family in tasks.FAMILIES}
    return summary


def retention(plan, prepared, role, before, after):
    ids = prepared['roles'][role]['ids']
    for values in (before, after):
        if [row['id'] for row in values] != ids:
            raise ValueError('Retention loss coverage differs from the frozen set')
    if [row['targets'] for row in before] != [row['targets'] for row in after]:
        raise ValueError('Retention target masks differ')
    gate = {**plan['quality_gate'], 'bootstrap_seed': plan['seeds']['bootstrap']}
    return continued.paired_losses([v['loss'] for v in before], [v['loss'] for v in after], gate)


def screen_decision(plan, prepared, baseline, candidate, tokenizer=None):
    prior = generation(plan, prepared, 'dev-prior', baseline['dev-prior']['answers'],
                       candidate['dev-prior']['answers'], tokenizer)
    result = {'prior': prior, 'passed': False}
    if prior['losses'] > plan['development_gate']['prior_losses_at_most']:
        result['reason'] = 'lost previously correct prior-development answers'
        return result
    if set(candidate) != set(DEVELOPMENT):
        raise ValueError('A viable screen must complete every development role')
    new = generation(plan, prepared, 'dev-new', baseline['dev-new']['answers'],
                     candidate['dev-new']['answers'], tokenizer)
    retain = retention(plan, prepared, 'dev-retention', baseline['dev-retention']['losses'],
                       candidate['dev-retention']['losses'])
    checks = {'new_gain': new['wins'] - new['losses'] >= plan['development_gate']['new_net_gain_at_least'],
              'family_floors': all(v['candidate_correct'] >= v['baseline_correct'] for v in new['families'].values()),
              'retention': retain['mean_delta'] <= plan['development_gate']['retention_mean_delta_at_most']}
    return {**result, 'new': new, 'retention': retain, 'checks': checks, 'passed': all(checks.values())}


def validate_selection(plan, prepared, selected, tokenizer=None):
    from .sharded import portable
    if (selected['format'] != SELECTION or selected['prepared'] != data.identity(prepared)
            or selected['parent'] != plan['parent']['checkpoint'] or selected['fast'] != plan['fast']['checkpoint']):
        raise ValueError('Selection belongs to a different consolidation')
    fast = selected['input_checkpoint']
    portable.validate(fast)
    if data.identity(fast) != plan['fast']['checkpoint']:
        raise ValueError('Selection changed the input weights or optimizer')
    attempts = selected['attempts']
    alphas = plan['method']['alphas']
    if not attempts or [row['alpha'] for row in attempts] != alphas[:len(attempts)]:
        raise ValueError('Selection skipped or reordered a declared alpha')
    for index, attempt in enumerate(attempts):
        actual = screen_decision(plan, prepared, selected['baseline'], attempt['outcomes'], tokenizer)
        if actual != attempt['decision'] or actual['passed'] != (index == len(attempts) - 1):
            raise ValueError('Selection must stop at the first complete passing development screen')
        checkpoint = attempt['checkpoint']
        portable.validate(checkpoint)
        if (checkpoint['job'] != job(prepared, attempt['alpha']) or checkpoint['step'] != plan['fast']['step']
                or checkpoint['parent'] != plan['fast']['checkpoint']
                or checkpoint['config'] != fast['config'] or checkpoint['boundaries'] != fast['boundaries']
                or checkpoint['optimizer'] != fast['optimizer']
                or len(checkpoint['shards']) != len(plan['boundaries']) - 1
                or checkpoint['transition'] != transition(plan, prepared, attempt['alpha'])):
            raise ValueError('Selected checkpoint does not bind the declared consolidation')
        if any(any(checkpoint['tensors'][name][key] != spec[key] for key in ('shape', 'born', 'group'))
               for name, spec in fast['tensors'].items()):
            raise ValueError('Consolidation changed tensor shapes, ages or optimizer groups')
    if selected['candidate'] != data.identity(attempts[-1]['checkpoint']):
        raise ValueError('Final candidate differs from the actual selected checkpoint')
    return True


def transition(plan, prepared, alpha):
    if type(alpha) is not float or alpha not in plan['method']['alphas']:
        raise ValueError('Undeclared consolidation coefficient')
    return {'kind': FORMAT, 'parent': plan['parent']['checkpoint'], 'fast': plan['fast']['checkpoint'],
            'prepared': data.identity(prepared), 'alpha': alpha,
            'optimizer': 'unchanged-fast-state', 'new_gradient_updates': 0}


def committed_selection(plan, prepared, selected):
    filename = path(plan, 'selection')
    raw = filename.read_bytes()
    if (plan['status'] != 'selection-committed' or git_bytes('HEAD', filename) != raw
            or json.loads(raw) != selected):
        raise ValueError('Final evaluation requires the actual Git-committed candidate selection')
    return validate_selection(plan, prepared, selected)
