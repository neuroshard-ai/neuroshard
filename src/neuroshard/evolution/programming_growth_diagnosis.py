"""Inference-only complementarity of two unchanged programming tails.

The unit-merge growth comparison remains failed. This measurement asks whether
the added tail solves any already-opened leftover extra that the incumbent
cannot. An oracle that always picks a successful tail is an upper bound, not a
deployable selector. The original 128-task final stays closed.
"""
from neuroshard.evolution import programming_fallback as fallback
from neuroshard.evolution import programming_growth as experiment
from neuroshard.evolution.programming_expert import extract_code
from neuroshard.evolution.reference_data import identity, sha256

FORMAT = experiment.FORMAT + '/diagnosis'
ADDED = experiment.ADDED
GROWTH_OUTPUTS = '7ba33b47e923978ff83c900894fb38425b4f4eb63b359051386f61cb1fd42077'
ADDED_EXPERT = '3abb54eecf62610a1116cc1b5eb67110bea17ee9608f946002ed9b4ad95703b6'
GROWTH_FREEZE_COMMIT = 'dcc66936a746f7fdc6b6dfa25d1c47ad6f366758'
VISIBLE_FAIL_TASK_IDS = [
    183, 323, 452, 449, 169, 276, 383, 158, 258, 466, 302, 339, 503, 371, 115,
    443, 348, 103, 54, 84, 249, 376, 291, 224, 265, 124, 277, 220, 283, 461,
    410, 353, 96, 307, 350, 209, 295, 468,
]


def extractable(output):
    try:
        extract_code(output['text'])
        return True
    except (ValueError, SyntaxError, TypeError, KeyError):
        return False


def growth_table(outputs):
    actual = [(out['id'], out['arm']) for out in outputs]
    if len(actual) != len(set(actual)):
        raise ValueError('Duplicated growth outputs')
    return {(out['id'], out['arm']): out for out in outputs}


def bind_diagnosis(plan, diagnosis, growth_outputs_digest, added_manifest):
    if diagnosis.get('format') != FORMAT:
        raise ValueError('Diagnosis plan does not bind this measurement')
    if diagnosis.get('growth_plan') != identity(plan):
        raise ValueError('Diagnosis is not bound to the failed growth plan')
    if diagnosis.get('growth_freeze_commit') != GROWTH_FREEZE_COMMIT:
        raise ValueError('Diagnosis must keep the failed growth freeze')
    if diagnosis.get('train') is not False or diagnosis.get('oracle_is_not_a_policy') is not True:
        raise ValueError('Diagnosis may not train or treat the oracle as a policy')
    if diagnosis.get('admission_evidence') is not False or diagnosis.get('original_final_opened') is not False:
        raise ValueError('Diagnosis is not admission evidence')
    if diagnosis.get('growth_outputs') != growth_outputs_digest or growth_outputs_digest != GROWTH_OUTPUTS:
        raise ValueError('Diagnosis must reuse the opened growth outputs')
    if (diagnosis.get('added_expert') != ADDED_EXPERT
            or identity(added_manifest) != ADDED_EXPERT):
        raise ValueError('Added checkpoint is not the failed merge tail')
    if diagnosis.get('incumbent_expert') != experiment.INCUMBENT_EXPERT:
        raise ValueError('Diagnosis incumbent is not the leftover fallback tail')
    if diagnosis.get('visible_fail_task_ids') != VISIBLE_FAIL_TASK_IDS:
        raise ValueError('Opened extra-decode IDs changed')
    experiment.bind_splits(plan)
    return {
        'plan': identity(plan),
        'diagnosis': identity(diagnosis),
        'added_expert': identity(added_manifest),
        'growth_outputs': growth_outputs_digest,
    }


def visible_fail_rows(rows, table, check):
    failed = []
    for row in rows:
        base = table[row['id'], 'base']
        if not fallback.passes(base, row, fallback.visible_tests(row), check):
            failed.append(row)
    return failed


def score_complementarity(preservation_rows, new_rows, growth_outputs, added_outputs,
                          plan, diagnosis, check):
    """Union of unchanged tails on opened extras. Not a serving policy."""
    experiment.bind_splits(plan)
    preservation_rows = experiment.load_role_rows(
        plan, preservation_rows, plan['splits']['preservation_task_ids'], role='preservation')
    new_rows = experiment.load_role_rows(
        plan, new_rows, plan['splits']['new_task_ids'], role='new')
    rows = list(preservation_rows) + list(new_rows)
    table = growth_table(growth_outputs)
    expected_growth = {(row['id'], arm) for row in rows for arm in ('base', experiment.INCUMBENT, experiment.MERGED)}
    if set(table) != expected_growth:
        raise ValueError('Growth outputs do not cover preservation and new leftover questions')
    growth = experiment.score_growth(preservation_rows, new_rows, growth_outputs, plan, check)
    expected_fail = diagnosis['visible_fail_task_ids']
    if growth['passed'] or growth['visible_fail_extras'][experiment.INCUMBENT] != len(expected_fail):
        raise ValueError('Diagnosis requires the failed unit-merge comparison')
    failed = visible_fail_rows(rows, table, check)
    if [row['task_id'] for row in failed] != expected_fail:
        raise ValueError('Visible-fail extras differ from the opened growth questions')
    added_table = growth_table(added_outputs)
    expected_added = {(row['id'], ADDED) for row in failed}
    if set(added_table) != expected_added:
        raise ValueError('Added extras must cover exactly the 38 opened extra-decode questions')
    preservation_ids = {row['id'] for row in preservation_rows}
    unique_incumbent = unique_added = both = neither = 0
    extract = {experiment.INCUMBENT: 0, ADDED: 0, experiment.MERGED: 0}
    extra_full = {experiment.INCUMBENT: 0, ADDED: 0, experiment.MERGED: 0}
    parent_full = incumbent_policy = oracle_policy = 0
    details = []
    for row in rows:
        base = table[row['id'], 'base']
        incumbent = table[row['id'], experiment.INCUMBENT]
        merged = table[row['id'], experiment.MERGED]
        visible = fallback.passes(base, row, fallback.visible_tests(row), check)
        full = row['tests']
        base_full = fallback.passes(base, row, full, check)
        parent_full += int(base_full)
        if visible:
            incumbent_policy += int(base_full)
            oracle_policy += int(base_full)
            details.append({
                'id': row['id'], 'task_id': row['task_id'],
                'set': 'preservation' if row['id'] in preservation_ids else 'new',
                'visible_base_passed': True, 'selected_extra': 'none',
                'extractable': None,
                'scores': {'base': base_full, 'incumbent': base_full, 'added': base_full},
                'unique': 'unused',
            })
            continue
        added = added_table[row['id'], ADDED]
        fallback.extra_attempt_recorded(
            added, path='expert', prompt_kind='original',
            original_prompt_ids=base.get('prompt_ids'))
        inc_full = fallback.passes(incumbent, row, full, check)
        add_full = fallback.passes(added, row, full, check)
        mer_full = fallback.passes(merged, row, full, check)
        extract[experiment.INCUMBENT] += int(extractable(incumbent))
        extract[ADDED] += int(extractable(added))
        extract[experiment.MERGED] += int(extractable(merged))
        extra_full[experiment.INCUMBENT] += int(inc_full)
        extra_full[ADDED] += int(add_full)
        extra_full[experiment.MERGED] += int(mer_full)
        incumbent_policy += int(inc_full)
        oracle_policy += int(inc_full or add_full)
        if inc_full and add_full:
            both += 1
            unique = 'both'
        elif inc_full:
            unique_incumbent += 1
            unique = experiment.INCUMBENT
        elif add_full:
            unique_added += 1
            unique = ADDED
        else:
            neither += 1
            unique = 'neither'
        details.append({
            'id': row['id'], 'task_id': row['task_id'],
            'set': 'preservation' if row['id'] in preservation_ids else 'new',
            'visible_base_passed': False, 'selected_extra': 'opened-extra',
            'extractable': {
                experiment.INCUMBENT: extractable(incumbent),
                ADDED: extractable(added),
                experiment.MERGED: extractable(merged),
            },
            'scores': {
                'base': base_full,
                experiment.INCUMBENT: inc_full,
                ADDED: add_full,
                experiment.MERGED: mer_full,
            },
            'unique': unique,
        })
    useful = unique_added > 0
    return {
        'format': FORMAT + '/score',
        'admission_evidence': False,
        'original_final_opened': False,
        'oracle_is_not_a_policy': True,
        'train': False,
        'growth_merge_passed': False,
        'visible_fail': len(failed),
        'extractable_extras': extract,
        'extra_full_test': extra_full,
        'unique_incumbent': unique_incumbent,
        'unique_added': unique_added,
        'both': both,
        'neither': neither,
        'parent_full_test': parent_full,
        'incumbent_policy': incumbent_policy,
        'oracle_policy': oracle_policy,
        'oracle_gain_vs_incumbent': unique_added,
        'useful_additional_coverage': useful,
        'next': 'selection' if useful else 'stop',
        'plan': identity(plan),
        'diagnosis': identity(diagnosis),
        'details': details,
    }
