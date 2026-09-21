"""Elect-sign disjoint mean of the two frozen programming tails, without trim.

TIES keep 0.2 restored extractable extras and scored 31/64, then missed unique
added 276 and 265. This method keeps sign election and the disjoint mean and
drops magnitude trim so small one-sided directions survive. It does not train.
It does not reopen the original 128-task final. It is not a retry of TIES keep.
"""
from pathlib import Path

from neuroshard.evolution import programming_fallback as fallback
from neuroshard.evolution import programming_growth as experiment
from neuroshard.evolution.programming_expert import extract_code
from neuroshard.evolution.reference_data import identity, sha256

FORMAT = 'neuroshard-programming-growth-elect-v1'
CONTRACT_IDENTITY = 'ebd0d2e4e6f5130147f5a02cc4bbf1dd96ff9d762b61721bc38cede2b503192a'
EXPERT_IDENTITY = '65058087178b044752b2e4bd7d432eeba8ad8c6c445025e42e09a385e4a82947'
GROWTH_PLAN = 'c5662eabc3f3fc719468f3f8948d124e2dd8ee11599640bf6789ef3ab2c5f1e8'
GROWTH_FREEZE_COMMIT = 'dcc66936a746f7fdc6b6dfa25d1c47ad6f366758'
GROWTH_OUTPUTS = '7ba33b47e923978ff83c900894fb38425b4f4eb63b359051386f61cb1fd42077'
ADDED_EXPERT = '3abb54eecf62610a1116cc1b5eb67110bea17ee9608f946002ed9b4ad95703b6'
INCUMBENT_EXPERT = experiment.INCUMBENT_EXPERT
LAMBDA = 1.0
VISIBLE_FAIL_TASK_IDS = [
    183, 323, 452, 449, 169, 276, 383, 158, 258, 466, 302, 339, 503, 371, 115,
    443, 348, 103, 54, 84, 249, 376, 291, 224, 265, 124, 277, 220, 283, 461,
    410, 353, 96, 307, 350, 209, 295, 468,
]
ELECT = 'elect'


def extractable(output):
    try:
        extract_code(output['text'])
        return True
    except (ValueError, SyntaxError, TypeError, KeyError):
        return False


def elect_merge(parent, incumbent, added, scale=LAMBDA):
    """θ_parent + λ · mean({untrimmed deltas whose sign matches the elected sign})."""
    if scale != LAMBDA:
        raise ValueError('elect-sign scale must stay at the frozen default')
    delta_inc = incumbent - parent
    delta_add = added - parent
    elected = (delta_inc + delta_add).sign()
    agree_inc = (delta_inc.sign() == elected) & (elected != 0)
    agree_add = (delta_add.sign() == elected) & (elected != 0)
    count = agree_inc.to(parent.dtype) + agree_add.to(parent.dtype)
    total = delta_inc * agree_inc.to(parent.dtype) + delta_add * agree_add.to(parent.dtype)
    merged = total / count.clamp(min=1)
    merged = merged * (count > 0).to(parent.dtype)
    return parent + scale * merged


def load_named(directory, manifest):
    from safetensors.torch import load_file
    tensors = {}
    for name, spec in manifest['tensors'].items():
        path = Path(directory) / spec['file']
        if sha256(path) != spec['sha256']:
            raise ValueError('elect-sign source tensor changed: ' + name)
        tensors[name] = load_file(path)['weight']
    return tensors


def merge_named(parent, incumbent, added, scale=LAMBDA):
    if set(parent) != set(incumbent) or set(parent) != set(added):
        raise ValueError('elect-sign merge requires identical tail tensor names')
    return {name: elect_merge(parent[name], incumbent[name], added[name], scale)
            for name in parent}


def bind_elect(plan, spec, growth_outputs_digest, added_manifest, incumbent_manifest):
    if identity(spec) != CONTRACT_IDENTITY:
        raise ValueError('elect-sign contract does not bind this measurement')
    if spec.get('expert_manifest') != EXPERT_IDENTITY:
        raise ValueError('elect-sign expert hashes changed')
    if spec.get('growth_plan') != identity(plan) or identity(plan) != GROWTH_PLAN:
        raise ValueError('elect-sign is not bound to the failed growth plan')
    if spec.get('growth_freeze_commit') != GROWTH_FREEZE_COMMIT:
        raise ValueError('elect-sign must keep the failed growth freeze')
    if spec.get('train') is not False or spec.get('admission_evidence') is not False:
        raise ValueError('elect-sign may not train or count as admission')
    if spec.get('scale') != LAMBDA:
        raise ValueError('elect-sign scale changed after freeze')
    if spec.get('trim') is not False or spec.get('keep') not in (None, False):
        raise ValueError('elect-sign must not trim')
    if spec.get('original_final_opened') is not False:
        raise ValueError('elect-sign does not open the original final')
    if spec.get('growth_outputs') != growth_outputs_digest or growth_outputs_digest != GROWTH_OUTPUTS:
        raise ValueError('elect-sign must reuse the opened growth outputs')
    if identity(added_manifest) != ADDED_EXPERT or spec.get('added_expert') != ADDED_EXPERT:
        raise ValueError('elect-sign added checkpoint is not the failed merge tail')
    if identity(incumbent_manifest) != INCUMBENT_EXPERT or spec.get('incumbent_expert') != INCUMBENT_EXPERT:
        raise ValueError('elect-sign incumbent is not the leftover fallback tail')
    if spec.get('visible_fail_task_ids') != VISIBLE_FAIL_TASK_IDS:
        raise ValueError('Opened extra-decode IDs changed')
    experiment.bind_splits(plan)
    return {
        'plan': identity(plan),
        'elect': identity(spec),
        'added_expert': identity(added_manifest),
        'incumbent_expert': identity(incumbent_manifest),
        'growth_outputs': growth_outputs_digest,
    }


def growth_table(outputs):
    actual = [(out['id'], out['arm']) for out in outputs]
    if len(actual) != len(set(actual)):
        raise ValueError('Duplicated growth outputs')
    return {(out['id'], out['arm']): out for out in outputs}


def score_elect_extras(preservation_rows, new_rows, growth_outputs, added_outputs, elect_outputs,
                       plan, spec, check, added_manifest, incumbent_manifest, growth_outputs_digest):
    bind_elect(plan, spec, growth_outputs_digest, added_manifest, incumbent_manifest)
    rows = list(preservation_rows) + list(new_rows)
    table = growth_table(list(growth_outputs) + list(added_outputs) + list(elect_outputs))
    by_task = {row['task_id']: row for row in rows}
    parent_correct = incumbent_correct = always_added = elect_correct = oracle = 0
    unique_added = unique_incumbent = both = neither = 0
    recovered_unique_added = 0
    preserved_incumbent = 0
    preserved_old = 0
    extract = {experiment.INCUMBENT: 0, experiment.ADDED: 0, ELECT: 0}
    required_old = plan['splits']['required_success_task_ids']
    details = []
    for task_id in spec['screen']['case_task_ids']:
        row = by_task[task_id]
        parent = table[row['id'], 'base']
        incumbent = table[row['id'], experiment.INCUMBENT]
        visible = fallback.passes(parent, row, fallback.visible_tests(row), check)
        full = row['tests']
        parent_full = fallback.passes(parent, row, full, check)
        if visible:
            added = None
            elect = None
            inc_full = add_full = elect_full = parent_full
        else:
            added = table[row['id'], experiment.ADDED]
            elect = table[row['id'], ELECT]
            inc_full = fallback.passes(incumbent, row, full, check)
            add_full = fallback.passes(added, row, full, check)
            elect_full = fallback.passes(elect, row, full, check)
            extract[experiment.INCUMBENT] += int(extractable(incumbent))
            extract[experiment.ADDED] += int(extractable(added))
            extract[ELECT] += int(extractable(elect))
        parent_correct += int(parent_full)
        incumbent_correct += int(inc_full)
        always_added += int(add_full)
        elect_correct += int(parent_full if visible else elect_full)
        oracle += int(parent_full if visible else inc_full or add_full)
        if visible:
            unique = 'unused'
        elif inc_full and add_full:
            unique = 'both'
            both += 1
        elif inc_full:
            unique = experiment.INCUMBENT
            unique_incumbent += 1
        elif add_full:
            unique = experiment.ADDED
            unique_added += 1
            recovered_unique_added += int(elect_full)
        else:
            unique = 'neither'
            neither += 1
        if inc_full and (visible or elect_full):
            preserved_incumbent += 1
        if row['task_id'] in required_old and (visible or (not visible and elect_full)):
            preserved_old += 1
        details.append({
            'id': row['id'],
            'task_id': task_id,
            'visible_base_passed': visible,
            'unique': unique,
            'extractable_elect': None if visible else extractable(elect),
            'scores': {
                'parent': parent_full,
                experiment.INCUMBENT: inc_full,
                experiment.ADDED: add_full,
                ELECT: elect_full,
            },
        })
    gates = spec['screen']
    passed = (
        extract[ELECT] == gates['extractable_elect_required']
        and recovered_unique_added == gates['unique_added_recovered_required']
        and preserved_incumbent == gates['incumbent_successes_preserved_required']
        and preserved_old == gates['old_successes_preserved_required']
        and elect_correct == gates['full_test_correct_required']
    )
    return {
        'format': FORMAT + '/screen-score',
        'admission_evidence': False,
        'gpu_authorized_by_screen': False,
        'train': False,
        'trim': False,
        'scale': LAMBDA,
        'parent_correct': parent_correct,
        'incumbent_correct': incumbent_correct,
        'always_added_correct': always_added,
        'elect_correct': elect_correct,
        'oracle_union_correct': oracle,
        'unique_added': unique_added,
        'unique_added_recovered': recovered_unique_added,
        'unique_incumbent': unique_incumbent,
        'both': both,
        'neither': neither,
        'incumbent_successes_preserved': preserved_incumbent,
        'old_successes_preserved': preserved_old,
        'extractable': extract,
        'passed': passed,
        'next': 'confirmation-freeze-eligible' if passed else 'stop-this-composition',
        'details': details,
    }
