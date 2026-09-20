"""Second admitted programming tail against the leftover fallback baseline.

Available capacity may grow by one disjoint expert. Each request still spends
at most one extra decode after the public example fails. Isolation of the new
tail must pass before the expanded system is scored against the complete
fallback baseline. The original 128-task final stays closed.
"""
import math
import random
import subprocess
from pathlib import Path

from neuroshard.evolution import programming_expert as parent
from neuroshard.evolution import programming_fallback as fallback
from neuroshard.evolution.reference_data import identity, sha256

FORMAT = 'neuroshard-programming-growth-v1'
PARENT_PLAN = fallback.PARENT_PLAN
PARENT_SELECTION = fallback.PARENT_SELECTION
INCUMBENT_EXPERT = fallback.PARENT_EXPERT
TOKENIZER = fallback.TOKENIZER
PARENT_FALLBACK_PLAN = '25a8cdcafdccd772cb2f0ea2b19c5934d9e19e6417e37ee7bdde912ffc4cb76a'
BASELINE_OUTPUTS = '3a34dea159358de4d910ff3f8ef23dee948ad98939c1b696a457afb962921037'
INCUMBENT = 'incumbent'
ADDED = 'added'
MERGED = 'merged'
EXECUTION_SOURCES = (
    'config/experiments/programming-expert-selection.json',
    'config/experiments/programming-expert.json',
    'config/experiments/programming-fallback.json',
    'config/experiments/programming-growth.json',
    'docs/learning-reference-requirements.txt',
    'scripts/prepare_programming_growth.py',
    'scripts/programming_sandbox.py',
    'scripts/run_programming_fallback.py',
    'scripts/run_programming_growth.py',
    'src/neuroshard/evolution/programming_expert.py',
    'src/neuroshard/evolution/programming_fallback.py',
    'src/neuroshard/evolution/programming_growth.py',
    'src/neuroshard/evolution/reference.py',
    'src/neuroshard/evolution/reference_data.py',
    'src/neuroshard/evolution/sharded/branch.py',
    'src/neuroshard/evolution/sharded/cached_inference.py',
    'src/neuroshard/evolution/sharded/model.py',
    'src/neuroshard/evolution/sharded/wire.py',
)


def leftover_comparison(plan):
    return {
        'seed': plan['leftover_seed'],
        'excluded_parent_pool_tasks': plan['excluded_parent_pool_tasks'],
        'parent_final_task_ids': plan['parent_final_task_ids'],
    }


def leftover_ranking(plan):
    return fallback.ranked_leftover_tasks(leftover_comparison(plan))


def bind_splits(plan):
    """Recompute leftover continuation splits from the frozen leftover ranking."""
    ranked = leftover_ranking(plan)
    preservation = ranked[:plan['counts']['preservation']]
    rest = ranked[plan['counts']['preservation']:]
    new = rest[:plan['counts']['new']]
    development = rest[plan['counts']['new']:plan['counts']['new'] + plan['counts']['development']]
    train = rest[plan['counts']['new'] + plan['counts']['development']:]
    splits = plan['splits']
    if preservation != splits['preservation_task_ids']:
        raise ValueError('Preservation leftover IDs changed')
    if new != splits['new_task_ids']:
        raise ValueError('New-answer leftover IDs changed')
    if development != splits['development_task_ids']:
        raise ValueError('Second-expert development IDs changed')
    if train != splits['train_task_ids']:
        raise ValueError('Second-expert training IDs changed')
    blocked = set(plan['excluded_parent_pool_tasks']) | set(plan['parent_final_task_ids'])
    used = preservation + new + development + train
    if len(set(used)) != len(used):
        raise ValueError('Growth splits overlap')
    if used != ranked:
        raise ValueError('Growth splits do not exhaust the leftover ranking')
    if any(n in blocked or not 11 <= n <= 510 for n in used):
        raise ValueError('Growth splits include a reserved or out-of-range task')
    if set(splits['incumbent_train_task_ids']) & set(used):
        raise ValueError('Second expert trains on the incumbent expert training IDs')
    required = splits['required_success_task_ids']
    if any(n not in set(preservation) for n in required):
        raise ValueError('Required successes are not in the leftover preservation set')
    if required != [n for n in preservation if n in set(required)]:
        raise ValueError('Required success IDs must keep leftover ranking order')
    merge_scales(plan)
    return splits


def merge_scales(plan):
    """Unit task-vector addition: θ_parent + (θ_inc − θ_parent) + (θ_add − θ_parent)."""
    merge = plan.get('merge')
    if merge != {'incumbent': 1, 'added': 1}:
        raise ValueError('Only unit task-vector addition of both tails is frozen')
    return merge


def task_vector_merge(parent, incumbent, added, incumbent_scale, added_scale):
    """θ_parent + λ_inc (θ_inc − θ_parent) + λ_add (θ_add − θ_parent)."""
    merge_scales({'merge': {'incumbent': incumbent_scale, 'added': added_scale}})
    return parent + incumbent_scale * (incumbent - parent) + added_scale * (added - parent)


def load_role_rows(plan, rows, task_ids, *, role):
    if not isinstance(rows, list) or len(rows) != len(task_ids):
        raise ValueError('Loaded ' + role + ' rows do not match the frozen count')
    actual = [row['task_id'] for row in rows]
    if actual != task_ids:
        raise ValueError('Loaded ' + role + ' task IDs differ from the committed split')
    if len(set(actual)) != len(actual) or len({row['id'] for row in rows}) != len(rows):
        raise ValueError(role + ' rows are not unique')
    expected = [identity({'dataset': parent.MBPP_SHA, 'task': n}) for n in task_ids]
    if [row['id'] for row in rows] != expected:
        raise ValueError(role + ' row identities do not match task IDs')
    if any(row.get('kind') != 'code' for row in rows):
        raise ValueError(role + ' rows must be coding tasks')
    bind_splits(plan)
    return rows


def require_committed(paths):
    for path in paths:
        try:
            tracked = subprocess.check_output(['git', 'show', 'HEAD:' + path], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            raise ValueError('Growth source is not committed: ' + path) from exc
        if tracked != Path(path).read_bytes():
            raise ValueError('Growth source is not committed: ' + path)


def source_digests():
    return {name: sha256(name) for name in EXECUTION_SOURCES}


def execution_freeze(plan, selection, source_commit):
    return {
        'format': FORMAT + '/freeze',
        'plan': identity(plan),
        'selection': identity(selection),
        'parent_plan': PARENT_PLAN,
        'parent_selection': PARENT_SELECTION,
        'parent_fallback_plan': PARENT_FALLBACK_PLAN,
        'incumbent_expert': INCUMBENT_EXPERT,
        'tokenizer': TOKENIZER,
        'source_commit': source_commit,
        'sources': source_digests(),
    }


def bind_execution(plan, selection, incumbent_manifest, tokenizer_digest):
    if plan.get('format') != FORMAT:
        raise ValueError('Growth plan does not bind this comparison')
    if (plan.get('parent_plan') != PARENT_PLAN
            or plan.get('parent_selection') != PARENT_SELECTION
            or identity(selection) != PARENT_SELECTION
            or selection.get('plan') != PARENT_PLAN):
        raise ValueError('Selected parent execution changed')
    if plan.get('tokenizer') != TOKENIZER or tokenizer_digest != TOKENIZER:
        raise ValueError('Tokenizer differs from the leftover fallback baseline')
    if (plan.get('incumbent_expert') != INCUMBENT_EXPERT
            or identity(incumbent_manifest) != INCUMBENT_EXPERT
            or incumbent_manifest.get('plan') != PARENT_PLAN
            or incumbent_manifest.get('selection') != PARENT_SELECTION):
        raise ValueError('Incumbent checkpoint is not the leftover fallback tail')
    if plan.get('parent_fallback_plan') != PARENT_FALLBACK_PLAN:
        raise ValueError('Growth is not bound to the leftover fallback plan')
    bind_splits(plan)
    return {
        'plan': identity(plan),
        'selection': identity(selection),
        'incumbent_expert': identity(incumbent_manifest),
        'tokenizer': tokenizer_digest,
    }


def validate_freeze(plan, selection, freeze):
    expected = execution_freeze(plan, selection, freeze.get('source_commit'))
    if freeze.get('format') != FORMAT + '/freeze' or freeze != expected:
        raise ValueError('Execution freeze does not match the committed growth comparison')
    require_committed([*EXECUTION_SOURCES, 'config/experiments/programming-growth-freeze.json'])


def _percentile(samples, *, positive=True):
    values = sorted(samples)
    if any(not math.isfinite(x) or x < 0 or (positive and x <= 0) for x in values):
        raise ValueError('Invalid serving measurements')
    return values[math.ceil(.95 * len(values)) - 1]


def score_isolation(rows, outputs, plan, check):
    """The new tail must be complementary on its own development slice first."""
    isolation_plan = {
        'seed': plan['seed'],
        'bootstrap_samples': plan['bootstrap_samples'],
        'gate': plan['isolation_gate'],
    }
    result = fallback.score_comparison(rows, outputs, isolation_plan, check)
    result['format'] = FORMAT + '/isolation'
    result['admission_evidence'] = False
    result['original_final_opened'] = False
    result['new_answers_opened'] = False
    return result


def score_growth(preservation_rows, new_rows, outputs, plan, check):
    """Expanded merged-tail extra versus the complete leftover fallback system."""
    bind_splits(plan)
    preservation_rows = load_role_rows(
        plan, preservation_rows, plan['splits']['preservation_task_ids'], role='preservation')
    new_rows = load_role_rows(plan, new_rows, plan['splits']['new_task_ids'], role='new')
    rows = list(preservation_rows) + list(new_rows)
    arms = ('base', INCUMBENT, MERGED)
    expected = {(row['id'], arm) for row in rows for arm in arms}
    actual = [(out['id'], out['arm']) for out in outputs]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Incomplete or duplicated growth coverage')
    table = {(out['id'], out['arm']): out for out in outputs}
    preservation_ids = {row['id'] for row in preservation_rows}
    required = {identity({'dataset': parent.MBPP_SHA, 'task': n})
                for n in plan['splits']['required_success_task_ids']}
    details = []
    vs_baseline_new = []
    preserved_successes = []
    baseline_usage, expanded_usage = [], []
    extras = {INCUMBENT: 0, MERGED: 0}
    for row in rows:
        current = {arm: table[row['id'], arm] for arm in arms}
        visible = fallback.visible_tests(row)
        full = row['tests']
        base_visible = fallback.passes(current['base'], row, visible, check)
        baseline_choice, baseline_answer = fallback.choose_after_visible_example(
            current['base'], current[INCUMBENT], row, check)
        expanded_choice, expanded_answer = fallback.choose_after_visible_example(
            current['base'], current[MERGED], row, check)
        if not base_visible:
            original_prompt = current['base'].get('prompt_ids')
            fallback.extra_attempt_recorded(
                current[INCUMBENT], path='expert', prompt_kind='original',
                original_prompt_ids=original_prompt)
            fallback.extra_attempt_recorded(
                current[MERGED], path='expert', prompt_kind='original',
                original_prompt_ids=original_prompt)
            extras[INCUMBENT] += 1
            extras[MERGED] += 1
        scores = {
            'base': fallback.passes(current['base'], row, full, check),
            'baseline': fallback.passes(baseline_answer, row, full, check),
            'expanded': fallback.passes(expanded_answer, row, full, check),
        }
        usage_baseline = fallback._usage(current['base'], current[INCUMBENT], base_visible)
        usage_expanded = fallback._usage(current['base'], current[MERGED], base_visible)
        baseline_usage.append(usage_baseline)
        expanded_usage.append(usage_expanded)
        if row['id'] in preservation_ids and row['id'] in required:
            preserved_successes.append(bool(scores['expanded']))
        if row['id'] not in preservation_ids:
            vs_baseline_new.append(int(scores['expanded']) - int(scores['baseline']))
        details.append({
            'id': row['id'], 'task_id': row['task_id'],
            'set': 'preservation' if row['id'] in preservation_ids else 'new',
            'visible_base_passed': base_visible,
            'selected_extra': MERGED if not base_visible else 'none',
            'baseline_choice': baseline_choice,
            'expanded_choice': expanded_choice,
            'scores': scores,
            'baseline_usage': usage_baseline,
            'expanded_usage': usage_expanded,
        })
    rng = random.Random(plan['seed'] + 1)
    boot = sorted(sum(rng.choices(vs_baseline_new, k=len(vs_baseline_new))) / len(vs_baseline_new)
                  for _ in range(plan['bootstrap_samples']))
    lower = boot[int(.05 * len(boot))]
    expanded_p95 = _percentile([row['seconds'] for row in expanded_usage])
    baseline_p95 = _percentile([row['seconds'] for row in baseline_usage])
    gates = {
        'preserved_successes': all(preserved_successes) and len(preserved_successes) == len(required),
        'gain_vs_baseline_new': (sum(vs_baseline_new) / len(vs_baseline_new)
                                 >= plan['growth_gate']['minimum_code_gain']),
        'gain_lower_vs_baseline_new': lower > 0,
        'latency_ratio': expanded_p95 <= baseline_p95 * plan['growth_gate']['maximum_p95_ratio'],
        'latency_absolute': expanded_p95 <= plan['growth_gate']['maximum_p95_seconds'],
        'single_extra_decode': True,
    }
    return {
        'format': FORMAT + '/score',
        'passed': all(gates.values()),
        'gates': gates,
        'preservation_tasks': len(preservation_rows),
        'new_tasks': len(new_rows),
        'required_successes': len(required),
        'preserved_required_successes': sum(preserved_successes),
        'net_vs_baseline_new': sum(vs_baseline_new),
        'gain_lower_95_vs_baseline_new': lower,
        'expanded_p95_seconds': expanded_p95,
        'baseline_p95_seconds': baseline_p95,
        'expanded_p95_input_tokens': _percentile([row['input_tokens'] for row in expanded_usage]),
        'baseline_p95_input_tokens': _percentile([row['input_tokens'] for row in baseline_usage]),
        'visible_fail_extras': extras,
        'maximum_extra_decodes': 1,
        'equal_computation': False,
        'task_vector_merge': True,
        'merge': merge_scales(plan),
        'original_final_opened': False,
        'details': details,
    }
