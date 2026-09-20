"""Complementary serving after the rejected programming expert.

The opened development 15/32 figure is not an admission result. This module
implements the frozen rule: keep the parent when its public example passes,
otherwise spend one extra generation. Extra attempts share an output-token cap
and a single extra decode; they are not equal computation, because the repair
prompt is longer. The expert fallback must beat both the parent and a parent
repair attempt. The original programming-expert final stays closed.
"""
import math
import random
import subprocess
from pathlib import Path

from neuroshard.evolution.programming_expert import MBPP_SHA, extract_code
from neuroshard.evolution.reference_data import identity, sha256

FORMAT = 'neuroshard-programming-fallback-v1'
PARENT_PLAN = 'b792b308e1bccbd8dd1c149f0be1c4eab146f4e837638549a5b0ff9d2d772476'
PARENT_SELECTION = 'fbff46def4eec6580a87def14153a0a139a4efd4f73c6c0aa47e6fb98f6bd393'
PARENT_EXPERT = '46bd2e76b975c403414df1ba8b610b82d737593935c07723783fa74187b7063b'
TOKENIZER = 'e9478f6c6191dbe1c442af64c81352c531f08c5320f266a6ff7349c0f552cd5e'
REPAIR_INSTRUCTION = (
    'The previous program failed this public example:\n{example}\n\n'
    'Write a corrected complete Python program. Return only the complete '
    'Python code, including needed imports.'
)
EXECUTION_SOURCES = (
    'config/experiments/programming-expert-selection.json',
    'config/experiments/programming-expert.json',
    'config/experiments/programming-fallback.json',
    'docs/learning-reference-requirements.txt',
    'scripts/prepare_programming_fallback.py',
    'scripts/programming_sandbox.py',
    'scripts/run_programming_fallback.py',
    'src/neuroshard/evolution/programming_expert.py',
    'src/neuroshard/evolution/programming_fallback.py',
    'src/neuroshard/evolution/reference.py',
    'src/neuroshard/evolution/reference_data.py',
    'src/neuroshard/evolution/sharded/branch.py',
    'src/neuroshard/evolution/sharded/cached_inference.py',
    'src/neuroshard/evolution/sharded/model.py',
    'src/neuroshard/evolution/sharded/wire.py',
)


def visible_tests(row):
    if not row.get('tests'):
        raise ValueError('A public example test is required')
    return row['tests'][:1]


def program_text(output):
    try:
        return extract_code(output['text'])
    except (ValueError, SyntaxError, TypeError, KeyError):
        return None


def passes(output, row, tests, check):
    program = program_text(output)
    if program is None:
        return False
    return bool(check(program, row.get('setup') or '', tests)['passed'])


def repair_messages(original_messages, failed_text, visible_test):
    """Second parent attempt: original prompt, failed reply, public example only."""
    if not original_messages or original_messages[-1]['role'] != 'user':
        raise ValueError('Repair starts from the original user prompt')
    if not isinstance(failed_text, str) or not failed_text:
        raise ValueError('Repair needs the failed first attempt')
    if not isinstance(visible_test, str) or 'assert ' not in visible_test:
        raise ValueError('Repair may cite only the public example test')
    return list(original_messages) + [
        {'role': 'assistant', 'content': failed_text},
        {'role': 'user', 'content': REPAIR_INSTRUCTION.format(example=visible_test)},
    ]


def choose_after_visible_example(base, specialist, row, check):
    """Keep the parent if the prompt example passes; otherwise take the extra try."""
    if passes(base, row, visible_tests(row), check):
        return 'base', base
    return 'specialist', specialist


def leftover_task_ids(plan, *, excluded, parent_final_tasks, all_pool_tasks):
    blocked = set(excluded) | set(parent_final_tasks)
    eligible = [n for n in all_pool_tasks if 11 <= n <= 510 and n not in blocked]
    seed = plan['comparison']['seed']
    ranked = sorted(eligible, key=lambda n: identity([seed, identity({'dataset': MBPP_SHA, 'task': n})]))
    chosen = ranked[:plan['comparison']['code_tasks']]
    if chosen != plan['comparison']['task_ids']:
        raise ValueError('Frozen leftover comparison IDs changed')
    return chosen


def load_comparison_rows(plan, rows):
    """Bind loaded rows to the committed leftover list, not preparation metadata."""
    comparison = plan['comparison']
    leftover_task_ids(
        plan, excluded=comparison['excluded_parent_pool_tasks'],
        parent_final_tasks=comparison['parent_final_task_ids'],
        all_pool_tasks=list(range(11, 511)))
    if not isinstance(rows, list) or len(rows) != comparison['code_tasks']:
        raise ValueError('Comparison does not contain the frozen leftover count')
    task_ids = [row['task_id'] for row in rows]
    if task_ids != comparison['task_ids']:
        raise ValueError('Loaded comparison task IDs differ from the committed leftover list')
    if len(set(task_ids)) != len(task_ids) or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Comparison rows are not unique')
    blocked = set(comparison['excluded_parent_pool_tasks']) | set(comparison['parent_final_task_ids'])
    if any(n in blocked or not 11 <= n <= 510 for n in task_ids):
        raise ValueError('Comparison includes a reserved or out-of-range task')
    expected = [identity({'dataset': MBPP_SHA, 'task': n}) for n in task_ids]
    if [row['id'] for row in rows] != expected:
        raise ValueError('Comparison row identities do not match leftover task IDs')
    if any(row.get('kind') != 'code' for row in rows):
        raise ValueError('Comparison rows must be coding tasks')
    return rows


def require_committed(paths):
    for path in paths:
        try:
            tracked = subprocess.check_output(['git', 'show', 'HEAD:' + path], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            raise ValueError('Comparison source is not committed: ' + path) from exc
        if tracked != Path(path).read_bytes():
            raise ValueError('Comparison source is not committed: ' + path)


def source_digests():
    return {name: sha256(name) for name in EXECUTION_SOURCES}


def execution_freeze(plan, selection, source_commit):
    return {
        'format': FORMAT + '/freeze',
        'plan': identity(plan),
        'selection': identity(selection),
        'parent_plan': PARENT_PLAN,
        'parent_selection': PARENT_SELECTION,
        'parent_expert': PARENT_EXPERT,
        'tokenizer': TOKENIZER,
        'source_commit': source_commit,
        'sources': source_digests(),
    }


def bind_execution(plan, selection, expert_manifest, tokenizer_digest):
    if plan.get('format') != FORMAT:
        raise ValueError('Fallback plan does not bind this comparison')
    if (plan.get('parent_plan') != PARENT_PLAN
            or plan.get('parent_selection') != PARENT_SELECTION
            or identity(selection) != PARENT_SELECTION
            or selection.get('plan') != PARENT_PLAN):
        raise ValueError('Selected parent execution changed')
    if plan.get('tokenizer') != TOKENIZER or tokenizer_digest != TOKENIZER:
        raise ValueError('Tokenizer differs from the rejected parent trial')
    if (plan.get('parent_expert') != PARENT_EXPERT
            or identity(expert_manifest) != PARENT_EXPERT
            or expert_manifest.get('plan') != PARENT_PLAN
            or expert_manifest.get('selection') != PARENT_SELECTION):
        raise ValueError('Expert checkpoint is not the rejected trial terminal')
    leftover_task_ids(
        plan, excluded=plan['comparison']['excluded_parent_pool_tasks'],
        parent_final_tasks=plan['comparison']['parent_final_task_ids'],
        all_pool_tasks=list(range(11, 511)))
    return {
        'plan': identity(plan),
        'selection': identity(selection),
        'parent_expert': identity(expert_manifest),
        'tokenizer': tokenizer_digest,
    }


def validate_freeze(plan, selection, freeze):
    expected = execution_freeze(plan, selection, freeze.get('source_commit'))
    if freeze.get('format') != FORMAT + '/freeze' or freeze != expected:
        raise ValueError('Execution freeze does not match the committed comparison')
    require_committed([*EXECUTION_SOURCES, 'config/experiments/programming-fallback-freeze.json'])


def extra_attempt_recorded(output, *, path, prompt_kind, original_prompt_ids):
    """A real extra decode may repeat tokens; a copied first attempt may not."""
    if not output.get('generated'):
        raise ValueError('Extra attempt did not record a generation call')
    if output.get('path') != path:
        raise ValueError('Extra attempt executed the wrong path')
    if output.get('prompt_kind') != prompt_kind:
        raise ValueError('Extra attempt used the wrong prompt')
    prompt_ids = output.get('prompt_ids')
    if not isinstance(prompt_ids, list) or not prompt_ids:
        raise ValueError('Extra attempt did not record its prompt')
    if prompt_kind == 'original' and prompt_ids != original_prompt_ids:
        raise ValueError('Expert fallback must use the original prompt tokens')
    if prompt_kind == 'repair' and prompt_ids == original_prompt_ids:
        raise ValueError('Parent repair reused the original prompt')
    if output.get('input_tokens') != len(prompt_ids) or output.get('output_tokens') != len(output.get('ids') or []):
        raise ValueError('Extra attempt token counts do not match the recorded call')


def _usage(base, extra, visible_pass):
    check = float(base.get('check_seconds') or 0)
    if check < 0 or not math.isfinite(check):
        raise ValueError('Invalid public-example checking time')
    if visible_pass:
        seconds = base['seconds'] + check
        input_tokens = base['input_tokens']
        output_tokens = base['output_tokens']
    else:
        seconds = base['seconds'] + extra['seconds'] + check
        input_tokens = base['input_tokens'] + extra['input_tokens']
        output_tokens = base['output_tokens'] + extra['output_tokens']
    return {'seconds': seconds, 'input_tokens': input_tokens,
            'output_tokens': output_tokens, 'check_seconds': check}


def score_comparison(rows, outputs, plan, check):
    """Paired executable scores for base, expert fallback, and parent repair."""
    arms = ('base', 'expert', 'repair')
    expected = {(row['id'], arm) for row in rows for arm in arms}
    actual = [(out['id'], out['arm']) for out in outputs]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Incomplete or duplicated comparison coverage')
    table = {(out['id'], out['arm']): out for out in outputs}
    details = []
    vs_base, vs_repair = [], []
    fallback_usage, repair_usage = [], []
    for row in rows:
        current = {arm: table[row['id'], arm] for arm in arms}
        visible = visible_tests(row)
        full = row['tests']
        base_visible = passes(current['base'], row, visible, check)
        choice, chosen = choose_after_visible_example(
            current['base'], current['expert'], row, check)
        repair_choice, repaired = ('base', current['base']) if base_visible else (
            'repair', current['repair'])
        scores = {
            'base': passes(current['base'], row, full, check),
            'expert_fallback': passes(chosen, row, full, check),
            'base_repair': passes(repaired, row, full, check),
        }
        original_prompt = current['base'].get('prompt_ids')
        if not isinstance(original_prompt, list) or not original_prompt:
            raise ValueError('Parent attempt did not record its prompt')
        if not base_visible:
            extra_attempt_recorded(
                current['expert'], path='expert', prompt_kind='original',
                original_prompt_ids=original_prompt)
            extra_attempt_recorded(
                current['repair'], path='parent', prompt_kind='repair',
                original_prompt_ids=original_prompt)
        vs_base.append(int(scores['expert_fallback']) - int(scores['base']))
        vs_repair.append(int(scores['expert_fallback']) - int(scores['base_repair']))
        fallback_usage.append(_usage(current['base'], current['expert'], base_visible))
        repair_usage.append(_usage(current['base'], current['repair'], base_visible))
        details.append({
            'id': row['id'], 'visible_base_passed': base_visible,
            'expert_choice': choice, 'repair_choice': repair_choice,
            'scores': scores,
            'repeated_expert_tokens': (not base_visible
                                       and current['expert']['ids'] == current['base']['ids']),
            'repeated_repair_tokens': (not base_visible
                                       and current['repair']['ids'] == current['base']['ids']),
            'fallback_usage': fallback_usage[-1],
            'repair_usage': repair_usage[-1],
        })
    rng = random.Random(plan['seed'] + 1)
    boot = sorted(sum(rng.choices(vs_repair, k=len(vs_repair))) / len(vs_repair)
                  for _ in range(plan['bootstrap_samples']))
    lower = boot[int(.05 * len(boot))]

    def percentile(samples, *, positive=True):
        values = sorted(samples)
        if any(not math.isfinite(x) or x < 0 or (positive and x <= 0) for x in values):
            raise ValueError('Invalid serving measurements')
        return values[math.ceil(.95 * len(values)) - 1]

    fallback_p95 = percentile([row['seconds'] for row in fallback_usage])
    repair_p95 = percentile([row['seconds'] for row in repair_usage])
    gates = {
        'gain_vs_base': sum(vs_base) / len(vs_base) >= plan['gate']['minimum_code_gain'],
        'gain_vs_repair': sum(vs_repair) / len(vs_repair) >= plan['gate']['minimum_code_gain'],
        'gain_lower_vs_repair': lower > 0,
        'latency_ratio': fallback_p95 <= repair_p95 * plan['gate']['maximum_p95_ratio'],
        'latency_absolute': fallback_p95 <= plan['gate']['maximum_p95_seconds'],
    }
    return {
        'format': FORMAT + '/score',
        'passed': all(gates.values()),
        'gates': gates,
        'code_tasks': len(rows),
        'net_vs_base': sum(vs_base),
        'net_vs_repair': sum(vs_repair),
        'gain_lower_95_vs_repair': lower,
        'fallback_p95_seconds': fallback_p95,
        'repair_p95_seconds': repair_p95,
        'fallback_p95_input_tokens': percentile([row['input_tokens'] for row in fallback_usage]),
        'repair_p95_input_tokens': percentile([row['input_tokens'] for row in repair_usage]),
        'fallback_p95_output_tokens': percentile([row['output_tokens'] for row in fallback_usage]),
        'repair_p95_output_tokens': percentile([row['output_tokens'] for row in repair_usage]),
        'check_p95_seconds': percentile([row['check_seconds'] for row in fallback_usage], positive=False),
        'equal_extra_attempt': True,
        'equal_output_token_cap': True,
        'equal_computation': False,
        'details': details,
        'original_final_opened': False,
    }


def score_opened_development_diagnostic(rows, outputs, check):
    """Recompute the posthoc 15/32 policy. Missing repair is recorded, not imagined."""
    table = {(out['id'], out['arm']): out for out in outputs}
    details = []
    correct = 0
    for row in rows:
        if row['kind'] != 'code':
            continue
        base, expert = table[row['id'], 'base'], table[row['id'], 'replacement']
        choice, chosen = choose_after_visible_example(base, expert, row, check)
        ok = passes(chosen, row, row['tests'], check)
        correct += int(ok)
        details.append({'id': row['id'], 'choice': choice, 'correct': ok,
                        'base': passes(base, row, row['tests'], check),
                        'expert': passes(expert, row, row['tests'], check)})
    return {'format': FORMAT + '/opened-development-diagnostic',
            'admission_evidence': False, 'new_method_not_frozen_before_observation': True,
            'missing_control': 'Parent repair generation under the same extra-attempt budget',
            'count': len(details), 'expert_fallback_correct': correct, 'details': details}
