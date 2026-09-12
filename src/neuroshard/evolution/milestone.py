"""Pre-registered learning milestone: public plan, frozen sealed set, fail-closed gates.

This module does not train. It refuses to treat an uncommitted sealed set as ready
for training, and it will not lower the published quality margins.
"""
import json
from pathlib import Path

FORMAT = 'neuroshard-learning-milestone-v1'
PLAN_PATH = Path(__file__).resolve().parents[3] / 'config/experiments/learning-milestone.json'
SELECTION_NAME = 'learning-milestone-selection.json'


def load(path=None):
    path = Path(path) if path else PLAN_PATH
    plan = json.loads(path.read_text())
    validate(plan)
    return plan


def validate(plan):
    if plan.get('format') != FORMAT:
        raise ValueError('Unsupported learning-milestone format')
    if plan.get('license') != 'Apache-2.0':
        raise ValueError('Milestone materials must stay under Apache-2.0')
    if plan.get('public_network') is None or 'unchanged' not in plan['public_network']:
        raise ValueError('This plan must not authorize a public-network change')
    open_source = plan['open_source']
    if open_source.get('secret_evaluation') is not False:
        raise ValueError('A secret evaluation set is incompatible with public reproduction')
    if not open_source.get('independent_reproduction'):
        raise ValueError('The plan must remain independently reproducible')
    if Path(open_source['selection_path']).name != SELECTION_NAME:
        raise ValueError('Selection artifacts belong in the published experiments directory')
    seed = plan['seed']
    if seed['growth_layers'] != 0 or seed['parameters'] != 134515008:
        raise ValueError('Learning uses the 135M seed with no growth')
    if plan['optimizer']['recipes'] != 1:
        raise ValueError('A second recipe requires a new sealed set and a new plan revision')
    if plan['optimizer']['learning_rate'] != 0.003 or plan['optimizer']['clip_norm'] != 1.0:
        raise ValueError('Optimizer constants differ from the frozen plan')
    evaluation = plan['evaluation']
    if evaluation['documents_per_role'] != 64 or evaluation['minimum_examples'] != 64:
        raise ValueError('Evaluation count differs from the frozen plan')
    if evaluation['z'] != 2.576 or evaluation['fresh_min_gain'] != 0.001 or evaluation['retention_margin'] != 0.02:
        raise ValueError('Quality margins cannot be relaxed')
    if evaluation['generation_is_pass_fail'] is not False:
        raise ValueError('Generations are published evidence, not a substitute for the loss gate')
    if plan['budget']['training_steps'] != 128 or plan['budget']['hosts'] != 1:
        raise ValueError('Learning budget differs from the frozen one-host recipe')
    learning = plan['learning']
    if learning['replay_fraction'] != 0.0 or learning['pass_roles'] != ['test', 'retention']:
        raise ValueError('Learning pass rule differs from the frozen plan')
    if plan['continual']['blocked_on'] != 'learning.pass' or plan['scaling']['blocked_on'] != 'learning.pass':
        raise ValueError('Continual learning and scaling start only after a learning pass')
    if plan['continual']['replay_source'] != 'trained-windows-of-learning':
        raise ValueError('Continual replay must reuse trained windows, not unused admitted ones')
    if plan['scaling']['primary'] != 'reliability':
        raise ValueError('Scaling success is recovery with measured overhead, not implied capacity')
    stop = plan['stop_rules']
    required = {
        'training_before_selection': 'forbidden',
        'margin_relaxation': 'forbidden',
        'second_recipe_on_same_sealed_set': 'forbidden',
        'growth_during_learning': 'forbidden',
        'continual_or_scaling_without_learning_pass': 'forbidden',
        'exhausted_budget_without_pass': 'fail the phase',
    }
    if any(stop.get(key) != value for key, value in required.items()):
        raise ValueError('Stop rules differ from the frozen plan')
    if plan['status'] not in ('plan-frozen', 'selection-committed', 'learning-running',
                              'learning-passed', 'learning-failed', 'complete'):
        raise ValueError('Unknown milestone status')
    if plan['status'] in ('selection-committed', 'learning-running', 'learning-passed', 'complete') and not committed_selection(plan):
        raise ValueError('Training status requires a committed sealed-set manifest')


def selection_path(plan):
    return Path(__file__).resolve().parents[3] / plan['open_source']['selection_path']


def committed_selection(plan):
    path = selection_path(plan)
    if not path.is_file():
        return False
    selection = json.loads(path.read_text())
    for field in ('format', 'baseline', 'documents', 'generation_prompts', 'plan_digest'):
        if field not in selection:
            raise ValueError('Incomplete sealed-set manifest')
    if selection.get('format') != 'neuroshard-learning-milestone-selection-v1':
        raise ValueError('Unsupported sealed-set format')
    documents = selection['documents']
    count = plan['evaluation']['documents_per_role']
    if any(len(documents.get(role, ())) != count for role in ('retention', 'fresh', 'test')):
        raise ValueError('Sealed set does not contain the declared document counts')
    if len(selection['generation_prompts']) != plan['evaluation']['generation_prompts']:
        raise ValueError('Generation probe count differs from the plan')
    return True


def training_allowed(plan):
    validate(plan)
    return plan['status'] == 'selection-committed' and committed_selection(plan)


def decide_learning(measurements, plan=None):
    """Pass only on sealed test improvement and frozen retention.

    `fresh` is reported and must be present; it cannot promote a failing test set.
    """
    from .evaluation import comparison
    plan = plan or load()
    evaluation = plan['evaluation']
    test = comparison(measurements['baseline']['test'], measurements['candidate']['test'],
                      -evaluation['fresh_min_gain'], evaluation['minimum_examples'], evaluation['z'])
    retention = comparison(measurements['baseline']['retention'], measurements['candidate']['retention'],
                           evaluation['retention_margin'], evaluation['minimum_examples'], evaluation['z'])
    fresh = comparison(measurements['baseline']['fresh'], measurements['candidate']['fresh'],
                        -evaluation['fresh_min_gain'], evaluation['minimum_examples'], evaluation['z'])
    return {
        'pass': test['passes'] and retention['passes'],
        'test': test,
        'retention': retention,
        'fresh': fresh,
        'scope': 'sealed test plus frozen retention; fresh is reported only',
    }
