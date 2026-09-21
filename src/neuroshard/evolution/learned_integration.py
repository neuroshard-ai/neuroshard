"""Learned integration of new sparse capacity.

The programming-growth campaign trained two last-four-layer tails independently
and then tried inexpensive selectors or weight arithmetic. Those challengers
failed their declared gates. This module pins the next experiment: train a new
module and its gate together, compare against matched-budget training of existing
capacity, and score generated answers on a fresh split. It does not train. It
does not authorize a GPU launch. It does not reopen the original 128-task final
or the opened 64 leftover cases.
"""
import json
from pathlib import Path

from neuroshard.evolution.reference_data import identity

FORMAT = 'neuroshard-learned-integration-v1'
CONTRACT_IDENTITY = '6801d1a260e622948fa90714d41be11223d36d6a5d7a000c7d173f74f174186f'
GROWTH_PLAN = 'c5662eabc3f3fc719468f3f8948d124e2dd8ee11599640bf6789ef3ab2c5f1e8'
GROWTH_FREEZE_COMMIT = 'dcc66936a746f7fdc6b6dfa25d1c47ad6f366758'
PARENT_FINAL_COUNT = 128
OPENED_DEVELOPMENT_COUNT = 64
ACTIVE_EXPERTS_PER_TOKEN = 1
STAGE1_EXPERTS_AFTER_EXPANSION = 2
EVALUATION_ROLES = ('development', 'confirmation', 'retention')
TRAINING_ROLES = ('train_new', 'train_replay')
UNIQUE_ADDED_HISTORY = (276, 503, 265)
UNIQUE_INCUMBENT_HISTORY = (249,)


def spec_path():
    marker = Path('config/experiments/learned-integration.json')
    for parent in Path(__file__).resolve().parents:
        candidate = parent / marker
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('learned-integration.json is not next to this source tree')


def load_spec(path=None):
    return json.loads(Path(path or spec_path()).read_text())


def bind_spec(spec):
    if identity(spec) != CONTRACT_IDENTITY:
        raise ValueError('Learned-integration contract does not bind this measurement')
    if spec.get('format') != FORMAT:
        raise ValueError('Learned-integration format changed')
    if spec.get('train') is not False or spec.get('gpu_launch_authorized') is not False:
        raise ValueError('This specification does not authorize training or a GPU launch')
    if spec.get('admission_evidence') is not False:
        raise ValueError('Learned integration may not count as admission')
    if spec.get('original_final_opened') is not False:
        raise ValueError('Learned integration does not open the original 128-task final')
    if spec.get('opened_64_reusable_as_evaluation') is not False:
        raise ValueError('Opened leftover cases remain development history')
    if spec.get('confirmation_opened') is not False:
        raise ValueError('Confirmation remains closed until a later execution freeze passes development')
    if spec.get('fresh_dataset_alone_is_not_the_experiment') is not True:
        raise ValueError('A fresh dataset alone is not this experiment')
    if spec.get('growth_plan') != GROWTH_PLAN:
        raise ValueError('Campaign close is not bound to the failed growth plan')
    if spec.get('growth_freeze_commit') != GROWTH_FREEZE_COMMIT:
        raise ValueError('Campaign close is not bound to the failed growth freeze')
    if spec.get('parent_final_task_count') != PARENT_FINAL_COUNT:
        raise ValueError('Original 128-task final identity changed')
    if spec.get('opened_development_task_count') != OPENED_DEVELOPMENT_COUNT:
        raise ValueError('Opened leftover-case count changed')
    architecture = spec.get('architecture') or {}
    if architecture.get('active_experts_per_token') != ACTIVE_EXPERTS_PER_TOKEN:
        raise ValueError('Active expert count must stay frozen at one')
    if architecture.get('experts_after_expansion') != STAGE1_EXPERTS_AFTER_EXPANSION:
        raise ValueError('Stage-1 expansion is exactly one added expert')
    if architecture.get('accepted_modules_frozen') is not True:
        raise ValueError('Accepted modules stay frozen while the new expert and gate train')
    if architecture.get('existing_moe_layer_is_this_runtime') is not False:
        raise ValueError('Existing MoE code is not this experiment runtime')
    if spec.get('control', {}).get('matched_data_and_steps') is not True:
        raise ValueError('The control must reuse the expansion data and step count')
    if spec.get('success', {}).get('loss_alone_is_success') is not False:
        raise ValueError('Lower loss is not success')
    if spec.get('stage2', {}).get('authorized') is not False:
        raise ValueError('Stage 2 is not authorized by this specification')
    if spec.get('gate', {}).get('development_must_pass_before_confirmation') is not True:
        raise ValueError('Confirmation cannot open before development passes')
    forbidden = set(spec['forbidden_evaluation_task_ids'])
    if len(forbidden) != spec['counts']['forbidden_evaluation']:
        raise ValueError('Forbidden-evaluation count changed')
    if any(task_id not in forbidden for task_id in UNIQUE_ADDED_HISTORY + UNIQUE_INCUMBENT_HISTORY):
        raise ValueError('Opened complementary cases must stay ineligible')
    seen = set()
    for role in TRAINING_ROLES + EVALUATION_ROLES:
        task_ids = spec['splits'][role]
        if len(task_ids) != spec['counts'][role]:
            raise ValueError(f'{role} split length changed')
        if len(set(task_ids)) != len(task_ids):
            raise ValueError(f'{role} split has duplicate task ids')
        overlap = set(task_ids) & (forbidden | seen)
        if overlap:
            raise ValueError('Evaluation or training split overlaps burned or other splits')
        seen.update(task_ids)
    return {
        'learned_integration': identity(spec),
        'growth_plan': GROWTH_PLAN,
        'gpu_launch_authorized': False,
        'admission_evidence': False,
    }
