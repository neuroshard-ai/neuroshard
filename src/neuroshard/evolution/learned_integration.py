"""Learned integration of new sparse capacity.

The programming-growth campaign trained two last-four-layer tails independently
and then tried inexpensive selectors or weight arithmetic. Those challengers
failed their declared gates. This module pins the next experiment and implements
its stage-1 method: train a new last-layer expert and its gate together, compare
against matched-budget training of existing capacity, and score generated
answers on a fresh split. It does not authorize a GPU launch. It does not reopen
the original 128-task final or the opened 64 leftover cases.
`neuroshard.core.model.moe` is not this runtime.
"""
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from neuroshard.evolution.programming_expert import MBPP_SHA, code_prompt, extract_code
from neuroshard.evolution.reference_data import identity, sha256

FORMAT = 'neuroshard-learned-integration-v1'
CONTRACT_IDENTITY = '6801d1a260e622948fa90714d41be11223d36d6a5d7a000c7d173f74f174186f'
GROWTH_PLAN = 'c5662eabc3f3fc719468f3f8948d124e2dd8ee11599640bf6789ef3ab2c5f1e8'
GROWTH_FREEZE_COMMIT = 'dcc66936a746f7fdc6b6dfa25d1c47ad6f366758'
PARENT_FINAL_COUNT = 128
OPENED_DEVELOPMENT_COUNT = 64
ACTIVE_EXPERTS_PER_TOKEN = 1
STAGE1_EXPERTS_AFTER_EXPANSION = 2
INCUMBENT_EXPERT = 0
ADDED_EXPERT = 1
ROUTER_INCUMBENT_BIAS = 1.0
EVALUATION_ROLES = ('development', 'confirmation', 'retention')
TRAINING_ROLES = ('train_new', 'train_replay')
UNIQUE_ADDED_HISTORY = (276, 503, 265)
UNIQUE_INCUMBENT_HISTORY = (249,)
METHOD_FORMAT = FORMAT + '/method'
METHOD_SOURCES = (
    'config/experiments/learned-integration.json',
    'docs/LEARNED_INTEGRATION.md',
    'scripts/prepare_learned_integration.py',
    'scripts/run_learned_integration.py',
    'src/neuroshard/evolution/learned_integration.py',
    'src/neuroshard/evolution/programming_expert.py',
    'scripts/programming_sandbox.py',
)


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


class SwiGLUExpert(nn.Module):
    """Last-layer MLP: silu(gate(x)) * up(x), then down. No bias."""

    def __init__(self, hidden, intermediate):
        super().__init__()
        if hidden < 1 or intermediate < 1:
            raise ValueError('Expert dimensions must be positive')
        self.hidden = hidden
        self.intermediate = intermediate
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, hidden):
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))

    @classmethod
    def from_expert(cls, expert):
        clone = cls(expert.hidden, expert.intermediate)
        clone.load_state_dict(expert.state_dict())
        return clone


class Top1Router(nn.Module):
    """Linear hidden-to-experts. Bias initially prefers the incumbent expert."""

    def __init__(self, hidden, n_experts=STAGE1_EXPERTS_AFTER_EXPANSION):
        super().__init__()
        if n_experts != STAGE1_EXPERTS_AFTER_EXPANSION:
            raise ValueError('Stage-1 expansion is exactly one added expert')
        self.weight = nn.Parameter(torch.zeros(n_experts, hidden))
        bias = torch.zeros(n_experts)
        bias[INCUMBENT_EXPERT] = ROUTER_INCUMBENT_BIAS
        bias[ADDED_EXPERT] = -ROUTER_INCUMBENT_BIAS
        self.bias = nn.Parameter(bias)

    def logits(self, hidden):
        return F.linear(hidden, self.weight, self.bias)

    def choices(self, hidden):
        return self.logits(hidden).argmax(dim=-1)


class LastLayerMixture(nn.Module):
    """Top-1 last-layer experts. Active expert count stays at one."""

    def __init__(self, experts, router):
        super().__init__()
        if len(experts) != STAGE1_EXPERTS_AFTER_EXPANSION:
            raise ValueError('Stage-1 expansion is exactly one added expert')
        self.experts = nn.ModuleList(experts)
        self.router = router
        self.last_choices = None

    def forward(self, hidden):
        flat = hidden.reshape(-1, hidden.shape[-1])
        logits = self.router.logits(flat)
        if self.training:
            gate = F.gumbel_softmax(logits, tau=1.0, hard=True)
            output = (gate[:, INCUMBENT_EXPERT:INCUMBENT_EXPERT + 1] * self.experts[INCUMBENT_EXPERT](flat)
                      + gate[:, ADDED_EXPERT:ADDED_EXPERT + 1] * self.experts[ADDED_EXPERT](flat))
            self.last_choices = gate.argmax(dim=-1).reshape(hidden.shape[:-1])
            return output.reshape(hidden.shape)
        choices = logits.argmax(dim=-1)
        output = torch.zeros_like(flat)
        for index, expert in enumerate(self.experts):
            selected = choices == index
            if bool(selected.any()):
                output[selected] = expert(flat[selected])
        self.last_choices = choices.reshape(hidden.shape[:-1])
        return output.reshape(hidden.shape)

    @classmethod
    def expand(cls, parent):
        incumbent = SwiGLUExpert.from_expert(parent)
        added = SwiGLUExpert.from_expert(parent)
        for parameter in incumbent.parameters():
            parameter.requires_grad_(False)
        return cls([incumbent, added], Top1Router(parent.hidden))

    def expansion_parameters(self):
        return list(self.experts[ADDED_EXPERT].parameters()) + list(self.router.parameters())

    def incumbent_parameters(self):
        return list(self.experts[INCUMBENT_EXPERT].parameters())


def control_from_parent(parent):
    return SwiGLUExpert.from_expert(parent)


def active_mlp_flops_per_token(hidden, intermediate, active=ACTIVE_EXPERTS_PER_TOKEN):
    if active != ACTIVE_EXPERTS_PER_TOKEN:
        raise ValueError('Active expert count must stay frozen at one')
    return 6 * hidden * intermediate * active


def matched_optimizers(expansion, control, learning_rate):
    return (
        torch.optim.AdamW(expansion.expansion_parameters(), lr=learning_rate),
        torch.optim.AdamW(control.parameters(), lr=learning_rate),
    )


def train_step(module, hidden, target, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss = F.mse_loss(module(hidden), target)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def train_matched(expansion, control, batches, *, steps, learning_rate):
    expansion_opt, control_opt = matched_optimizers(expansion, control, learning_rate)
    history = []
    for step in range(steps):
        hidden, target = batches[step % len(batches)]
        history.append({
            'step': step,
            'expansion_loss': train_step(expansion, hidden, target, expansion_opt),
            'control_loss': train_step(control, hidden, target, control_opt),
        })
    return history


def generated_passed(text, setup, tests, check):
    try:
        program = extract_code(text)
    except (ValueError, SyntaxError):
        return False
    return bool(check(program, setup, tests)['passed'])


def _align_role(rows, spec, role):
    expected = spec['splits'][role]
    actual = [row['task_id'] for row in rows]
    if actual != expected:
        raise ValueError(role + ' rows are not the frozen split')
    if any(row.get('active_experts_per_token', ACTIVE_EXPERTS_PER_TOKEN) != ACTIVE_EXPERTS_PER_TOKEN
           for row in rows):
        raise ValueError('Active expert count must stay frozen at one')
    return rows


def percentile_95(samples):
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError('Invalid serving measurements')
    ordered = sorted(samples)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def score_gate(parent, control, expansion, spec, *, role, general, check=None):
    bind_spec(spec)
    if role == 'confirmation':
        raise ValueError('Confirmation remains closed until a later execution freeze passes development')
    if role != 'development':
        raise ValueError('Only development may be scored before confirmation opens')
    parent = _align_role(parent, spec, role)
    control = _align_role(control, spec, role)
    expansion = _align_role(expansion, spec, role)
    if general is None:
        raise ValueError('General retention identities are recorded in a later execution freeze')
    if len(general) != spec['general_retention']['documents']:
        raise ValueError('General retention count changed')
    def code_count(rows):
        if check is None:
            return sum(row['passed'] for row in rows)
        return sum(generated_passed(row['text'], row['setup'], row['tests'], check) for row in rows)

    parent_code = code_count(parent)
    control_code = code_count(control)
    expansion_code = code_count(expansion)
    retained = sum(1 for row in general if row['expansion'] == row['parent'])
    expansion_p95 = percentile_95([row['seconds'] for row in expansion])
    control_p95 = percentile_95([row['seconds'] for row in control])
    expansion_mem = percentile_95([row['peak_memory_bytes'] for row in expansion])
    control_mem = percentile_95([row['peak_memory_bytes'] for row in control])
    gate = spec['gate']
    gates = {
        'code_gain_vs_control': expansion_code >= control_code + gate['minimum_confirmation_gain_tasks_vs_control'],
        'general_exact_match': retained == gate['general_exact_match'],
        'latency_ratio': expansion_p95 <= control_p95 * gate['maximum_p95_latency_ratio_vs_control'],
        'latency_absolute': expansion_p95 <= gate['maximum_p95_seconds'],
        'memory_ratio': expansion_mem <= control_mem * gate['maximum_peak_memory_ratio_vs_control'],
        'active_experts': all(row.get('active_experts_per_token', 1) == 1 for row in expansion),
        'loss_alone_is_success': False,
    }
    return {
        'format': FORMAT + '/score',
        'role': role,
        'passed': all(value is True for key, value in gates.items() if key != 'loss_alone_is_success'),
        'gates': gates,
        'parent_code': parent_code,
        'control_code': control_code,
        'expansion_code': expansion_code,
        'general_preserved': retained,
        'expansion_p95_seconds': expansion_p95,
        'control_p95_seconds': control_p95,
        'expansion_p95_memory': expansion_mem,
        'control_p95_memory': control_mem,
        'admission_evidence': False,
        'confirmation_opened': False,
    }


def score_retention(parent, expansion, spec):
    bind_spec(spec)
    parent = _align_role(parent, spec, 'retention')
    expansion = _align_role(expansion, spec, 'retention')
    parent_code = sum(row['passed'] for row in parent)
    expansion_code = sum(row['passed'] for row in expansion)
    return {
        'retention_code_parent': parent_code,
        'retention_code_expansion': expansion_code,
        'preserves': expansion_code >= parent_code,
    }


def complete_development_gate(code_score, retention_score, spec):
    bind_spec(spec)
    if code_score.get('role') != 'development' or code_score.get('confirmation_opened') is not False:
        raise ValueError('Development gate is not bound to this score')
    gates = dict(code_score['gates'])
    gates['retention_code_at_least_parent'] = retention_score['preserves']
    passed = all(value is True for key, value in gates.items() if key != 'loss_alone_is_success')
    return {
        'format': FORMAT + '/development-gate',
        'passed': passed,
        'gates': gates,
        'confirmation_opened': False,
        'admission_evidence': False,
        'next': 'confirmation-execution-freeze-eligible' if passed else 'stop',
    }


def code_rows(task_ids, raw, gold_check):
    rows = []
    for number in task_ids:
        row = raw[number]
        verdict = gold_check(row['code'], row['test_setup_code'], row['test_list'])
        if not verdict['passed']:
            raise ValueError('Frozen gold program failed: %s' % number)
        rows.append({
            'id': identity({'dataset': MBPP_SHA, 'task': number}),
            'task_id': number,
            'kind': 'code',
            'messages': code_prompt(row),
            'reference': row['code'],
            'setup': row['test_setup_code'],
            'tests': row['test_list'],
        })
    return rows


def load_mbpp(path, spec):
    if sha256(path) != spec['corpus']['sha256']:
        raise ValueError('Source differs from the pinned benchmark')
    raw = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        raw[row['task_id']] = row
    return raw


def training_schedule(spec):
    bind_spec(spec)
    import random
    new = spec['splits']['train_new']
    replay = spec['splits']['train_replay']
    combined = [('new', i) for i in range(len(new))] + [('replay', i) for i in range(len(replay))]
    rng = random.Random(spec['training']['seed'])
    rng.shuffle(combined)
    batches = [combined[i:i + spec['training']['batch_documents']]
               for i in range(0, len(combined), spec['training']['batch_documents'])]
    schedule = []
    while len(schedule) < spec['training']['steps']:
        epoch = list(range(len(batches)))
        rng.shuffle(epoch)
        schedule.extend(epoch)
    return {'batches': batches, 'schedule': schedule[:spec['training']['steps']]}


def method_freeze():
    return {
        'format': METHOD_FORMAT,
        'spec': CONTRACT_IDENTITY,
        'train': False,
        'gpu_launch_authorized': False,
        'admission_evidence': False,
        'confirmation_opened': False,
        'existing_moe_layer_is_this_runtime': False,
        'active_experts_per_token': ACTIVE_EXPERTS_PER_TOKEN,
        'files': {name: sha256(name) for name in METHOD_SOURCES},
    }


def bind_method_freeze(freeze, spec):
    bind_spec(spec)
    expected = method_freeze()
    if freeze != expected:
        raise ValueError('Method freeze does not match the committed stage-1 method')
    if freeze.get('gpu_launch_authorized') is not False or freeze.get('train') is not False:
        raise ValueError('Method freeze does not authorize training or a GPU launch')
    return identity(freeze)


def refuse_launch(spec, freeze):
    bind_method_freeze(freeze, spec)
    raise ValueError('This specification does not authorize training or a GPU launch')
