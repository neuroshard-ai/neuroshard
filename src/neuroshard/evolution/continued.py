"""Frozen continued-learning contract: artifact hashes, generated-answer gain, no growth.

This module does not train and does not settle native rewards. Training is forbidden
until the prepared artifact freeze is Git-committed. Quality scoring cannot promote
serving or mint tokens.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

from . import grounded_tasks as tasks
from .reference_data import identity

FORMAT = 'neuroshard-continued-learning-v1'
PREPARED_FORMAT = 'neuroshard-continued-learning-prepared-v1'
SELECTION_FORMAT = 'neuroshard-continued-learning-selection-v1'
PLAN_PATH = Path(__file__).resolve().parents[3] / 'config/experiments/continued-learning.json'
PREPARED_NAME = 'continued-learning-prepared.json'
SELECTION_NAME = 'continued-learning-selection.json'
DIGEST = re.compile(r'[0-9a-f]{64}')
STATUSES = ('plan-frozen', 'prepared-committed', 'learning-running',
            'selection-committed', 'learning-passed', 'learning-failed')
RUNTIME_KEYS = (
    'device', 'python', 'machine', 'torch', 'transformers', 'tokenizers', 'safetensors',
    'numpy', 'jinja2', 'threads', 'cuda', 'cpu_dispatch', 'mkl_instructions', 'gpu',
    'parameters', 'optimizer', 'autocast', 'attention', 'deterministic_algorithms',
    'allocator', 'cross_device_exact', 'torch_build',
)
SOURCE_PATHS = (
    'scripts/run_sharded_training.py',
    'scripts/prepare_continued_learning.py',
    'scripts/score_continued_learning.py',
    'docs/learning-reference-requirements.txt',
    'src/neuroshard/dataflow/store.py',
    'src/neuroshard/dataflow/collect.py',
    'src/neuroshard/evolution/continued.py',
    'src/neuroshard/evolution/reference.py',
    'src/neuroshard/evolution/reference_data.py',
    'src/neuroshard/evolution/grounded_tasks.py',
    'src/neuroshard/evolution/data.py',
    'src/neuroshard/evolution/sharded/__init__.py',
    'src/neuroshard/evolution/sharded/model.py',
    'src/neuroshard/evolution/sharded/training.py',
    'src/neuroshard/evolution/sharded/wire.py',
    'src/neuroshard/evolution/sharded/checkpoint.py',
    'src/neuroshard/evolution/sharded/portable.py',
    'src/neuroshard/evolution/sharded/guarded.py',
    'src/neuroshard/evolution/sharded/transcript.py',
    'src/neuroshard/evolution/sharded/continued_job.py',
)


def load(path=None):
    path = Path(path) if path else PLAN_PATH
    plan = json.loads(path.read_text())
    validate(plan)
    return plan


def digest(value):
    if not isinstance(value, str) or DIGEST.fullmatch(value) is None:
        raise ValueError('Invalid content digest')
    return value


def validate(plan):
    if plan.get('format') != FORMAT:
        raise ValueError('Unsupported continued-learning format')
    if plan.get('license') != 'Apache-2.0':
        raise ValueError('Continued-learning materials must stay under Apache-2.0')
    if plan.get('purpose') != 'development':
        raise ValueError('This plan supports development experiments only')
    if plan.get('public_network') is None or 'unchanged' not in plan['public_network']:
        raise ValueError('This plan must not authorize a public-network change')
    if plan['status'] not in STATUSES:
        raise ValueError('Unknown continued-learning status')
    open_source = plan['open_source']
    if open_source.get('secret_evaluation') is not False:
        raise ValueError('A secret evaluation set is incompatible with public reproduction')
    if not open_source.get('independent_reproduction'):
        raise ValueError('The plan must remain independently reproducible')
    if Path(open_source['prepared_path']).name != PREPARED_NAME:
        raise ValueError('Prepared artifacts belong in the published experiments directory')
    if Path(open_source['selection_path']).name != SELECTION_NAME:
        raise ValueError('Selection artifacts belong in the published experiments directory')
    parent = plan['parent']
    if (parent['parameters'] != 1711376384 or parent['layers'] != 24
            or parent['step'] != 128 or parent['boundaries'] != [0, 6, 15, 24]
            or parent['includes_optimizer'] is not True):
        raise ValueError('Parent must be the passing 24-layer phase-A portable checkpoint with Adam')
    for field in ('checkpoint', 'state_root'):
        digest(parent[field])
    if parent['checkpoint'] != '094138fb6e3e2a8962f8455b0bf81de3f2fbe82029222df76aa71e2e22e14d49':
        raise ValueError('Parent checkpoint identity differs from the frozen phase-A root')
    if parent['state_root'] != '497dfd36ff77ab7db04b8bac2e66b7a548763db51552cc28ecd33171bd6d9f21':
        raise ValueError('Parent learned-state root differs from the frozen phase-A root')
    reference = plan['reference']
    if (reference['kind'] != 'frozen-parent-checkpoint'
            or reference['checkpoint'] != parent['checkpoint']
            or reference['state_root'] != parent['state_root']
            or reference['kl_strength'] != 2.0):
        raise ValueError('The frozen teacher must be the phase-A checkpoint')
    digest(plan['tokenizer'])
    digest(plan['config_sha256'])
    if plan['tokenizer'] != 'e9478f6c6191dbe1c442af64c81352c531f08c5320f266a6ff7349c0f552cd5e':
        raise ValueError('Tokenizer identity differs from the frozen parent job')
    if plan['config_sha256'] != '67a3cb445147cdba56909efb817e7e0022c22bb3356e07676b68a4bca7caba31':
        raise ValueError('Architecture config digest differs from the frozen parent job')
    if plan['parameters'] != parent['parameters'] or plan['allowed_layers'] != [24]:
        raise ValueError('Continued learning keeps the 1.7B 24-layer layout')
    method = plan['method']
    if (method['growth_layers'] != 0 or method['inherit_adam'] is not True
            or method['reinitialize_adam'] is not False
            or method['new_task_family_cycle'] != ['total', 'total', 'lookup', 'filter', 'sort', 'total']
            or method['new_task_weight'] != 8 or method['total_task_weight'] != 16):
        raise ValueError('Method constants differ from the frozen recipe')
    training = plan['training']
    if (training['steps'] != 96 or training['batch_documents'] != 64
            or training['learning_rate'] != 0.000001 or training['warmup_steps'] != 8
            or training['weight_decay'] != 0.01 or training['clip_norm'] != 1.0
            or training['seed'] != 2026091500):
        raise ValueError('Optimizer constants differ from the frozen recipe')
    if (plan['parent_step'] != 128 or plan['additional_steps'] != 96
            or plan['final_step'] != 224 or plan['checkpoints'] != [160, 192, 224]):
        raise ValueError('Cursor and endpoints differ from the frozen recipe')
    if (plan['new_tasks'] != 3072 or plan['trained_replay'] != 1536
            or plan['conversation_replay'] != 1536):
        raise ValueError('Training mix differs from the frozen recipe')
    if plan['new_tasks'] + plan['trained_replay'] + plan['conversation_replay'] != 96 * 64:
        raise ValueError('Training mix must fill every scheduled step exactly once')
    if (plan['test_new_cases'] != 256 or plan['test_prior_cases'] != 128
            or plan['retention_cases'] != 128 or plan['development_cases'] != 64
            or plan['test_new_per_family'] != 64 or plan['test_prior_per_family'] != 32
            or plan['families'] != ['lookup', 'filter', 'total', 'sort']):
        raise ValueError('Evaluation counts differ from the frozen plan')
    if plan['fresh_retention_source']['start'] != 22000:
        raise ValueError('Final conversation retention must use the unused SmolTalk cursor')
    if plan['development_retention_source']['start'] != 19000:
        raise ValueError('Development retention must not reuse the previous final scan')
    prior = plan['prior_exclusion']
    digest(prior['inputs_sha256'])
    digest(prior['prepared'])
    digest(prior['role_ids_digest'])
    if prior['inputs_sha256'] != prior['prepared']:
        raise ValueError('Prior prepared identity must match the committed inputs file')
    if prior['prepared'] != 'f1c85dc732ce8e1152da6eade09af70acfca2e0eaa68a7cc42febd76e9cdddfd':
        raise ValueError('Prior adaptive job identity differs from the recorded freeze')
    gate = plan['quality_gate']
    if (gate['primary'] != 'generated-answer-improvement' or gate['min_net_gain'] != 8
            or gate['max_one_sided_p'] != 0.05
            or gate['family_floor'] != 'candidate_correct >= baseline_correct'
            or gate['retention_upper_at_most_nats'] != 0.02
            or gate['response_loss_is_acceptance'] is not False
            or gate['lock_before_final'] is not True
            or gate['failed_final'] != 'reported-failure'
            or gate['recipes'] != 1
            or gate['bootstrap_samples'] != 10000 or gate['confidence'] != 0.95):
        raise ValueError('Quality margins cannot be relaxed')
    if plan['rewards_and_serving']['promotion_mints'] != 0:
        raise ValueError('Promotion must mint nothing')
    if plan['rewards_and_serving']['failed_quality_keeps_training_reward'] is not True:
        raise ValueError('Failed quality must not claw back verified training payment')
    settlement = plan['settlement']
    if (settlement['status'] != 'not-implemented' or settlement['order'] != 'after-quality-pass'
            or settlement['existing_checkpoint_insufficient'] is not True
            or settlement['requires_activation'] is not True
            or settlement['requires_reserved_windows'] is not True
            or settlement['requires_signed_worker_receipts'] is not True):
        raise ValueError('Settlement remains a later activated replay, not a checkpoint import')
    required = {
        'training_before_prepared': 'forbidden',
        'training_without_artifact_freeze': 'forbidden',
        'margin_relaxation': 'forbidden',
        'second_recipe_on_same_finals': 'forbidden',
        'growth_during_continued_learning': 'forbidden',
        'reuse_of_adaptive_finals': 'forbidden',
        'payment_by_checkpoint_import': 'forbidden',
        'final_eval_before_lock': 'forbidden',
        'exhausted_budget_without_pass': 'fail the phase',
    }
    if any(plan['stop_rules'].get(key) != value for key, value in required.items()):
        raise ValueError('Stop rules differ from the frozen plan')
    runtime = plan['runtime']
    if any(key not in runtime for key in RUNTIME_KEYS):
        raise ValueError('Incomplete runtime freeze')
    if runtime['gpu'] != 'NVIDIA A10G' or runtime['cuda'] != '12.8' or runtime['torch'] != '2.9.1+cu128':
        raise ValueError('Runtime freeze differs from the parent A10G CUDA 12.8 profile')
    return plan


def repo_root():
    return PLAN_PATH.parents[2]


def prepared_path(plan):
    relative = Path(plan['open_source']['prepared_path'])
    if relative.as_posix() != 'config/experiments/' + PREPARED_NAME:
        raise ValueError('Prepared freeze must use the public experiments path')
    return repo_root() / relative


def selection_path(plan):
    relative = Path(plan['open_source']['selection_path'])
    if relative.as_posix() != 'config/experiments/' + SELECTION_NAME:
        raise ValueError('Selection must use the public experiments path')
    return repo_root() / relative


def git_bytes(ref, path):
    if ref != 'HEAD' and re.fullmatch('[0-9a-f]{40}', ref) is None:
        raise ValueError('Pin a complete Git commit')
    relative = Path(path).resolve().relative_to(repo_root().resolve()).as_posix()
    try:
        return subprocess.check_output(['git', '-C', str(repo_root()), 'show', f'{ref}:{relative}'],
                                       stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        raise ValueError('Frozen plan, prepared set and selection must be committed to Git') from None


def implementation_digest():
    values = {name: hashlib.sha256((repo_root() / name).read_bytes()).hexdigest() for name in SOURCE_PATHS}
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def prior_role_ids(plan, prior=None):
    path = repo_root() / plan['prior_exclusion']['inputs_path']
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != plan['prior_exclusion']['inputs_sha256']:
        raise ValueError('Prior adaptive inputs file changed')
    prior = prior or json.loads(raw)
    ids = sorted({item for role in prior['roles'].values() for item in role['ids']})
    if identity(ids) != plan['prior_exclusion']['role_ids_digest']:
        raise ValueError('Prior evaluation and training identities changed')
    for role, digest_value in plan['prior_exclusion']['role_sha256'].items():
        if prior['roles'][role]['sha256'] != digest_value:
            raise ValueError('Prior role bytes changed')
    return set(ids)


def overlap_ids(*groups):
    seen = set()
    for group in groups:
        values = list(group)
        if len(set(values)) != len(values):
            raise ValueError('Repeated identity inside a prepared role')
        extra = set(values)
        if seen.intersection(extra):
            raise ValueError('Prepared identity overlaps a frozen exclusion set')
        seen.update(extra)
    return seen


def content_keys(messages):
    """Identity of rendered content, independent of source row or hidden task fields."""
    from .data import normalized
    return {identity(['document', [
        [message['role'], normalized(message['content'])] for message in messages]])} | {
        identity(['prompt', normalized(message['content'])])
        for message in messages if message['role'] == 'user'}


class ContentExclusion:
    def __init__(self):
        self.keys = set()

    def add(self, messages):
        keys = content_keys(messages)
        fresh = self.keys.isdisjoint(keys)
        self.keys.update(keys)
        return fresh


def paired_losses(before, after, gate):
    import numpy as np
    if not before or len(before) != len(after):
        raise ValueError('Require aligned nonempty losses')
    differences = np.asarray(after, dtype=np.float64) - np.asarray(before, dtype=np.float64)
    if not bool(np.isfinite(differences).all()):
        raise ValueError('Nonfinite quality measurements')
    generator = np.random.default_rng(gate['bootstrap_seed'])
    means = []
    for start in range(0, gate['bootstrap_samples'], 100):
        count = min(100, gate['bootstrap_samples'] - start)
        indices = generator.integers(0, len(differences), size=(count, len(differences)))
        means.extend(differences[indices].mean(axis=1).tolist())
    return {
        'documents': len(differences),
        'baseline_mean': float(sum(before) / len(before)),
        'candidate_mean': float(sum(after) / len(after)),
        'mean_delta': float(differences.mean()),
        'upper': float(np.quantile(means, gate['confidence'], method='linear')),
        'confidence': gate['confidence'],
        'bootstrap_samples': len(means),
        'bootstrap_seed': gate['bootstrap_seed'],
    }


def family_pairs(records, before, after):
    by_id = {row['id']: row['task']['family'] for row in records}
    if [row['id'] for row in records] != [row['id'] for row in before] or [row['id'] for row in before] != [row['id'] for row in after]:
        raise ValueError('Generated answers must cover the frozen evaluation identities in order')
    groups = {family: {'before': [], 'after': []} for family in tasks.FAMILIES}
    for left, right in zip(before, after):
        family = by_id[left['id']]
        groups[family]['before'].append(left)
        groups[family]['after'].append(right)
    return groups


def generation_decision(records, before, after, plan, *, primary):
    gate = plan['quality_gate']
    paired = tasks.paired_accuracy(before, after)
    families = {}
    floors = []
    for family, group in family_pairs(records, before, after).items():
        if not group['before']:
            raise ValueError('Every task family must appear in the scored set')
        row = tasks.paired_accuracy(group['before'], group['after'])
        row['passed'] = row['candidate_correct'] >= row['baseline_correct']
        families[family] = row
        floors.append(row['passed'])
    expected = plan['test_new_per_family'] if primary else plan['test_prior_per_family']
    if any(families[family]['documents'] != expected for family in plan['families']):
        raise ValueError('Family evaluation counts differ from the frozen plan')
    passed = all(floors)
    if primary:
        passed = (passed and paired['wins'] - paired['losses'] >= gate['min_net_gain']
                  and paired['one_sided_p'] < gate['max_one_sided_p'])
    paired.update(families=families, passed=passed, primary=primary)
    return paired


def validate_prepared(prepared, plan=None):
    plan = plan or load()
    validate(plan)
    if prepared.get('format') != PREPARED_FORMAT:
        raise ValueError('Unsupported prepared freeze format')
    for field in ('plan_digest', 'implementation_digest', 'parent_checkpoint',
                  'parent_state_root', 'reference_checkpoint', 'tokenizer', 'config_sha256'):
        digest(prepared[field])
    if prepared['parent_checkpoint'] != plan['parent']['checkpoint']:
        raise ValueError('Prepared parent checkpoint differs from the frozen plan')
    if prepared['parent_state_root'] != plan['parent']['state_root']:
        raise ValueError('Prepared parent state root differs from the frozen plan')
    if prepared['reference_checkpoint'] != plan['reference']['checkpoint']:
        raise ValueError('Prepared reference checkpoint differs from the frozen parent')
    if prepared['tokenizer'] != plan['tokenizer'] or prepared['config_sha256'] != plan['config_sha256']:
        raise ValueError('Tokenizer or architecture freeze differs from the plan')
    if prepared['runtime'] != {key: plan['runtime'][key] for key in RUNTIME_KEYS}:
        raise ValueError('Prepared runtime freeze differs from the plan')
    roles = prepared['roles']
    required = set(plan['development_roles'] + plan['final_roles'] + ['train'])
    if set(roles) != required:
        raise ValueError('Prepared roles differ from the frozen plan')
    counts = {
        'train': plan['new_tasks'] + plan['trained_replay'] + plan['conversation_replay'],
        'dev-new': plan['development_cases'],
        'dev-prior': plan['development_cases'],
        'dev-retention': plan['development_cases'],
        'test-new': plan['test_new_cases'],
        'test-prior': plan['test_prior_cases'],
        'retention': plan['retention_cases'],
    }
    prior_ids = prior_role_ids(plan)
    prior = json.loads((repo_root() / plan['prior_exclusion']['inputs_path']).read_bytes())
    task_replay = prepared['trained_replay_ids']
    conversation_replay = prepared['conversation_replay_ids']
    replay = overlap_ids(task_replay, conversation_replay)
    if (len(task_replay) != plan['trained_replay']
            or len(conversation_replay) != plan['conversation_replay']
            or not replay <= set(prior['roles']['train-a']['ids'])):
        raise ValueError('Replay must contain the declared previously trained phase-A windows')
    groups = []
    for role, count in counts.items():
        spec = roles[role]
        digest(spec['sha256'])
        if spec['count'] != count or len(spec['ids']) != count:
            raise ValueError(f'Prepared {role} count differs from the frozen plan')
        if spec['file'] != role + '.jsonl':
            raise ValueError('Prepared role must use its declared local filename')
        groups.append(spec['ids'])
    overlap_ids(*groups)
    if set(roles['train']['ids']) & prior_ids != replay:
        raise ValueError('Only the declared trained replay may overlap prior identities')
    if any(set(spec['ids']) & prior_ids for role, spec in roles.items() if role != 'train'):
        raise ValueError('Fresh evaluation overlaps previously exposed identities')
    if prepared['plan'] != {k: v for k, v in plan.items() if k != 'status'}:
        raise ValueError('Prepared recipe differs from the frozen plan')
    if set(prepared['sources']) != set(SOURCE_PATHS):
        raise ValueError('Incomplete prepared source inventory')
    if identity(prepared['sources']) != prepared['implementation_digest']:
        raise ValueError('Prepared source inventory differs from its commitment')
    if len(prepared['schedule']) != plan['additional_steps']:
        raise ValueError('Training schedule length differs from the frozen step budget')
    if any(len(step['indices']) != plan['training']['batch_documents'] for step in prepared['schedule']):
        raise ValueError('Training schedule does not use the frozen batch size')
    indices = []
    for step in prepared['schedule']:
        if step['role'] != 'train' or any(type(i) is not int or not 0 <= i < counts['train'] for i in step['indices']):
            raise ValueError('Invalid training schedule indices or role')
        if sum(roles['train']['ids'][i] in replay for i in step['indices']) != 32:
            raise ValueError('Every update requires its declared 32 replay anchors')
        indices.extend(step['indices'])
    if sorted(indices) != list(range(counts['train'])):
        raise ValueError('Schedule must train every declared window exactly once')
    return prepared


def validate_selection(selection, plan, prepared, checkpoint=None):
    required = {'format', 'prepared', 'plan_digest', 'implementation_digest',
                'baseline', 'candidate', 'step', 'frozen_before_final_evaluation', 'development'}
    if not required <= set(selection):
        raise ValueError('Incomplete candidate lock')
    if selection['format'] != SELECTION_FORMAT:
        raise ValueError('Unsupported selection format')
    for field in ('prepared', 'plan_digest', 'implementation_digest', 'baseline', 'candidate'):
        digest(selection[field])
    if selection['prepared'] != identity(prepared):
        raise ValueError('Selection binds another prepared freeze')
    if any(selection[field] != prepared[field] for field in ('plan_digest', 'implementation_digest')):
        raise ValueError('Selection binds another plan or evaluator')
    if selection['baseline'] != plan['parent']['checkpoint']:
        raise ValueError('Selection baseline must be the frozen parent checkpoint')
    if selection['step'] != plan['final_step'] or selection['frozen_before_final_evaluation'] is not True:
        raise ValueError('Only the predetermined final step may be locked')
    if selection['candidate'] == selection['baseline']:
        raise ValueError('A locked candidate must be a later checkpoint than its parent')
    if checkpoint is not None:
        if (identity(checkpoint) != selection['candidate']
                or checkpoint['step'] != plan['final_step']
                or checkpoint['job'] != job_identity(prepared, plan['runtime'])
                or checkpoint['boundaries'] != plan['parent']['boundaries']
                or checkpoint.get('transition') is not None):
            raise ValueError('Selected manifest is not the actual fixed-size final checkpoint')
    development = selection['development']
    validate_development(development, plan, prepared, selection['candidate'])
    return selection


def committed_prepared(plan, supplied=None):
    path = prepared_path(plan)
    if not path.is_file():
        return False
    raw = path.read_bytes()
    if git_bytes('HEAD', path) != raw or git_bytes('HEAD', PLAN_PATH) != PLAN_PATH.read_bytes():
        raise ValueError('Training requires unchanged Git-committed plan and prepared bytes')
    prepared = json.loads(raw)
    if json.loads(PLAN_PATH.read_bytes()) != plan:
        raise ValueError('Supplied plan differs from the committed plan')
    if supplied is not None and supplied != prepared:
        raise ValueError('Supplied prepared artifact differs from the committed freeze')
    validate_prepared(prepared, plan)
    frozen = json.loads(git_bytes(prepared['plan_commit'], PLAN_PATH))
    if hashlib.sha256(git_bytes(prepared['plan_commit'], PLAN_PATH)).hexdigest() != prepared['plan_digest']:
        raise ValueError('Prepared freeze does not bind the frozen plan bytes')
    if frozen['status'] != 'plan-frozen' or {**plan, 'status': 'plan-frozen'} != frozen:
        raise ValueError('Frozen plan constants changed after preparation')
    if implementation_digest() != prepared['implementation_digest']:
        raise ValueError('Continued-learning implementation changed after sealing')
    if any(git_bytes('HEAD', repo_root() / name) != (repo_root() / name).read_bytes()
           for name in SOURCE_PATHS):
        raise ValueError('Numerical sources must also be committed to Git')
    return True


def committed_selection(plan, supplied=None):
    path = selection_path(plan)
    if not path.is_file():
        return False
    raw = path.read_bytes()
    if git_bytes('HEAD', path) != raw:
        raise ValueError('Final evaluation requires an unchanged Git-committed selection')
    prepared = json.loads(prepared_path(plan).read_bytes())
    if supplied is not None and json.loads(raw) != supplied:
        raise ValueError('Supplied selection differs from the committed candidate lock')
    validate_selection(json.loads(raw), plan, prepared)
    return True


def training_allowed(plan):
    validate(plan)
    return (plan['status'] in ('prepared-committed', 'learning-running')
            and not selection_path(plan).exists() and committed_prepared(plan))


def final_evaluation_allowed(plan):
    validate(plan)
    return plan['status'] == 'selection-committed' and committed_prepared(plan) and committed_selection(plan)


def preflight(plan, prepared, parent, checkpoint, command, roles=None, selection=None):
    """Run before loading tokenizer assets, CUDA or any model tensor, including on resume."""
    if command not in ('train', 'evaluate'):
        raise ValueError('Unknown continued-learning operation')
    if command == 'train' and not training_allowed(plan):
        raise ValueError('Training requires a Git-committed prepared freeze without a final lock')
    if not committed_prepared(plan, prepared):
        raise ValueError('Execution requires the committed prepared artifact')
    if identity(parent) != plan['parent']['checkpoint']:
        raise ValueError('Parent/reference differs from the frozen phase-A checkpoint')
    inherited = identity(checkpoint) == identity(parent)
    if not inherited:
        if (checkpoint['job'] != job_identity(prepared, plan['runtime'])
                or checkpoint['step'] not in plan['checkpoints']
                or checkpoint['boundaries'] != parent['boundaries']
                or checkpoint['config'] != parent['config']
                or checkpoint.get('transition') is not None):
            raise ValueError('Resume is not a declared checkpoint of the frozen job')
    if command == 'train':
        if checkpoint['step'] >= plan['final_step']:
            raise ValueError('The final endpoint cannot be trained further')
        return
    roles = roles or plan['development_roles']
    if len(set(roles)) != len(roles):
        raise ValueError('Repeated evaluation role')
    if set(roles) <= set(plan['development_roles']):
        return
    if (selection is None or plan['status'] != 'selection-committed'
            or not committed_selection(plan, selection)):
        raise ValueError('Final evaluation requires the Git-committed candidate lock')
    if set(roles) - set(plan['final_roles']):
        raise ValueError('Unsupported or mixed final evaluation roles')
    if not inherited:
        validate_selection(selection, plan, prepared, checkpoint)


def assert_training_freeze(plan, prepared, checkpoint, tokenizer, runtime):
    """Refuse training unless every frozen artifact matches the committed plan."""
    validate(plan)
    validate_prepared(prepared, plan)
    if not training_allowed(plan):
        raise ValueError('Training requires a Git-committed prepared freeze')
    assert_artifact_freeze(plan, prepared, checkpoint, tokenizer, runtime)


def job_identity(prepared, runtime):
    return identity({'prepared': identity(prepared), 'runtime': runtime})


def assert_artifact_freeze(plan, prepared, checkpoint, tokenizer, runtime):
    if not committed_prepared(plan, prepared):
        raise ValueError('Execution requires a Git-committed prepared freeze')
    if identity(checkpoint) != plan['parent']['checkpoint']:
        raise ValueError('Parent checkpoint bytes differ from the frozen artifact')
    if checkpoint['state_root'] != plan['parent']['state_root']:
        raise ValueError('Parent learned-state root differs from the frozen artifact')
    if checkpoint['step'] != plan['parent']['step'] or checkpoint['boundaries'] != plan['parent']['boundaries']:
        raise ValueError('Parent cursor or layout differs from the frozen artifact')
    if not any(checkpoint['step'] - spec['born'] > 0 for spec in checkpoint['tensors'].values()):
        raise ValueError('Parent checkpoint is missing aged Adam state')
    if tokenizer != plan['tokenizer']:
        raise ValueError('Tokenizer identity differs from the frozen artifact')
    profile = {key: runtime[key] for key in RUNTIME_KEYS}
    if profile != plan['runtime']:
        raise ValueError('Runtime profile differs from the frozen artifact')
    if prepared['implementation_digest'] != implementation_digest():
        raise ValueError('Numerical source differs from the frozen job')


def read_role(folder, prepared, role):
    from .reference_data import read_records
    spec = prepared['roles'][role]
    rows = read_records(Path(folder) / spec['file'], spec['sha256'])
    if len(rows) != spec['count'] or [row['id'] for row in rows] != spec['ids']:
        raise ValueError('Role records differ from the committed identity list')
    if role == 'train':
        replay = set(prepared['trained_replay_ids'] + prepared['conversation_replay_ids'])
        if any(type(row['distill']) is not bool or row['distill'] != (row['id'] in replay) for row in rows):
            raise ValueError('Replay anchor flags differ from the committed windows')
    return rows


def development_decision(plan, prepared, checkpoint, before, after):
    expected = prepared['roles']['dev-retention']['ids']
    if [row['id'] for row in before] != expected or [row['id'] for row in after] != expected:
        raise ValueError('Development retention must cover the frozen records in order')
    import math
    deltas = [b['loss'] - a['loss'] for a, b in zip(before, after)]
    if not deltas or not all(math.isfinite(v) for v in deltas):
        raise ValueError('Invalid development measurements')
    mean = math.fsum(deltas) / len(deltas)
    return {'format': 'neuroshard-continued-development-v1', 'prepared': identity(prepared),
            'baseline': plan['parent']['checkpoint'], 'candidate': checkpoint,
            'before': before, 'after': after, 'mean_delta': mean,
            'passed': mean <= plan['quality_gate']['development_abort_retention_mean_delta']}


def validate_development(report, plan, prepared, checkpoint):
    expected = development_decision(plan, prepared, checkpoint, report['before'], report['after'])
    if report != expected or not report['passed']:
        raise ValueError('Candidate failed its frozen development retention gate')
    return report


def decide(plan, records, baseline, candidate):
    """Score the locked candidate. Generation gain is primary; loss cannot pass the run."""
    validate(plan)
    gate = plan['quality_gate']
    generation = generation_decision(records['test-new'], baseline['test-new']['answers'],
                                     candidate['test-new']['answers'], plan, primary=True)
    prior = generation_decision(records['test-prior'], baseline['test-prior']['answers'],
                                candidate['test-prior']['answers'], plan, primary=False)
    retention = paired_losses(baseline['retention']['losses'], candidate['retention']['losses'], gate)
    retention['passed'] = retention['upper'] <= gate['retention_upper_at_most_nats']
    outcomes = {
        'test-new': {
            'generation': generation,
            'loss': paired_losses(baseline['test-new']['losses'], candidate['test-new']['losses'], gate),
            'passed': generation['passed'],
        },
        'test-prior': {
            'generation': prior,
            'loss': paired_losses(baseline['test-prior']['losses'], candidate['test-prior']['losses'], gate),
            'passed': prior['passed'],
        },
        'retention': retention,
    }
    return {
        'format': 'neuroshard-continued-quality-v1',
        'passed': all(row['passed'] for row in outcomes.values()),
        'outcomes': outcomes,
        'scope': 'Generated-answer gain on fresh examples of four public task families, with per-family floors and public conversation retention; not general assistant quality',
        'primary': 'test-new generated answers',
        'settlement': 'not authorized by this decision',
        'serving': 'not authorized by this decision',
    }
