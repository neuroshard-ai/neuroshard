"""Pre-registered learning milestone: public plan, frozen sealed set, fail-closed gates.

This module does not train. It refuses to treat an uncommitted sealed set as ready
for training, and it will not lower the published quality margins.
"""
import hashlib
import json
import re
import subprocess
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
    fixed = {'optimizer': {'kind': 'sgd', 'batch_windows_per_step': 2},
             'data': {'sequence_length': 128, 'context_tokens': 64, 'response_tokens': 64,
                      'max_windows': 4, 'target_mode': 'response'},
             'budget': {'wall_clock_hours': 72, 'disk_gib': 256, 'worker_processes': 3,
                        'worker_capacity': 48_000_000, 'include_evaluation_forwards': True,
                        'include_retries_and_rejected_work': True},
             'evaluation': {'generation_prompts': 20, 'generation_max_tokens': 32},
             'learning': {'train_start': 8192, 'train_scan_limit': 2048, 'train_documents': 256,
                          'heldout_start': 3072, 'heldout_scan_limit': 2048},
             'continual': {'replay_fraction': .25}, 'scaling': {'steps': 16}}
    if any(plan[section][key] != value for section, values in fixed.items() for key, value in values.items()):
        raise ValueError('Execution constants differ from the frozen plan')
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
    relative = Path(plan['open_source']['selection_path'])
    if relative.as_posix() != 'config/experiments/' + SELECTION_NAME:
        raise ValueError('Selection must use the public experiments path')
    return PLAN_PATH.parents[2] / relative


def git_bytes(ref, path):
    """Read committed bytes without consulting the index or following a ref option."""
    if ref != 'HEAD' and re.fullmatch('[0-9a-f]{40}', ref) is None:
        raise ValueError('Pin a complete Git commit')
    repo = PLAN_PATH.parents[2]
    relative = Path(path).resolve().relative_to(repo.resolve()).as_posix()
    try:
        return subprocess.check_output(['git', '-C', str(repo), 'show', f'{ref}:{relative}'],
                                       stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        raise ValueError('Sealed selection and plan must be committed to Git') from None


def implementation_digest():
    """Bind the numerical package, driver and dependency lock before preparation."""
    repo = PLAN_PATH.parents[2]
    paths = [*sorted((repo/'src/neuroshard').rglob('*.py')),
             repo/'scripts/run_learning_milestone.py', repo/'docs/evolution-requirements.txt',
             repo/'docs/llm-requirements.txt', repo/'pyproject.toml']
    values = {p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_selection(selection, plan):
    """Validate all role identities and the prescribed batch schedule, without training."""
    required = {'format', 'baseline', 'imported_model_root', 'tokenizer_root', 'documents',
                'generation_prompts', 'plan_digest', 'plan_commit', 'implementation_digest',
                'training_batches', 'source_windows'}
    if not required <= set(selection):
        raise ValueError('Incomplete sealed-set manifest')
    if selection['format'] != 'neuroshard-learning-milestone-selection-v1':
        raise ValueError('Unsupported sealed-set format')
    for field in ('baseline', 'imported_model_root', 'tokenizer_root', 'plan_digest', 'implementation_digest'):
        if not isinstance(selection[field], str) or re.fullmatch('[0-9a-f]{64}', selection[field]) is None:
            raise ValueError('Invalid sealed content digest')
    if selection['imported_model_root'] != plan['seed']['imported_model_root']:
        raise ValueError('Selection must start from the prescribed seed')
    from neuroshard.dataflow.store import canonical
    sources = {}
    for name, role in (('heldout', 'heldout'), ('train', 'train')):
        spec = {k: plan['data'][k] for k in ('repo', 'revision', 'license')}
        spec.update(split=plan['learning'][name+'_split'], role=role)
        key = hashlib.sha256(canonical(spec)).hexdigest()
        sources[key] = {'source': key, 'spec': spec, 'start': plan['learning'][name+'_start'],
                        'end': plan['learning'][name+'_start']+plan['learning'][name+'_scan_limit']}
    if selection['source_windows'] != list(sources.values()):
        raise ValueError('Sealed source ranges differ from the frozen scan')
    counts = {'train': plan['learning']['train_documents'],
              **{role: plan['evaluation']['documents_per_role'] for role in ('retention', 'fresh', 'test')}}
    documents = selection['documents']
    if set(documents) != set(counts) or any(len(documents[role]) != count for role, count in counts.items()):
        raise ValueError('Sealed set does not contain the declared document counts')
    seen, windows, positions = set(), set(), set()
    for role, items in documents.items():
        for item in items:
            if not {'id', 'row', 'source', 'object', 'windows', 'omitted_targets'} <= set(item):
                raise ValueError('Incomplete document provenance')
            for key in (item['id'], item['source'], item['object'], *item['windows']):
                if not isinstance(key, str) or re.fullmatch('[0-9a-f]{64}', key) is None:
                    raise ValueError('Invalid document or window digest')
            position = (item['source'], item['row'])
            if item['id'] in seen or position in positions or type(item['row']) is not int:
                raise ValueError('Document or source row reused across sealed roles')
            source = sources.get(item['source'])
            if (source is None or source['spec']['role'] != ('train' if role == 'train' else 'heldout') or
                    not source['start'] <= item['row'] < source['end']):
                raise ValueError('Document lies outside its declared source role or cursor window')
            if item['omitted_targets'] != 0 or not 1 <= len(item['windows']) <= plan['data']['max_windows']:
                raise ValueError('Select complete documents within the window bound')
            if role != 'train' and ('retention', 'fresh', 'test')[int(item['id'], 16) % 3] != role:
                raise ValueError('Held-out role differs from the document partition')
            if len(set(item['windows'])) != len(item['windows']) or windows.intersection(item['windows']):
                raise ValueError('Repeated sealed token window')
            seen.add(item['id'])
            positions.add(position)
            windows.update(item['windows'])
    batches = selection['training_batches']
    training_windows = {w for d in documents['train'] for w in d['windows']}
    if (len(batches) != plan['budget']['training_steps'] or
            any(len(batch) != plan['optimizer']['batch_windows_per_step'] for batch in batches) or
            any(w not in training_windows for batch in batches for w in batch)):
        raise ValueError('Training schedule must use only the committed training pool')
    if len({w for batch in batches for w in batch}) != sum(map(len, batches)):
        raise ValueError('Phase one must not repeat a scheduled training window')
    prompts = selection['generation_prompts']
    expected_ids = sorted(d['id'] for d in documents['test'])[:plan['evaluation']['generation_prompts']]
    if [p.get('document') for p in prompts] != expected_ids:
        raise ValueError('Generation probes must use the first sorted sealed test documents')
    if any(not isinstance(p.get('prompt'), str) or not p['prompt'] for p in prompts):
        raise ValueError('Generation probes require their original public prompt text')


def committed_selection(plan):
    path = selection_path(plan)
    if not path.is_file():
        return False
    raw = path.read_bytes()
    if git_bytes('HEAD', path) != raw or git_bytes('HEAD', PLAN_PATH) != PLAN_PATH.read_bytes():
        raise ValueError('Training requires unchanged Git-committed plan and selection bytes')
    selection = json.loads(raw)
    validate_selection(selection, plan)
    frozen = git_bytes(selection['plan_commit'], PLAN_PATH)
    if hashlib.sha256(frozen).hexdigest() != selection['plan_digest']:
        raise ValueError('Selection does not bind the frozen plan bytes')
    original = json.loads(frozen)
    if original['status'] != 'plan-frozen' or {**plan, 'status': 'plan-frozen'} != original:
        raise ValueError('Frozen plan constants changed after selection')
    if implementation_digest() != selection['implementation_digest']:
        raise ValueError('Milestone implementation changed after sealing')
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
    validate(plan)
    evaluation = plan['evaluation']
    if (set(measurements) != {'baseline', 'candidate'} or
            any(set(groups) != {'retention', 'fresh', 'test'} for groups in measurements.values()) or
            any(len(values) != evaluation['documents_per_role']
                for groups in measurements.values() for values in groups.values())):
        raise ValueError('Score every sealed document exactly once per side')
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
