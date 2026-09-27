"""Development evaluation of both trained systems on the canonical parent's CPU runtime.

Each system selects once per episode from the frozen parent feature, then runs
the whole episode with the chosen model. The parent control is the canonical
re-baseline, pinned by its result digest; protected successes come from it.
Each arm also answers every original anchor with selection forced on, which
measures forgetting directly; the routed system serves anchors with the parent.
Each arm runs alone in a fresh worker at the baseline's thread count, so numerics and latency match it.
"""

import importlib.metadata
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution import assistant_workflow_baseline as first
from neuroshard.evolution import assistant_workflow_canonical as canonical
from neuroshard.evolution import granite_reference as reference
from neuroshard.evolution import granite_tokenizer
from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, launch, read, save, sha256, verify_artifacts,
)

ARMS = ('update', 'addition')
PLAN = 'config/experiments/assistant-experience-learning.json'
EXECUTION = 'config/experiments/assistant-experience-development-execution.json'
SCRIPT = 'scripts/run_assistant_experience_development.py'
PROFILE = 'assistant-experience-development'
UPLOADED = '.arms'


def verify_arms(directory, pinned):
    """Uploaded checkpoints and gates must match the digests pinned before evaluation."""
    directory = Path(directory)
    for arm in ARMS:
        manifest = read(directory / f'{arm}-checkpoint' / 'manifest.json')
        if manifest['trainable_sha256'] != pinned[arm]['trainable_sha256'] or manifest['arm'] != arm:
            raise ValueError(f'{arm} checkpoint differs from the pinned training result')
    if sha256(directory / 'integration.json') != pinned['integration_sha256']:
        raise ValueError('integration gates differ from the pinned result')
    return read(directory / 'integration.json')


def evaluate_arm(parent, model, tokenizer, gate, feature, cases, policy, anchor_plan):
    """Selected complete episodes for one system, plus its forced-arm anchor answers.

    ``feature`` runs the parent at the first request; its time is served latency.
    """
    from neuroshard.evolution import assistant_selector as selector

    respond = {'parent': first.native_responder(parent, tokenizer, policy),
               'arm': first.native_responder(model, tokenizer, policy)}
    rows = []
    for case in cases:
        started = time.monotonic()
        chosen = 'arm' if selector.choose(gate, feature(case)) else 'parent'
        selection = time.monotonic() - started
        rows.append({**workflow.execute(case, respond[chosen], policy), 'selected': chosen,
                     'selection_seconds': selection})
    anchors = [reference.generate(model, tokenizer, anchor_plan, task, 'baseline') for task in anchor_plan['tasks']]
    return {'episodes': rows, 'forced_anchors': anchors}


def forgetting(anchor_rows, protected_ids):
    passed = {row['id'] for row in anchor_rows if row['passed']}
    return {'correct': len(passed), 'lost_protected': sorted(set(protected_ids) - passed)}


def committed_sources(root=ROOT):
    execution = read(root / EXECUTION)
    for name, digest in execution['contracts'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'changed evaluation contract: {name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in execution['sources']:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root) != (root / name).read_bytes():
            raise ValueError(f'uncommitted evaluation source: {name}')
        sources[name] = sha256(root / name)
    return {'commit': commit, 'sources': sources}


def configure():
    if 'torch' in sys.modules:
        raise ValueError('configure the evaluation runtime before importing torch')
    for key, value in read(ROOT / EXECUTION)['environment'].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def freeze():
    """The canonical parent's CPU runtime, byte for byte in packages and environment."""
    source = committed_sources()
    execution = read(ROOT / EXECUTION)
    baseline = read(ROOT / canonical.EXECUTION)
    if any(execution[key] != baseline[key] for key in ('packages', 'python', 'required_cpu_flags', 'environment')):
        raise ValueError('evaluation runtime differs from the canonical parent runtime')
    packages = {key: importlib.metadata.version(key) for key in execution['packages']}
    if packages != execution['packages'] or platform.python_version() != execution['python']:
        raise ValueError('evaluation runtime differs')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('evaluation requires Linux x86_64')
    cpu = Path('/proc/cpuinfo').read_text().split()
    if any(flag not in cpu for flag in execution['required_cpu_flags']):
        raise ValueError('evaluation CPU lacks required instructions')
    if any(os.environ.get(key) != value for key, value in execution['environment'].items()):
        raise ValueError('evaluation numerical environment differs')
    return {**source, 'packages': packages, 'python': platform.python_version()}


def worker(request_path):
    configure()
    request_path = Path(request_path)
    request = read(request_path)
    if freeze() != request['freeze']:
        raise ValueError('evaluation worker differs from freeze')
    import torch
    from neuroshard.evolution import assistant_experience_run as accelerator
    from neuroshard.evolution import assistant_experience_train as trainer

    execution = read(ROOT / EXECUTION)
    arm, phase = request['model'], request['phase']
    if (phase, arm) != ('prepare', 'baseline') and (arm not in ARMS or phase != 'development'):
        raise ValueError('unsupported evaluation worker role')
    plan = read(ROOT / PLAN)
    policy = read(ROOT / plan['policy'])
    torch.set_num_threads(execution['threads'])
    torch.set_num_interop_threads(1)
    reply = {'binding': request['binding'], 'arm': arm, 'execution_completed': False}
    started = time.monotonic()
    tokenizer = None
    try:
        inventory = read(ROOT / reference.ARTIFACTS)['models']['baseline']
        directory = Path(request['models']) / 'baseline'
        if phase == 'prepare':
            reply['file_state'] = verify_artifacts(directory, inventory, download=True)
            reply['execution_completed'] = True
            return
        arms = ROOT / UPLOADED
        gate = verify_arms(arms, execution['arms'])['arms'][arm]['gate']
        state = verify_artifacts(directory, inventory, download=False)
        tokenizer, report = granite_tokenizer.load(directory)
        reply['tokenizer'] = report
        parent, _ = reference.load_model(directory, 'baseline')
        cases = first.load_cases(read(ROOT / canonical.PLAN))
        model, _ = reference.load_model(directory, 'baseline')
        reply['checkpoint'] = trainer.load_trainable(model, arm, plan['training'], arms / f'{arm}-checkpoint')
        reply['served_projections_converted'] = trainer.serving(model, plan['training'])

        def feature(case):
            return accelerator.boundary_feature(parent, tokenizer, policy, case, 'cpu')

        reply.update(evaluate_arm(parent, model, tokenizer, gate, feature, cases, policy, read(ROOT / reference.PLAN)))
        if file_state(directory, inventory) != state:
            raise ValueError('parent checkpoint changed during evaluation')
        reply['execution_completed'] = True
    except Exception as error:
        reply['error'] = f'{type(error).__name__}: {error}'
    finally:
        reply.update(checked_encodes=tokenizer.checked_encodes if tokenizer else 0,
                     wall_seconds=time.monotonic() - started, process_cpu_seconds=time.process_time(),
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        save(request_path.parent / 'reply.json', reply, exclusive=True)


def assess(plan, cases, canonical_result, replies):
    from neuroshard.evolution import assistant_experience_gate as gate

    policy = read(ROOT / plan['policy'])
    by_id = {case['id']: case for case in cases}
    systems = {}
    for arm in ARMS:
        rows = replies[arm]['episodes']
        for row in rows:
            if workflow.score(by_id[row['id']], row, policy) != row['score']:
                raise ValueError('evaluation outcome rescore differs')
        systems[arm] = rows
    report = gate.development(plan, cases, canonical_result['primary']['episodes'], systems['update'],
                              systems['addition'], canonical_result['report']['protected_workflow_ids'])
    anchors = canonical_result['report']['protected_anchor_ids']
    report['forced_anchor_forgetting'] = {arm: forgetting(replies[arm]['forced_anchors'], anchors) for arm in ARMS}
    report['selected_arm_episodes'] = {arm: sum(r['selected'] == 'arm' for r in systems[arm]) for arm in ARMS}
    return report


def run(home, models):
    configure()
    source = freeze()
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    execution = read(ROOT / EXECUTION)
    plan = read(ROOT / PLAN)
    binding = {'freeze': source, 'profile': PROFILE, 'plan_sha256': sha256(ROOT / PLAN)}
    save(home / 'binding.json', binding, exclusive=True)
    result = {'binding': binding, 'execution_completed': False, 'checklist_credit': False,
              'admission_evidence': False, 'confirmation_opened': False}
    try:
        verify_arms(ROOT / UPLOADED, execution['arms'])
        prepared = launch(home, models, binding, 'baseline', 'prepare', execution['prepare_seconds'],
                          execution['memory_bytes'], worker_script=SCRIPT)
        if not prepared['execution_completed']:
            raise ValueError(prepared.get('error', 'parent artifacts were not prepared'))
        # One fresh worker at a time, as in the canonical baseline, so latency is comparable.
        replies = {arm: launch(home, models, binding, arm, 'development', execution['worker_seconds'],
                               execution['memory_bytes'], worker_script=SCRIPT) for arm in ARMS}
        result['replies'] = replies
        failed = [arm for arm in ARMS if not replies[arm]['execution_completed']]
        if failed:
            raise ValueError(f'incomplete evaluation for {failed}')
        canonical_result = read(ROOT / execution['canonical_result']['path'])
        if sha256(ROOT / execution['canonical_result']['path']) != execution['canonical_result']['sha256']:
            raise ValueError('canonical parent result changed')
        result['report'] = assess(plan, first.load_cases(read(ROOT / canonical.PLAN)), canonical_result, replies)
        result['execution_completed'] = True
    except Exception as error:
        result['error'] = str(error)
    save(home / 'result.json', result, exclusive=True)
    return result
