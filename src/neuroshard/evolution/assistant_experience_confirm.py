"""Confirmation of the verified-experience systems on a sealed split, once.

It opens only when the pinned development report records a complete pass. Three
hosts run one system each so latency conditions match the canonical baseline:
the parent control under its declared serving (recompute unless the execution
says otherwise), and each routed system under the declared serving runtime. The
gate is assessed after all three finish.
"""

import importlib.metadata
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from neuroshard.evolution import assistant_experience_eval as development
from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution import assistant_workflow_baseline as first
from neuroshard.evolution import assistant_workflow_canonical as canonical
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution import granite_reference as reference
from neuroshard.evolution import granite_tokenizer
from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, identity, launch, read, save, sha256, verify_artifacts,
)

EXECUTION = 'config/experiments/assistant-experience-confirmation-execution.json'
SCRIPT = 'scripts/run_assistant_experience_confirmation.py'
SYSTEMS = ('parent', 'update', 'addition')
PROFILES = {f'assistant-experience-confirmation-{system}': system for system in SYSTEMS}


def opened(execution):
    """The declared confirmation split, only behind a pinned complete development pass."""
    pinned = execution['development_report']
    report_path = ROOT / pinned['path']
    report = read(report_path)
    if sha256(report_path) != pinned['sha256'] or not all(report[key] for key in pinned.get('requires', ['development_passed'])):
        raise ValueError('confirmation is sealed without a pinned complete development pass')
    split = execution.get('split', 'confirmation')
    if 'data' in execution:
        manifest = read(ROOT / execution['data'])
        if manifest['split'] != split:
            raise ValueError('confirmation manifest is for another split')
    else:
        manifest = read(ROOT / read(ROOT / development.PLAN)['data'])['splits'][split]
    cases = data.cases(split)
    if identity(cases) != manifest['sha256'] or [c['id'] for c in cases] != manifest['case_ids']:
        raise ValueError('confirmation data differs from the frozen split')
    return cases


def committed_sources(root=ROOT):
    execution = read(root / EXECUTION)
    for name, digest in execution['contracts'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'changed confirmation contract: {name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in execution['sources']:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root) != (root / name).read_bytes():
            raise ValueError(f'uncommitted confirmation source: {name}')
        sources[name] = sha256(root / name)
    return {'commit': commit, 'sources': sources}


def configure():
    if 'torch' in sys.modules:
        raise ValueError('configure the confirmation runtime before importing torch')
    for key, value in read(ROOT / EXECUTION)['environment'].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def freeze():
    source = committed_sources()
    execution = read(ROOT / EXECUTION)
    baseline = read(ROOT / canonical.EXECUTION)
    if any(execution[key] != baseline[key] for key in ('packages', 'python', 'required_cpu_flags', 'environment')):
        raise ValueError('confirmation runtime differs from the canonical parent runtime')
    packages = {key: importlib.metadata.version(key) for key in execution['packages']}
    if packages != execution['packages'] or platform.python_version() != execution['python']:
        raise ValueError('confirmation runtime differs')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('confirmation requires Linux x86_64')
    cpu = Path('/proc/cpuinfo').read_text().split()
    if any(flag not in cpu for flag in execution['required_cpu_flags']):
        raise ValueError('confirmation CPU lacks required instructions')
    if any(os.environ.get(key) != value for key, value in execution['environment'].items()):
        raise ValueError('confirmation numerical environment differs')
    return {**source, 'packages': packages, 'python': platform.python_version()}


def worker(request_path):
    configure()
    request_path = Path(request_path)
    request = read(request_path)
    if freeze() != request['freeze']:
        raise ValueError('confirmation worker differs from freeze')
    import torch
    from neuroshard.evolution import assistant_experience_run as accelerator
    from neuroshard.evolution import assistant_experience_train as trainer
    from neuroshard.evolution.assistant_serving import cached_responder

    execution = read(ROOT / EXECUTION)
    system, phase = request['model'], request['phase']
    if (phase, system) != ('prepare', 'baseline') and (system not in SYSTEMS or phase != 'confirmation'):
        raise ValueError('unsupported confirmation worker role')
    plan = read(ROOT / development.PLAN)
    policy = read(ROOT / plan['policy'])
    torch.set_num_threads(execution['threads'])
    torch.set_num_interop_threads(1)
    reply = {'binding': request['binding'], 'system': system, 'execution_completed': False}
    started = time.monotonic()
    tokenizer = None
    try:
        inventory = read(ROOT / reference.ARTIFACTS)['models']['baseline']
        directory = Path(request['models']) / 'baseline'
        if phase == 'prepare':
            reply['file_state'] = verify_artifacts(directory, inventory, download=True)
            reply['execution_completed'] = True
            return
        cases = opened(execution)
        state = verify_artifacts(directory, inventory, download=False)
        tokenizer, report = granite_tokenizer.load(directory)
        reply['tokenizer'] = report
        parent, _ = reference.load_model(directory, 'baseline')
        if system == 'parent' and execution.get('parent_serving') == 'prefix-cache':
            reply['episodes'] = [workflow.execute(case, cached_responder(parent, tokenizer, policy), policy)
                                 for case in cases]
            reply['serving'] = 'prefix-cache'
        elif system == 'parent':
            respond = first.native_responder(parent, tokenizer, policy)
            reply['episodes'] = [workflow.execute(case, respond, policy) for case in cases]
            reply['serving'] = 'recompute'
        else:
            arms = ROOT / development.UPLOADED
            gate = development.verify_arms(arms, execution['arms'])['arms'][system]['gate']
            model, _ = reference.load_model(directory, 'baseline')
            reply['checkpoint'] = trainer.load_trainable(model, system, plan['training'], arms / f'{system}-checkpoint')
            reply['served_projections_converted'] = trainer.serving(model, plan['training'])

            def feature(case):
                return accelerator.boundary_feature(parent, tokenizer, policy, case, 'cpu')

            result = development.evaluate_arm(parent, model, tokenizer, gate, feature, cases, policy,
                                              {'tasks': []}, cached_responder)
            reply['episodes'] = result['episodes']
            reply['serving'] = 'prefix-cache'
        if file_state(directory, inventory) != state:
            raise ValueError('parent checkpoint changed during confirmation')
        reply['execution_completed'] = True
    except Exception as error:
        reply['error'] = f'{type(error).__name__}: {error}'
    finally:
        reply.update(checked_encodes=tokenizer.checked_encodes if tokenizer else 0,
                     wall_seconds=time.monotonic() - started, process_cpu_seconds=time.process_time(),
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        save(request_path.parent / 'reply.json', reply, exclusive=True)


def run(home, models, system):
    configure()
    source = freeze()
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    execution = read(ROOT / EXECUTION)
    if system not in SYSTEMS:
        raise ValueError('unknown confirmation system')
    binding = {'freeze': source, 'profile': f'assistant-experience-confirmation-{system}',
               'plan_sha256': sha256(ROOT / development.PLAN)}
    save(home / 'binding.json', binding, exclusive=True)
    result = {'binding': binding, 'system': system, 'execution_completed': False, 'checklist_credit': False,
              'admission_evidence': False}
    try:
        opened(execution)
        if system != 'parent':
            development.verify_arms(ROOT / development.UPLOADED, execution['arms'])
        prepared = launch(home, models, binding, 'baseline', 'prepare', execution['prepare_seconds'],
                          execution['memory_bytes'], worker_script=SCRIPT)
        if not prepared['execution_completed']:
            raise ValueError(prepared.get('error', 'parent artifacts were not prepared'))
        reply = launch(home, models, binding, system, 'confirmation', execution['worker_seconds'],
                       execution['memory_bytes'], worker_script=SCRIPT)
        result['reply'] = reply
        if not reply['execution_completed']:
            raise ValueError(reply.get('error', 'incomplete confirmation'))
        result['execution_completed'] = True
    except Exception as error:
        result['error'] = str(error)
    save(home / 'result.json', result, exclusive=True)
    return result


def assess(plan, cases, replies, section='confirmation_gate'):
    """The declared confirmation gate from three completed systems, every episode rescored."""
    from neuroshard.evolution import assistant_experience_gate as gate

    policy = read(ROOT / plan['policy'])
    by_id = {case['id']: case for case in cases}
    rows = {}
    for system in SYSTEMS:
        rows[system] = replies[system]['episodes']
        for row in rows[system]:
            if workflow.score(by_id[row['id']], row, policy) != row['score']:
                raise ValueError('confirmation outcome rescore differs')
    protected = sorted(row['id'] for row in rows['parent'] if row['score']['passed'])
    report = gate.confirmation(plan, cases, rows['parent'], rows['update'], rows['addition'], protected, section)
    report['selected_arm_episodes'] = {s: sum(r['selected'] == 'arm' for r in rows[s]) for s in ('update', 'addition')}
    return report
