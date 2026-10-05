"""Stage 0 of repeated growth: the parent and the accepted version under the calendar interface.

Before any scheduling training is declared, one CPU host serves the opened development
cases of the scheduling, cross-capability and drafting grammars under the calendar
policy: first the unchanged parent, then the accepted A2 version (the parent plus the
round-4 update, selected per episode by its gate). Each runs in a fresh worker at the
canonical eight threads with prefix-cache serving. Nothing is trained and no sealed
split opens. Drafting outcomes are compared, case by case, with the pinned results of
the same systems under the drafting interface.
"""

import importlib.metadata
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from neuroshard.evolution import assistant_schedule_data as schedule
from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution import assistant_workflow_canonical as canonical
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution import granite_reference as reference
from neuroshard.evolution import granite_tokenizer
from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, identity, launch, read, save, sha256, verify_artifacts,
)

PLAN = 'config/experiments/assistant-growth.json'
EXECUTION = 'config/experiments/assistant-growth-baseline-execution.json'
SCRIPT = 'scripts/run_assistant_growth_baseline.py'
PROFILE = 'assistant-growth-baseline'
UPLOADED = '.arms'
SYSTEMS = ('parent', 'update')
SETS = {'scheduling': ('schedule', 'development'), 'cross': ('schedule', 'cross-development'),
        'drafting': ('drafting', 'development')}


def opened(plan):
    """The opened development cases of each grammar, each checked against its frozen manifest."""
    manifests = {'schedule': read(ROOT / plan['cohort2']['data'])['splits'],
                 'drafting': read(ROOT / read(ROOT / plan['cohort1']['learning'])['data'])['splits']}
    sets = {}
    for name, (grammar, split) in SETS.items():
        cases = (schedule if grammar == 'schedule' else data).cases(split)
        frozen = manifests[grammar][split]
        if identity(cases) != frozen['sha256'] or [c['id'] for c in cases] != frozen['case_ids']:
            raise ValueError(f'{name} cases differ from their frozen split')
        sets[name] = cases
    return sets


def committed_sources(root=ROOT):
    execution = read(root / EXECUTION)
    for name, digest in execution['contracts'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'changed growth contract: {name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in execution['sources']:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root) != (root / name).read_bytes():
            raise ValueError(f'uncommitted growth source: {name}')
        sources[name] = sha256(root / name)
    return {'commit': commit, 'sources': sources}


def configure():
    if 'torch' in sys.modules:
        raise ValueError('configure the growth runtime before importing torch')
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
        raise ValueError('growth runtime differs from the canonical parent runtime')
    packages = {key: importlib.metadata.version(key) for key in execution['packages']}
    if packages != execution['packages'] or platform.python_version() != execution['python']:
        raise ValueError('growth runtime differs')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('growth evaluation requires Linux x86_64')
    cpu = Path('/proc/cpuinfo').read_text().split()
    if any(flag not in cpu for flag in execution['required_cpu_flags']):
        raise ValueError('growth CPU lacks required instructions')
    if any(os.environ.get(key) != value for key, value in execution['environment'].items()):
        raise ValueError('growth numerical environment differs')
    return {**source, 'packages': packages, 'python': platform.python_version()}


def worker(request_path):
    configure()
    request_path = Path(request_path)
    request = read(request_path)
    if freeze() != request['freeze']:
        raise ValueError('growth worker differs from freeze')
    import torch
    from neuroshard.evolution import assistant_experience_eval as development
    from neuroshard.evolution import assistant_experience_run as accelerator
    from neuroshard.evolution import assistant_experience_train as trainer
    from neuroshard.evolution.assistant_serving import cached_responder

    execution = read(ROOT / EXECUTION)
    plan = read(ROOT / PLAN)
    system, phase = request['model'], request['phase']
    if (phase, system) != ('prepare', 'baseline') and (system not in SYSTEMS or phase != 'baseline'):
        raise ValueError('unsupported growth worker role')
    policy = read(ROOT / plan['cohort2']['policy'])
    training = read(ROOT / plan['cohort1']['learning'])['training']
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
        sets = opened(plan)
        state = verify_artifacts(directory, inventory, download=False)
        tokenizer, report = granite_tokenizer.load(directory)
        reply['tokenizer'] = report
        parent, _ = reference.load_model(directory, 'baseline')
        if system == 'parent':
            reply['episodes'] = {name: [workflow.execute(case, cached_responder(parent, tokenizer, policy), policy)
                                        for case in cases] for name, cases in sets.items()}
        else:
            arms = ROOT / UPLOADED
            gate = development.verify_arms(arms, execution['arms'])['arms']['update']['gate']
            model, _ = reference.load_model(directory, 'baseline')
            reply['checkpoint'] = trainer.load_trainable(model, 'update', training, arms / 'update-checkpoint')
            reply['served_projections_converted'] = trainer.serving(model, training)

            def feature(case):
                return accelerator.boundary_feature(parent, tokenizer, policy, case, 'cpu')

            reply['episodes'] = {name: development.evaluate_arm(parent, model, tokenizer, gate, feature, cases, policy,
                                                                {'tasks': []}, cached_responder)['episodes']
                                 for name, cases in sets.items()}
        reply['serving'] = 'prefix-cache'
        if file_state(directory, inventory) != state:
            raise ValueError('parent checkpoint changed during the growth baseline')
        reply['execution_completed'] = True
    except Exception as error:
        reply['error'] = f'{type(error).__name__}: {error}'
    finally:
        reply.update(checked_encodes=tokenizer.checked_encodes if tokenizer else 0,
                     wall_seconds=time.monotonic() - started, process_cpu_seconds=time.process_time(),
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        save(request_path.parent / 'reply.json', reply, exclusive=True)


def assess(plan, sets, replies, references):
    """Every episode rescored; outcomes per grammar and family, latency, and the drafting interface effect.

    ``references`` holds each system's pinned drafting episodes under the drafting interface.
    """
    from neuroshard.evolution import assistant_experience_gate as gate

    policy = read(ROOT / plan['cohort2']['policy'])
    report = {'systems': {}, 'drafting_interface_effect': {}}
    for system in SYSTEMS:
        outcomes = {}
        for name, cases in sets.items():
            rows = replies[system]['episodes'][name]
            by_id = {case['id']: case for case in cases}
            if [row['id'] for row in rows] != [case['id'] for case in cases]:
                raise ValueError('every growth episode must run, in order')
            for row in rows:
                if workflow.score(by_id[row['id']], row, policy) != row['score']:
                    raise ValueError('growth outcome rescore differs')
            passed = gate.passed(rows)
            families = sorted({case['family'] for case in cases})
            outcomes[name] = {'correct': sum(passed.values()), 'cases': len(cases),
                              'by_family': {f: sum(passed[c['id']] for c in cases if c['family'] == f) for f in families},
                              'p95_seconds': gate.p95(rows, routed=system != 'parent')}
            if system != 'parent':
                outcomes[name]['selected_arm'] = sum(row['selected'] == 'arm' for row in rows)
        report['systems'][system] = outcomes
        earlier = gate.passed(references[system])
        report['drafting_interface_effect'][system] = {
            'drafting_interface_correct': sum(earlier.values()),
            **gate.paired(gate.passed(replies[system]['episodes']['drafting']), earlier)}
    rules = plan['stage0']
    current = report['systems']['update']
    report['headroom'] = (current['scheduling']['correct'] <= rules['headroom_maximum']
                          and report['systems']['parent']['scheduling']['correct'] <= rules['headroom_maximum'])
    report['stage1_may_be_declared'] = report['headroom']
    return report


def references(plan, execution):
    """Each system's pinned drafting development episodes under the drafting interface."""
    rows = {}
    for system, pinned in execution['drafting_references'].items():
        if sha256(ROOT / pinned['path']) != pinned['sha256']:
            raise ValueError(f'{system} drafting reference changed')
        result = read(ROOT / pinned['path'])
        rows[system] = result['primary']['episodes'] if system == 'parent' else result['replies']['update']['episodes']
    return rows


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
              'admission_evidence': False, 'sealed_opened': False}
    try:
        from neuroshard.evolution import assistant_experience_eval as development

        sets = opened(plan)
        pinned = references(plan, execution)
        development.verify_arms(ROOT / UPLOADED, execution['arms'])
        prepared = launch(home, models, binding, 'baseline', 'prepare', execution['prepare_seconds'],
                          execution['memory_bytes'], worker_script=SCRIPT)
        if not prepared['execution_completed']:
            raise ValueError(prepared.get('error', 'parent artifacts were not prepared'))
        # One fresh worker at a time, as in the canonical baseline, so latency is comparable.
        replies = {system: launch(home, models, binding, system, 'baseline', execution['worker_seconds'],
                                  execution['memory_bytes'], worker_script=SCRIPT) for system in SYSTEMS}
        result['replies'] = replies
        failed = [system for system in SYSTEMS if not replies[system]['execution_completed']]
        if failed:
            raise ValueError(f'incomplete growth baseline for {failed}')
        result['report'] = assess(plan, sets, replies, pinned)
        result['execution_completed'] = True
    except Exception as error:
        result['error'] = str(error)
    save(home / 'result.json', result, exclusive=True)
    return result
