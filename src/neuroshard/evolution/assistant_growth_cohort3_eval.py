"""A3 cohort 3 development: the upgraded system routed turn by turn on the opened development cases.

One CPU host on the canonical runtime with prefix-cache serving. The upgraded system is
cohort 2's accepted system with L3 on its drafting route: the pinned router sends each user
turn to drafting, where A2's gate chooses once per episode between the parent and U1 with
L3, or to scheduling, U1 with L2 under the free-slot policy. The previous system's
development episodes are pinned; both systems' episodes are rescored and compared case by
case after the host finishes. Nothing is trained and no sealed split opens.
"""

import importlib.metadata
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from neuroshard.evolution import assistant_growth_baseline as baseline
from neuroshard.evolution import assistant_growth_confirm as stage1_confirmation
from neuroshard.evolution import assistant_growth_eval as stage1_development
from neuroshard.evolution import assistant_routing as routing
from neuroshard.evolution import assistant_workflow_canonical as canonical
from neuroshard.evolution import granite_reference as reference
from neuroshard.evolution import granite_tokenizer
from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, identity, launch, read, save, sha256, verify_artifacts,
)

DECLARATION = 'config/experiments/assistant-growth-cohort3.json'
EXECUTION = 'config/experiments/assistant-growth-cohort3-development-execution.json'
SCRIPT = 'scripts/run_assistant_growth_cohort3_development.py'
PROFILE = 'assistant-growth-cohort3-development'
UPLOADED = '.units'
# Each system's units on the drafting route and on the scheduling route.
SYSTEMS = {'upgrade': ('L3', 'L2'), 'previous': ('U1', 'L2')}


def policies():
    """The drafting and scheduling routes' policies, as cohort 3 declares them."""
    return {route: read(ROOT / path) for route, path in read(ROOT / DECLARATION)['policies'].items()}


def needed(system):
    """The checkpoints a system loads: its units, and U1, which L2 and L3 are trained on top of."""
    return sorted(set(SYSTEMS[system]) | {'U1'})


def load_units(load_parent, spec, directory, execution, units):
    """Each unit on a fresh parent: U1 served as plain projections, and L2 or L3 attached on top of it."""
    from neuroshard.evolution import assistant_experience_train as trainer

    def load(unit):
        model = load_parent()
        trainer.load_trainable(model, 'update', spec, Path(directory) / execution['units']['U1']['checkpoint'])
        trainer.serving(model, spec)
        if unit != 'U1':
            trainer.load_trainable(model, 'addition', spec, Path(directory) / execution['units'][unit]['checkpoint'])
        return model.eval()

    return {unit: load(unit) for unit in sorted(set(units))}


def serve(directory, tokenizer, execution, system, sets, spec):
    """One system routed turn by turn on every set, behind A2's per-episode gate and the pinned router."""
    names = needed(system)
    gates = stage1_confirmation.verify(ROOT / UPLOADED, execution, names)
    a2_gate, turn_gate = gates['a2']['arms']['update']['gate'], gates['router']['gate']
    parent, _ = reference.load_model(directory, 'baseline')
    models = load_units(lambda: reference.load_model(directory, 'baseline')[0], spec, ROOT / UPLOADED, execution,
                        SYSTEMS[system])
    route_policies = policies()
    episodes = {name: stage1_development.routed_episodes(parent, models, tokenizer, None, a2_gate, turn_gate, cases,
                                                  units=SYSTEMS[system], route_policies=route_policies)
                for name, cases in sets.items()}
    return {'units': {unit: execution['units'][unit]['trainable_sha256'] for unit in names},
            'selector': {'rule': turn_gate['rule'], 'gate_sha256': identity(turn_gate)}, 'episodes': episodes}


def committed_sources(root=ROOT):
    execution = read(root / EXECUTION)
    for name, digest in execution['contracts'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'changed cohort-3 development contract: {name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in execution['sources']:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root) != (root / name).read_bytes():
            raise ValueError(f'uncommitted cohort-3 development source: {name}')
        sources[name] = sha256(root / name)
    return {'commit': commit, 'sources': sources}


def configure():
    if 'torch' in sys.modules:
        raise ValueError('configure the development runtime before importing torch')
    for key, value in read(ROOT / EXECUTION)['environment'].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def runtime(execution):
    """The canonical parent's CPU runtime, byte for byte in packages and environment."""
    pinned = read(ROOT / canonical.EXECUTION)
    if any(execution[key] != pinned[key] for key in ('packages', 'python', 'required_cpu_flags', 'environment')):
        raise ValueError('cohort-3 runtime differs from the canonical parent runtime')
    packages = {key: importlib.metadata.version(key) for key in execution['packages']}
    if packages != execution['packages'] or platform.python_version() != execution['python']:
        raise ValueError('cohort-3 runtime differs')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('cohort 3 requires Linux x86_64')
    cpu = Path('/proc/cpuinfo').read_text().split()
    if any(flag not in cpu for flag in execution['required_cpu_flags']):
        raise ValueError('cohort-3 CPU lacks required instructions')
    if any(os.environ.get(key) != value for key, value in execution['environment'].items()):
        raise ValueError('cohort-3 numerical environment differs')
    return {'packages': packages, 'python': platform.python_version()}


def freeze():
    return {**committed_sources(), **runtime(read(ROOT / EXECUTION))}


def worker(request_path):
    configure()
    request_path = Path(request_path)
    request = read(request_path)
    if freeze() != request['freeze']:
        raise ValueError('development worker differs from freeze')
    import torch
    from neuroshard.evolution import assistant_growth_run as growth

    execution = read(ROOT / EXECUTION)
    plan = read(ROOT / growth.PLAN)
    system, phase = request['model'], request['phase']
    if (phase, system) != ('prepare', 'baseline') and (system, phase) != ('upgrade', 'development'):
        raise ValueError('unsupported development worker role')
    spec = growth.stage_spec(plan, read(ROOT / read(ROOT / plan['plan'])['cohort1']['learning']))
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
        sets = baseline.opened(read(ROOT / plan['plan']))
        state = verify_artifacts(directory, inventory, download=False)
        tokenizer, report = granite_tokenizer.load(directory)
        reply['tokenizer'] = report
        reply.update(serve(directory, tokenizer, execution, system, sets, spec))
        reply['serving'] = 'prefix-cache'
        if file_state(directory, inventory) != state:
            raise ValueError('parent checkpoint changed during development')
        reply['execution_completed'] = True
    except Exception as error:
        reply['error'] = f'{type(error).__name__}: {error}'
    finally:
        reply.update(checked_encodes=tokenizer.checked_encodes if tokenizer else 0,
                     wall_seconds=time.monotonic() - started, process_cpu_seconds=time.process_time(),
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        save(request_path.parent / 'reply.json', reply, exclusive=True)


def run(home, models):
    configure()
    source = freeze()
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    execution = read(ROOT / EXECUTION)
    binding = {'freeze': source, 'profile': PROFILE, 'declaration_sha256': sha256(ROOT / DECLARATION)}
    save(home / 'binding.json', binding, exclusive=True)
    result = {'binding': binding, 'system': 'upgrade', 'execution_completed': False, 'checklist_credit': False,
              'admission_evidence': False, 'confirmation_opened': False}
    try:
        stage1_confirmation.verify(ROOT / UPLOADED, execution, needed('upgrade'))
        prepared = launch(home, models, binding, 'baseline', 'prepare', execution['prepare_seconds'],
                          execution['memory_bytes'], worker_script=SCRIPT)
        if not prepared['execution_completed']:
            raise ValueError(prepared.get('error', 'parent artifacts were not prepared'))
        reply = launch(home, models, binding, 'upgrade', 'development', execution['worker_seconds'],
                       execution['memory_bytes'], worker_script=SCRIPT)
        result['reply'] = reply
        if not reply['execution_completed']:
            raise ValueError(reply.get('error', 'incomplete development'))
        result['execution_completed'] = True
    except Exception as error:
        result['error'] = str(error)
    save(home / 'result.json', result, exclusive=True)
    return result


def previous_episodes(declaration):
    """The previous system's pinned development episodes."""
    pinned = declaration['previous']['development']
    if sha256(ROOT / pinned['path']) != pinned['sha256']:
        raise ValueError("the previous system's development episodes differ from their pin")
    return read(ROOT / pinned['path'])['reply']['episodes']


def compare(sets, episodes, route_policies):
    """Both systems' episodes on each set, rescored, with the cases the upgrade gained and lost."""
    from neuroshard.evolution import assistant_experience_gate as gate

    report = {}
    for name, cases in sets.items():
        by_id = {case['id']: case for case in cases}
        passed = {}
        for system, rows in episodes.items():
            if [row['id'] for row in rows[name]] != [case['id'] for case in cases]:
                raise ValueError('every episode must run, in order')
            for row in rows[name]:
                if routing.score(by_id[row['id']], row, route_policies) != row['score']:
                    raise ValueError('outcome rescore differs')
            passed[system] = gate.passed(rows[name])
        report[name] = {'upgrade': sum(passed['upgrade'].values()), 'previous': sum(passed['previous'].values()),
                        'cases': len(cases), **gate.paired(passed['upgrade'], passed['previous'])}
    return report


def assess(declaration, sets, reply):
    """The declared development gate, after the host finishes: every episode of both systems rescored."""
    from neuroshard.evolution import granite_context_reference as context

    report = {'sets': compare(sets, {'upgrade': reply['episodes'], 'previous': previous_episodes(declaration)},
                              policies())}
    rules = declaration['development_gate']
    p95 = context.percentile([stage1_development.latency(row) for rows in reply['episodes'].values() for row in rows], .95)
    checks = {'drafting': report['sets']['drafting']['upgrade'] >= rules['minimum_drafting'],
              'lost': not any(row['lost'] for row in report['sets'].values()),
              'p95': p95 <= rules['p95_seconds']}
    report.update(p95_seconds=p95, checks=checks, passed=all(checks.values()))
    return report
