"""A3 stage 1: the message-feature router, shifted toward scheduling by a held-out calibration.

Labels, data, feature and the centroid rule are the message-feature router's. A turn that
needs the calendar always fails on the drafting route, while a drafting turn on the
scheduling route can still succeed, so the router's threshold moves toward scheduling. The
shift is the largest under which at most a declared share of the drafting training turns,
each scored by the rule fitted without its fold, would go to scheduling. Development runs
only if the shift sends fewer integration turns that need the calendar to drafting.
"""

import importlib.metadata
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from neuroshard.evolution import assistant_growth_round4 as round4
from neuroshard.evolution import assistant_growth_router as router
from neuroshard.evolution import assistant_growth_router3 as router3
from neuroshard.evolution import assistant_growth_run as growth
from neuroshard.evolution import assistant_workflow_canonical as canonical
from neuroshard.evolution import granite_reference as reference
from neuroshard.evolution import granite_tokenizer
from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, identity, launch, read, save, sha256, verify_artifacts,
)

DECLARATION = 'config/experiments/assistant-growth-router4.json'
EXECUTION = 'config/experiments/assistant-growth-router4-execution.json'
SCRIPT = 'scripts/run_assistant_growth_router4.py'
PROFILE = 'assistant-growth-router4'


def calibrate(features, rows, drafting, epsilon, rate, folds=4):
    """The largest shift toward scheduling under which at most ``rate`` of the held-out ``drafting`` turns cross.

    Each fold of training cases is scored by the centroid rule fitted without it, the folds
    of the reported held-out accuracy. Returns the shift, never negative, the held-out
    margins and the number of drafting turns allowed to cross.
    """
    from neuroshard.evolution import assistant_selector as selector

    cases = sorted({key.rsplit('#', 1)[0] for key in rows})
    fold = {case: index % folds for index, case in enumerate(cases)}
    margins = {}
    for k in range(folds):
        train = {key: row for key, row in rows.items() if fold[key.rsplit('#', 1)[0]] != k}
        gate = selector.fit_centroid({key: features[key] for key in train}, train, epsilon)
        margins.update({key: selector.margin(gate, features[key]) for key in rows if fold[key.rsplit('#', 1)[0]] == k})
    ranked = sorted((margins[key] for key in drafting), reverse=True)
    allowed = math.floor(rate * len(ranked))
    if not ranked or allowed >= len(ranked):
        raise ValueError('the calibration needs more drafting turns than it may let cross')
    return max(0.0, -ranked[allowed]), margins, allowed


def routes(rows, scheduling, drafting=()):
    """Where a set of decisions sends the labelled turns: ``scheduling`` maps each turn to True for scheduling."""
    calendar = sorted(key for key, (target, _) in rows.items() if target == 1.0)
    other = sorted(key for key in rows if rows[key][0] != 1.0)
    summary = {'calendar_turns': len(calendar), 'calendar_to_drafting': [key for key in calendar if not scheduling[key]],
               'other_turns': len(other), 'other_to_scheduling': [key for key in other if scheduling[key]],
               'accuracy': sum(scheduling[key] == (rows[key][0] == 1.0) for key in rows) / len(rows)}
    if drafting:
        summary.update(drafting_turns=len(drafting),
                       drafting_to_scheduling=sum(bool(scheduling[key]) for key in drafting))
    return summary


def calibrated(features, rows, check, check_rows, drafting, epsilon, rate):
    """The shifted router, and where it and the unshifted router send the held-out and integration turns."""
    from neuroshard.evolution import assistant_selector as selector

    gate, report = router.fit(features, rows, check, check_rows, epsilon, router3.FEATURE)
    shift, margins, allowed = calibrate(features, rows, drafting, epsilon, rate)
    final = selector.shifted(gate, shift)
    held = {name: routes(rows, {key: margins[key] > -value for key in rows}, drafting)
            for name, value in (('unshifted', 0.0), ('shifted', shift))}
    integration = {name: routes(check_rows, {key: selector.choose(chosen, check[key]) for key in check_rows})
                   for name, chosen in (('unshifted', gate), ('shifted', final))}
    report.update(unshifted_gate_sha256=report.pop('gate_sha256'), gate_sha256=identity(final),
                  calibration={'rate': rate, 'drafting_turns': len(drafting), 'allowed_crossings': allowed,
                               'shift': shift, 'held_out': held, 'integration': integration},
                  development_runs=(len(integration['shifted']['calendar_to_drafting'])
                                    < len(integration['unshifted']['calendar_to_drafting'])))
    return final, report, margins


def committed_sources(root=ROOT):
    execution = read(root / EXECUTION)
    for name, digest in execution['contracts'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'changed router contract: {name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in execution['sources']:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root) != (root / name).read_bytes():
            raise ValueError(f'uncommitted router source: {name}')
        sources[name] = sha256(root / name)
    return {'commit': commit, 'sources': sources}


def configure():
    if 'torch' in sys.modules:
        raise ValueError('configure the router runtime before importing torch')
    for key, value in read(ROOT / EXECUTION)['environment'].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def freeze():
    """The canonical parent's CPU runtime, byte for byte in packages and environment."""
    source = committed_sources()
    execution = read(ROOT / EXECUTION)
    pinned = read(ROOT / canonical.EXECUTION)
    if any(execution[key] != pinned[key] for key in ('packages', 'python', 'required_cpu_flags', 'environment')):
        raise ValueError('router runtime differs from the canonical parent runtime')
    packages = {key: importlib.metadata.version(key) for key in execution['packages']}
    if packages != execution['packages'] or platform.python_version() != execution['python']:
        raise ValueError('router runtime differs')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('the router requires Linux x86_64')
    cpu = Path('/proc/cpuinfo').read_text().split()
    if any(flag not in cpu for flag in execution['required_cpu_flags']):
        raise ValueError('router CPU lacks required instructions')
    if any(os.environ.get(key) != value for key, value in execution['environment'].items()):
        raise ValueError('router numerical environment differs')
    return {**source, 'packages': packages, 'python': platform.python_version()}


def worker(request_path):
    configure()
    request_path = Path(request_path)
    request = read(request_path)
    if freeze() != request['freeze']:
        raise ValueError('router worker differs from freeze')
    import torch

    execution = read(ROOT / EXECUTION)
    phase = request['phase']
    if (request['model'], phase) not in (('baseline', 'prepare'), ('baseline', 'router')):
        raise ValueError('unsupported router worker role')
    torch.set_num_threads(execution['threads'])
    torch.set_num_interop_threads(1)
    reply = {'binding': request['binding'], 'execution_completed': False}
    started = time.monotonic()
    tokenizer = None
    try:
        inventory = read(ROOT / reference.ARTIFACTS)['models']['baseline']
        directory = Path(request['models']) / 'baseline'
        if phase == 'prepare':
            reply['file_state'] = verify_artifacts(directory, inventory, download=True)
            reply['execution_completed'] = True
            return
        stage = round4.plan()
        _, learning, policies = growth.contracts(stage)
        state = verify_artifacts(directory, inventory, download=False)
        tokenizer, report = granite_tokenizer.load(directory)
        reply['tokenizer'] = report
        parent, _ = reference.load_model(directory, 'baseline')
        rows, texts = router.scheduling_turns(stage, policies['scheduling'])
        drafting_rows, drafting_texts = router.drafting_turns(learning)
        rows.update(drafting_rows)
        texts.update(drafting_texts)
        check_rows, check_texts = router.integration_turns(stage, learning)
        begun = time.monotonic()
        features = router3.message_features(parent, tokenizer, policies['drafting'], texts)
        check = router3.message_features(parent, tokenizer, policies['drafting'], check_texts)
        reply['feature_seconds'] = time.monotonic() - begun
        rate = read(ROOT / DECLARATION)['router']['calibration']['rate']
        gate, reply['router'], margins = calibrated(features, rows, check, check_rows, sorted(drafting_rows),
                                                    stage['integration']['recipe']['epsilon'], rate)
        save(request_path.parent / 'router.json', {'gate': gate, 'report': reply['router'],
                                                    'held_out_margins': margins}, exclusive=True)
        if file_state(directory, inventory) != state:
            raise ValueError('parent checkpoint changed while fitting the router')
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
    binding = {'freeze': source, 'profile': PROFILE, 'plan_sha256': sha256(ROOT / growth.PLAN),
               'declaration_sha256': sha256(ROOT / DECLARATION)}
    save(home / 'binding.json', binding, exclusive=True)
    result = {'binding': binding, 'execution_completed': False, 'checklist_credit': False,
              'admission_evidence': False, 'development_opened': False, 'confirmation_opened': False}
    try:
        prepared = launch(home, models, binding, 'baseline', 'prepare', execution['prepare_seconds'],
                          execution['memory_bytes'], worker_script=SCRIPT)
        if not prepared['execution_completed']:
            raise ValueError(prepared.get('error', 'parent artifacts were not prepared'))
        reply = launch(home, models, binding, 'baseline', 'router', execution['worker_seconds'],
                       execution['memory_bytes'], worker_script=SCRIPT)
        result['reply'] = reply
        if not reply['execution_completed']:
            raise ValueError(reply.get('error', 'incomplete router fit'))
        result['execution_completed'] = True
    except Exception as error:
        result['error'] = str(error)
    save(home / 'result.json', result, exclusive=True)
    return result
