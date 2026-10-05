"""A3 stage 1: a turn router learned from what each turn needs, fitted once on the serving CPU runtime.

A turn is labelled for the scheduling route if its verified correct solution uses the
calendar. Every training scheduling case is solved by its demonstration in the calendar
workspace, and a turn's label is whether that turn's calls include a calendar tool.
Every turn of the drafting training cases is labelled for the drafting route: their goals
are drafts, and the drafting interface has no calendar. Features are the parent's
final-layer states at each user message, rendered as served. The router is the centroid
rule. Its accuracy on held-out training cases and on the integration turns is reported,
not gated. One router serves every version, since what a turn needs does not depend on
the unit that serves it.
"""

import importlib.metadata
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from neuroshard.evolution import assistant_calendar as calendar
from neuroshard.evolution import assistant_growth_eval as development
from neuroshard.evolution import assistant_growth_round3 as round3
from neuroshard.evolution import assistant_growth_round4 as round4
from neuroshard.evolution import assistant_growth_run as growth
from neuroshard.evolution import assistant_routing as routing
from neuroshard.evolution import assistant_workflow_canonical as canonical
from neuroshard.evolution import granite_reference as reference
from neuroshard.evolution import granite_tokenizer
from neuroshard.evolution.assistant_workflow_data import public_case
from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, identity, launch, read, save, sha256, verify_artifacts,
)

DECLARATION = 'config/experiments/assistant-growth-router2.json'
EXECUTION = 'config/experiments/assistant-growth-router2-execution.json'
SCRIPT = 'scripts/run_assistant_growth_router.py'
PROFILE = 'assistant-growth-router'
CALENDAR_TOOLS = ('list_busy', 'add_minutes', 'save_meeting')


def scheduling_turns(stage, policy):
    """Each training scheduling turn, labelled by whether its demonstration's calls use the calendar."""
    rows, texts = {}, {}
    for case in round4.training_cases(stage):
        _, result = round3.demonstration(case, policy)
        previous = 0
        for turn, (round_, user) in enumerate(zip(result['rounds'], public_case(case)['user_turns'])):
            calls = result['calls'][previous:round_['call_count']]
            previous = round_['call_count']
            key = routing.turn_key(case['id'], turn)
            rows[key] = (1.0 if any(c['call']['name'] in CALENDAR_TOOLS for c in calls) else 0.0, 1.0)
            texts[key] = user
    return rows, texts


def drafting_turns(learning, split='train'):
    """Each turn of the drafting training cases, labelled for the drafting route."""
    from neuroshard.evolution import assistant_experience_run as accelerator

    rows, texts = {}, {}
    for case in accelerator.split_cases(learning, split):
        for turn, user in enumerate(public_case(case)['user_turns']):
            key = routing.turn_key(case['id'], turn)
            rows[key], texts[key] = (0.0, 1.0), user
    return rows, texts


def integration_turns(stage, learning):
    """Each integration turn, labelled by whether its goal includes a meeting; a check, not training data."""
    rows, texts = {}, {}
    for case in growth.integration_cases(stage, learning):
        for turn, (spec, user) in enumerate(zip(case['turns'], public_case(case)['user_turns'])):
            needs = bool(case.get('capability')) and calendar.goals(spec['expected'])[1] is not None
            key = routing.turn_key(case['id'], turn)
            rows[key], texts[key] = (1.0 if needs else 0.0, 1.0), user
    return rows, texts


def fit(features, rows, check_features, check_rows, epsilon):
    """The centroid router, its held-out accuracy, and its errors on the integration turns."""
    from neuroshard.evolution import assistant_selector as selector

    gate = selector.fit_centroid(features, rows, epsilon)
    errors = sorted(key for key, (target, _) in check_rows.items()
                    if selector.choose(gate, check_features[key]) != (target == 1.0))
    report = {'turns': len(rows), 'scheduling_turns': sum(target == 1.0 for target, _ in rows.values()),
              'counts': gate['counts'], 'held_out_accuracy': development.held_out(features, rows, epsilon),
              'integration_turns': len(check_rows),
              'integration_accuracy': 1 - len(errors) / len(check_rows), 'integration_errors': errors,
              'features_sha256': identity(features), 'gate_sha256': identity(gate)}
    return gate, report


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
        rows, texts = scheduling_turns(stage, policies['scheduling'])
        drafting_rows, drafting_texts = drafting_turns(learning)
        rows.update(drafting_rows)
        texts.update(drafting_texts)
        check_rows, check_texts = integration_turns(stage, learning)
        begun = time.monotonic()
        drafting = policies['drafting']
        features = {key: routing.turn_feature(parent, tokenizer, drafting, texts[key], 'cpu') for key in sorted(rows)}
        check = {key: routing.turn_feature(parent, tokenizer, drafting, check_texts[key], 'cpu')
                 for key in sorted(check_rows)}
        reply['feature_seconds'] = time.monotonic() - begun
        gate, reply['router'] = fit(features, rows, check, check_rows, stage['integration']['recipe']['epsilon'])
        save(request_path.parent / 'router.json', {'gate': gate, 'report': reply['router']}, exclusive=True)
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
