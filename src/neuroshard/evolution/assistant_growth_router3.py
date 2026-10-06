"""A3 stage 1: the turn-need router refitted on what the user wrote, once, on the serving CPU runtime.

The labels and data are the previous router's: a turn needs the scheduling route if its
verified correct solution uses the calendar. The feature changes. It is the parent's mean
final-layer state over the user message and the reply header, after a prefix every turn
shares, the instruction, tools and user header, which is computed once and reused. The
router file names this feature, so development and confirmation compute it the same way.
"""

import importlib.metadata
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from neuroshard.evolution import assistant_growth_round4 as round4
from neuroshard.evolution import assistant_growth_router as router
from neuroshard.evolution import assistant_growth_run as growth
from neuroshard.evolution import assistant_routing as routing
from neuroshard.evolution import assistant_workflow_canonical as canonical
from neuroshard.evolution import granite_reference as reference
from neuroshard.evolution import granite_tokenizer
from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, identity, launch, read, save, sha256, verify_artifacts,
)

DECLARATION = 'config/experiments/assistant-growth-router3.json'
EXECUTION = 'config/experiments/assistant-growth-router3-execution.json'
SCRIPT = 'scripts/run_assistant_growth_router3.py'
PROFILE = 'assistant-growth-router3'
FEATURE = 'message-mean'


def message_features(parent, tokenizer, policy, texts):
    """Each turn's message feature, the shared prefix run once."""
    prefix = routing.message_prefix(parent, tokenizer, policy, 'cpu')
    return {key: routing.message_feature(parent, tokenizer, policy, texts[key], 'cpu', prefix) for key in sorted(texts)}


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
        features = message_features(parent, tokenizer, policies['drafting'], texts)
        check = message_features(parent, tokenizer, policies['drafting'], check_texts)
        reply['feature_seconds'] = time.monotonic() - begun
        gate, reply['router'] = router.fit(features, rows, check, check_rows,
                                           stage['integration']['recipe']['epsilon'], FEATURE)
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
