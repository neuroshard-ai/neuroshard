"""The general route's baseline: the unchanged Granite parent on the user's documents and on general chat.

One canonical CPU host. The parent (no update, no module) serves the opened development splits of
the documents and chat tasks under their declared policies with prefix-cache serving, and its
accepted router feature is computed for every train and development turn of both, so a three-way
router (general, drafting, scheduling) can be fitted and judged locally. Nothing is trained; no
confirmation split is opened.
"""

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, launch, read, save, sha256, verify_artifacts,
)

DECLARATION = 'config/experiments/assistant-general-baseline.json'
EXECUTION = 'config/experiments/assistant-general-baseline-execution.json'
SCRIPT = 'scripts/run_assistant_general_baseline.py'
PROFILE = 'assistant-general-baseline'
POLICIES = {'documents': 'config/experiments/assistant-general-policy-documents.json',
            'chat': 'config/experiments/assistant-general-policy-chat.json'}


def feature_texts():
    """Every train and development user turn of both general sets, sorted."""
    from neuroshard.evolution import assistant_chat_data as chat
    from neuroshard.evolution import assistant_documents_data as docs

    return sorted({turn['user'] for module in (docs, chat) for split in ('train', 'development')
                   for case in module.cases(split) for turn in case['turns']})


def committed_sources(root=ROOT):
    execution = read(root / EXECUTION)
    for name, digest in execution['contracts'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'changed general baseline contract: {name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in execution['sources']:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root) != (root / name).read_bytes():
            raise ValueError(f'uncommitted general baseline source: {name}')
        sources[name] = sha256(root / name)
    return {'commit': commit, 'sources': sources}


def configure():
    if 'torch' in sys.modules:
        raise ValueError('configure the general baseline runtime before importing torch')
    for key, value in read(ROOT / EXECUTION)['environment'].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def freeze():
    """The canonical parent's CPU runtime, byte for byte in packages and environment."""
    from neuroshard.evolution import assistant_workflow_canonical as canonical

    source = committed_sources()
    execution = read(ROOT / EXECUTION)
    pinned = read(ROOT / canonical.EXECUTION)
    if any(execution[key] != pinned[key] for key in ('packages', 'python', 'required_cpu_flags', 'environment')):
        raise ValueError('general baseline runtime differs from the canonical parent runtime')
    packages = {key: importlib.metadata.version(key) for key in execution['packages']}
    if packages != execution['packages'] or platform.python_version() != execution['python']:
        raise ValueError('general baseline runtime differs')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('general baseline requires Linux x86_64')
    cpu = Path('/proc/cpuinfo').read_text().split()
    if any(flag not in cpu for flag in execution['required_cpu_flags']):
        raise ValueError('general baseline CPU lacks required instructions')
    if any(os.environ.get(key) != value for key, value in execution['environment'].items()):
        raise ValueError('general baseline numerical environment differs')
    return {**source, 'packages': packages, 'python': platform.python_version()}


def worker(request_path):
    configure()
    request_path = Path(request_path)
    request = read(request_path)
    if freeze() != request['freeze']:
        raise ValueError('general baseline worker differs from freeze')
    import numpy as np
    import torch

    from neuroshard.evolution import assistant_chat_data as chat
    from neuroshard.evolution import assistant_documents_data as docs
    from neuroshard.evolution import assistant_general as general
    from neuroshard.evolution import assistant_growth_round4 as round4
    from neuroshard.evolution import assistant_growth_run as growth
    from neuroshard.evolution import granite_reference as reference
    from neuroshard.evolution import granite_tokenizer
    from neuroshard.evolution import router_scaling_granite as features
    from neuroshard.evolution.assistant_serving import cached_responder

    execution = read(ROOT / EXECUTION)
    phase = request['phase']
    if (request['model'], phase) not in (('baseline', 'prepare'), ('baseline', 'general')):
        raise ValueError('unsupported general baseline worker role')
    torch.set_num_threads(execution['threads'])
    torch.set_num_interop_threads(1)
    reply = {'binding': request['binding'], 'execution_completed': False}
    started = time.monotonic()
    tokenizer = None
    home = request_path.parent
    try:
        inventory = read(ROOT / reference.ARTIFACTS)['models']['baseline']
        directory = Path(request['models']) / 'baseline'
        if phase == 'prepare':
            reply['file_state'] = verify_artifacts(directory, inventory, download=True)
            reply['execution_completed'] = True
            return
        state = verify_artifacts(directory, inventory, download=False)
        tokenizer, report = granite_tokenizer.load(directory)
        reply['tokenizer'] = report
        parent, _ = reference.load_model(directory, 'baseline')
        policies = {name: read(ROOT / path) for name, path in POLICIES.items()}
        _, _, routes = growth.contracts(round4.plan())
        texts = feature_texts()
        begun = time.monotonic()
        computed, _ = features.layer_features(parent, tokenizer, routes['drafting'], texts, layers=())
        np.savez(home / 'features.npz', **{'layer-1': np.asarray(computed['layer-1'], dtype=np.float32)})
        (home / 'texts.json').write_text(json.dumps(texts))
        reply['feature_seconds'], reply['feature_texts'] = time.monotonic() - begun, len(texts)
        episodes = {'documents': [], 'chat': []}
        for name, cases, run in (('chat', chat.cases('development'), general.chat_episode),
                                 ('documents', docs.cases('development'), general.documents_episode)):
            for case in cases:
                respond = cached_responder(parent, tokenizer, policies[name])
                episodes[name].append(run(case, respond, policies[name]))
                save(home / 'progress.json', {'set': name, 'done': len(episodes[name]), 'cases': len(cases)})
        reply['episodes'] = episodes
        reply['summary'] = {name: general.summary(rows) for name, rows in episodes.items()}
        if file_state(directory, inventory) != state:
            raise ValueError('parent checkpoint changed during the baseline')
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
    result = {'binding': binding, 'execution_completed': False, 'checklist_credit': False,
              'admission_evidence': False, 'confirmation_opened': False}
    try:
        prepared = launch(home, models, binding, 'baseline', 'prepare', execution['prepare_seconds'],
                          execution['memory_bytes'], worker_script=SCRIPT)
        if not prepared['execution_completed']:
            raise ValueError(prepared.get('error', 'parent artifacts were not prepared'))
        reply = launch(home, models, binding, 'baseline', 'general', execution['worker_seconds'],
                       execution['memory_bytes'], worker_script=SCRIPT)
        result['reply'] = {k: v for k, v in reply.items() if k != 'episodes'}
        if not reply['execution_completed']:
            raise ValueError(reply.get('error', 'incomplete general baseline'))
        result['execution_completed'] = True
    except Exception as error:
        result['error'] = str(error)
    save(home / 'result.json', result, exclusive=True)
    return result
