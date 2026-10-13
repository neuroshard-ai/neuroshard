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
# Later runs of the same worker: each declares the sets and policies it serves.
PROFILES = {PROFILE: (DECLARATION, EXECUTION),
            'assistant-general-documents2': ('config/experiments/assistant-general-documents2.json',
                                             'config/experiments/assistant-general-documents2-execution.json'),
            'assistant-general-confirmation': ('config/experiments/assistant-general-confirmation.json',
                                               'config/experiments/assistant-general-confirmation-execution.json'),
            **{f'assistant-general-{name}': (f'config/experiments/assistant-general-{name}.json',
                                             f'config/experiments/assistant-general-{name}-execution.json')
               for name in ('v3-development', 'v3-confirmation', 'l4-confirmation', 'l7-confirmation', 'l8-confirmation')}}
UPLOADED = '.units'


def runs(declaration):
    """``[(set name, split, policy path)]``; the first baseline declared none and served both development sets."""
    if 'runs' in declaration:
        return [(row['set'], row['split'], row['policy'], row.get('module')) for row in declaration['runs']]
    return [('chat', 'development', POLICIES['chat'], None), ('documents', 'development', POLICIES['documents'], None)]


def module_model(load_parent, declaration, name):
    """The parent with a declared low-rank module attached, checked against its pinned digest."""
    from neuroshard.evolution import assistant_experience_train as trainer
    from neuroshard.evolution import assistant_growth_run as growth

    pin = declaration['modules'][name]
    plan = read(ROOT / growth.PLAN)
    settings = growth.stage_spec(plan, read(ROOT / read(ROOT / plan['plan'])['cohort1']['learning']))
    model = load_parent()
    manifest = trainer.load_trainable(model, 'addition', settings, ROOT / UPLOADED / name)
    if manifest['trainable_sha256'] != pin['trainable_sha256']:
        raise ValueError(f'{name} differs from its pinned digest')
    return model.eval()


def feature_texts():
    """Every train and development user turn of both general sets, sorted."""
    from neuroshard.evolution import assistant_chat_data as chat
    from neuroshard.evolution import assistant_documents_data as docs

    return sorted({turn['user'] for module in (docs, chat) for split in ('train', 'development')
                   for case in module.cases(split) for turn in case['turns']})


def committed_sources(root=ROOT, profile=PROFILE):
    execution = read(root / PROFILES[profile][1])
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


def configure(profile=PROFILE):
    if 'torch' in sys.modules:
        raise ValueError('configure the general baseline runtime before importing torch')
    for key, value in read(ROOT / PROFILES[profile][1])['environment'].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def freeze(profile=PROFILE):
    """The canonical parent's CPU runtime, byte for byte in packages and environment."""
    from neuroshard.evolution import assistant_workflow_canonical as canonical

    source = committed_sources(profile=profile)
    execution = read(ROOT / PROFILES[profile][1])
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
    request_path = Path(request_path)
    request = read(request_path)
    profile = request['profile']
    configure(profile)
    if freeze(profile) != request['freeze']:
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

    declaration_path, execution_path = PROFILES[profile]
    execution, declaration = read(ROOT / execution_path), read(ROOT / declaration_path)
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
        if declaration.get('features', True):
            _, _, routes = growth.contracts(round4.plan())
            texts = feature_texts()
            begun = time.monotonic()
            computed, _ = features.layer_features(parent, tokenizer, routes['drafting'], texts, layers=())
            np.savez(home / 'features.npz', **{'layer-1': np.asarray(computed['layer-1'], dtype=np.float32)})
            (home / 'texts.json').write_text(json.dumps(texts))
            reply['feature_seconds'], reply['feature_texts'] = time.monotonic() - begun, len(texts)
        episodes = {}
        modules = {'chat': (chat, general.chat_episode), 'documents': (docs, general.documents_episode)}
        attached = {}
        for name, split, path, unit in runs(declaration):
            module, run = modules[name]
            policy, cases = read(ROOT / path), module.cases(split)
            key = f'{name}-{split}' + (f'-{unit}' if unit else '')
            if unit and unit not in attached:
                attached[unit] = module_model(lambda: reference.load_model(directory, 'baseline')[0], declaration, unit)
            model = attached[unit] if unit else parent
            episodes[key] = []
            for case in cases:
                episodes[key].append(run(case, cached_responder(model, tokenizer, policy), policy))
                save(home / 'progress.json', {'set': key, 'done': len(episodes[key]), 'cases': len(cases)})
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


def run(home, models, profile=PROFILE):
    configure(profile)
    source = freeze(profile)
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    declaration_path, execution_path = PROFILES[profile]
    execution = read(ROOT / execution_path)
    binding = {'freeze': source, 'profile': profile, 'declaration_sha256': sha256(ROOT / declaration_path)}
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
