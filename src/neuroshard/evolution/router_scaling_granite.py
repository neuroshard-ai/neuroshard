"""The router scaling study's features from the pinned Granite 4.1 3B parent, on one canonical CPU host.

The local study used SmolLM2-135M as a stand-in. This worker computes the accepted router
feature, the parent's mean final-layer state over each user message after the drafting
policy's shared prefix (`assistant_routing.message_feature`), for every study text, and the
same mean at hidden layers 20 and 30 from the same forward pass. The first texts are also
computed with `assistant_routing.message_feature` itself and must match bit for bit, so the
final layer is the served feature, not an approximation of it.

Nothing is trained, routed or served; no development or confirmation data is read. The
analysis of the features runs locally (scripts/run_router_scaling.py --features).
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

DECLARATION = 'config/experiments/router-scaling-granite.json'
EXECUTION = 'config/experiments/router-scaling-granite-execution.json'
SCRIPT = 'scripts/run_router_scaling_granite.py'
PROFILE = 'router-scaling-granite'
LAYERS = (20, 30)
CHECKED = 8


def committed_sources(root=ROOT):
    execution = read(root / EXECUTION)
    for name, digest in execution['contracts'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'changed router scaling contract: {name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in execution['sources']:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root) != (root / name).read_bytes():
            raise ValueError(f'uncommitted router scaling source: {name}')
        sources[name] = sha256(root / name)
    return {'commit': commit, 'sources': sources}


def configure():
    if 'torch' in sys.modules:
        raise ValueError('configure the router scaling runtime before importing torch')
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
        raise ValueError('router scaling runtime differs from the canonical parent runtime')
    packages = {key: importlib.metadata.version(key) for key in execution['packages']}
    if packages != execution['packages'] or platform.python_version() != execution['python']:
        raise ValueError('router scaling runtime differs')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('router scaling requires Linux x86_64')
    cpu = Path('/proc/cpuinfo').read_text().split()
    if any(flag not in cpu for flag in execution['required_cpu_flags']):
        raise ValueError('router scaling CPU lacks required instructions')
    if any(os.environ.get(key) != value for key, value in execution['environment'].items()):
        raise ValueError('router scaling numerical environment differs')
    return {**source, 'packages': packages, 'python': platform.python_version()}


def layer_features(parent, tokenizer, policy, texts, layers=LAYERS):
    """``{'layer-1': [...], 'layer20': [...], ...}``: the accepted final feature and middle-layer means, one pass per text."""
    import torch

    from neuroshard.evolution import assistant_routing as routing

    prefix = routing.message_prefix(parent, tokenizer, policy, 'cpu')
    names = ['layer-1', *(f'layer{layer}' for layer in layers)]
    out = {name: [] for name in names}
    for text in texts:
        ids = routing.turn_ids(tokenizer, policy, text)
        if ids[:len(prefix['ids'])] != prefix['ids'] or len(ids) == len(prefix['ids']):
            raise ValueError('a turn does not extend the shared prefix')
        try:
            with torch.no_grad():
                result = parent.model(input_ids=torch.tensor([ids[len(prefix['ids']):]]),
                                      past_key_values=prefix['cache'], use_cache=True, output_hidden_states=True)
            out['layer-1'].append(result.last_hidden_state[0].float().mean(dim=0).tolist())
            for layer in layers:
                out[f'layer{layer}'].append(result.hidden_states[layer][0].float().mean(dim=0).tolist())
        finally:
            prefix['cache'].crop(len(prefix['ids']))
    return out, prefix


def worker(request_path):
    configure()
    request_path = Path(request_path)
    request = read(request_path)
    if freeze() != request['freeze']:
        raise ValueError('router scaling worker differs from freeze')
    import numpy as np
    import torch

    from neuroshard.evolution import assistant_growth_round4 as round4
    from neuroshard.evolution import assistant_growth_run as growth
    from neuroshard.evolution import assistant_routing as routing
    from neuroshard.evolution import granite_reference as reference
    from neuroshard.evolution import granite_tokenizer
    from neuroshard.evolution import router_scaling_study as study

    execution = read(ROOT / EXECUTION)
    phase = request['phase']
    if (request['model'], phase) not in (('baseline', 'prepare'), ('baseline', 'features')):
        raise ValueError('unsupported router scaling worker role')
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
        _, _, policies = growth.contracts(round4.plan())
        state = verify_artifacts(directory, inventory, download=False)
        tokenizer, report = granite_tokenizer.load(directory)
        reply['tokenizer'] = report
        parent, _ = reference.load_model(directory, 'baseline')
        texts = study.texts()
        declared = read(ROOT / DECLARATION)
        if len(texts) != declared['texts'] or sha256_texts(texts) != declared['texts_sha256']:
            raise ValueError('study texts differ from the declaration')
        begun = time.monotonic()
        features, prefix = layer_features(parent, tokenizer, policies['drafting'], texts)
        reply['feature_seconds'] = time.monotonic() - begun
        accepted = [routing.message_feature(parent, tokenizer, policies['drafting'], text, 'cpu', prefix)
                    for text in texts[:CHECKED]]
        reply['accepted_feature_checked'] = CHECKED
        reply['accepted_feature_identical'] = accepted == features['layer-1'][:CHECKED]
        if not reply['accepted_feature_identical']:
            raise ValueError('final-layer feature differs from the accepted router feature')
        home = request_path.parent
        np.savez(home / 'features.npz', **{name: np.asarray(values, dtype=np.float32)
                                           for name, values in features.items()})
        (home / 'texts.json').write_text(json.dumps(texts))
        reply['features_sha256'] = sha256(home / 'features.npz')
        reply['layers'] = sorted(features)
        reply['texts'] = len(texts)
        if file_state(directory, inventory) != state:
            raise ValueError('parent checkpoint changed while computing features')
        reply['execution_completed'] = True
    except Exception as error:
        reply['error'] = f'{type(error).__name__}: {error}'
    finally:
        reply.update(checked_encodes=tokenizer.checked_encodes if tokenizer else 0,
                     wall_seconds=time.monotonic() - started, process_cpu_seconds=time.process_time(),
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        save(request_path.parent / 'reply.json', reply, exclusive=True)


def sha256_texts(texts):
    import hashlib

    return hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()


def run(home, models):
    configure()
    source = freeze()
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    execution = read(ROOT / EXECUTION)
    binding = {'freeze': source, 'profile': PROFILE, 'declaration_sha256': sha256(ROOT / DECLARATION)}
    save(home / 'binding.json', binding, exclusive=True)
    result = {'binding': binding, 'execution_completed': False, 'checklist_credit': False,
              'admission_evidence': False, 'development_opened': False, 'confirmation_opened': False}
    try:
        prepared = launch(home, models, binding, 'baseline', 'prepare', execution['prepare_seconds'],
                          execution['memory_bytes'], worker_script=SCRIPT)
        if not prepared['execution_completed']:
            raise ValueError(prepared.get('error', 'parent artifacts were not prepared'))
        reply = launch(home, models, binding, 'baseline', 'features', execution['worker_seconds'],
                       execution['memory_bytes'], worker_script=SCRIPT)
        result['reply'] = reply
        if not reply['execution_completed']:
            raise ValueError(reply.get('error', 'incomplete feature computation'))
        result['execution_completed'] = True
    except Exception as error:
        result['error'] = str(error)
    save(home / 'result.json', result, exclusive=True)
    return result
