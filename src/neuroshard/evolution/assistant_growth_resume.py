"""A3 stage 1, resumed: the declared integration alone, on a second GPU host, from the first host's saved units.

Used only if the stage-1 worker stopped after saving U2 and L2 and before saving its
integration. U1 is the accepted update as pinned by A2; U2 and L2 are re-verified against
the first host's training manifests. Collection and training do not run again, and no
development or confirmation goal is read.
"""

import importlib.metadata
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from neuroshard.evolution import assistant_growth_run as growth
from neuroshard.evolution import granite_tokenizer
from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, launch, read, save, sha256, verify_artifacts,
)

DECLARATION = 'config/experiments/assistant-growth-stage1-resume.json'
EXECUTION = 'config/experiments/assistant-growth-resume-execution.json'
SCRIPT = 'scripts/run_assistant_growth_resume.py'
PROFILE = 'assistant-growth-resume-gpu'
TRAINED = 'trained'


def verify_trained(directory, pinned):
    """U2 and L2 as the first host saved them, each matching its training manifest digest."""
    for unit, digest in pinned.items():
        if read(Path(directory) / growth.CHECKPOINTS[unit] / 'manifest.json')['trainable_sha256'] != digest:
            raise ValueError(f'{unit} differs from the first host\'s training manifest')


def committed_sources(root=ROOT):
    execution = read(root / EXECUTION)
    for name, digest in execution['contracts'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'changed resumption contract: {name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in execution['sources']:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root) != (root / name).read_bytes():
            raise ValueError(f'uncommitted resumption source: {name}')
        sources[name] = sha256(root / name)
    return {'commit': commit, 'sources': sources}


def configure():
    if 'torch' in sys.modules:
        raise ValueError('configure the accelerator runtime before importing torch')
    for key, value in read(ROOT / EXECUTION)['environment'].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def freeze():
    """The stage-1 GPU runtime, byte for byte in packages, GPUs and environment."""
    from neuroshard.evolution import assistant_experience_run as run

    source = committed_sources()
    execution = read(ROOT / EXECUTION)
    pinned = read(ROOT / growth.EXECUTION)
    keys = ('packages', 'python', 'gpus', 'environment', 'worker_environment', 'tokenizer_pipeline_sha256', 'execution')
    if any(execution[key] != pinned[key] for key in keys):
        raise ValueError('resumption runtime differs from the stage-1 runtime')
    packages = {key: importlib.metadata.version(key) for key in execution['packages']}
    if packages != execution['packages'] or platform.python_version() != execution['python']:
        raise ValueError('accelerator runtime differs')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('accelerator runtime requires Linux x86_64')
    gpus = run.gpu_names()
    if len(gpus) != 1 or gpus[0] not in execution['gpus']:
        raise ValueError('accelerator is not one declared GPU')
    if any(os.environ.get(key) != value for key, value in execution['environment'].items()):
        raise ValueError('accelerator environment differs')
    return {**source, 'packages': packages, 'python': platform.python_version(), 'gpus': gpus}


def worker(request_path):
    configure()
    request_path = Path(request_path)
    request = read(request_path)
    if freeze() != request['freeze']:
        raise ValueError('resumption worker differs from freeze')
    import torch
    from transformers import AutoModelForCausalLM

    execution = read(ROOT / EXECUTION)
    plan = read(ROOT / growth.PLAN)
    growth_plan, learning, policies = growth.contracts(plan)
    gpu = request['freeze']['gpus'][0]
    parameters = {**execution['execution'], **execution['gpus'][gpu]}
    pinned = growth_plan['cohort1']['trainable_sha256']
    home = request_path.parent
    reply = {'binding': request['binding'], 'execution_completed': False, 'phases': {}, 'gpu': gpu,
             'max_batch': parameters['max_batch']}
    started = time.monotonic()
    tokenizer = None
    try:
        trained = ROOT / growth.UPLOADED / TRAINED
        verify_trained(trained, execution['units'])
        inventory = read(ROOT / growth.ARTIFACTS)['models']['baseline']
        directory = Path(request['models']) / 'baseline'
        state = verify_artifacts(directory, inventory, download=True)
        tokenizer, report = granite_tokenizer.load(directory)
        if report['pipeline_sha256'] != execution['tokenizer_pipeline_sha256']:
            raise ValueError('tokenizer pipeline differs from the canonical baseline')
        reply['tokenizer'] = report

        def load_parent():
            model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.bfloat16, local_files_only=True,
                                                         attn_implementation=parameters['attention'])
            if sum(p.numel() for p in model.parameters()) != inventory['parameters']:
                raise ValueError('parent parameter inventory differs')
            return model.to(parameters['device']).eval()

        spec = growth.stage_spec(plan, learning)
        begun = time.monotonic()
        gates, _ = growth.integrate(load_parent, growth.unit_loaders(load_parent, spec, trained, pinned), tokenizer,
                                    plan, learning, policies, parameters, home)
        reply['gates'] = {version: {k: v for k, v in gate.items() if k != 'weight'} for version, gate in gates.items()}
        reply['phases']['integrate'] = time.monotonic() - begun
        if file_state(directory, inventory) != state:
            raise ValueError('parent checkpoint changed during the resumed integration')
        reply['execution_completed'] = True
    except Exception as error:
        reply['error'] = f'{type(error).__name__}: {error}'
    finally:
        reply.update(checked_encodes=tokenizer.checked_encodes if tokenizer else 0,
                     wall_seconds=time.monotonic() - started, process_cpu_seconds=time.process_time(),
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                     peak_gpu_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
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
        verify_trained(ROOT / growth.UPLOADED / TRAINED, execution['units'])
        reply = launch(home, models, binding, 'baseline', 'integration', execution['worker_seconds'],
                       execution['memory_bytes'], worker_script=SCRIPT, environment=execution['worker_environment'])
        result['reply'] = reply
        if not reply['execution_completed']:
            raise ValueError(reply.get('error', 'incomplete resumed integration'))
        result['execution_completed'] = True
    except Exception as error:
        result['error'] = str(error)
    save(home / 'result.json', result, exclusive=True)
    return result
