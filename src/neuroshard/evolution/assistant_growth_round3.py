"""A3 stage 1, round 3, on one GPU host: demonstrations as scheduling experience, then stage 1's training and integration.

Each training case is solved once by its demonstration, executed in the calendar workspace
under the calendar policy and verified by the scorer; nothing is sampled. U2 and L2 then
train from the accepted version exactly as declared for stage 1, on the declared mixture
with the demonstrations as scheduling experience, and every route runs the integration
cases to fit the turn selectors. No development or confirmation goal is read.
"""

import collections
import importlib.metadata
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from neuroshard.evolution import assistant_experience as experience
from neuroshard.evolution import assistant_growth_run as growth
from neuroshard.evolution import assistant_schedule_demonstration as demonstrations
from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution import granite_tokenizer
from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, identity, launch, read, save, sha256, verify_artifacts,
)

DECLARATION = 'config/experiments/assistant-growth-round3.json'
EXECUTION = 'config/experiments/assistant-growth-round3-execution.json'
SCRIPT = 'scripts/run_assistant_growth_round3.py'
PROFILE = 'assistant-growth-round3-gpu'


def plan():
    """The stage-1 plan with round 3's experience; training and integration are stage 1's."""
    return {**read(ROOT / growth.PLAN), 'round3': read(ROOT / DECLARATION)}


def demonstration(case, policy):
    """One training case solved by its demonstration in the calendar workspace, verified by the scorer."""
    replies = iter(demonstrations.texts(case))

    def respond(messages, tools):
        return {'text': next(replies), 'terminated': True, 'executed': True, 'token_ids': [], 'input_token_ids': [],
                'prompt_sha256': identity({'messages': messages, 'tools': tools}), 'seconds': 0.0,
                'model': 'demonstration'}

    result = workflow.execute(case, respond, policy)
    if not result['score']['passed'] or next(replies, None) is not None:
        raise ValueError(f"the demonstration does not solve {case['id']}")
    return {**experience.trajectory(case, result, policy, policy, sample=0), 'demonstration': True}, result


def collect(tokenizer, stage, policies, home):
    """Every training case's verified demonstration, as encoded scheduling experience."""
    from neuroshard.evolution import assistant_experience_train as trainer

    policy = policies['scheduling']
    cases = [case for split in stage['collection']['splits'] for case in growth.scheduling_cases(stage, split)]
    items, rows = [], []
    for case in cases:
        item, result = demonstration(case, policy)
        items.append(item)
        rows.append({'case_id': case['id'], 'sample': 0, 'policy_sha256': identity(policy), 'result': result})
    report = {'cases': len(cases), 'complete': sum(t['complete'] for t in items),
              'by_family': dict(collections.Counter(t['family'] for t in items)),
              'replies': sum(t['model_calls'] for t in items),
              'demonstrations_sha256': growth.write_rows(home / 'demonstrations.jsonl.gz', rows),
              'trajectories_sha256': growth.write_rows(home / 'trajectories.jsonl.gz', items)}
    save(home / 'experience.json', report, exclusive=True)
    tools = workflow.interface(policy).TOOLS
    return [trainer.encode(tokenizer, t, tools) for t in items], report


def committed_sources(root=ROOT):
    execution = read(root / EXECUTION)
    for name, digest in execution['contracts'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'changed round-3 contract: {name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in execution['sources']:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root) != (root / name).read_bytes():
            raise ValueError(f'uncommitted round-3 source: {name}')
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
    keys = ('packages', 'python', 'gpus', 'environment', 'worker_environment', 'tokenizer_pipeline_sha256',
            'execution', 'drafting_collection')
    if any(execution[key] != pinned[key] for key in keys):
        raise ValueError('round-3 runtime differs from the stage-1 runtime')
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
        raise ValueError('round-3 worker differs from freeze')
    import torch
    from transformers import AutoModelForCausalLM

    execution = read(ROOT / EXECUTION)
    stage = plan()
    growth_plan, learning, policies = growth.contracts(stage)
    gpu = request['freeze']['gpus'][0]
    parameters = {**execution['execution'], **execution['gpus'][gpu]}
    pinned = growth_plan['cohort1']['trainable_sha256']
    home = request_path.parent
    reply = {'binding': request['binding'], 'execution_completed': False, 'phases': {}, 'gpu': gpu,
             'max_batch': parameters['max_batch']}
    started = time.monotonic()
    tokenizer = None
    try:
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

        spec = growth.stage_spec(stage, learning)
        begun = time.monotonic()
        scheduling_rows, collected = collect(tokenizer, stage, policies, home)
        reply['collection'] = collected
        reply['phases']['demonstrate'] = time.monotonic() - begun
        begun = time.monotonic()
        drafting_rows, replay_rows = growth.drafting_experience(tokenizer, learning, policies, execution, home)
        reply['phases']['verify_drafting'] = time.monotonic() - begun
        begun = time.monotonic()
        rows = {'scheduling': scheduling_rows, 'drafting': drafting_rows, 'replay': replay_rows}
        reply['training'] = growth.train(load_parent, stage, learning, rows, parameters, home, pinned)
        reply['phases']['train'] = time.monotonic() - begun
        begun = time.monotonic()
        gates, _ = growth.integrate(load_parent, growth.unit_loaders(load_parent, spec, home, pinned), tokenizer,
                                    stage, learning, policies, parameters, home)
        reply['gates'] = {version: {k: v for k, v in gate.items() if k != 'weight'} for version, gate in gates.items()}
        reply['phases']['integrate'] = time.monotonic() - begun
        if file_state(directory, inventory) != state:
            raise ValueError('parent checkpoint changed during round 3')
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
        reply = launch(home, models, binding, 'baseline', 'round3', execution['worker_seconds'],
                       execution['memory_bytes'], worker_script=SCRIPT, environment=execution['worker_environment'])
        result['reply'] = reply
        if not reply['execution_completed']:
            raise ValueError(reply.get('error', 'incomplete round-3 execution'))
        result['execution_completed'] = True
    except Exception as error:
        result['error'] = str(error)
    save(home / 'result.json', result, exclusive=True)
    return result
