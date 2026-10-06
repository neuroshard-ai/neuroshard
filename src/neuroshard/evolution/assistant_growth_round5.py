"""A3 stage 1, round 5, on one GPU host: round 4's training under the free-slot calendar interface.

The 1,152 training cases of round 4 are each solved once under the free-slot interface,
executed and verified by the scorer, and every saved meeting must start at the first
window the tool returned for its date. U2 and L2 train from the accepted version on
round 4's mixture for round 4's steps. Every route runs the integration cases, and so does
the accepted version on the scheduling route: the control, given the same tool. No
development or confirmation goal is read.
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

from neuroshard.evolution import assistant_growth_run as growth
from neuroshard.evolution import assistant_schedule_data as schedule
from neuroshard.evolution import assistant_schedule_demonstration as demonstrations
from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution import granite_tokenizer
from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, identity, launch, read, save, sha256, verify_artifacts,
)

DECLARATION = 'config/experiments/assistant-growth-round5.json'
EXECUTION = 'config/experiments/assistant-growth-round5-execution.json'
SCRIPT = 'scripts/run_assistant_growth_round5.py'
PROFILE = 'assistant-growth-round5-gpu'
ROUTE_RUNS = growth.ROUTE_RUNS + (('U1', 'scheduling'),)


def plan():
    """The stage-1 plan with round 5's experience and steps; everything else is stage 1's."""
    stage = read(ROOT / growth.PLAN)
    declared = read(ROOT / DECLARATION)
    return {**stage, 'round5': declared, 'training': {**stage['training'], 'steps': declared['training']['steps']}}


def contracts(stage):
    """Stage 1's contracts with the free-slot policy on the scheduling route."""
    growth_plan, learning, policies = growth.contracts(stage)
    return growth_plan, learning, {**policies, 'scheduling': read(ROOT / stage['round5']['interface']['policy'])}


def training_cases(stage):
    """Every declared training case, each split checked against its frozen manifest."""
    cases = [case for split in stage['collection']['splits'] for case in growth.scheduling_cases(stage, split)]
    frozen = read(ROOT / stage['round5']['experience']['data'])['splits']
    for split in stage['round5']['experience']['splits']:
        if split not in schedule.GROWTH:
            raise ValueError('round 5 adds only the further training splits')
        added = schedule.cases(split)
        if identity(added) != frozen[split]['sha256'] or [c['id'] for c in added] != frozen[split]['case_ids']:
            raise ValueError('scheduling data differs from its frozen split')
        cases += added
    return cases


def first_windows(result):
    """True when every saved meeting starts at the first window the tool last returned for its date."""
    windows = {}
    for row in result['calls']:
        name, args = row['call']['name'], row['call']['arguments']
        if name == 'free_slots' and 'free' in row['result']:
            windows[args['date']] = row['result']['free']
        elif name == 'save_meeting' and not (windows.get(args['date']) and
                                             windows[args['date']][0][0] == args['start_time']):
            return False
    return True


def demonstration(case, policy):
    """One training case solved under the free-slot interface, verified by the scorer and by the tool's windows."""
    from neuroshard.evolution import assistant_experience as experience

    replies = iter(demonstrations.slot_texts(case))

    def respond(messages, tools):
        return {'text': next(replies), 'terminated': True, 'executed': True, 'token_ids': [], 'input_token_ids': [],
                'prompt_sha256': identity({'messages': messages, 'tools': tools}), 'seconds': 0.0,
                'model': 'demonstration'}

    result = workflow.execute(case, respond, policy)
    if not result['score']['passed'] or next(replies, None) is not None or not first_windows(result):
        raise ValueError(f"the demonstration does not solve {case['id']}")
    return {**experience.trajectory(case, result, policy, policy, sample=0), 'demonstration': True}, result


def collect(tokenizer, stage, policies, home):
    """Every training case's verified demonstration, as encoded scheduling experience."""
    from neuroshard.evolution import assistant_experience_train as trainer

    policy = policies['scheduling']
    cases = training_cases(stage)
    items, rows = [], []
    for case in cases:
        item, result = demonstration(case, policy)
        items.append(item)
        rows.append({'case_id': case['id'], 'sample': 0, 'policy_sha256': identity(policy), 'result': result})
    report = {'cases': len(items), 'complete': sum(t['complete'] for t in items),
              'by_family': dict(collections.Counter(t['family'] for t in items)),
              'by_split': dict(collections.Counter(case['split'] for case in cases)),
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
            raise ValueError(f'changed round-5 contract: {name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in execution['sources']:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root) != (root / name).read_bytes():
            raise ValueError(f'uncommitted round-5 source: {name}')
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
        raise ValueError('round-5 runtime differs from the stage-1 runtime')
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
        raise ValueError('round-5 worker differs from freeze')
    import torch
    from transformers import AutoModelForCausalLM

    execution = read(ROOT / EXECUTION)
    stage = plan()
    growth_plan, learning, policies = contracts(stage)
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
                                    stage, learning, policies, parameters, home, route_runs=ROUTE_RUNS)
        reply['gates'] = {version: {k: v for k, v in gate.items() if k != 'weight'} for version, gate in gates.items()}
        reply['phases']['integrate'] = time.monotonic() - begun
        if file_state(directory, inventory) != state:
            raise ValueError('parent checkpoint changed during round 5')
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
        reply = launch(home, models, binding, 'baseline', 'round5', execution['worker_seconds'],
                       execution['memory_bytes'], worker_script=SCRIPT, environment=execution['worker_environment'])
        result['reply'] = reply
        if not reply['execution_completed']:
            raise ValueError(reply.get('error', 'incomplete round-5 execution'))
        result['execution_completed'] = True
    except Exception as error:
        result['error'] = str(error)
    save(home / 'result.json', result, exclusive=True)
    return result
