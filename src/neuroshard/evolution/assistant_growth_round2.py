"""A3 stage 1, round 2, on one GPU host: budget-aware coaching, then the stage-1 training and integration unchanged.

Stage 1's natural rollouts came from the same sampler under the same policy; they are
re-verified here instead of sampled again, and decide which training cases are coached.
Each such case gets coached rollouts under the round-2 card, which teaches two
independent calls per reply and the conflict rule. Verified successes are kept without
the near-policy filter, up to the declared number per case. If too few training cases
have a complete verified trajectory, the round stops before training. Otherwise U2 and
L2 train from the accepted version exactly as declared for stage 1, and every route runs
the integration cases to fit the turn selectors. No development or confirmation goal is read.
"""

import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from neuroshard.evolution import assistant_experience as experience
from neuroshard.evolution import assistant_growth_run as growth
from neuroshard.evolution import assistant_rollout as rollout
from neuroshard.evolution import granite_tokenizer
from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, identity, launch, read, save, sha256, verify_artifacts,
)

DECLARATION = 'config/experiments/assistant-growth-round2.json'
EXECUTION = 'config/experiments/assistant-growth-round2-execution.json'
SCRIPT = 'scripts/run_assistant_growth_round2.py'
PROFILE = 'assistant-growth-round2-gpu'


def plan():
    """The stage-1 plan with round 2's collection; training and integration are stage 1's."""
    return {**read(ROOT / growth.PLAN), 'round2': read(ROOT / DECLARATION)}


def natural(stage, policy, path):
    """Stage 1's natural scheduling rollouts, each re-verified, as experience and as the coaching trigger."""
    pinned = stage['round2']['collection']['natural']
    if sha256(path) != pinned['sha256']:
        raise ValueError('stage-1 rollouts differ from their pinned digest')
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        recorded = [json.loads(line) for line in handle]
    if identity(recorded) != pinned['rows_sha256']:
        raise ValueError('stage-1 rollouts differ from the rows the stage-1 report records')
    cases = {case['id']: case for split in stage['collection']['splits'] for case in growth.scheduling_cases(stage, split)}
    expected = {(case_id, sample) for case_id in cases for sample in range(stage['collection']['samples_per_case'])}
    rows, seen = [], set()
    for row in recorded:
        if row['policy_sha256'] != identity(policy):
            continue
        key = (row['case_id'], row['sample'])
        if key not in expected or key in seen:
            raise ValueError('stage-1 natural rollouts differ from the declared collection')
        seen.add(key)
        rows.append(row)
    if seen != expected:
        raise ValueError('stage-1 natural rollouts are incomplete')
    accepted = [t for row in rows if (t := experience.trajectory(cases[row['case_id']], row['result'], policy, policy,
                                                                 sample=row['sample'])) is not None]
    return cases, accepted


def collect(sampler, tokenizer, stage, policies, execution, home):
    """Coached rollouts for every case without a complete natural success; verified successes, no near-policy filter."""
    round2 = stage['round2']['collection']
    policy = policies['scheduling']
    cases, accepted = natural(stage, policy, ROOT / growth.UPLOADED / round2['natural']['file'])
    pending = [case for case in cases.values() if experience.needs_coaching(case, accepted)]
    card = experience.coached(policy, round2['card'])
    batcher = rollout.Batcher(sampler, tokenizer, growth.sampling(policy, round2['temperature'], round2['top_p']),
                              max_batch=execution['max_batch'], device=execution['device'], seed=round2['seed'])
    first = stage['collection']['samples_per_case']
    started = time.monotonic()
    try:
        coached = rollout.rollouts([(case, card, first + sample) for case in pending
                                    for sample in range(round2['samples_per_case'])],
                                   batcher.respond, workers=execution['workers'],
                                   progress=growth.reporter(home, 'collect-coached'))
    finally:
        batcher.close()
    natural_count = len(accepted)
    accepted += [t for row in coached if (t := experience.trajectory(
        cases[row['case_id']], row['result'], card, policy, sample=row['sample'], coaching=True)) is not None]
    chosen = experience.select(accepted, round2['selection_per_case'])
    complete = sorted({t['case_id'] for t in chosen if t['complete']})
    report = {'summary': experience.summary(list(cases.values()), len(coached), chosen),
              'natural_verified': natural_count, 'coached_verified': len(accepted) - natural_count,
              'coached_cases': [case['id'] for case in pending], 'complete_cases': len(complete),
              'seconds': time.monotonic() - started, 'batches': len(batcher.batches),
              'model_calls': sum(r['result']['model_calls'] for r in coached),
              'rollouts_sha256': growth.write_rows(home / 'rollouts.jsonl.gz', coached),
              'trajectories_sha256': growth.write_rows(home / 'trajectories.jsonl.gz', chosen)}
    save(home / 'experience.json', report, exclusive=True)
    from neuroshard.evolution import assistant_experience_train as trainer
    from neuroshard.evolution import assistant_workflow as workflow

    tools = workflow.interface(policy).TOOLS
    return [trainer.encode(tokenizer, t, tools) for t in chosen], report


def committed_sources(root=ROOT):
    execution = read(root / EXECUTION)
    for name, digest in execution['contracts'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'changed round-2 contract: {name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in execution['sources']:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root) != (root / name).read_bytes():
            raise ValueError(f'uncommitted round-2 source: {name}')
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
        raise ValueError('round-2 runtime differs from the stage-1 runtime')
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
        raise ValueError('round-2 worker differs from freeze')
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
        sampler = growth.accepted(load_parent, spec, ROOT / growth.UPLOADED, pinned)
        scheduling_rows, collected = collect(sampler, tokenizer, stage, policies, parameters, home)
        reply['collection'] = {key: value for key, value in collected.items() if key != 'coached_cases'}
        del sampler
        growth.release_accelerator()
        reply['phases']['collect'] = time.monotonic() - begun
        minimum = stage['round2']['stop']['minimum_complete_cases']
        if collected['complete_cases'] < minimum:
            reply['stopped'] = f"{collected['complete_cases']} training cases have a complete verified trajectory; at least {minimum} are needed"
        else:
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
            raise ValueError('parent checkpoint changed during round 2')
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
        reply = launch(home, models, binding, 'baseline', 'round2', execution['worker_seconds'],
                       execution['memory_bytes'], worker_script=SCRIPT, environment=execution['worker_environment'])
        result['reply'] = reply
        if not reply['execution_completed']:
            raise ValueError(reply.get('error', 'incomplete round-2 execution'))
        result['execution_completed'] = True
    except Exception as error:
        result['error'] = str(error)
    save(home / 'result.json', result, exclusive=True)
    return result
