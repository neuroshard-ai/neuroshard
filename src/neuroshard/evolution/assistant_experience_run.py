"""One accelerator host: collect verified experience, record replay, train both arms, fit gates.

Development and confirmation evaluation are not here; they run on the CPU
runtime of the canonical parent baseline so device numerics cannot flip a
protected parent success. Every phase writes its inventory before the next.
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
from neuroshard.evolution import assistant_replay as replay
from neuroshard.evolution import assistant_rollout as rollout
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution import assistant_workspace as sandbox
from neuroshard.evolution import granite_tokenizer
from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, identity, launch, read, save, sha256, verify_artifacts,
)

PLAN = 'config/experiments/assistant-experience-learning.json'
EXECUTION = 'config/experiments/assistant-experience-execution.json'
ARTIFACTS = 'config/experiments/granite-reference-artifacts.json'
SCRIPT = 'scripts/run_assistant_experience.py'
PROFILE = 'assistant-experience-gpu'


def split_cases(plan, split):
    if split not in ('train', 'integration'):
        raise ValueError('accelerator phases may not access development or confirmation goals')
    manifest = read(ROOT / plan['data'])['splits'][split]
    cases = data.cases(split)
    if identity(cases) != manifest['sha256'] or [c['id'] for c in cases] != manifest['case_ids']:
        raise ValueError('workflow data differs from frozen split')
    return cases


def write_rows(path, rows):
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
    return identity(rows)


def sampling(policy, execution, temperature):
    return {**policy['generation'], 'temperature': temperature, 'top_p': execution['top_p']}


def collect(model, tokenizer, plan, policy, execution, home):
    """Uncoached rollouts, coached retries where needed, parent likelihood and selection."""
    from neuroshard.evolution import assistant_experience_train as trainer

    cases = split_cases(plan, 'train')
    by_id = {case['id']: case for case in cases}
    samples = execution['samples_per_case']
    card_policy = experience.coached(policy, plan['coaching']['card'])
    batcher = rollout.Batcher(model, tokenizer, sampling(policy, execution, execution['temperature']),
                              max_batch=execution['max_batch'], device=execution['device'], seed=execution['seed'])
    started = time.monotonic()
    try:
        natural = rollout.rollouts([(c, policy, s) for c in cases for s in range(samples)], batcher.respond,
                                   workers=execution['workers'])
        accepted = [t for row in natural if (t := experience.trajectory(
            by_id[row['case_id']], row['result'], policy, policy, sample=row['sample'])) is not None]
        pending = [c for c in cases if experience.needs_coaching(c, accepted)]
        coached_rows = rollout.rollouts([(c, card_policy, samples + s) for c in pending for s in range(samples)],
                                        batcher.respond, workers=execution['workers'])
        accepted += [t for row in coached_rows if (t := experience.trajectory(
            by_id[row['case_id']], row['result'], card_policy, policy, sample=row['sample'], coaching=True)) is not None]
    finally:
        batcher.close()
    sequences = {t['transcript_sha256']: trainer.encode(tokenizer, t, sandbox.TOOLS) for t in accepted}
    nll = {key: trainer.negative_log_likelihood(model, value, execution['device']) for key, value in sequences.items()}
    kept, ceiling = experience.near_policy(accepted, nll)
    chosen = experience.select(kept, plan['experience_selection_per_case'])
    report = {'summary': experience.summary(cases, len(natural) + len(coached_rows), chosen),
              'accepted_before_filter': len(accepted), 'near_policy_ceiling': ceiling,
              'coached_cases': [c['id'] for c in pending], 'seconds': time.monotonic() - started,
              'batches': len(batcher.batches), 'model_calls': sum(r['result']['model_calls'] for r in natural + coached_rows),
              'input_tokens': sum(r['result']['input_tokens'] for r in natural + coached_rows),
              'output_tokens': sum(r['result']['output_tokens'] for r in natural + coached_rows),
              'rollouts_sha256': write_rows(home / 'rollouts.jsonl.gz', natural + coached_rows),
              'trajectories_sha256': write_rows(home / 'trajectories.jsonl.gz', chosen),
              'nll': {t['transcript_sha256']: nll[t['transcript_sha256']] for t in chosen}}
    save(home / 'experience.json', report, exclusive=True)
    return [sequences[t['transcript_sha256']] for t in chosen], report


def record_replay(model, tokenizer, plan, execution, home):
    """Parent greedy answers to the generated replay prompts, before any training."""
    from neuroshard.evolution import assistant_experience_train as trainer

    anchors = read(ROOT / 'config/experiments/granite-reference.json')
    prompts = replay.prompts(execution['replay_seed'], anchors['tasks'] + anchors['reference_tasks'])
    greedy = {'max_input_tokens': execution['replay_max_input_tokens'],
              'max_new_tokens': execution['replay_max_new_tokens'], 'temperature': 0, 'top_p': 1.0}
    batcher = rollout.Batcher(model, tokenizer, greedy, max_batch=execution['max_batch'], device=execution['device'])
    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=execution['workers']) as pool:
            responses = list(pool.map(lambda p: batcher.respond(p['messages'], p['tools']), prompts))
    finally:
        batcher.close()
    items = [item for p, r in zip(prompts, responses) if (item := replay.replay_item(p, r)) is not None]
    sequences = [trainer.encode(tokenizer, item, item['tools']) for item in items]
    save(home / 'replay.json', {'prompts': len(prompts), 'terminated': len(items),
                                'items_sha256': write_rows(home / 'replay.jsonl.gz', items)}, exclusive=True)
    return sequences


def train_arms(load_parent, plan, execution, experience_rows, replay_rows, home):
    from neuroshard.evolution import assistant_experience_train as trainer

    spec = {**plan['training'], 'seed': plan['training']['seed']}
    manifests = {}
    for arm in ('update', 'addition'):
        model = load_parent()
        started = time.monotonic()
        trainable, receipt = trainer.train(model, arm, experience_rows, replay_rows, spec, device=execution['device'])
        roots = {'experience': identity([r['sha256'] for r in experience_rows]),
                 'replay': identity([r['sha256'] for r in replay_rows]), 'plan': identity(plan)}
        manifests[arm] = {**trainer.checkpoint(home / f'{arm}-checkpoint', trainable, receipt, roots),
                          'seconds': time.monotonic() - started, 'tokens_processed': receipt['tokens_processed']}
        del model, trainable
    save(home / 'training.json', manifests, exclusive=True)
    return manifests


def boundary_feature(model, tokenizer, policy, case, device):
    """Frozen parent final-layer state at the first assistant-generation boundary."""
    import torch

    messages = [{'role': 'system', 'content': policy['system_instruction']},
                {'role': 'user', 'content': data.public_case(case)['user_turns'][0]}]
    prompt = tokenizer.apply_chat_template(messages, tools=sandbox.TOOLS, add_generation_prompt=True, tokenize=False)
    ids = torch.tensor([tokenizer(prompt, add_special_tokens=False)['input_ids']], device=device)
    with torch.no_grad():
        return model.model(input_ids=ids).last_hidden_state[0, -1].float().cpu().tolist()


def outcomes(model, tokenizer, policy, execution, cases, seed):
    """One greedy and the declared number of sampled complete episodes per integration case."""
    results = {case['id']: [] for case in cases}
    for temperature, count in ((0, 1), (execution['temperature'], execution['integration_samples'])):
        batcher = rollout.Batcher(model, tokenizer, sampling(policy, execution, temperature),
                                  max_batch=execution['max_batch'], device=execution['device'], seed=seed)
        try:
            rows = rollout.rollouts([(c, policy, s) for c in cases for s in range(count)], batcher.respond,
                                    workers=execution['workers'])
        finally:
            batcher.close()
        for row in rows:
            results[row['case_id']].append(row['result']['score']['passed'])
    return results


def integrate(load_parent, tokenizer, plan, policy, execution, home):
    from neuroshard.evolution import assistant_experience_train as trainer
    from neuroshard.evolution import assistant_selector as selector

    cases = split_cases(plan, 'integration')
    parent = load_parent()
    features = {c['id']: boundary_feature(parent, tokenizer, policy, c, execution['device']) for c in cases}
    parent_outcomes = outcomes(parent, tokenizer, policy, execution, cases, execution['seed'] + 1)
    del parent
    recipe = plan['selection_recipe']
    gates = {}
    for arm in ('update', 'addition'):
        model = load_parent()
        trainer.load_trainable(model, arm, plan['training'], home / f'{arm}-checkpoint')
        arm_outcomes = outcomes(model, tokenizer, policy, execution, cases, execution['seed'] + 2)
        gates[arm] = {'gate': selector.fit(features, selector.targets(parent_outcomes, arm_outcomes), recipe),
                      'outcomes': arm_outcomes}
        del model
    save(home / 'integration.json', {'parent_outcomes': parent_outcomes, 'arms': gates,
                                     'features_sha256': identity(features)}, exclusive=True)
    return gates


def committed_sources(root=ROOT):
    execution = read(root / EXECUTION)
    for name, digest in execution['contracts'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'changed experience contract: {name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in execution['sources']:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root) != (root / name).read_bytes():
            raise ValueError(f'uncommitted experience source: {name}')
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


def gpu_names():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True)
    return [line.strip() for line in output.splitlines() if line.strip()]


def freeze():
    source = committed_sources()
    execution = read(ROOT / EXECUTION)
    packages = {key: importlib.metadata.version(key) for key in execution['packages']}
    if packages != execution['packages'] or platform.python_version() != execution['python']:
        raise ValueError('accelerator runtime differs')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('accelerator runtime requires Linux x86_64')
    if gpu_names() != execution['gpus']:
        raise ValueError('accelerator differs from the declared GPU')
    if any(os.environ.get(key) != value for key, value in execution['environment'].items()):
        raise ValueError('accelerator environment differs')
    return {**source, 'packages': packages, 'python': platform.python_version(), 'gpus': execution['gpus']}


def worker(request_path):
    configure()
    request_path = Path(request_path)
    request = read(request_path)
    if freeze() != request['freeze']:
        raise ValueError('experience worker differs from freeze')
    import torch
    from transformers import AutoModelForCausalLM

    execution = read(ROOT / EXECUTION)
    parameters = execution['execution']
    plan = read(ROOT / PLAN)
    policy = read(ROOT / plan['policy'])
    home = request_path.parent
    reply = {'binding': request['binding'], 'execution_completed': False, 'phases': {}}
    started = time.monotonic()
    tokenizer = None
    try:
        inventory = read(ROOT / ARTIFACTS)['models']['baseline']
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

        parent = load_parent()
        begun = time.monotonic()
        experience_rows, _ = collect(parent, tokenizer, plan, policy, parameters, home)
        reply['phases']['collect'] = time.monotonic() - begun
        begun = time.monotonic()
        replay_rows = record_replay(parent, tokenizer, plan, parameters, home)
        reply['phases']['replay'] = time.monotonic() - begun
        del parent
        torch.cuda.empty_cache()
        begun = time.monotonic()
        reply['training'] = train_arms(load_parent, plan, parameters, experience_rows, replay_rows, home)
        reply['phases']['train'] = time.monotonic() - begun
        begun = time.monotonic()
        gates = integrate(load_parent, tokenizer, plan, policy, parameters, home)
        reply['gates'] = {arm: value['gate'] for arm, value in gates.items()}
        reply['phases']['integrate'] = time.monotonic() - begun
        if file_state(directory, inventory) != state:
            raise ValueError('parent checkpoint changed during experience execution')
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
    binding = {'freeze': source, 'profile': PROFILE, 'plan_sha256': sha256(ROOT / PLAN)}
    save(home / 'binding.json', binding, exclusive=True)
    result = {'binding': binding, 'execution_completed': False, 'checklist_credit': False,
              'admission_evidence': False, 'confirmation_opened': False}
    try:
        reply = launch(home, models, binding, 'baseline', 'experience', execution['worker_seconds'],
                       execution['memory_bytes'], worker_script=SCRIPT, environment=execution['worker_environment'])
        result['reply'] = reply
        if not reply['execution_completed']:
            raise ValueError(reply.get('error', 'incomplete experience execution'))
        result['execution_completed'] = True
    except Exception as error:
        result['error'] = str(error)
    save(home / 'result.json', result, exclusive=True)
    return result
