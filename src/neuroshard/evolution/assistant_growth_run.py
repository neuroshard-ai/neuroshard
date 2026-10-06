"""A3 stage 1 on one GPU host: scheduling experience, two arms trained from the accepted version, and turn selectors.

The accepted version (the parent plus U1) samples the scheduling and cross training
cases under the calendar policy, with coached retries where it never succeeds alone.
Two arms then train from it on one declared mixture of that experience, A2's pinned
drafting experience and A2's parent replay: U2 continues U1's tensors, and L2 is a
low-rank module on top of U1. Each version's drafting and scheduling routes run every
integration case alone, and each version's turn selector is fitted from their per-turn
success rates. No development or confirmation goal is read.
"""

import importlib.metadata
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from neuroshard.evolution import assistant_experience as experience
from neuroshard.evolution import assistant_routing as routing
from neuroshard.evolution import assistant_rollout as rollout
from neuroshard.evolution import assistant_schedule_data as schedule
from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution import granite_tokenizer
from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, identity, launch, read, save, sha256, verify_artifacts,
)

PLAN = 'config/experiments/assistant-growth-stage1.json'
EXECUTION = 'config/experiments/assistant-growth-execution.json'
ARTIFACTS = 'config/experiments/granite-reference-artifacts.json'
SCRIPT = 'scripts/run_assistant_growth.py'
PROFILE = 'assistant-growth-gpu'
UPLOADED = '.growth'
# Each version's units for drafting turns and for scheduling turns.
VERSIONS = {'separate_update': ('U1', 'U2'), 'separate_module': ('U1', 'L2'), 'shared': ('U2', 'U2')}
ROUTE_RUNS = (('U1', 'drafting'), ('U2', 'drafting'), ('U2', 'scheduling'), ('L2', 'scheduling'))
CHECKPOINTS = {'U2': 'update-checkpoint', 'L2': 'module-checkpoint'}


def contracts(plan):
    """The A3 plan, the A2 learning contract and the policies the routes are served under."""
    growth = read(ROOT / plan['plan'])
    learning = read(ROOT / growth['cohort1']['learning'])
    policies = {'drafting': read(ROOT / learning['policy']), 'scheduling': read(ROOT / growth['cohort2']['policy'])}
    return growth, learning, policies


def scheduling_cases(plan, split):
    """Scheduling training or integration cases, checked against their frozen split."""
    if split not in (*schedule.TRAINING, 'integration', 'cross-integration'):
        raise ValueError('stage 1 may not access development or confirmation goals')
    frozen = read(ROOT / plan['collection']['data'])['splits'][split]
    cases = schedule.cases(split)
    if identity(cases) != frozen['sha256'] or [c['id'] for c in cases] != frozen['case_ids']:
        raise ValueError('scheduling data differs from its frozen split')
    return cases


def drafting_cases(learning, split):
    from neuroshard.evolution import assistant_experience_run as run

    return run.split_cases(learning, split)


def stage_spec(plan, learning):
    """The declared stage-1 schedule on the A2 layers and projections, as a trainer spec."""
    training = plan['training']
    mixture = training['mixture']
    return {**learning['training'],
            **{key: training[key] for key in ('steps', 'microbatch', 'gradient_accumulation', 'warmup_steps',
                                              'gradient_clip', 'weight_decay', 'betas', 'epsilon', 'seed')},
            'learning_rates': {'update': training['learning_rates']['update'],
                               'addition': training['learning_rates']['module']},
            'mixture': {'experience': mixture['scheduling'], 'drafting': mixture['drafting'],
                        'replay': mixture['preservation']}}


def accepted(load_parent, spec, directory, pinned):
    """The accepted version: a fresh parent with U1's tensors served as plain projections."""
    from neuroshard.evolution import assistant_experience_train as trainer

    model = load_parent()
    manifest = trainer.load_trainable(model, 'update', spec, Path(directory) / 'update-checkpoint')
    if manifest['trainable_sha256'] != pinned:
        raise ValueError('accepted update differs from its pinned digest')
    trainer.serving(model, spec)
    return model.eval()


def unit_loaders(load_parent, spec, home, pinned):
    """Loaders for the three units: U1 as accepted, U2 as trained, and L2 attached on top of U1."""
    from neuroshard.evolution import assistant_experience_train as trainer

    def u1():
        return accepted(load_parent, spec, ROOT / UPLOADED, pinned)

    def u2():
        model = load_parent()
        trainer.load_trainable(model, 'update', spec, Path(home) / CHECKPOINTS['U2'])
        trainer.serving(model, spec)
        return model.eval()

    def l2():
        model = u1()
        trainer.load_trainable(model, 'addition', spec, Path(home) / CHECKPOINTS['L2'])
        return model.eval()

    return {'U1': u1, 'U2': u2, 'L2': l2}


def sampling(policy, temperature, top_p):
    return {**policy['generation'], 'temperature': temperature, 'top_p': top_p}


def collect(sampler, tokenizer, plan, learning, policies, execution, home):
    """Natural and coached scheduling rollouts, the drafting reference, the near-policy check and selection."""
    from neuroshard.evolution import assistant_experience_train as trainer

    stage = plan['collection']
    cases = [case for split in stage['splits'] for case in scheduling_cases(plan, split)]
    reference = drafting_cases(learning, 'train')[:64]
    by_id = {case['id']: case for case in cases + reference}
    calendar_policy, drafting_policy = policies['scheduling'], policies['drafting']
    card_policy = experience.coached(calendar_policy, stage['coaching']['card'])
    executed = {identity(p): p for p in (calendar_policy, drafting_policy, card_policy)}
    samples = stage['samples_per_case']
    batcher = rollout.Batcher(sampler, tokenizer, sampling(calendar_policy, stage['temperature'], stage['top_p']),
                              max_batch=execution['max_batch'], device=execution['device'], seed=stage['seed'])
    started = time.monotonic()
    try:
        natural = rollout.rollouts([(c, calendar_policy, s) for c in cases for s in range(samples)]
                                   + [(c, drafting_policy, s) for c in reference for s in range(2)],
                                   batcher.respond, workers=execution['workers'], progress=reporter(home, 'collect-natural'))
        accepted_rows = [t for row in natural if (t := experience.trajectory(
            by_id[row['case_id']], row['result'], executed[row['policy_sha256']], executed[row['policy_sha256']],
            sample=row['sample'])) is not None]
        pending = [c for c in cases if experience.needs_coaching(c, accepted_rows)]
        coached = rollout.rollouts([(c, card_policy, samples + s) for c in pending for s in range(samples)],
                                   batcher.respond, workers=execution['workers'], progress=reporter(home, 'collect-coached'))
        accepted_rows += [t for row in coached if (t := experience.trajectory(
            by_id[row['case_id']], row['result'], card_policy, calendar_policy, sample=row['sample'],
            coaching=True)) is not None]
    finally:
        batcher.close()
    save(home / 'progress.json', {'phase': 'sampler-likelihood', 'accepted': len(accepted_rows), 'unix': time.time()})
    tools = {identity(p): workflow.interface(p).TOOLS for p in (calendar_policy, drafting_policy)}
    sequences = {t['transcript_sha256']: trainer.encode(tokenizer, t, tools[t['train_policy_sha256']]) for t in accepted_rows}
    nll = {key: trainer.negative_log_likelihood(sampler, value, execution['device']) for key, value in sequences.items()}
    kept, ceiling = experience.near_policy(accepted_rows, nll)
    scheduling_ids = {case['id'] for case in cases}
    chosen = experience.select([t for t in kept if t['case_id'] in scheduling_ids], stage['selection_per_case'])
    report = {'summary': experience.summary(cases, len(natural) + len(coached) - 2 * len(reference), chosen),
              'drafting_reference': {'cases': len(reference), 'natural_successes': sum(
                  t['case_id'] not in scheduling_ids for t in accepted_rows)},
              'accepted_before_filter': sum(t['case_id'] in scheduling_ids for t in accepted_rows),
              'near_policy_ceiling': ceiling, 'coached_cases': [c['id'] for c in pending],
              'seconds': time.monotonic() - started, 'batches': len(batcher.batches),
              'model_calls': sum(r['result']['model_calls'] for r in natural + coached),
              'rollouts_sha256': write_rows(home / 'rollouts.jsonl.gz', natural + coached),
              'trajectories_sha256': write_rows(home / 'trajectories.jsonl.gz', chosen),
              'nll': {t['transcript_sha256']: nll[t['transcript_sha256']] for t in chosen}}
    save(home / 'experience.json', report, exclusive=True)
    return [sequences[t['transcript_sha256']] for t in chosen], report


def write_rows(path, rows):
    from neuroshard.evolution import assistant_experience_run as run

    return run.write_rows(path, rows)


def reporter(home, phase):
    def report(done, total):
        if done == total or done % 50 == 0:
            save(home / 'progress.json', {'phase': phase, 'done': done, 'total': total, 'unix': time.time()})
    return report


def drafting_experience(tokenizer, learning, policies, execution, home):
    """A2's pinned round-1 drafting trajectories and parent replay, re-verified from their recorded rollouts."""
    from neuroshard.evolution import assistant_experience_run as run

    pinned = execution['drafting_collection']
    return run.load_collection(ROOT / UPLOADED, pinned['files'], tokenizer, learning, policies['drafting'],
                               {'replay_seed': pinned['replay_seed']}, home)


def train(load_parent, plan, learning, rows, execution, home, pinned):
    """U2 continues U1's tensors and L2 trains on top of U1, both on the same declared mixture."""
    from neuroshard.evolution import assistant_experience_train as trainer

    spec = stage_spec(plan, learning)
    roots = {name: identity([r['sha256'] for r in value]) for name, value in rows.items()}
    manifests = {}
    for unit, arm in (('U2', 'update'), ('L2', 'addition')):
        save(home / 'progress.json', {'phase': f'train-{unit}', 'unix': time.time()})
        started = time.monotonic()
        if unit == 'U2':
            model = load_parent()
            manifest, trainable = trainer.resume(model, 'update', spec, ROOT / UPLOADED / 'update-checkpoint')
            if manifest['trainable_sha256'] != pinned:
                raise ValueError('accepted update differs from its pinned digest')
        else:
            model, trainable = accepted(load_parent, spec, ROOT / UPLOADED, pinned), None
        trained, receipt = trainer.train(model, arm, rows['scheduling'], rows['replay'], spec, device=execution['device'],
                                         trainable=trainable, extra={'drafting': rows['drafting']})
        manifests[unit] = {**trainer.checkpoint(Path(home) / CHECKPOINTS[unit], trained, receipt,
                                                {**roots, 'plan': identity(plan)}),
                           'seconds': time.monotonic() - started, 'tokens_processed': receipt['tokens_processed']}
        del model, trained, trainable
        release_accelerator()
    save(home / 'training.json', manifests, exclusive=True)
    return manifests


def release_accelerator():
    from neuroshard.evolution import assistant_experience_run as run

    run.release_accelerator()


def integration_cases(plan, learning):
    return (scheduling_cases(plan, 'integration') + scheduling_cases(plan, 'cross-integration')
            + drafting_cases(learning, 'integration'))


def integrate(load_parent, loaders, tokenizer, plan, learning, policies, execution, home, route_runs=ROUTE_RUNS):
    """Every route alone on every integration case, then each version's turn selector."""
    from neuroshard.evolution import assistant_selector as selector

    stage = plan['integration']
    cases = integration_cases(plan, learning)
    parent = load_parent()
    features = {routing.turn_key(case['id'], turn): routing.turn_feature(parent, tokenizer, policies['drafting'], user,
                                                                         execution['device'])
                for case in cases for turn, user in enumerate(data.public_case(case)['user_turns'])}
    del parent
    release_accelerator()
    outcomes = {}
    for index, (unit, route) in enumerate(route_runs):
        model = loaders[unit]()
        runs = {case['id']: [] for case in cases}
        for temperature, count in ((0, stage['greedy']), (stage['temperature'], stage['samples'])):
            batcher = rollout.Batcher(model, tokenizer, sampling(policies[route], temperature, 0.95),
                                      max_batch=execution['max_batch'], device=execution['device'],
                                      seed=stage['seed'] + index)
            try:
                rows = routing.rollouts([(c, policies[route], s) for c in cases for s in range(count)], batcher.respond,
                                        workers=execution['workers'], progress=reporter(home, f'integrate-{unit}-{route}'))
            finally:
                batcher.close()
            for row in rows:
                runs[row['case_id']].append(row['result']['score']['round_successes'])
        outcomes[f'{unit}-{route}'] = runs
        del model
        release_accelerator()
    turns = {case['id']: len(case['turns']) for case in cases}
    gates = {version: selector.fit(features, routing.turn_targets(outcomes[f'{drafting}-drafting'],
                                                                  outcomes[f'{scheduling}-scheduling'], turns),
                                   stage['recipe'])
             for version, (drafting, scheduling) in VERSIONS.items()}
    save(home / 'integration.json', {'outcomes': outcomes, 'gates': gates, 'features_sha256': identity(features)},
         exclusive=True)
    return gates, outcomes


def committed_sources(root=ROOT):
    execution = read(root / EXECUTION)
    for name, digest in execution['contracts'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'changed growth contract: {name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in execution['sources']:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root) != (root / name).read_bytes():
            raise ValueError(f'uncommitted growth source: {name}')
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
    from neuroshard.evolution import assistant_experience_run as run

    source = committed_sources()
    execution = read(ROOT / EXECUTION)
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
        raise ValueError('growth worker differs from freeze')
    import torch
    from transformers import AutoModelForCausalLM

    execution = read(ROOT / EXECUTION)
    plan = read(ROOT / PLAN)
    growth, learning, policies = contracts(plan)
    gpu = request['freeze']['gpus'][0]
    parameters = {**execution['execution'], **execution['gpus'][gpu]}
    pinned = growth['cohort1']['trainable_sha256']
    home = request_path.parent
    reply = {'binding': request['binding'], 'execution_completed': False, 'phases': {}, 'gpu': gpu,
             'max_batch': parameters['max_batch']}
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

        spec = stage_spec(plan, learning)
        begun = time.monotonic()
        sampler = accepted(load_parent, spec, ROOT / UPLOADED, pinned)
        scheduling_rows, collected = collect(sampler, tokenizer, plan, learning, policies, parameters, home)
        reply['collection'] = {**collected['summary'], 'near_policy_ceiling': collected['near_policy_ceiling'],
                               'drafting_reference': collected['drafting_reference']}
        del sampler
        release_accelerator()
        if not scheduling_rows:
            raise ValueError('no verified scheduling experience')
        reply['phases']['collect'] = time.monotonic() - begun
        begun = time.monotonic()
        drafting_rows, replay_rows = drafting_experience(tokenizer, learning, policies, execution, home)
        reply['phases']['verify_drafting'] = time.monotonic() - begun
        begun = time.monotonic()
        rows = {'scheduling': scheduling_rows, 'drafting': drafting_rows, 'replay': replay_rows}
        reply['training'] = train(load_parent, plan, learning, rows, parameters, home, pinned)
        reply['phases']['train'] = time.monotonic() - begun
        begun = time.monotonic()
        gates, _ = integrate(load_parent, unit_loaders(load_parent, spec, home, pinned), tokenizer, plan, learning,
                             policies, parameters, home)
        reply['gates'] = {version: {k: v for k, v in gate.items() if k != 'weight'} for version, gate in gates.items()}
        reply['phases']['integrate'] = time.monotonic() - begun
        if file_state(directory, inventory) != state:
            raise ValueError('parent checkpoint changed during the growth execution')
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
              'admission_evidence': False, 'development_opened': False, 'confirmation_opened': False}
    try:
        reply = launch(home, models, binding, 'baseline', 'growth', execution['worker_seconds'],
                       execution['memory_bytes'], worker_script=SCRIPT, environment=execution['worker_environment'])
        result['reply'] = reply
        if not reply['execution_completed']:
            raise ValueError(reply.get('error', 'incomplete growth execution'))
        result['execution_completed'] = True
    except Exception as error:
        result['error'] = str(error)
    save(home / 'result.json', result, exclusive=True)
    return result
