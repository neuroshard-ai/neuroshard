"""A3 cohort 3 on one GPU host: an upgrade of drafting, learned from verified demonstrations.

Every drafting training case is solved once by the drafting solver and verified by the
scorer under the drafting policy; a demonstration that fails, or that reads an unapproved
revision, stops the job before training. L3, a new low-rank module on top of the accepted
update U1, trains on the demonstrations and A2's parent replay. U1 and L3 then run the
drafting integration cases alone on the drafting route. No development or confirmation
goal is read.
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

from neuroshard.evolution import assistant_drafting_demonstration as demonstrations
from neuroshard.evolution import assistant_growth_run as growth
from neuroshard.evolution import assistant_rollout as rollout
from neuroshard.evolution import assistant_routing as routing
from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution import granite_tokenizer
from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, identity, launch, read, save, sha256, verify_artifacts,
)

DECLARATION = 'config/experiments/assistant-growth-cohort3.json'
EXECUTION = 'config/experiments/assistant-growth-cohort3-execution.json'
SCRIPT = 'scripts/run_assistant_growth_cohort3.py'
PROFILE = 'assistant-growth-cohort3-gpu'
CHECKPOINT = 'module-checkpoint'


def plan():
    """The stage-1 plan with cohort 3's declaration; its training and integration rules are cohort 3's."""
    return {**read(ROOT / growth.PLAN), 'cohort3': read(ROOT / DECLARATION)}


def contracts(stage):
    """The A3 plan, the A2 learning contract and the two route policies cohort 3 declares."""
    growth_plan, learning, _ = growth.contracts(stage)
    declared = stage['cohort3']['policies']
    return growth_plan, learning, {route: read(ROOT / path) for route, path in declared.items()}


def spec(stage, learning):
    """Stage 1's trainer settings with cohort 3's steps, seed and mixture."""
    training = stage['cohort3']['training']
    return {**growth.stage_spec(stage, learning), 'steps': training['steps'], 'seed': training['seed'],
            'mixture': training['mixture']}


def training_cases(stage, learning):
    """Every declared drafting training case, each split checked against its frozen manifest."""
    return [case for split in stage['cohort3']['experience']['splits'] for case in growth.drafting_cases(learning, split)]


def demonstration(case, policy):
    """One training case solved by the drafting solver and verified by the scorer; no unapproved revision is read."""
    from neuroshard.evolution import assistant_experience as experience

    replies = iter(demonstrations.texts(case))

    def respond(messages, tools):
        return {'text': next(replies), 'terminated': True, 'executed': True, 'token_ids': [], 'input_token_ids': [],
                'prompt_sha256': identity({'messages': messages, 'tools': tools}), 'seconds': 0.0,
                'model': 'demonstration'}

    result = workflow.execute(case, respond, policy)
    unapproved = any(row['call']['name'] == 'read_document' and row['result'].get('status') != 'approved'
                     for row in result['calls'])
    if not result['score']['passed'] or next(replies, None) is not None or unapproved:
        raise ValueError(f"the demonstration does not solve {case['id']}")
    return {**experience.trajectory(case, result, policy, policy, sample=0), 'demonstration': True}, result


def collect(tokenizer, stage, learning, policies, home):
    """Every drafting training case's verified demonstration, as encoded experience."""
    from neuroshard.evolution import assistant_experience_train as trainer

    policy = policies['drafting']
    cases = training_cases(stage, learning)
    items, rows = [], []
    for case in cases:
        item, result = demonstration(case, policy)
        items.append(item)
        rows.append({'case_id': case['id'], 'sample': 0, 'policy_sha256': identity(policy), 'result': result})
    replies = [m['content'] for t in items for m in t['messages'] if m['role'] == 'assistant']
    report = {'cases': len(items), 'complete': sum(t['complete'] for t in items),
              'by_family': dict(collections.Counter(t['family'] for t in items)),
              'by_split': dict(collections.Counter(case['split'] for case in cases)),
              'replies': len(replies), 'two_call_replies': sum(text.count('<tool_call>') == 2 for text in replies),
              'demonstrations_sha256': growth.write_rows(home / 'demonstrations.jsonl.gz', rows),
              'trajectories_sha256': growth.write_rows(home / 'trajectories.jsonl.gz', items)}
    save(home / 'experience.json', report, exclusive=True)
    return [trainer.encode(tokenizer, t, workflow.interface(policy).TOOLS) for t in items], report


def train(load_parent, stage, learning, rows, parameters, home, pinned):
    """L3 attaches to the accepted version and trains on the declared mixture; U1 stays frozen."""
    from neuroshard.evolution import assistant_experience_train as trainer

    settings = spec(stage, learning)
    save(home / 'progress.json', {'phase': 'train-L3', 'unix': time.time()})
    started = time.monotonic()
    model = growth.accepted(load_parent, settings, ROOT / growth.UPLOADED, pinned)
    trained, receipt = trainer.train(model, stage['cohort3']['training']['arm'], rows['experience'], rows['replay'],
                                     settings, device=parameters['device'])
    roots = {name: identity([r['sha256'] for r in value]) for name, value in rows.items()}
    manifest = {**trainer.checkpoint(Path(home) / CHECKPOINT, trained, receipt, {**roots, 'plan': identity(stage)}),
                'seconds': time.monotonic() - started, 'tokens_processed': receipt['tokens_processed']}
    del model, trained
    growth.release_accelerator()
    save(home / 'training.json', {'L3': manifest}, exclusive=True)
    return {'L3': manifest}


def unit_loaders(load_parent, settings, home, pinned):
    """U1 as accepted, and L3 attached on top of it."""
    from neuroshard.evolution import assistant_experience_train as trainer

    def u1():
        return growth.accepted(load_parent, settings, ROOT / growth.UPLOADED, pinned)

    def l3():
        model = u1()
        trainer.load_trainable(model, 'addition', settings, Path(home) / CHECKPOINT)
        return model.eval()

    return {'U1': u1, 'L3': l3}


def integrate(loaders, tokenizer, stage, learning, policies, parameters, home):
    """Each declared route alone on every drafting integration case, greedily and sampled."""
    rules, declared = stage['integration'], stage['cohort3']['integration']
    cases = growth.drafting_cases(learning, 'integration')
    policy = policies['drafting']
    outcomes, greedy = {}, []
    for index, (unit, route) in enumerate(declared['routes']):
        model = loaders[unit]()
        runs = {case['id']: [] for case in cases}
        for temperature, count in ((0, rules['greedy']), (rules['temperature'], rules['samples'])):
            batcher = rollout.Batcher(model, tokenizer, growth.sampling(policy, temperature, 0.95),
                                      max_batch=parameters['max_batch'], device=parameters['device'],
                                      seed=declared['seed'] + index)
            try:
                rows = routing.rollouts([(c, policy, s) for c in cases for s in range(count)], batcher.respond,
                                        workers=parameters['workers'],
                                        progress=growth.reporter(home, f'integrate-{unit}-{route}'))
            finally:
                batcher.close()
            for row in rows:
                runs[row['case_id']].append(row['result']['score']['round_successes'])
            if temperature == 0:
                greedy += [{**row, 'unit': unit} for row in rows]
        outcomes[f'{unit}-{route}'] = runs
        del model
        growth.release_accelerator()
    turns = {case['id']: len(case['turns']) for case in cases}

    def solved(case, run):
        return len(run) == turns[case] and all(run)

    passed = {name: {case: solved(case, runs[0]) for case, runs in by_case.items()} for name, by_case in outcomes.items()}
    before, after = passed['U1-drafting'], passed['L3-drafting']
    summary = {'greedy': {name: sum(values.values()) for name, values in passed.items()},
               'sampled': {name: sum(solved(case, run) for case, runs in by_case.items() for run in runs[1:])
                           for name, by_case in outcomes.items()},
               'gained': sorted(case for case in after if after[case] and not before[case]),
               'lost': sorted(case for case in after if before[case] and not after[case]), 'cases': len(cases)}
    save(home / 'integration.json', {'outcomes': outcomes, 'summary': summary,
                                     'greedy_sha256': growth.write_rows(home / 'integration-greedy.jsonl.gz', greedy)},
         exclusive=True)
    return summary


def committed_sources(root=ROOT):
    execution = read(root / EXECUTION)
    for name, digest in execution['contracts'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'changed cohort-3 contract: {name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in execution['sources']:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root) != (root / name).read_bytes():
            raise ValueError(f'uncommitted cohort-3 source: {name}')
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
        raise ValueError('cohort-3 runtime differs from the stage-1 runtime')
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
        raise ValueError('cohort-3 worker differs from freeze')
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

        begun = time.monotonic()
        experience_rows, reply['collection'] = collect(tokenizer, stage, learning, policies, home)
        reply['phases']['demonstrate'] = time.monotonic() - begun
        begun = time.monotonic()
        _, replay_rows = growth.drafting_experience(tokenizer, learning, policies, execution, home)
        reply['phases']['verify_replay'] = time.monotonic() - begun
        begun = time.monotonic()
        reply['training'] = train(load_parent, stage, learning, {'experience': experience_rows, 'replay': replay_rows},
                                  parameters, home, pinned)
        reply['phases']['train'] = time.monotonic() - begun
        begun = time.monotonic()
        reply['integration'] = integrate(unit_loaders(load_parent, spec(stage, learning), home, pinned), tokenizer,
                                         stage, learning, policies, parameters, home)
        reply['phases']['integrate'] = time.monotonic() - begun
        if file_state(directory, inventory) != state:
            raise ValueError('parent checkpoint changed during cohort 3')
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
        reply = launch(home, models, binding, 'baseline', 'cohort3', execution['worker_seconds'],
                       execution['memory_bytes'], worker_script=SCRIPT, environment=execution['worker_environment'])
        result['reply'] = reply
        if not reply['execution_completed']:
            raise ValueError(reply.get('error', 'incomplete cohort-3 execution'))
        result['execution_completed'] = True
    except Exception as error:
        result['error'] = str(error)
    save(home / 'result.json', result, exclusive=True)
    return result
