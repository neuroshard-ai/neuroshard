"""The general route's documents unit L4 on one GPU host: learned from verified demonstrations, judged on development.

L4 is a low-rank module on the unchanged parent, attached to the A2 layers and projections. It
learns from the documents solver's verified demonstrations of every training case, mixed with the
parent's own verified chat replies on the chat training split, so that what the parent already
does well on general chat is replayed rather than overwritten. Then the parent and the parent with
L4 run the documents and chat development cases greedily, side by side on the same accelerator.
No confirmation goal is read; L4 is pinned before any sealed evaluation.
"""

import collections
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
    ROOT, file_state, identity, launch, read, save, sha256, verify_artifacts,
)

DECLARATION = 'config/experiments/assistant-general-l4.json'
EXECUTION = 'config/experiments/assistant-general-l4-execution.json'
SCRIPT = 'scripts/run_assistant_general_l4.py'
PROFILE = 'assistant-general-l4-gpu'
CHECKPOINT = 'module-checkpoint'
# Later documents units on the same worker: each declares its experience, replay and development splits.
PROFILES = {PROFILE: (DECLARATION, EXECUTION),
            'assistant-general-l5-gpu': ('config/experiments/assistant-general-l5.json',
                                         'config/experiments/assistant-general-l5-execution.json'),
            'assistant-general-l6-gpu': ('config/experiments/assistant-general-l6.json',
                                         'config/experiments/assistant-general-l6-execution.json'),
            'assistant-general-l7-gpu': ('config/experiments/assistant-general-l7.json',
                                         'config/experiments/assistant-general-l7-execution.json'),
            'assistant-general-l8-gpu': ('config/experiments/assistant-general-l8.json',
                                         'config/experiments/assistant-general-l8-execution.json'),
            'assistant-general-c1-gpu': ('config/experiments/assistant-general-c1.json',
                                         'config/experiments/assistant-general-c1-execution.json'),
            'assistant-general-l9-gpu': ('config/experiments/assistant-general-l9.json',
                                         'config/experiments/assistant-general-l9-execution.json')}


def declared(profile=PROFILE):
    return read(ROOT / PROFILES[profile][0])


def splits(declaration):
    """(documents experience, chat replay, development) splits; L4 declared none and used the authored ones."""
    rows = declaration.get('splits', {})
    return (rows.get('experience', 'train'), rows.get('replay', 'train'),
            {'documents': rows.get('documents_development', 'development'),
             'chat': rows.get('chat_development', 'development')})


def policies(declaration=None):
    declaration = declaration or declared()
    return {name: read(ROOT / path) for name, path in declaration['policies'].items()}


def spec(declaration=None):
    """A2's trainer settings on its layers and projections, with L4's steps, seed and mixture."""
    from neuroshard.evolution import assistant_growth_run as growth

    declaration = declaration or declared()
    plan = read(ROOT / growth.PLAN)
    base = growth.stage_spec(plan, read(ROOT / read(ROOT / plan['plan'])['cohort1']['learning']))
    training = declaration['training']
    return {**base, 'steps': training['steps'], 'seed': training['seed'], 'mixture': training['mixture']}


def demonstrations(tokenizer, policy, home, split='train'):
    """Every documents training case's verified demonstration, encoded under the documents tools."""
    from neuroshard.evolution import assistant_documents as documents
    from neuroshard.evolution import assistant_documents_data as data
    from neuroshard.evolution import assistant_documents_demonstration as demo
    from neuroshard.evolution import assistant_experience_train as trainer
    from neuroshard.evolution import assistant_growth_run as growth

    items = []
    for case in [c for name in ([split] if isinstance(split, str) else split) for c in data.cases(name)]:
        row = demo.demonstration(case, policy)
        items.append(demo.trajectory(case, row, policy))
    report = {'cases': len(items), 'by_family': dict(collections.Counter(t['family'] for t in items)),
              'trajectories_sha256': growth.write_rows(home / 'documents-demonstrations.jsonl.gz', items)}
    from neuroshard.evolution import assistant_general as general

    tools = documents.tools(general.version_of(policy))
    return [trainer.encode(tokenizer, t, tools) for t in items], report


def chat_replay(model, tokenizer, policy, parameters, home, split='train', seed=None, temperature=0.0, attempts=1,
                name='chat-replay'):
    """The parent's chat replies on the chat training split(s); only those that pass are kept.

    The first attempt is greedy. With ``attempts > 1`` each case that has not passed is sampled again at
    ``temperature`` (fresh seed per attempt) and its first passing reply is kept: verified experience from
    the parent's own distribution, never from another model."""
    from concurrent.futures import ThreadPoolExecutor

    from neuroshard.evolution import assistant_chat_data as chat
    from neuroshard.evolution import assistant_experience_train as trainer
    from neuroshard.evolution import assistant_general as general
    from neuroshard.evolution import assistant_growth_run as growth
    from neuroshard.evolution import assistant_rollout as rollout

    cases = [c for split_name in ([split] if isinstance(split, str) else split) for c in chat.cases(split_name)]
    seed = declared()['training']['seed'] if seed is None else seed
    rows, attempt_of = [None] * len(cases), [None] * len(cases)
    for attempt in range(attempts):
        pending = [i for i, row in enumerate(rows) if row is None or not row['passed']]
        if not pending:
            break
        batcher = rollout.Batcher(model, tokenizer, growth.sampling(policy, temperature if attempt else 0, 0.95 if attempt
                                                                    else 1.0), max_batch=parameters['max_batch'],
                                  device=parameters['device'], seed=seed + attempt)
        try:
            with ThreadPoolExecutor(max_workers=parameters['workers']) as pool:
                fresh = list(pool.map(lambda i: general.chat_episode(cases[i], batcher.respond, policy), pending))
        finally:
            batcher.close()
        for i, row in zip(pending, fresh):
            rows[i], attempt_of[i] = row, attempt
    from neuroshard.evolution import assistant_workspace as base

    kit = general.ChatTools if policy.get('interface') == 'chat-tools/1' else general.Calculator
    tools = kit.TOOLS
    items = []
    for case, row in zip(cases, rows):
        if not row['passed']:
            continue
        messages = [{'role': 'system', 'content': policy['system_instruction']},
                    {'role': 'user', 'content': case['turns'][0]['user']}]
        calls, clean = iter(row['calls']), True
        for generated in row['generations']:
            messages.append({'role': 'assistant', 'content': generated['text']})
            try:
                proposed = base.parse_calls(generated['text'], kit.REGISTRY)
            except (ValueError, TypeError, RecursionError):
                clean = False
                break
            for _ in proposed:
                messages.append({'role': 'tool', 'content': json.dumps(next(calls)['result'], sort_keys=True)})
        if not clean or next(calls, None) is not None:
            continue
        items.append({'case_id': case['id'], 'family': case['family'], 'messages': messages,
                      'trainable': [m['role'] == 'assistant' for m in messages]})
    report = {'cases': len(cases), 'passed': len(items),
              'by_family': dict(collections.Counter(t['family'] for t in items)),
              'by_attempt': dict(collections.Counter(attempt_of[i] for i, row in enumerate(rows) if row['passed'])),
              'replay_sha256': growth.write_rows(home / f'{name}.jsonl.gz', items)}
    return [trainer.encode(tokenizer, t, tools) for t in items], report


def integrate(models, tokenizer, declaration, parameters, home):
    """Each model greedily on the documents and chat development cases, on the same accelerator."""
    from concurrent.futures import ThreadPoolExecutor

    from neuroshard.evolution import assistant_chat_data as chat
    from neuroshard.evolution import assistant_documents_data as docs
    from neuroshard.evolution import assistant_general as general
    from neuroshard.evolution import assistant_growth_run as growth
    from neuroshard.evolution import assistant_rollout as rollout

    routes = policies(declaration)
    out = {}
    for name, load in models.items():
        model = load()
        out[name] = {}
        kinds = declaration.get('integrate', ['documents', 'chat'])
        for kind, module, run in (('documents', docs, general.documents_episode), ('chat', chat, general.chat_episode)):
            if kind not in kinds:
                continue
            policy, cases = routes[kind], module.cases(splits(declaration)[2][kind])
            batcher = rollout.Batcher(model, tokenizer, growth.sampling(policy, 0, 1.0),
                                      max_batch=parameters['max_batch'], device=parameters['device'],
                                      seed=declaration['training']['seed'])
            try:
                with ThreadPoolExecutor(max_workers=parameters['workers']) as pool:
                    rows = list(pool.map(lambda case: run(case, batcher.respond, policy), cases))
            finally:
                batcher.close()
            out[name][kind] = rows
            save(home / 'progress.json', {'phase': f'integrate-{name}-{kind}', 'unix': time.time()})
        del model
        growth.release_accelerator()
    summary = {name: {kind: general.summary(rows) for kind, rows in by_kind.items()} for name, by_kind in out.items()}
    save(home / 'integration.json', {'summary': summary, 'episodes': out}, exclusive=True)
    return summary


def committed_sources(root=ROOT, profile=PROFILE):
    execution = read(root / PROFILES[profile][1])
    for name, digest in execution['contracts'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'changed L4 contract: {name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in execution['sources']:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root) != (root / name).read_bytes():
            raise ValueError(f'uncommitted L4 source: {name}')
        sources[name] = sha256(root / name)
    return {'commit': commit, 'sources': sources}


def configure(profile=PROFILE):
    if 'torch' in sys.modules:
        raise ValueError('configure the accelerator runtime before importing torch')
    for key, value in read(ROOT / PROFILES[profile][1])['environment'].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def freeze(profile=PROFILE):
    """The stage-1 GPU runtime, byte for byte in packages, GPUs and environment."""
    from neuroshard.evolution import assistant_experience_run as run
    from neuroshard.evolution import assistant_growth_run as growth

    source = committed_sources(profile=profile)
    execution = read(ROOT / PROFILES[profile][1])
    pinned = read(ROOT / growth.EXECUTION)
    keys = ('packages', 'python', 'gpus', 'environment', 'worker_environment', 'tokenizer_pipeline_sha256', 'execution')
    if any(execution[key] != pinned[key] for key in keys):
        raise ValueError('L4 runtime differs from the stage-1 runtime')
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
    request_path = Path(request_path)
    request = read(request_path)
    profile = request['profile'] if request['profile'] in PROFILES else PROFILE
    configure(profile)
    if freeze(profile) != request['freeze']:
        raise ValueError('L4 worker differs from freeze')
    import torch
    from transformers import AutoModelForCausalLM

    from neuroshard.evolution import assistant_experience_train as trainer
    from neuroshard.evolution import assistant_growth_run as growth
    from neuroshard.evolution import granite_tokenizer

    declaration_path, execution_path = PROFILES[profile]
    execution, declaration = read(ROOT / execution_path), read(ROOT / declaration_path)
    experience_split, replay_split, _ = splits(declaration)
    gpu = request['freeze']['gpus'][0]
    parameters = {**execution['execution'], **execution['gpus'][gpu]}
    home = request_path.parent
    reply = {'binding': request['binding'], 'execution_completed': False, 'phases': {}, 'gpu': gpu}
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

        routes, settings = policies(declaration), spec(declaration)
        begun = time.monotonic()
        if declaration.get('route') == 'chat':
            # A chat unit: its experience is the parent's own verified chat replies, rejection-sampled.
            sampling = declaration['experience']['sampling']
            parent = load_parent()
            experience, reply['chat_experience'] = chat_replay(
                parent, tokenizer, routes['chat'], parameters, home, experience_split,
                declaration['training']['seed'] + 1000, sampling['temperature'], sampling['attempts'], 'chat-experience')
            del parent
            growth.release_accelerator()
            if reply['chat_experience']['passed'] < declaration['training']['minimum_experience']:
                raise ValueError('too few verified chat replies to learn from')
        else:
            experience, reply['demonstrations'] = demonstrations(tokenizer, routes['documents'], home, experience_split)
        reply['phases']['demonstrate'] = time.monotonic() - begun
        begun = time.monotonic()
        parent = load_parent()
        replay, reply['chat_replay'] = chat_replay(parent, tokenizer, routes['chat'], parameters, home, replay_split,
                                                   declaration['training']['seed'])
        del parent
        growth.release_accelerator()
        reply['phases']['replay'] = time.monotonic() - begun
        if reply['chat_replay']['passed'] < declaration['training']['minimum_replay']:
            raise ValueError('too few verified chat replies to replay')
        begun = time.monotonic()
        model = load_parent()
        trained, receipt = trainer.train(model, 'addition', experience, replay, settings, device=parameters['device'])
        manifest = trainer.checkpoint(home / CHECKPOINT, trained, receipt,
                                      {'experience': identity([r['sha256'] for r in experience]),
                                       'replay': identity([r['sha256'] for r in replay]),
                                       'declaration': sha256(ROOT / declaration_path)})
        reply['training'] = {**manifest, 'tokens_processed': receipt['tokens_processed'],
                             'final_losses': receipt['losses'][-8:]}
        del model, trained
        growth.release_accelerator()
        reply['phases']['train'] = time.monotonic() - begun
        begun = time.monotonic()

        def with_l4():
            model = load_parent()
            trainer.load_trainable(model, 'addition', settings, home / CHECKPOINT)
            return model.eval()

        reply['integration'] = integrate({'parent': load_parent, 'L4': with_l4}, tokenizer, declaration,
                                         parameters, home)
        reply['phases']['integrate'] = time.monotonic() - begun
        if file_state(directory, inventory) != state:
            raise ValueError('parent checkpoint changed during L4')
        reply['execution_completed'] = True
    except Exception as error:
        reply['error'] = f'{type(error).__name__}: {error}'
    finally:
        reply.update(checked_encodes=tokenizer.checked_encodes if tokenizer else 0,
                     wall_seconds=time.monotonic() - started, process_cpu_seconds=time.process_time(),
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                     peak_gpu_bytes=torch.cuda.max_memory_allocated() if 'torch' in sys.modules
                     and torch.cuda.is_available() else 0)
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
        reply = launch(home, models, binding, 'baseline', 'l4', execution['worker_seconds'],
                       execution['memory_bytes'], worker_script=SCRIPT, environment=execution['worker_environment'])
        result['reply'] = {k: v for k, v in reply.items() if k != 'integration_episodes'}
        if not reply['execution_completed']:
            raise ValueError(reply.get('error', 'incomplete L4 execution'))
        result['execution_completed'] = True
    except Exception as error:
        result['error'] = str(error)
    save(home / 'result.json', result, exclusive=True)
    return result
