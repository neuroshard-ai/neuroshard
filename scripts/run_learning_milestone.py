#!/usr/bin/env python3
"""Prepare a Git-sealed learning experiment, then train and score one fixed recipe.

This research driver has no ledger, network-upgrade or token-payment interface.
It preserves worker databases, coordinator journals and every accepted checkpoint.
"""
import argparse
import copy
import ctypes
import fcntl
import hashlib
import json
import math
import os
import platform
import secrets
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from neuroshard.dataflow.store import canonical
from neuroshard.evolution import milestone
from neuroshard.evolution.objects import Objects, digest


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    with temporary.open('wb') as output:
        output.write(canonical(value))
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def emit(phase, **fields):
    print(json.dumps({'phase': phase, **fields}), flush=True)


def home_path(value):
    path = Path(value).absolute()
    root = milestone.PLAN_PATH.parents[2]/'.neuroshard'
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Use an experiment home under this checkout’s .neuroshard directory')
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def clean_source():
    repo = milestone.PLAN_PATH.parents[2]
    paths = ['src/neuroshard', 'scripts/run_learning_milestone.py',
             'docs/evolution-requirements.txt', 'docs/llm-requirements.txt', 'pyproject.toml',
             'config/experiments/learning-milestone.json']
    dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--', *paths])
    if dirty:
        raise ValueError('Commit the implementation and frozen plan before preparing data')
    return subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()


def runtime_profile():
    from importlib.metadata import version
    expected = {'torch': '2.9.1+cpu', 'numpy': '2.2.6', 'transformers': '4.57.3',
                'tokenizers': '0.22.1', 'safetensors': '0.7.0', 'huggingface-hub': '0.36.0',
                'jinja2': '3.1.6', 'pyarrow': '25.0.1'}
    if platform.system() != 'Linux' or platform.machine() != 'x86_64' or not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise ValueError('Use the published Linux x86_64, Python 3.10–3.12 numerical profile')
    actual = {name: version(name) for name in expected}
    if actual != expected:
        raise ValueError('Installed numerical or data dependencies differ from the frozen profile')
    return {'libraries': actual, 'cpu_dispatch': 'DEFAULT', 'mkl_instructions': 'SSE4_2', 'threads': 1}


def collect_to(corpus, source, end):
    while True:
        cursor = corpus.db.execute('SELECT cursor FROM sources WHERE id=?', (source,)).fetchone()[0]
        if cursor >= end:
            if cursor != end:
                raise ValueError('Collector advanced beyond its frozen scan limit')
            return
        result = corpus.collect(source, min(256, end-cursor))
        emit('preparing_source', source=source, cursor=result['end'], rejected=result['rejected'])
        if result['end'] == cursor:
            raise ValueError('Pinned source exhausted before the frozen scan end')


def select_documents(corpus, plan):
    """Select by public identity only; protect held-out windows before training."""
    selected, seen_tokens = {}, set()
    quotas = {'test': 64, 'retention': 64, 'fresh': 64, 'train': 256}
    rejected = {'incomplete': 0, 'repeated_tokens': 0}
    for role, count in quotas.items():
        selected[role] = []
        rows = corpus.db.execute('SELECT id,source,row,object FROM documents WHERE role=? ORDER BY id',
                                 (role,)).fetchall()
        for identity, source, position, object_root in rows:
            document = corpus.store.json(object_root)
            prepared = corpus.codec.response_windows(document['messages'], plan['data']['context_tokens'],
                                                     plan['data']['response_tokens'], plan['data']['max_windows'])
            if prepared['truncated'] or not prepared['windows']:
                rejected['incomplete'] += 1
                continue
            roots = [corpus.store.put_json({'document': identity, **window, 'role': role, 'source': source})
                     for window in prepared['windows']]
            token_roots = [digest(canonical({'input_ids': [w['tokens']], 'labels': [w['labels']]}))
                           for w in prepared['windows']]
            if len(set(token_roots)) != len(token_roots) or seen_tokens.intersection(token_roots):
                rejected['repeated_tokens'] += 1
                continue
            seen_tokens.update(token_roots)
            selected[role].append({'id': identity, 'source': source, 'row': position,
                                   'object': object_root, 'windows': roots, 'omitted_targets': 0})
            if len(selected[role]) == count:
                break
        if len(selected[role]) != count:
            raise ValueError(f'Frozen scan cannot fill {role}: {len(selected[role])}/{count}; do not lower n')
    return selected, rejected


def prepare(args):
    from transformers import AutoTokenizer
    from neuroshard.evolution.data import TextCorpus
    from neuroshard.evolution.model import from_pretrained
    from neuroshard.evolution.text import TextCodec, bind_model
    plan = milestone.load()
    profile = runtime_profile()
    reconstruct = args.command == 'reconstruct'
    committed = None
    if reconstruct:
        if not milestone.committed_selection(plan):
            raise ValueError('Reconstruction requires the committed selection')
        committed = json.loads(milestone.selection_path(plan).read_bytes())
        identity = {k: committed[k] for k in ('plan_commit', 'plan_digest', 'implementation_digest')}
    else:
        if plan['status'] != 'plan-frozen' or milestone.selection_path(plan).exists():
            raise ValueError('Preparation requires a frozen plan with no existing selection')
        commit = clean_source()
        identity = {'plan_commit': commit, 'plan_digest': digest(milestone.PLAN_PATH.read_bytes()),
                    'implementation_digest': milestone.implementation_digest()}
    home = home_path(args.home)
    if (home/'run.json').exists():
        raise ValueError('Do not prepare data over a started training run')
    plan_digest = identity['plan_digest']
    marker = home/'preparation.json'
    if marker.exists() and json.loads(marker.read_bytes()) != identity:
        raise ValueError('Retained preparation belongs to another plan or implementation')
    save(marker, identity)
    store = Objects(home/'objects')
    initial, model = from_pretrained(args.model_dir, store)
    if initial != plan['seed']['imported_model_root'] or model['parameters'] != plan['seed']['parameters']:
        raise ValueError('Imported seed differs from the frozen plan')
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    codec = TextCodec(tokenizer, store)
    baseline, _ = bind_model(store, initial, codec)
    corpus_home = home/'corpus'
    corpus_home.mkdir(exist_ok=True)
    if args.upstream_cache and not (corpus_home/'upstream').exists():
        (corpus_home/'upstream').symlink_to(args.upstream_cache.resolve(), target_is_directory=True)
    corpus = TextCorpus(corpus_home, store, codec, plan['data']['sequence_length'], plan['data']['max_windows'])
    try:
        source_windows = []
        for name, role in (('heldout', 'heldout'), ('train', 'train')):
            spec = {k: plan['data'][k] for k in ('repo', 'revision', 'license')}
            spec.update(split=plan['learning'][name+'_split'], role=role)
            start = plan['learning'][name+'_start']
            end = start + plan['learning'][name+'_scan_limit']
            source = corpus.register(spec, initial_cursor=start)
            collect_to(corpus, source, end)
            source_windows.append({'source': source, 'spec': spec, 'start': start, 'end': end})
        documents, rejected = select_documents(corpus, plan)
    finally:
        corpus.db.close()
    # Cover all 256 selected training documents once, choosing one complete
    # response window per document by a fixed public hash, before any model run.
    ordered = sorted(documents['train'], key=lambda d: digest(canonical([plan_digest, 'document', d['id']])))
    schedule = [min(d['windows'], key=lambda w: digest(canonical([plan_digest, 'window', w]))) for d in ordered]
    prompts = []
    for document in sorted(documents['test'], key=lambda d: d['id'])[:plan['evaluation']['generation_prompts']]:
        messages = store.json(document['object'])['messages']
        prompt = next(m['content'] for m in reversed(messages) if m['role'] == 'user')
        rendered = codec._chat(codec.messages([{'role': 'user', 'content': prompt}]), True)
        # The existing generation executor accepts 192 input tokens. The
        # original prompt and any explicit tail truncation are sealed together.
        prompts.append({'document': document['id'], 'prompt': prompt, 'input_ids': rendered[-192:],
                        'original_tokens': len(rendered), 'truncated': len(rendered) > 192,
                        'input_policy': 'chat-template-then-last-192-tokens-v1'})
    selection = {'format': 'neuroshard-learning-milestone-selection-v1', **identity,
                 'runtime_profile': profile,
                 'baseline': baseline, 'imported_model_root': initial, 'tokenizer_root': codec.root,
                 'source_windows': source_windows, 'documents': documents, 'generation_prompts': prompts,
                 'training_batches': [schedule[i:i+2] for i in range(0, len(schedule), 2)],
                 'selection_rejections': rejected,
                 'schedule_policy': 'one hash-selected window per training document, in hash order'}
    milestone.validate_selection(selection, plan)
    if reconstruct:
        if selection != committed:
            raise ValueError('Reconstructed public selection differs from the committed bytes')
    else:
        save(milestone.selection_path(plan), selection)
    save(home/'selection.json', selection)
    emit('selection_reconstructed' if reconstruct else 'selection_prepared',
         path=str(milestone.selection_path(plan)), sha256=digest(canonical(selection)),
         counts={r: len(d) for r, d in documents.items()},
         truncated_generation_prompts=sum(p['truncated'] for p in prompts),
         training='forbidden until this selection and selection-committed plan status are committed')


def verify_artifacts(store, selection, plan):
    from neuroshard.evolution.data import normalized
    from neuroshard.evolution.text import TextCodec
    codec = TextCodec.load(store, selection['tokenizer_root'])
    model = store.json(selection['baseline'])
    codec.check_model(model)
    if model['parameters'] != plan['seed']['parameters'] or model['parent'] != selection['imported_model_root']:
        raise ValueError('Baseline is not the text-bound frozen seed')
    for role, documents in selection['documents'].items():
        for item in documents:
            doc = store.json(item['object'])
            if (doc['id'] != item['id'] or doc['row'] != item['row'] or doc['source'] != item['source'] or
                    doc['license'] != plan['data']['license'] or
                    digest(normalized('\n'.join(m['content'] for m in doc['messages'])).encode()) != item['id']):
                raise ValueError('Document identity differs from its sealed provenance')
            prepared = codec.response_windows(doc['messages'], plan['data']['context_tokens'],
                                              plan['data']['response_tokens'], plan['data']['max_windows'])
            expected = [{'document': item['id'], **w, 'role': role, 'source': item['source']}
                        for w in prepared['windows']]
            if prepared['truncated'] or [digest(canonical(w)) for w in expected] != item['windows']:
                raise ValueError('Sealed windows differ from complete re-tokenization')
            for key in item['windows']:
                store.get(key)
    by_id = {d['id']: d for d in selection['documents']['test']}
    for probe in selection['generation_prompts']:
        doc = store.json(by_id[probe['document']]['object'])
        prompt = next(m['content'] for m in reversed(doc['messages']) if m['role'] == 'user')
        rendered = codec._chat(codec.messages([{'role': 'user', 'content': prompt}]), True)
        if probe != {'document': probe['document'], 'prompt': prompt, 'input_ids': rendered[-192:],
                     'original_tokens': len(rendered), 'truncated': len(rendered) > 192,
                     'input_policy': 'chat-template-then-last-192-tokens-v1'}:
            raise ValueError('Generation probe differs from its sealed test document')
    return codec


class Budget:
    def __init__(self, home, state, plan, clock=time.time):
        self.home, self.state, self.plan, self.clock = home, state, plan, clock

    def check(self):
        if self.clock() >= self.state['deadline']:
            raise ValueError('The original wall-clock budget is exhausted')
        used = sum(p.stat().st_size for p in self.home.rglob('*') if p.is_file())
        if used >= self.plan['budget']['disk_gib']*1024**3:
            raise ValueError('Experiment disk budget is exhausted')
        if shutil.disk_usage(self.home).free < 2*1024**3:
            raise ValueError('The filesystem cannot safely complete another operation')
        self.state['peak_disk_bytes'] = max(self.state.get('peak_disk_bytes', 0), used)


class Workers:
    def __init__(self, home, store, base_port):
        self.home, self.store, self.base_port = home, store, base_port
        self.processes, self.endpoints = [], []
        self.homes = [home/f'worker{i}' for i in range(3)]
        self.transferred = {'sent_object_bytes': 0, 'received_object_bytes': 0}

    def start(self):
        from neuroshard.evolution.transport import Endpoint
        if not 1024 <= self.base_port <= 65533:
            raise ValueError('Choose three unprivileged worker ports')
        for i, home in enumerate(self.homes):
            port = self.base_port+i
            with socket.socket() as probe:
                if probe.connect_ex(('127.0.0.1', port)) == 0:
                    raise ValueError('Worker port already has a listener')
            home.mkdir(exist_ok=True, mode=0o700)
            objects = home/'objects'
            if objects.exists():
                if objects.resolve() != self.store.root.resolve():
                    raise ValueError('Retained worker uses another object store')
            else:
                objects.symlink_to(self.store.root.resolve(), target_is_directory=True)
            token_path = home/'rpc.token'
            if not token_path.exists():
                fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, 'w') as f:
                    f.write(secrets.token_hex(32)+'\n')
            command = [sys.executable, str(Path(__file__).resolve()), '_worker', '--home', str(home),
                       '--port', str(port), '--parent-pid', str(os.getpid())]
            with (home/'worker.log').open('ab') as log:
                process = subprocess.Popen(command, stdout=log, stderr=log)
            self.processes.append(process)
            deadline = time.monotonic()+60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise ValueError('Worker exited during startup; inspect its retained log')
                with socket.socket() as probe:
                    if probe.connect_ex(('127.0.0.1', port)) == 0:
                        break
                time.sleep(.1)
            else:
                raise ValueError('Worker startup timed out')
            self.endpoints.append(Endpoint(f'http://127.0.0.1:{port}', token_path.read_text().strip(), self.store))
        return self

    def close(self):
        for endpoint in self.endpoints:
            self.transferred['sent_object_bytes'] += endpoint.sent
            self.transferred['received_object_bytes'] += endpoint.received
            self.store.fetchers.remove(endpoint.fetch)
            endpoint.http.close()
        self.endpoints.clear()
        for process in reversed(self.processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        self.processes.clear()


def recover_record(store, homes, session, step, clip_norm):
    """Recover an accepted step even if the coordinator died before logging it."""
    from neuroshard.evolution.pipeline import validate_record
    from neuroshard.evolution.verification import work_identity
    results = []
    for home in homes:
        db = sqlite3.connect(f'file:{home}/worker.sqlite?mode=ro', uri=True)
        try:
            rows = db.execute('SELECT result FROM operations WHERE session=? AND step=? AND position=3',
                              (session, step)).fetchall()
        finally:
            db.close()
        if len(rows) != 1 or rows[0][0] is None:
            raise ValueError('Accepted coordinator step lacks a durable worker receipt')
        results.append(json.loads(rows[0][0]))
    traces = [store.json(r['trace']) for r in results]
    parent = traces[0]['parent']
    model = copy.deepcopy(store.json(parent))
    model['parent'] = parent
    for result in results:
        model['components'].update(result['components'])
    record = {'parent': parent, 'model_root': store.put_json(model), 'batch': traces[0]['input'], 'step': step,
              'traces': [r['trace'] for r in results], 'norms': [t['norm_squared_hex'] for t in traces],
              'clip_norm_hex': float(clip_norm).hex(), 'learning_rate_hex': traces[0]['learning_rate_hex'],
              'scale_hex': traces[0]['scale_hex'], 'loss_hex': traces[0]['loss_hex']}
    root = store.put_json(record)
    validate_record(store, root)
    return {**record, 'record_root': root, 'work_identity': work_identity(store, root)}


def reconcile_steps(store, workers, selection, state, clip_norm):
    from neuroshard.evolution.batches import from_windows
    journal = json.loads((workers.home/'coordinator.json').read_bytes())
    while len(state['steps']) < journal['step']:
        step = len(state['steps'])
        record = recover_record(store, workers.homes, state['session'], step, clip_norm)
        batch = from_windows(store, selection['training_batches'][step], selection['tokenizer_root'])
        expected_parent = state['steps'][-1]['model_root'] if state['steps'] else selection['baseline']
        if record['parent'] != expected_parent or record['batch'] != digest(canonical(batch)):
            raise ValueError('Recovered step differs from its sealed batch or parent')
        if record['work_identity'] in {s['work_identity'] for s in state['steps']}:
            raise ValueError('Duplicate numerical work identity')
        state['steps'].append(record)
    expected = state['steps'][-1]['model_root'] if state['steps'] else selection['baseline']
    if len(state['steps']) != journal['step'] or expected != journal['model_root']:
        raise ValueError('Receipt history and coordinator journal disagree')


def score(pipe, store, documents, measurements, budget, persist):
    """Persist each complete window result; aggregate one observation per document."""
    from neuroshard.evolution.batches import from_windows
    for document in documents:
        for key in document['windows']:
            if key in measurements:
                continue
            budget.check()
            started = time.monotonic()
            result = pipe.evaluate(from_windows(store, [key], pipe.model['tokenizer_root']))
            value = float.fromhex(result['loss_hex'])
            if not math.isfinite(value):
                raise ValueError('Nonfinite sealed evaluation loss')
            measurements[key] = {'loss_hex': result['loss_hex'], 'seconds': time.monotonic()-started}
            persist()
    values = []
    for document in documents:
        weighted, counts = [], []
        for key in document['windows']:
            count = sum(x != -100 for x in store.json(key)['labels'][1:])
            weighted.append(float.fromhex(measurements[key]['loss_hex'])*count)
            counts.append(count)
        values.append(math.fsum(weighted)/sum(counts))
    return values


def run(args):
    import requests
    from neuroshard.evolution.batches import from_windows
    from neuroshard.evolution.pipeline import Pipeline
    plan = milestone.load()
    if not milestone.training_allowed(plan):
        raise ValueError('Training requires the exact Git-committed selection and selection-committed status')
    home = home_path(args.home)
    selection_raw = milestone.selection_path(plan).read_bytes()
    selection = json.loads(selection_raw)
    if runtime_profile() != selection['runtime_profile']:
        raise ValueError('Runtime differs from the sealed preparation profile')
    store = Objects(home/'objects')
    state_path = home/'run.json'
    identity = {'selection_sha256': digest(selection_raw), 'implementation_digest': milestone.implementation_digest()}
    if state_path.exists():
        state = json.loads(state_path.read_bytes())
        if any(state[k] != v for k, v in identity.items()):
            raise ValueError('Retained run belongs to another selection or implementation')
        if state['status'] == 'failed':
            raise ValueError('This run has failed; its budget and sealed set cannot be reset')
        if state['status'] == 'complete':
            if not (home/'result.json').exists():
                raise ValueError('Completed run has lost its durable result')
            emit('already_complete', result=str(home/'result.json'), decision=state['decision'])
            return
    else:
        now = time.time()
        state = {**identity, 'status': 'running', 'started': now,
                 'deadline': now+plan['budget']['wall_clock_hours']*3600,
                 'session': 'learning-'+digest(selection_raw)[:24], 'steps': [], 'retries': [],
                 'measurements': {}, 'generations': {}, 'peak_disk_bytes': 0}
        save(state_path, state)
    budget = Budget(home, state, plan)
    workers = Workers(home, store, args.base_port)
    def persist():
        save(state_path, state)
    # SIGALRM makes the wall-clock deadline apply inside a slow RPC as well.
    def timeout(*_):
        raise ValueError('The original wall-clock budget is exhausted')
    old_alarm = signal.signal(signal.SIGALRM, timeout)
    remaining = state['deadline']-time.time()
    if remaining > 0:
        signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        budget.check()
        codec = verify_artifacts(store, selection, plan)
        workers.start()
        pipe = Pipeline(store, selection['baseline'], workers.endpoints, [plan['budget']['worker_capacity']]*3,
                        state['session'], plan['optimizer']['learning_rate'], plan['optimizer']['clip_norm'],
                        journal=home/'coordinator.json')
        pipe.persist()
        reconcile_steps(store, workers, selection, state, plan['optimizer']['clip_norm'])
        persist()
        for step in range(pipe.step, plan['budget']['training_steps']):
            budget.check()
            batch = from_windows(store, selection['training_batches'][step], selection['tokenizer_root'])
            started = time.monotonic()
            result = pipe.train(batch)
            reconcile_steps(store, workers, selection, state, plan['optimizer']['clip_norm'])
            state['steps'][-1]['observed_seconds'] = time.monotonic()-started
            persist()
            emit('training', step=pipe.step, loss=float.fromhex(result['loss_hex']), model_root=pipe.model_root)
        candidate = pipe.model_root
        pipe.close()
        # Read the selected candidate only after the complete fixed recipe.
        values = {}
        for side, root in (('baseline', selection['baseline']), ('candidate', candidate)):
            pipe = Pipeline(store, root, workers.endpoints, [plan['budget']['worker_capacity']]*3,
                            state['session']+'-score-'+side)
            values[side] = {}
            for role in ('retention', 'fresh', 'test'):
                cached = state['measurements'].setdefault(side, {}).setdefault(role, {})
                values[side][role] = score(pipe, store, selection['documents'][role], cached, budget, persist)
                emit('scored', side=side, role=role, documents=len(values[side][role]))
            generations = state['generations'].setdefault(side, [])
            for index in range(len(generations), len(selection['generation_prompts'])):
                budget.check()
                probe = selection['generation_prompts'][index]
                started = time.monotonic()
                output = pipe.generate(probe['input_ids'], plan['evaluation']['generation_max_tokens'],
                                       (codec.tokenizer.eos_token_id,))
                generations.append({**probe, **output, 'text': codec.decode(output['token_ids']),
                                    'seconds': time.monotonic()-started})
                persist()
                emit('generation', side=side, index=index+1, output_tokens=len(output['token_ids']))
            pipe.close()
        decision = milestone.decide_learning(values, plan)
        budget.check()
        state.update(status='complete', candidate=candidate, decision=decision, finished=time.time())
        result = {**identity, 'format': 'neuroshard-learning-milestone-result-v1', 'phase': 'learning',
                  'baseline': selection['baseline'], 'candidate': candidate, 'decision': decision,
                  'document_losses': values, 'generations': state['generations'], 'training_steps': len(state['steps']),
                  'trained_windows': [w for batch in selection['training_batches'] for w in batch],
                  'steps': state['steps'], 'elapsed_seconds': state['finished']-state['started'],
                  'retries': state['retries'], 'peak_disk_bytes': state['peak_disk_bytes'],
                  'hosts': 1, 'worker_processes': 3, 'shared_local_artifact_store': True,
                  'runtime_profile': selection['runtime_profile'], 'python_version': platform.python_version(),
                  'source_windows': selection['source_windows'],
                  'token_issuance': 0, 'public_network_changed': False}
        save(home/'result.json', result)
        persist()
        emit('complete', result=str(home/'result.json'), decision=decision)
    except (requests.RequestException, ConnectionError, TimeoutError) as exc:
        state['retries'].append({'time': time.time(), 'type': type(exc).__name__,
                                 'message': str(exc), 'accepted_steps': len(state['steps'])})
        if len(state['retries']) >= 3:
            state.update(status='failed', failure={'type': type(exc).__name__, 'message': str(exc)}, finished=time.time())
            persist()
            raise
        persist()
        workers.close()
        emit('recovering_transport', attempts=len(state['retries']), accepted_steps=len(state['steps']))
        time.sleep(2)
        run(args)
    except (KeyboardInterrupt, SystemExit):
        # A deliberate stop preserves the original deadline and every journal.
        persist()
        raise
    except Exception as exc:
        state.update(status='failed', failure={'type': type(exc).__name__, 'message': str(exc)}, finished=time.time())
        persist()
        emit('failed', reason=str(exc), retained_home=str(home))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        workers.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='Select and reconstruct public data; never train')
    prep.add_argument('--home', type=Path, required=True)
    prep.add_argument('--model-dir', type=Path, required=True)
    prep.add_argument('--upstream-cache', type=Path)
    rebuild = commands.add_parser('reconstruct', help='Rebuild the committed inputs from pinned public sources')
    rebuild.add_argument('--home', type=Path, required=True)
    rebuild.add_argument('--model-dir', type=Path, required=True)
    rebuild.add_argument('--upstream-cache', type=Path)
    execute = commands.add_parser('run', help='Train, score and publish all outputs in the retained experiment home')
    execute.add_argument('--home', type=Path, required=True)
    execute.add_argument('--base-port', type=int, default=55301)
    worker = commands.add_parser('_worker', help=argparse.SUPPRESS)
    worker.add_argument('--home', type=Path, required=True)
    worker.add_argument('--port', type=int, required=True)
    worker.add_argument('--parent-pid', type=int, required=True)
    args = parser.parse_args()
    if args.command == '_worker':
        from neuroshard.evolution.transport import serve
        if ctypes.CDLL(None).prctl(1, signal.SIGTERM) != 0 or os.getppid() != args.parent_pid:
            raise SystemExit('Worker parent is no longer present')
        serve(args.home, args.port, args.home/'rpc.token', 48_000_000)
        return
    home = home_path(args.home)
    with (home/'driver.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another driver owns this experiment home') from None
        (prepare if args.command in ('prepare', 'reconstruct') else run)(args)


if __name__ == '__main__':
    main()
