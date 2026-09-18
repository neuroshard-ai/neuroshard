#!/usr/bin/env python3
"""Installed preparation and numerical backend for the native ordinary trial.

The existing publisher and audit daemons own their separate signing journals.
This backend launches actual partition work and ordinary answering; it cannot
approve a job, settle a claim, mint tokens or move the serving root itself.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

from neuroshard.dataflow.store import canonical, LocalStore
from neuroshard.demo import protocol
from neuroshard.evolution import auditing, expert_admission, expert_data, expert_lifecycle as life
from neuroshard.evolution import expert_work, ordinary_cohorts, ordinary_operation, expert_source
from neuroshard.evolution.committed_state import read as committed_state
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.reference_data import identity, save
from ordinary_cloud import Cloud, REMOTE, REPO, PUBLIC

ROOT = Path(__file__).resolve().parents[1]


class Backend:
    def __init__(self, home, actor=0):
        self.home, self.actor = Path(home).resolve(), actor
        if type(actor) is not int or actor not in range(4):
            raise ValueError('Use one configured publisher or audit role')
        self.cloud = Cloud(home)
        self.freeze = json.loads((self.home/'operation.json').read_bytes())
        self.store = Objects(self.home/'compiled/objects')
        self.profile = self.store.json(self.freeze['executor'])
        self.transport = LocalStore(self.home/'feed-objects')
        self.feed = expert_source.Feed(self.transport, self.freeze['feed_head'])
        self.public_feed = LocalStore(self.home/'public-feed-cache')
        if any((self.home/'metadata-publications').glob('*.json')):
            discovery = json.loads((self.home/'feed-head.json').read_bytes())
            self.advance_feed(discovery['entry'])
        self.jobs = self.home/'jobs'
        self.jobs.mkdir(exist_ok=True)
        self.keys = [protocol.Identity.load_or_create(self.home/'keys'/('prefix-'+str(rank)+'.key')) for rank in range(3)]
        self.training_key = protocol.Identity.load_or_create(self.home/'keys/training.key')
        self.preparation = None

    def publish_metadata(self):
        """Mirror small protocol/source objects and verify their public bytes."""
        journal = self.home/'metadata-publications'
        journal.mkdir(exist_ok=True)
        known = set()
        for path in journal.glob('*.json'):
            known.update(json.loads(path.read_bytes())['objects'])
        values = {}
        for paths in (self.store.root.glob('*/*'), self.transport.root.glob('*')):
            for path in paths:
                if len(path.name) == 64 and path.name not in known:
                    raw = path.read_bytes()
                    from neuroshard.evolution.objects import digest
                    if digest(raw) != path.name:
                        raise ValueError('Local metadata changed before publication')
                    values[path.name] = raw
        if not values:
            return
        self.cloud.bundle(3, {'metadata/'+key: raw for key, raw in values.items()})
        inventory = {key: {'bytes': len(raw), 'path': REMOTE+'/metadata/'+key} for key, raw in values.items()}
        self.publish(3, inventory, journal/(identity(sorted(values))+'.json'))

    def advance_feed(self, entry):
        """Read the immutable published head/window chain into a fresh cache."""
        from neuroshard.evolution.sharded.retained_objects import restore
        if type(entry) is not int or not 0 <= entry < len(self.freeze['feed_heads']):
            raise ValueError('Unknown prospectively frozen source discovery step')
        required = set()
        for key in self.freeze['feed_heads'][:entry+1]:
            required.add(key)
            manifest = json.loads(self.transport.get(key))
            for window_key in manifest['windows']:
                required.add(window_key)
                descriptor = json.loads(self.transport.get(window_key))
                required.add(descriptor['records']['sha256'])
        def fetch(key):
            raw = self.transport.get(key)
            return restore(key, len(raw), [PUBLIC+key], self.public_feed.path(key), max_seconds=180)
        with ThreadPoolExecutor(max_workers=4) as pool:
            observations = list(pool.map(fetch, sorted(required)))
        self.feed = expert_source.Feed(self.public_feed, self.freeze['feed_heads'][0])
        for key in self.freeze['feed_heads'][1:entry+1]:
            self.feed.advance(key)
        save(self.home/'feed-head.json', {'head': self.feed.head, 'entry': entry,
            'public_objects_checked': len(observations), 'bytes': sum(row['bytes'] for row in observations)})
        if getattr(self, 'preparation', None) is not None:
            self.preparation.feed = self.feed

    def state(self):
        config = json.loads((self.home/'native.json').read_bytes())
        return committed_state(config['home'], config['genesis_sha256'])

    def preparer(self):
        if self.preparation is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.home/'compiled/seed', local_files_only=True)
            self.preparation = ordinary_operation.Preparation(self.home/'preparation', self.freeze, self.store,
                tokenizer, self.feed, self.initialize)
        return self.preparation

    def initialize(self, parent, plan, prepared, directory):
        key = identity(prepared)
        folder = REMOTE+'/initialization/'+key
        values = {'parent.json': parent, 'plan.json': plan, 'prepared.json': prepared}
        self.cloud.bundle(3, {'initialization/'+key+'/'+name: value for name, value in values.items()})
        config = {'action': 'initialize', **{name: folder+'/'+name+'.json' for name in ('parent', 'plan', 'prepared')},
            'paths': {'inputs': folder, 'objects': REMOTE+'/objects', 'checkpoint_store': folder+'/checkpoints'}}
        result = self.cloud.execute(3, config)
        save(directory/'initialization.json', result)
        initial = result['result']
        self.publish_checkpoint(3, initial, folder+'/checkpoints', directory/'initial-publication.json')
        return initial

    def context(self, job, *, branch=None):
        key = identity(job) if branch is None else identity({'job': identity(job), 'branch': branch})
        directory = self.jobs/key
        directory.mkdir(exist_ok=True)
        prepared = self.store.json(job['work']['prepared'])
        plan = self.store.json(prepared['plan'])
        values = {'job.json': job, 'work.json': job['work'], 'prepared.json': prepared,
                  'plan.json': plan, 'parent.json': job['work']['parent']}
        rows = {role+'.jsonl': self.store.get(spec['sha256']) for role, spec in prepared['roles'].items()}
        marker = directory/'installed.json'
        expected = {'job': key, 'inputs': identity(prepared)}
        # Quality auditors share one immutable context. Serialize its first
        # installation and leave existing readers' files untouched on retries.
        with (directory/'installation.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if marker.exists():
                if (json.loads(marker.read_bytes()) != expected
                        or any((directory/name).read_bytes() != canonical(value)
                               for name, value in values.items())
                        or any((directory/name).read_bytes() != raw for name, raw in rows.items())):
                    raise ValueError('Installed immutable job context differs from its commitment')
            else:
                for name, value in values.items():
                    save(directory/name, value)
                for name, raw in rows.items():
                    with (directory/name).open('wb') as output:
                        output.write(raw)
                        output.flush()
                        os.fsync(output.fileno())
                files = {'jobs/'+key+'/'+name: value for name, value in {**values, **rows}.items()}
                with ThreadPoolExecutor(max_workers=7) as pool:
                    list(pool.map(lambda rank: self.cloud.bundle(rank, files), range(7)))
                save(marker, expected)
        return {'key': key, 'directory': directory, 'remote': REMOTE+'/jobs/'+key,
                'job': job, 'prepared': prepared, 'plan': plan}

    def publish(self, physical, inventory, destination):
        request = {'objects': {key: {'bytes': spec['bytes']} for key, spec in inventory.items()}}
        signed = subprocess.run(['sudo', sys.executable, str(ROOT/'scripts/sign_ordinary_objects.py')],
            input=canonical(request), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=90)
        capabilities = json.loads(signed.stdout)
        value = self.cloud.assets(physical, {'action': 'publish', 'objects': {
            key: {**capabilities[key], 'path': spec['path']} for key, spec in inventory.items()}})
        save(destination, value)
        if value['all_hashes_verified'] is not True:
            raise ValueError('Preserve every boundary before using its publication receipt')
        return value

    def publish_checkpoint(self, physical, checkpoint, directory, destination):
        inventory = {spec['sha256']: {'bytes': spec['bytes'], 'path': directory+'/'+checkpoint['checkpoint']
            +'/shard-'+str(checkpoint['step']).zfill(6)+'/'+spec['sha256']+'.safetensors'}
            for spec in checkpoint['tensors'].values()}
        return self.publish(physical, inventory, destination)

    def prefix(self, ctx, claim=None):
        from neuroshard.evolution.sharded import expert_execution, prefix_execution
        actor = self.actor
        invocation = uuid.uuid4().hex
        base = ctx['remote']+'/actor-'+str(actor)
        production = base+'/prefix-'+invocation
        reports, incoming, transfer = [], None, []
        for rank in range(3):
            destination = production+'/rank-'+str(rank)
            config = {'action': 'prefix_stage', 'rank': rank, 'profile': ctx['remote']+'/work.json',
                'plan': ctx['remote']+'/plan.json', 'prepared': ctx['remote']+'/prepared.json',
                'paths': {'inputs': ctx['remote'], 'objects': REMOTE+'/objects', 'checkpoint_store': base+'/checkpoints'},
                'home': destination, 'incoming': incoming, 'max_seconds': 1800}
            result = self.cloud.execute(rank, config)
            reports.append(result['result'])
            if rank < 2:
                target = production+'/incoming-'+str(rank)
                transfer.append(self.cloud.copy_directory(rank, rank+1, destination+'/features', target+'/features'))
                self.cloud.put(rank+1, target+'/report.json', reports[-1])
                incoming = {'features': target+'/features', 'report': target+'/report.json'}
        bank_path = production+'/rank-2/features'
        bank = json.loads(self.cloud.read(2, bank_path+'/index.json'))
        records = expert_execution.training_records(ctx['plan'], ctx['prepared'], ctx['directory'], ctx['job']['work']['parent'])
        value = prefix_execution.owned_production(ctx['job']['work'], ctx['plan'], ctx['prepared'], records, reports, bank)
        transfer.append(self.cloud.copy_directory(2, 3+actor, bank_path, base+'/bank'))
        evidence = {'production': value, 'reports': reports, 'bank': bank, 'transfers': transfer}
        save(ctx['directory']/('prefix-'+str(actor)+'.json'), evidence)
        if claim is not None:
            return prefix_execution.owned_verdict(claim, ctx['job']['work'], ctx['plan'], ctx['prepared'], value)
        # Preserve the real producer's feature packets as well as the bank root.
        inventory = {spec['sha256']: {'bytes': spec['bytes'], 'path': bank_path+'/'+spec['file']}
                     for batch in bank['batches'] for spec in batch['files']}
        self.publish(2, inventory, ctx['directory']/'features-publication.json')
        return value

    def training(self, ctx, work, count=None, claim=None):
        from neuroshard.evolution.sharded import expert_execution
        actor, base = self.actor, ctx['remote']+'/actor-'+str(self.actor)
        profile = expert_work.resolve_prefix(ctx['job']['work'], work['feature_root'], work['batch_roots'])
        self.cloud.put(3+actor, base+'/profile.json', profile)
        self.cloud.put(3+actor, base+'/before.json', work['checkpoint'])
        config = {'action': 'training_audit' if claim is not None else 'training',
            'profile': base+'/profile.json', 'before': base+'/before.json',
            'plan': ctx['remote']+'/plan.json', 'prepared': ctx['remote']+'/prepared.json',
            'paths': {'inputs': ctx['remote'], 'objects': REMOTE+'/objects',
                      'checkpoint_store': base+'/checkpoints', 'bank_home': base+'/bank'}, 'max_seconds': 1800}
        if claim is not None:
            config['claim'] = base+'/claim-'+claim['id']+'.json'
            self.cloud.put(3+actor, config['claim'], claim)
        else:
            config['updates'] = count
        result = self.cloud.execute(3+actor, config)
        label = 'audit-'+claim['id'] if claim is not None else 'produce-'+str(work['checkpoint']['step'])
        save(ctx['directory']/(label+'-actor-'+str(actor)+'.json'), result)
        if claim is None:
            checkpoint = result['result']['window']['output']
            self.publish_checkpoint(3+actor, checkpoint, base+'/checkpoints',
                ctx['directory']/('publication-'+str(checkpoint['step'])+'.json'))
        return result['result']

    def assets_for_graph(self, graph):
        placement = self.cloud.placement(graph)
        requests = {}
        for route in graph['descriptor']['rules']:
            physical = placement[route['owner']]
            requests.setdefault(physical, {})
            for spec in graph['experts'][route['id']]['tensors'].values():
                key = spec['sha256']
                requests[physical][key] = {'bytes': spec['bytes'], 'path': REMOTE+'/objects/'+key+'.safetensors'}
        with ThreadPoolExecutor(max_workers=4) as pool:
            return list(pool.map(lambda item: self.cloud.assets(item[0], {'action': 'fetch', 'objects': item[1]}), requests.items()))

    def start_service(self, graph, baseline, quality, slot, label):
        self.assets_for_graph(graph)
        inputs = {spec['file']: self.store.get(spec['sha256']) for spec in quality['roles'].values()}
        key = identity({'graph': identity(graph), 'slot': slot, 'label': label, 'invocation': uuid.uuid4().hex})
        service = self.cloud.service(key, graph, baseline, self.profile, quality, inputs, self.store, slot=slot)
        save(self.home/('service-slot-'+str(slot)+'.json'), service)
        return service

    @contextmanager
    def evaluation_slot(self):
        """Bound concurrent full services while every auditor executes afresh."""
        count = self.freeze['resources']['parallel_evaluations']
        if type(count) is not int or not 1 <= count <= 2:
            raise ValueError('Bound full-model evaluation concurrency')
        acquired = None
        try:
            while acquired is None:
                self.cloud.remaining()
                for index in range(count):
                    lock = (self.home/('evaluation-slot-'+str(index)+'.lock')).open('a')
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        lock.close()
                    else:
                        acquired = lock
                        break
                if acquired is None:
                    time.sleep(1)
            yield
        finally:
            if acquired is not None:
                acquired.close()

    def quality(self, ctx, checkpoint, claim=None):
        with self.evaluation_slot():
            return self.evaluate_quality(ctx, checkpoint, claim)

    def evaluate_quality(self, ctx, checkpoint, claim=None):
        graph = life.materialize_graph(ctx['job']['lifecycle']['candidate_template'], checkpoint)
        quality = self.store.json(ctx['job']['lifecycle']['quality']['policy_root'])
        service = self.start_service(graph, ctx['job']['lifecycle']['serving_graph'], quality,
                                     4+self.actor, 'quality-'+ctx['key']+'-'+str(self.actor))
        try:
            request = {'id': identity({'job': ctx['key'], 'quality': self.actor, 'claim': claim and claim['id']}),
                       'kind': 'quality_audit' if claim else 'evaluate'}
            if claim is not None:
                request['claim'] = claim
            result = self.cloud.query(service, request)
            if result['status'] != 'completed':
                raise ValueError('The complete ordinary quality execution was unavailable')
            save(ctx['directory']/('quality-'+str(self.actor)+'.json'), result)
            return result['report'] if claim is not None else result['result']
        finally:
            self.cloud.stop(service)

    def serving(self, state):
        graph = state['expert_lifecycle']['serving_graph']
        path = self.home/'accepted-service.json'
        with (self.home/'accepted-service.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            previous = json.loads(path.read_bytes()) if path.exists() else None
            if previous and previous['graph'] == identity(graph):
                if all(self.cloud.active(physical, previous['units'][rank])
                       for rank, physical in enumerate(previous['placement'])):
                    return previous
            quality = ordinary_operation.prior_quality(state, self.store)
            slot = 1 if previous and previous['slot'] == 0 else 0
            service = self.start_service(graph, graph, quality, slot, 'accepted')
            save(path, service)
            if previous:
                self.cloud.stop(previous)
            return service

    def probe(self, state, label):
        service = self.serving(state)
        request = {'id': identity({'label': label, 'graph': state['serving_root'], 'id': uuid.uuid4().hex}),
            'kind': 'generate', 'graph': state['serving_root'],
            'question': [{'role': 'user', 'content': 'What is the name of the NeuroShard client package?'}],
            'max_tokens': 64}
        value = self.cloud.query(service, request, timeout=180)
        if value['status'] != 'completed' or value['result']['text'].strip().rstrip('.') != 'neuroshard-ai':
            raise ValueError('The accepted ordinary model did not preserve its live serving probe')
        folder = self.home/'serving-probes'
        folder.mkdir(exist_ok=True)
        save(folder/(request['id']+'.json'), {'label': label, 'height': state['height'],
            'serving_root': state['serving_root'], 'response': value})
        return value

    def prepare(self, request):
        state = request['state']
        sequence = ordinary_operation.outcome_sequence(state, self.freeze['entries'])
        save(self.home/'sequence.json', sequence)
        self.probe(state, 'prepare-next-cohort')
        if sequence['status'] != 'next':
            return {'job': None}
        if sequence['entry'] == 1 and not any(row.get('phase') == 'data_rejected' for row in request['history']):
            spec = self.freeze['sources'][self.freeze['entries'][1]['name']+'/train']
            expected = self.store.json(spec['records'])
            altered = copy.deepcopy(expected)
            altered[0]['messages'][-1]['content'] = 'This substituted answer is not supported by the pinned source.'
            bad = expert_source.publish_window(self.transport, spec['source'], 0, altered)
            head = self.transport.put(canonical({'format': expert_source.FORMAT, 'previous': None, 'windows': [bad]}))
            alternative = expert_source.Feed(self.transport, head)
            try:
                ordinary_cohorts.verify_feed_rows(self.preparer().evidence, expected,
                    list(alternative(spec['source'], 0, len(expected))))
            except ValueError as error:
                review = {'mechanical_checks_passed': False, 'source_correspondence_passed': False,
                          'reason': str(error), 'feed': head, 'window': bad, 'hashes_internally_valid': True}
                save(self.home/'rejected-source.json', review)
                self.publish_metadata()
                return {'rejected_data': {'entry': head, 'review': review}}
            raise ValueError('The installed curator accepted a substituted source target')
        if sequence['entry'] == 1:
            from ordinary_comparison import start_growth
            start_growth(self.home, self.freeze['entries'][1]['name'])
        if sequence['entry'] == 2:
            from ordinary_comparison import run
            comparison = run(self, state)
            if comparison.get('pending') is True:
                return {'job': None}
            if not comparison['passed']:
                save(self.home/'comparison-stop.json', comparison)
                return {'job': None}
        self.advance_feed(sequence['entry'])
        job = self.preparer().prepare(state, self.freeze['entries'][sequence['entry']])
        self.publish_metadata()
        return {'job': job}

    def execute(self, request):
        ctx = self.context(request['job'])
        phase = request['phase']
        # Starting the probe before work establishes an already loaded accepted
        # service. The supervisor also sends probes during long execution.
        self.probe(self.state(), 'before-'+phase)
        if phase == 'prefix':
            produced = self.prefix(ctx)
            output = {name: produced[name] for name in ('feature_root', 'batch_roots')}
            fields = {**output, 'transcript_root': produced['transcript_root'], 'workers': [key.sign(
                expert_work.receipt(request['chain_id'], request['assignment'], output,
                    produced['transcript_root'], rank)) for rank, key in enumerate(self.keys)]}
        elif phase == 'training':
            produced = self.training(ctx, request['work'], request['stages'])
            fields = {**produced, 'worker': self.training_key.sign(expert_work.receipt(request['chain_id'],
                request['assignment'], produced['window']['output']['checkpoint'], identity(produced['window']), 0))}
        elif phase == 'quality':
            from neuroshard.evolution.sharded import graph_quality
            graph = life.materialize_graph(request['job']['lifecycle']['candidate_template'], request['work']['checkpoint'])
            measured = self.quality(ctx, request['work']['checkpoint'])
            report = {'format': life.FORMAT+'/quality', 'policy_root': request['job']['lifecycle']['quality']['policy_root'],
                'baseline_graph': identity(request['job']['lifecycle']['serving_graph']), 'candidate_graph': identity(graph),
                'prepared': request['job']['work']['prepared'], 'passed': measured['decision']['passed'],
                'results_root': identity(measured)}
            claim = {'kind': 'expert_quality', 'graph': graph, 'model_root': identity(graph),
                'executor_root': graph['executor_root'], 'record_root': None, 'report': report,
                'baseline_graph': request['job']['lifecycle']['serving_graph'],
                'stages': request['job']['lifecycle']['quality']['stages']}
            fields = {'report': report, 'transcript_root': identity(graph_quality.quality_transcript(claim, measured))}
        else:
            raise ValueError('Unknown installed publisher phase')
        self.probe(self.state(), 'after-'+phase)
        return fields

    def audit(self, claim):
        state = self.state()
        actual = state['candidate']
        if actual is None or actual['id'] != claim['id'] or auditing.coverage(actual) != auditing.coverage(claim):
            raise ValueError('Replay the actual currently committed claim')
        admission = expert_admission.bookkeeping(state)
        job = admission['active']['job'] if admission['active'] else {
            'work': state['manifest']['expert_work'], 'lifecycle': state['manifest']['expert_lifecycle'], 'data': admission['data']}
        ctx = self.context(job)
        save(ctx['directory']/('claim-'+claim['id']+'-actor-'+str(self.actor)+'.json'), claim)
        if claim['kind'] == 'expert_features':
            return self.prefix(ctx, claim)
        if claim['kind'] == 'expert_training':
            return self.training(ctx, state['expert_work'], claim=claim)
        if claim['kind'] == 'expert_quality':
            return self.quality(ctx, state['expert_work']['checkpoint'], claim)
        if claim['kind'] == 'expert_inference':
            with self.evaluation_slot():
                return self.inference_audit(claim)
        raise ValueError('The installed auditor does not support this claim')

    def inference_audit(self, claim):
        state = self.state()
        quality = ordinary_operation.prior_quality(state, self.store)
        service = self.start_service(claim['graph'], claim['graph'], quality, 4+self.actor,
                                     'inference-audit-'+claim['id'])
        try:
            result = self.cloud.query(service, {'id': claim['id'], 'kind': 'inference_audit', 'claim': claim})
            if result['status'] != 'completed':
                raise ValueError('Ordinary inference replay was unavailable')
            return result['report']
        finally:
            self.cloud.stop(service)

    def review(self, request):
        proposal = request['proposal']
        preparer = self.preparer()
        report = expert_data.review(request['state'], proposal['job'],
            self.store.json(self.freeze['data_policy']), self.store, preparer.tokenizer, preparer.reviewed)
        return {'proposal': proposal['id'], 'job': identity(proposal['job']),
                'approve': True, 'review': {**report, 'source_evidence': self.freeze['source_evidence']}}


def invoke(home, actor, audit, request):
    """Keep failure locations even when the operator suppresses child stderr."""
    try:
        backend = Backend(home, actor)
        if audit:
            return backend.audit(request)
        if request['phase'] == 'prepare':
            return backend.prepare(request)
        if request['phase'] == 'review':
            return backend.review(request)
        return backend.execute(request)
    except Exception as error:
        # Do not copy exception messages, commands, inputs, or credentials.
        evidence = {'actor': actor, 'audit': audit, 'request_root': identity(request),
                    'exception': type(error).__name__,
                    'frames': [{'file': frame.filename, 'line': frame.lineno, 'function': frame.name}
                               for frame in traceback.extract_tb(error.__traceback__)[-16:]]}
        try:
            save(Path(home)/'backend-failures'/(str(time.time_ns())+'-'+str(actor)+'.json'), evidence)
        except OSError:
            pass
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--actor', type=int, required=True)
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    request = json.load(sys.stdin)
    result = invoke(args.home, args.actor, args.audit, request)
    sys.stdout.write(json.dumps(result))
