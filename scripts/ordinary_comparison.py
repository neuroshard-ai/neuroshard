"""Compare isolated addition with fixed capacity under an equal host budget.

The control recomputes the identical learning trajectory and three audits; no
shadow execution is submitted for issuance. Both arms receive the same full
host/disk interval, including verification and a two-replica serving workload.
This measures one declared workload and interval, not optimal lifetime cost.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import copy
import hashlib
import json
from pathlib import Path
import time

from neuroshard.dataflow.store import canonical
from neuroshard.evolution import answering, expert_lifecycle as life, expert_work, ordinary_cohorts
from neuroshard.evolution.reference_data import identity, save
from neuroshard.evolution.sharded import graph_quality


def resources(cloud):
    """Observe OS byte counters and retained disk usage on every actual owner."""
    def one(rank):
        script = '''import json,pathlib,os,subprocess
from datetime import datetime,timezone
folder='/home/ubuntu/native-expert-live'
network={}
for iface in pathlib.Path('/sys/class/net').iterdir():
    if iface.name!='lo':
        network[iface.name]={key:int((iface/'statistics'/key).read_text()) for key in ('rx_bytes','tx_bytes')}
disk=os.statvfs(folder)
print(json.dumps({'time':datetime.now(timezone.utc).isoformat(),'network':network,
    'retained_directory_bytes':int(subprocess.check_output(['du','-sb',folder]).split()[0]),
    'filesystem_used_bytes':(disk.f_blocks-disk.f_bfree)*disk.f_frsize}))
'''
        return str(rank), json.loads(cloud.command(rank, ['python3', '-c', script], timeout=120).stdout)
    with ThreadPoolExecutor(max_workers=7) as pool:
        return dict(pool.map(one, range(7)))


def start_growth(home, job_label):
    from ordinary_cloud import Cloud
    path = Path(home)/'comparison-growth-start.json'
    value = {'label': job_label, 'started': datetime.now(timezone.utc).isoformat()}
    if not path.exists():
        value['resources'] = resources(Cloud(home))
        save(path, value)
    return json.loads(path.read_bytes())


def control_job(backend, original):
    graph = copy.deepcopy(answering.core(original['lifecycle']['serving_graph']))
    initial = original['work']['checkpoint']
    graph['experts']['planner'] = copy.deepcopy(initial)
    graph['descriptor']['previous_graph'] = identity(original['lifecycle']['serving_graph']['descriptor'])
    for expert in graph['descriptor']['experts']:
        if expert['id'] == 'planner':
            expert['checkpoint'] = initial['checkpoint']
    graph = ordinary_cohorts.bind_policy(graph,
        backend.store.json(backend.freeze['comparison']['answering_policy']), backend.store)
    job = copy.deepcopy(original)
    quality = backend.store.json(original['lifecycle']['quality']['policy_root'])
    quality['candidate_template'] = graph
    graph_quality.validate_policy(quality)
    job['lifecycle']['candidate_template'] = graph
    job['lifecycle']['quality']['policy_root'] = backend.store.put_json(quality)
    return job


def benchmark(backend, graph, baseline, quality, until, label):
    """Spend the remaining equal interval on the predeclared prompt workload."""
    fitting = json.loads((backend.home/'compiled/selector-fitting.json').read_bytes())['rows']
    prompts = []
    for route in ('parent', 'directory', 'protocol', 'planner', backend.freeze['comparison']['cohort']):
        prompts.extend(row['question'] for row in sorted(
            [row for row in fitting if row['route'] == route], key=lambda row: row['id'])[:16])
    if len(prompts) != 80:
        raise ValueError('Serve the same eighty training-only workload prompts in both arms')
    services = []
    folder = backend.home/'comparison'/label
    folder.mkdir(parents=True, exist_ok=True)
    try:
        for replica in range(2):
            services.append(backend.start_service(graph, baseline, quality, 8+replica, label+'-benchmark'))

        def serve(replica):
            count, tokens, seconds, response_bytes = 0, 0, 0., 0
            transcript = hashlib.sha256()
            with (folder/('replica-'+str(replica)+'.jsonl')).open('ab') as log:
                while datetime.now(timezone.utc).timestamp() < until:
                    question = prompts[count % len(prompts)]
                    key = identity({'arm': label, 'replica': replica, 'index': count, 'question': question})
                    response = backend.cloud.query(services[replica], {'id': key, 'kind': 'generate',
                        'graph': identity(graph), 'question': [{'role': 'user', 'content': question}], 'max_tokens': 64}, timeout=300)
                    if response['status'] != 'completed':
                        raise ValueError('A benchmark replica failed to answer')
                    value = response['result']
                    raw = canonical(value)
                    transcript.update(raw)
                    generated = value['answering']['generated_tokens']
                    event = {'id': key, 'question': identity(question), 'response': identity(value),
                        'tokens': generated, 'seconds': response['seconds'], 'bytes': len(raw)}
                    log.write(canonical(event)+b'\n')
                    log.flush()
                    if count < len(prompts):
                        save(folder/('sample-'+str(replica)+'-'+str(count)+'.json'), response)
                    count += 1
                    tokens += generated
                    seconds += response['seconds']
                    response_bytes += len(raw)
            finished = datetime.now(timezone.utc).timestamp()
            return {'requests': count, 'tokens': tokens, 'execution_seconds': seconds,
                    'finished': finished, 'last_request_overrun_seconds': max(0., finished-until),
                    'response_bytes': response_bytes, 'transcript_sha256': transcript.hexdigest()}

        with ThreadPoolExecutor(max_workers=2) as pool:
            values = list(pool.map(serve, range(2)))
        return {'replicas': values, 'workload': identity(prompts), 'graph': identity(graph),
                'requests': sum(value['requests'] for value in values), 'tokens': sum(value['tokens'] for value in values)}
    finally:
        for service in services:
            backend.cloud.stop(service)


def run(backend, state):
    from ordinary_campaign_backend import Backend
    home = backend.home/'comparison'
    home.mkdir(exist_ok=True)
    done = home/'result.json'
    if done.exists():
        return json.loads(done.read_bytes())
    native = state['expert_lifecycle']['admission']['active']['job']
    ctx = backend.context(native)
    if life.training_expert(native['lifecycle']['candidate_template']) != backend.freeze['comparison']['cohort']:
        raise ValueError('Compare the declared first full cohort before starting later jobs')
    budget = backend.freeze['comparison']['seconds_per_arm']
    growth_started = datetime.fromisoformat(json.loads((backend.home/'comparison-growth-start.json').read_bytes())['started']).timestamp()
    growth_end = growth_started+budget
    growth_file = home/'growth.json'
    if not growth_file.exists():
        if datetime.now(timezone.utc).timestamp() >= growth_end:
            value = {'passed': False, 'reason': 'The growth trajectory exceeded its prospectively fixed total host interval.'}
            save(done, value)
            return value
        quality = backend.store.json(native['lifecycle']['quality']['policy_root'])
        completed = life.materialize_graph(native['lifecycle']['candidate_template'], state['expert_work']['checkpoint'])
        growth = benchmark(backend, completed, native['lifecycle']['serving_graph'], quality, growth_end, 'growth')
        save(growth_file, {'started': growth_started, 'deadline': growth_end, 'serving': growth,
                           'resources_after': resources(backend.cloud),
                           'quality': json.loads((ctx['directory']/'quality-0.json').read_bytes())['result']})
        # Each arm fits the installed backend's four-hour call bound; both
        # arms together need not. Resume the control through the next ordinary
        # publisher poll, using the durable growth result and unchanged budget.
        return {'pending': True, 'completed_arm': 'growth'}
    started_path = home/'control-start.json'
    if not started_path.exists():
        save(started_path, {'started': datetime.now(timezone.utc).timestamp(),
                           'resources': resources(backend.cloud)})
    control_started = json.loads(started_path.read_bytes())['started']
    control_end = control_started+budget

    def remaining():
        if datetime.now(timezone.utc).timestamp() >= control_end:
            raise TimeoutError('The fixed-capacity control exceeded its equal total host interval')
        backend.cloud.remaining()

    # Every claim below was actually committed and independently audited on the
    # growth trajectory. The control re-executes it without signing or payment.
    claims = [json.loads(path.read_bytes()) for path in ctx['directory'].glob('claim-*-actor-1.json')]
    prefix = [claim for claim in claims if claim['kind'] == 'expert_features']
    windows = sorted((claim for claim in claims if claim['kind'] == 'expert_training'),
                     key=lambda claim: claim['input_checkpoint']['step'])
    if (len(prefix) != 1 or len(windows) != 32 or [claim['input_checkpoint']['step'] for claim in windows] != list(range(0, 128, 4))):
        raise ValueError('Compare all actual prefix work and thirty-two native training windows')
    actors = [Backend(backend.home, actor) for actor in range(4)]
    contexts = [actor.context(native, branch='matched-control') for actor in actors]
    prefix_marker = home/'control-prefix.json'
    if not prefix_marker.exists():
        remaining()
        produced = actors[0].prefix(contexts[0])
        with ThreadPoolExecutor(max_workers=3) as pool:
            reports = list(pool.map(lambda index: actors[index].prefix(contexts[index], prefix[0]), range(1, 4)))
        if any(not expert_work.replay_report(prefix[0], report)['valid'] for report in reports):
            raise ValueError('A fresh control prefix audit disagreed')
        save(prefix_marker, {'production': produced, 'reports': reports})
    for claim in windows:
        marker = home/('control-window-'+str(claim['input_checkpoint']['step'])+'.json')
        if marker.exists():
            continue
        remaining()
        work = {**state['expert_work'], 'checkpoint': claim['input_checkpoint']}
        produced = actors[0].training(contexts[0], work, 4)
        if produced['window'] != claim['window'] or produced['intermediates'] != claim['intermediates']:
            raise ValueError('Repeated control computation changed its fixed numerical trajectory')
        with ThreadPoolExecutor(max_workers=3) as pool:
            reports = list(pool.map(lambda index: actors[index].training(contexts[index], work, claim=claim), range(1, 4)))
        if any(not expert_work.replay_report(claim, report)['valid'] for report in reports):
            raise ValueError('A fresh control training audit disagreed')
        save(marker, {'production': produced, 'reports': reports, 'issued_atoms': 0})
    controlled = control_job(backend, native)
    checkpoint = state['expert_work']['checkpoint']
    graph = life.materialize_graph(controlled['lifecycle']['candidate_template'], checkpoint)
    quality = backend.store.json(controlled['lifecycle']['quality']['policy_root'])
    quality_marker = home/'control-quality.json'
    if not quality_marker.exists():
        remaining()
        context_quality = actors[0].context(controlled, branch='matched-control-quality')
        measured = actors[0].quality(context_quality, checkpoint)
        report = {'format': life.FORMAT+'/quality', 'policy_root': identity(quality),
            'baseline_graph': identity(controlled['lifecycle']['serving_graph']), 'candidate_graph': identity(graph),
            'prepared': controlled['work']['prepared'], 'passed': measured['decision']['passed'], 'results_root': identity(measured)}
        claim = {'kind': 'expert_quality', 'graph': graph, 'model_root': identity(graph),
            'executor_root': graph['executor_root'], 'record_root': None, 'report': report,
            'baseline_graph': controlled['lifecycle']['serving_graph'], 'stages': graph_quality.stages(quality)}
        claim['record_root'] = identity(graph_quality.quality_transcript(claim, measured))
        claim['id'] = identity({'shadow_quality': claim})
        with ThreadPoolExecutor(max_workers=3) as pool:
            reports = list(pool.map(lambda index: actors[index].quality(
                actors[index].context(controlled, branch='matched-control-quality'), checkpoint, claim), range(1, 4)))
        if any(not life.replay_report(claim, result)['valid'] for result in reports):
            raise ValueError('A fresh complete control quality audit disagreed')
        save(quality_marker, {'result': measured, 'reports': reports})
    remaining()
    control_serving = benchmark(backend, graph, controlled['lifecycle']['serving_graph'], quality,
                                 control_end, 'fixed-capacity')
    growth = json.loads(growth_file.read_bytes())
    control = json.loads(quality_marker.read_bytes())['result']
    result = {'passed': True, 'comparison_completed': True, 'seconds_per_arm': budget,
        'hosts_per_arm': 7, 'disk_gib_per_host': backend.freeze['resources']['disk_gib'],
        'growth': {**growth, 'resources_before': json.loads((backend.home/'comparison-growth-start.json').read_bytes())['resources']},
        'control': {'started': control_started, 'deadline': control_end,
                                     'resources_before': json.loads(started_path.read_bytes())['resources'],
                                     'resources_after': resources(backend.cloud),
                                     'quality': control, 'serving': control_serving},
        'shadow_issued_atoms': 0,
        'growth_preferred_by_preservation': bool(growth['quality']['decision']['passed']
            and growth['quality']['retention']['lost_correct'] == 0
            and control['retention']['lost_correct'] > 0),
        'scope': 'Equal declared host/disk budget including complete fresh verification. One frozen workload; not a lifetime-cost or optimal-control claim.'}
    save(done, result)
    return result
