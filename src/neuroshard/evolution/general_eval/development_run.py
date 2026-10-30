"""Step 0: the served ``a4-general`` on the open development split, on one canonical CPU host.

Declared in ``config/experiments/general-eval-development.json`` before execution. The worker:

1. loads the served stack as serving does: the frozen Granite parent (router features, message-mean of the
   drafting policy's cached prefix), A3's released router gate and the general router (each checked against
   the a4 descriptor's pins), the a4 route policies (pinned), and the parent with L9 (documents) and with
   C1 (chat), each digest-checked, behind one ``cached_responder`` per route per conversation;
2. FIRST reproduces the first five documents and first five chat conversations of confirmation7 through that
   stack, comparing each to its accepted run in every generated token, tool call and result, final reply,
   route and (documents) workspace snapshot. Any difference stops the run with ``execution_completed`` false;
3. then runs the 120 development items once each, in area round-robin order, writing every transcript.

A3's drafting and scheduling units are not loaded: a general item routed there with no user files fails
in the Session before any generation (as it would in serving); one that reaches generation is recorded as
``a3-route-not-loaded``. Numbers are development numbers, never a baseline claim.
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

from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, launch, read, save, sha256, verify_artifacts)

PROFILE = 'general-eval-development'
DECLARATION = 'config/experiments/general-eval-development.json'
EXECUTION = 'config/experiments/general-eval-development-execution.json'
SCRIPT = 'scripts/run_general_eval_development.py'
PROFILES = {PROFILE: (DECLARATION, EXECUTION)}
UPLOADED = '.units'
A3_ROUTER = UPLOADED + '/router.json'
EPISODES = 'config/experiments/assistant-general-l9-confirmation-episodes.json.gz'
ACCEPTED_SETS = {'documents': 'documents-confirmation7-L9', 'chat': 'chat-confirmation7-C1'}


class RouteNotLoaded(RuntimeError):
    pass


# ---------------------------------------------------------------- logic (tested with fakes)

def reproduction_work(count_per_route=5):
    """``[(route, case, accepted projection)]``: the first cases of each confirmation7 set, accepted runs."""
    from neuroshard.assistant import release_check_general as release
    from neuroshard.evolution import assistant_chat_data as chat
    from neuroshard.evolution import assistant_documents_data as docs

    episodes = json.loads(gzip.decompress((ROOT / EPISODES).read_bytes()))['episodes']
    work = []
    for route, module in (('documents', docs), ('chat', chat)):
        cases, rows = module.cases('confirmation7'), episodes[ACCEPTED_SETS[route]]
        if [c['id'] for c in cases] != [r['id'] for r in rows]:
            raise ValueError(f'accepted {route} episodes do not match confirmation7')
        work += [(route, case, release.accepted(row, route)) for case, row in list(zip(cases, rows))[:count_per_route]]
    return work


def first_difference(actual, expected):
    if actual.get('route') != expected.get('route'):
        return f'route {actual.get("route")} != {expected.get("route")}'
    for key in ('final_text', 'calls', 'snapshot'):
        if actual.get(key) != expected.get(key):
            return key
    got, want = actual['generations'], expected['generations']
    if len(got) != len(want):
        return f'{len(got)} generations != {len(want)}'
    for index, (g, w) in enumerate(zip(got, want)):
        for field in g:
            if g[field] != w.get(field):
                return f'generation {index} {field}'
    return None


def reproduce(system, work):
    from neuroshard.assistant import release_check_general as release
    from neuroshard.evolution.modular_reference_execution import identity

    rows = []
    for route, case, expected in work:
        started = time.monotonic()
        world = case['world'] if route == 'documents' else {'documents': []}
        outcome = system.converse(world, [turn['user'] for turn in case['turns']])
        actual = release.general_projection(outcome['route'], outcome['final_text'], outcome['calls'],
                                            outcome['generations'], outcome['snapshot'] if route == 'documents'
                                            else None)
        difference = first_difference(actual, expected)
        rows.append({'id': case['id'], 'route': route, 'chosen_route': outcome['route'], 'matched': difference is None,
                     'first_difference': difference, 'actual_sha256': identity(actual),
                     'expected_sha256': identity(expected), 'seconds': round(time.monotonic() - started, 3)})
        if difference is not None:
            break  # the first mismatch stops the attempt, as in the release check
    return rows


def interleave(items):
    """Area round-robin, so a run stopped by its time budget still covers every area."""
    from neuroshard.evolution.general_eval.items import AREAS

    queues = {area: [item for item in items if item['area'] == area] for area in AREAS}
    out = []
    while any(queues.values()):
        for area in AREAS:
            if queues[area]:
                out.append(queues[area].pop(0))
    return out


def served_system(policies, route, models, tokenizer, responder_factory):
    """The served stack: ``models`` maps a loaded route to its parent+unit model; other routes are not loaded."""
    from neuroshard.evolution.general_eval import served
    from neuroshard.evolution.general_eval.runner import ServedSystem

    def responder(name):
        if name not in models:
            raise RouteNotLoaded(f'a3-route-not-loaded: {name}')
        return responder_factory(models[name], tokenizer, policies[name])

    return ServedSystem(policies, route, responder, name='a4-general', modules=served.MODULES)


def execute(system, home, *, work, items, split_digest, cost, deadline, clock=time.monotonic):
    """Reproduction, then development. Returns the reply fields; writes evidence under ``home``."""
    from neuroshard.evolution.general_eval import runner

    home = Path(home)
    (home / 'transcripts').mkdir(parents=True, exist_ok=True)
    reply = {'execution_completed': False, 'baseline_claim': False, 'label': 'development, not a baseline claim'}
    save(home / 'status.json', {'phase': 'reproduction', 'done': 0, 'total': len(work)})
    rows = reproduce(system, work)
    save(home / 'reproduction.json', {'rows': rows, 'matched': sum(r['matched'] for r in rows), 'planned': len(work)})
    reply['reproduction'] = {'matched': sum(r['matched'] for r in rows), 'planned': len(work),
                             'first_mismatch': next((r for r in rows if not r['matched']), None)}
    if len(rows) != len(work) or not all(r['matched'] for r in rows):
        reply['error'] = 'reproduction differs from the accepted confirmation7 runs; development not run'
        return reply
    ordered = interleave(items)

    def progress(done, total, row, transcript):
        save(home / 'transcripts' / f'{row["id"]}.json', transcript)
        save(home / 'status.json', {'phase': 'development', 'done': done, 'total': total, 'last': row['id']})

    report, _ = runner.run(ordered, system, split='development', split_digest=split_digest, cost=cost,
                           progress=progress, stop=lambda: clock() >= deadline)
    report['label'] = 'development, not a baseline claim'
    histogram = {}
    for row in report['per_item']:
        histogram[str(row['route'])] = histogram.get(str(row['route']), 0) + 1
    report['route_histogram'] = dict(sorted(histogram.items()))
    seconds = [row['seconds'] for row in report['per_item'] if not row['unsupported']]
    report['seconds_per_item'] = {'items': len(seconds), 'mean': sum(seconds) / len(seconds) if seconds else None,
                                  'max': max(seconds, default=None),
                                  'by_area': {area: round(sum(r['seconds'] for r in report['per_item']
                                                              if r['area'] == area) / max(1, data['cases']), 3)
                                              for area, data in report['areas'].items()}}
    save(home / 'general-eval-development-report.json', report)
    reply['report'] = {k: v for k, v in report.items() if k != 'per_item'}
    reply['items_completed'] = report['items']
    if report['items'] != len(items):
        reply['error'] = f'time budget reached after {report["items"]} of {len(items)} items'
        return reply
    reply['execution_completed'] = True
    return reply


# ---------------------------------------------------------------- canonical host plumbing

def committed_sources(root=ROOT, profile=PROFILE):
    execution = read(root / PROFILES[profile][1])
    for name, digest in execution['contracts'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'changed step-0 contract: {name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in execution['sources']:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root) != (root / name).read_bytes():
            raise ValueError(f'uncommitted step-0 source: {name}')
        sources[name] = sha256(root / name)
    return {'commit': commit, 'sources': sources}


def configure(profile=PROFILE):
    if 'torch' in sys.modules:
        raise ValueError('configure the step-0 runtime before importing torch')
    for key, value in read(ROOT / PROFILES[profile][1])['environment'].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def freeze(profile=PROFILE):
    """The canonical parent's CPU runtime, byte for byte in packages and environment."""
    from neuroshard.evolution import assistant_workflow_canonical as canonical

    source = committed_sources(profile=profile)
    execution = read(ROOT / PROFILES[profile][1])
    pinned = read(ROOT / canonical.EXECUTION)
    if any(execution[key] != pinned[key] for key in ('packages', 'python', 'required_cpu_flags', 'environment')):
        raise ValueError('step-0 runtime differs from the canonical parent runtime')
    packages = {key: importlib.metadata.version(key) for key in execution['packages']}
    if packages != execution['packages'] or platform.python_version() != execution['python']:
        raise ValueError('step-0 runtime differs')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('step 0 requires Linux x86_64')
    cpu = Path('/proc/cpuinfo').read_text().split()
    if any(flag not in cpu for flag in execution['required_cpu_flags']):
        raise ValueError('step-0 CPU lacks required instructions')
    if any(os.environ.get(key) != value for key, value in execution['environment'].items()):
        raise ValueError('step-0 numerical environment differs')
    return {**source, 'packages': packages, 'python': platform.python_version()}


def worker(request_path):
    request_path = Path(request_path)
    request = read(request_path)
    profile = request['profile']
    configure(profile)
    if freeze(profile) != request['freeze']:
        raise ValueError('step-0 worker differs from freeze')
    import torch

    from neuroshard.evolution import assistant_general_baseline as baseline
    from neuroshard.evolution import granite_reference as reference
    from neuroshard.evolution import granite_tokenizer
    from neuroshard.evolution.assistant_serving import cached_responder
    from neuroshard.evolution.general_eval import items as schema
    from neuroshard.evolution.general_eval import served

    declaration_path, execution_path = PROFILES[profile]
    execution, declaration = read(ROOT / execution_path), read(ROOT / declaration_path)
    if request['phase'] not in ('prepare', 'evaluate') or request['model'] != 'baseline':
        raise ValueError('unsupported step-0 worker role')
    torch.set_num_threads(execution['threads'])
    torch.set_num_interop_threads(1)
    reply = {'binding': request['binding'], 'execution_completed': False}
    started = time.monotonic()
    tokenizer = None
    home = request_path.parent
    try:
        inventory = read(ROOT / reference.ARTIFACTS)['models']['baseline']
        directory = Path(request['models']) / 'baseline'
        if request['phase'] == 'prepare':
            reply['file_state'] = verify_artifacts(directory, inventory, download=True)
            reply['execution_completed'] = True
            return
        state = verify_artifacts(directory, inventory, download=False)
        tokenizer, report = granite_tokenizer.load(directory)
        reply['tokenizer'] = report
        policies, network = served.a4_policies()
        a3_router, general_router = served.routers(network, ROOT / A3_ROUTER)
        parent, _ = reference.load_model(directory, 'baseline')
        route = served.local_route(parent, tokenizer, policies, a3_router, general_router, served.route_names(network))
        load_parent = lambda: reference.load_model(directory, 'baseline')[0]  # noqa: E731
        models = {name: baseline.module_model(load_parent, declaration, unit)
                  for name, unit in declaration['routes_loaded'].items()}
        system = served_system(policies, route, models, tokenizer, cached_responder)
        split = schema.load_split('development')
        if schema.split_digest(split) != declaration['development_split_sha256']:
            raise ValueError('the development split differs from its declared digest')
        reply.update(execute(system, home, work=reproduction_work(declaration['reproduction_per_route']),
                             items=split, split_digest=schema.split_digest(split),
                             cost={'instance_type': 'r7i.4xlarge', 'usd_per_hour': 1.0584,
                                   'note': 'per-item wall time on the canonical host; dollars from the receipt'},
                             deadline=started + execution['worker_seconds'] - execution['finish_reserve_seconds']))
        if file_state(directory, inventory) != state:
            raise ValueError('parent checkpoint changed during step 0')
    except Exception as error:
        reply['execution_completed'] = False
        reply['error'] = f'{type(error).__name__}: {error}'
    finally:
        reply.update(checked_encodes=tokenizer.checked_encodes if tokenizer else 0,
                     wall_seconds=time.monotonic() - started, process_cpu_seconds=time.process_time(),
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        save(home / 'reply.json', reply, exclusive=True)


def run(home, models, profile=PROFILE):
    configure(profile)
    source = freeze(profile)
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    declaration_path, execution_path = PROFILES[profile]
    execution = read(ROOT / execution_path)
    binding = {'freeze': source, 'profile': profile, 'declaration_sha256': sha256(ROOT / declaration_path)}
    save(home / 'binding.json', binding, exclusive=True)
    result = {'binding': binding, 'execution_completed': False, 'baseline_claim': False, 'sealed_opened': False,
              'training': False, 'label': 'development, not a baseline claim'}
    try:
        prepared = launch(home, models, binding, 'baseline', 'prepare', execution['prepare_seconds'],
                          execution['memory_bytes'], worker_script=SCRIPT)
        if not prepared['execution_completed']:
            raise ValueError(prepared.get('error', 'parent artifacts were not prepared'))
        reply = launch(home, models, binding, 'baseline', 'evaluate', execution['worker_seconds'],
                       execution['memory_bytes'], worker_script=SCRIPT)
        result['reply'] = reply
        if not reply['execution_completed']:
            raise ValueError(reply.get('error', 'incomplete step 0'))
        result['execution_completed'] = True
    except Exception as error:
        result['error'] = str(error)
    save(home / 'result.json', result, exclusive=True)
    return result
