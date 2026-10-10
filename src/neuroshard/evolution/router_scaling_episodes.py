"""Reworded real requests served end to end by the accepted A3 system, under the pinned and the context router.

The pinned A3 router sends 33 of 38 reworded drafting turns of the router scaling study to
the scheduling route. Labels cannot say what that costs: the calendar workspace keeps every
drafting tool, so a drafting turn on the scheduling route may still succeed. This run serves
the accepted system (A2's per-episode gate, L3 on the drafting route, L2 on the scheduling route
under the free-slot policy, prefix-cache serving) on real integration cases whose opening
request is reworded with every rule clause kept verbatim, so each keeps its workspace and
expected outcomes.

Both routers' route plans are computed on the host before serving, from the parent's accepted
router feature. Routes depend only on user text, so a plan fixes the whole episode; where the
two plans agree the episode is served once and counted for both. The original requests of the
reworded drafting cases are also served, so the wording's effect on the units is separated from
routing. Sets run in a declared priority order and stop cleanly before the worker's deadline.
Nothing is trained; no development, sealed or confirmation split is read.
"""

import importlib.metadata
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

DECLARATION = 'config/experiments/router-scaling-episodes.json'
EXECUTION = 'config/experiments/router-scaling-episodes-execution.json'
ROUTER = 'config/experiments/router-scaling-episodes-router.json'
SCRIPT = 'scripts/run_router_scaling_episodes.py'
PROFILE = 'router-scaling-episodes'
# A second, paired run serves the original wording of the reworded scheduling and cross cases.
PROFILES = {PROFILE: (DECLARATION, EXECUTION),
            'router-scaling-episodes-originals': ('config/experiments/router-scaling-episodes-originals.json',
                                                  'config/experiments/router-scaling-episodes-originals-execution.json')}
UPLOADED = '.units'
SYSTEM = ('L3', 'L2')
NEEDED = ('L2', 'L3', 'U1')
REWORD_SEED = 67000


def selected(cases, per_family):
    """The first ``per_family`` cases of each family, in split order."""
    counts, out = {}, []
    for case in cases:
        if counts.get(case['family'], 0) < per_family:
            counts[case['family']] = counts.get(case['family'], 0) + 1
            out.append(case)
    return out


def sets(declaration=None):
    """The declared sets in priority order: ``[(name, cases)]``."""
    from neuroshard.evolution import assistant_growth_run as growth
    from neuroshard.evolution import assistant_growth_round4 as round4
    from neuroshard.evolution import router_scaling_reworded as reworded

    declaration = declaration or read(ROOT / DECLARATION)
    stage = round4.plan()
    _, learning = growth.contracts(stage)[:2]
    drafting = selected(growth.drafting_cases(learning, 'integration'), declaration['drafting_per_family'])
    cross = growth.scheduling_cases(stage, 'cross-integration')
    scheduling = selected(growth.scheduling_cases(stage, 'integration'), declaration['scheduling_per_family'])
    built = {'reworded-drafting': reworded.reworded(drafting, REWORD_SEED),
             'reworded-cross': reworded.reworded(cross, REWORD_SEED),
             'original-drafting': drafting,
             'reworded-scheduling': reworded.reworded(scheduling, REWORD_SEED),
             'original-cross': cross,
             'original-scheduling': scheduling}
    return [(name, built[name]) for name in declaration['priority']]


def profile_files(profile):
    """The declaration and execution of a profile."""
    if profile not in PROFILES:
        raise ValueError(f'unknown episodes profile: {profile}')
    return PROFILES[profile]


def route_plans(parent, tokenizer, drafting_policy, pinned_gate, context_router, cases):
    """Each case's per-turn routes under the pinned centroid router and the context router."""
    from neuroshard.evolution import assistant_routing as routing
    from neuroshard.evolution import assistant_selector as selector
    from neuroshard.evolution import assistant_turn_router as turn_router
    from neuroshard.evolution.assistant_workflow_data import public_case

    prefix = routing.message_prefix(parent, tokenizer, drafting_policy, 'cpu')
    plans = {}
    for case in cases:
        users = public_case(case)['user_turns']
        features = [routing.message_feature(parent, tokenizer, drafting_policy, user, 'cpu', prefix) for user in users]
        pinned = ['scheduling' if selector.choose(pinned_gate, feature) else 'drafting' for feature in features]
        context = turn_router.route_many(context_router, turn_router.conversation_features(features))
        plans[case['id']] = {'pinned': pinned, 'context': context}
    return plans


def serve(parent, models, tokenizer, a2_gate, route_policies, case, plan):
    """One episode of the accepted system with a fixed per-turn route plan."""
    from neuroshard.evolution import assistant_experience_run as accelerator
    from neuroshard.evolution import assistant_routing as routing
    from neuroshard.evolution import assistant_selector as selector
    from neuroshard.evolution.assistant_serving import cached_responder

    drafting, scheduling = route_policies['drafting'], route_policies['scheduling']
    chosen = time.monotonic()
    arm = selector.choose(a2_gate, accelerator.boundary_feature(parent, tokenizer, drafting, case, 'cpu'))
    a2_seconds = time.monotonic() - chosen
    routes = {'drafting': (cached_responder(models[SYSTEM[0]] if arm else parent, tokenizer, drafting), drafting),
              'scheduling': (cached_responder(models[SYSTEM[1]], tokenizer, scheduling), scheduling)}
    row = routing.execute(case, routes, lambda turn, user: plan[turn])
    return {**row, 'a2_selected': 'arm' if arm else 'parent', 'a2_selection_seconds': a2_seconds, 'plan': list(plan)}


def committed_sources(root=ROOT, profile=PROFILE):
    execution = read(root / profile_files(profile)[1])
    for name, digest in execution['contracts'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'changed episodes contract: {name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in execution['sources']:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root) != (root / name).read_bytes():
            raise ValueError(f'uncommitted episodes source: {name}')
        sources[name] = sha256(root / name)
    return {'commit': commit, 'sources': sources}


def configure(profile=PROFILE):
    if 'torch' in sys.modules:
        raise ValueError('configure the episodes runtime before importing torch')
    for key, value in read(ROOT / profile_files(profile)[1])['environment'].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def freeze(profile=PROFILE):
    from neuroshard.evolution import assistant_workflow_canonical as canonical

    source = committed_sources(profile=profile)
    execution = read(ROOT / profile_files(profile)[1])
    pinned = read(ROOT / canonical.EXECUTION)
    if any(execution[key] != pinned[key] for key in ('packages', 'python', 'required_cpu_flags', 'environment')):
        raise ValueError('episodes runtime differs from the canonical parent runtime')
    packages = {key: importlib.metadata.version(key) for key in execution['packages']}
    if packages != execution['packages'] or platform.python_version() != execution['python']:
        raise ValueError('episodes runtime differs')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('episodes require Linux x86_64')
    cpu = Path('/proc/cpuinfo').read_text().split()
    if any(flag not in cpu for flag in execution['required_cpu_flags']):
        raise ValueError('episodes CPU lacks required instructions')
    if any(os.environ.get(key) != value for key, value in execution['environment'].items()):
        raise ValueError('episodes numerical environment differs')
    return {**source, 'packages': packages, 'python': platform.python_version()}


def worker(request_path):
    request_path = Path(request_path)
    request = read(request_path)
    profile = request['profile']
    configure(profile)
    if freeze(profile) != request['freeze']:
        raise ValueError('episodes worker differs from freeze')
    import torch

    from neuroshard.evolution import assistant_growth_cohort3_eval as cohort3
    from neuroshard.evolution import assistant_growth_confirm as stage1_confirmation
    from neuroshard.evolution import assistant_growth_run as growth
    from neuroshard.evolution import assistant_turn_router as turn_router
    from neuroshard.evolution import granite_reference as reference
    from neuroshard.evolution import granite_tokenizer

    declaration_path, execution_path = profile_files(profile)
    execution = read(ROOT / execution_path)
    phase = request['phase']
    if (request['model'], phase) not in (('baseline', 'prepare'), ('upgrade', 'episodes')):
        raise ValueError('unsupported episodes worker role')
    torch.set_num_threads(execution['threads'])
    torch.set_num_interop_threads(1)
    reply = {'binding': request['binding'], 'execution_completed': False}
    started = time.monotonic()
    deadline = started + request['seconds'] - execution['stop_margin_seconds']
    tokenizer = None
    try:
        inventory = read(ROOT / reference.ARTIFACTS)['models']['baseline']
        directory = Path(request['models']) / 'baseline'
        if phase == 'prepare':
            reply['file_state'] = verify_artifacts(directory, inventory, download=True)
            reply['execution_completed'] = True
            return
        plan = read(ROOT / growth.PLAN)
        spec = growth.stage_spec(plan, read(ROOT / read(ROOT / plan['plan'])['cohort1']['learning']))
        state = verify_artifacts(directory, inventory, download=False)
        tokenizer, report = granite_tokenizer.load(directory)
        reply['tokenizer'] = report
        gates = stage1_confirmation.verify(ROOT / UPLOADED, execution, NEEDED)
        a2_gate, pinned_gate = gates['a2']['arms']['update']['gate'], gates['router']['gate']
        context_router = turn_router.verify(read(ROOT / ROUTER)['router'])
        route_policies = cohort3.policies()
        parent, _ = reference.load_model(directory, 'baseline')
        models = cohort3.load_units(lambda: reference.load_model(directory, 'baseline')[0], spec, ROOT / UPLOADED,
                                    execution, SYSTEM)
        reply['units'] = {unit: execution['units'][unit]['trainable_sha256'] for unit in NEEDED}
        reply['routers'] = {'pinned': identity(pinned_gate), 'context': context_router['sha256']}
        declared = sets(read(ROOT / declaration_path))
        begun = time.monotonic()
        everything = [case for _, cases in declared for case in cases]
        reply['plans'] = route_plans(parent, tokenizer, route_policies['drafting'], pinned_gate, context_router,
                                     everything)
        reply['plan_seconds'] = time.monotonic() - begun
        reply['episodes'], reply['stopped_before'] = {}, None
        for name, cases in declared:
            rows = []
            for case in cases:
                plans = reply['plans'][case['id']]
                distinct = [('pinned', plans['pinned'])] + ([('context', plans['context'])]
                                                           if plans['context'] != plans['pinned'] else [])
                if time.monotonic() + execution['episode_reserve_seconds'] * len(distinct) > deadline:
                    reply['stopped_before'] = {'set': name, 'case': case['id']}
                    break
                for label, route_plan in distinct:
                    row = serve(parent, models, tokenizer, a2_gate, route_policies, case, route_plan)
                    rows.append({**row, 'routers': ['pinned', 'context'] if len(distinct) == 1 else [label]})
                save(request_path.parent / 'progress.json',
                     {'set': name, 'case': case['id'], 'episodes': sum(map(len, reply['episodes'].values())) + len(rows)})
            reply['episodes'][name] = rows
            if reply['stopped_before']:
                break
        reply['serving'] = 'prefix-cache'
        if file_state(directory, inventory) != state:
            raise ValueError('parent checkpoint changed while serving')
        reply['execution_completed'] = True
    except Exception as error:
        reply['error'] = f'{type(error).__name__}: {error}'
    finally:
        reply.update(checked_encodes=tokenizer.checked_encodes if tokenizer else 0,
                     wall_seconds=time.monotonic() - started, process_cpu_seconds=time.process_time(),
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        save(request_path.parent / 'reply.json', reply, exclusive=True)


def run(home, models, profile=PROFILE):
    from neuroshard.evolution import assistant_growth_confirm as stage1_confirmation

    configure(profile)
    source = freeze(profile)
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    declaration_path, execution_path = profile_files(profile)
    execution = read(ROOT / execution_path)
    binding = {'freeze': source, 'profile': profile, 'declaration_sha256': sha256(ROOT / declaration_path)}
    save(home / 'binding.json', binding, exclusive=True)
    result = {'binding': binding, 'execution_completed': False, 'checklist_credit': False,
              'admission_evidence': False, 'development_opened': False, 'confirmation_opened': False}
    try:
        stage1_confirmation.verify(ROOT / UPLOADED, execution, NEEDED)
        prepared = launch(home, models, binding, 'baseline', 'prepare', execution['prepare_seconds'],
                          execution['memory_bytes'], worker_script=SCRIPT)
        if not prepared['execution_completed']:
            raise ValueError(prepared.get('error', 'parent artifacts were not prepared'))
        reply = launch(home, models, binding, 'upgrade', 'episodes', execution['worker_seconds'],
                       execution['memory_bytes'], worker_script=SCRIPT)
        result['reply'] = reply
        if not reply['execution_completed']:
            raise ValueError(reply.get('error', 'incomplete episodes'))
        result['execution_completed'] = True
    except Exception as error:
        result['error'] = str(error)
    save(home / 'result.json', result, exclusive=True)
    return result


def assess(result, route_policies=None, declaration=None):
    """Every episode rescored from its transcript; per set, each router's passes and the cases they differ on."""
    from neuroshard.evolution import assistant_growth_cohort3_eval as cohort3
    from neuroshard.evolution import assistant_routing as routing

    route_policies = route_policies or cohort3.policies()
    reply = result['reply']
    by_id = {case['id']: case for _, cases in sets(declaration) for case in cases}
    report = {}
    for name, rows in reply['episodes'].items():
        passed = {'pinned': {}, 'context': {}}
        for row in rows:
            case = by_id[row['id']]
            if routing.score(case, row, route_policies) != row['score']:
                raise ValueError(f'rescore differs: {row["id"]}')
            if [r['route'] for r in row['rounds']] != row['plan'][:len(row['rounds'])]:
                raise ValueError(f'served routes differ from the plan: {row["id"]}')
            for router in row['routers']:
                passed[router][row['id']] = row['score']['passed']
        cases = sorted(passed['pinned'])
        if sorted(passed['context']) != cases:
            raise ValueError(f'{name}: routers cover different cases')
        report[name] = {'cases': len(cases), 'pinned': sum(passed['pinned'].values()),
                        'context': sum(passed['context'].values()),
                        'gained': sorted(c for c in cases if passed['context'][c] and not passed['pinned'][c]),
                        'lost': sorted(c for c in cases if passed['pinned'][c] and not passed['context'][c]),
                        'plans_differ': sum(reply['plans'][c]['pinned'] != reply['plans'][c]['context'] for c in cases)}
    return report
