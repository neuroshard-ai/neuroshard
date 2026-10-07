"""A3 stage 1 development: each version routed turn by turn on the opened development cases.

One CPU host per version, on the canonical runtime with prefix-cache serving. A host
loads the parent for the selection features, the version's drafting unit and its
scheduling unit. Each user turn goes to the route the version's selector names.
Drafting turns are served as A2 served them: its gate chooses, once per episode, between
the parent and the drafting unit. Nothing is trained and no sealed split opens; the
development gate is assessed after every version finishes.
"""

import importlib.metadata
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from neuroshard.evolution import assistant_growth_baseline as baseline
from neuroshard.evolution import assistant_routing as routing
from neuroshard.evolution import assistant_workflow_canonical as canonical
from neuroshard.evolution import granite_reference as reference
from neuroshard.evolution import granite_tokenizer
from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, identity, launch, read, save, sha256, verify_artifacts,
)

PLAN = 'config/experiments/assistant-growth-stage1.json'
EXECUTION = 'config/experiments/assistant-growth-development-execution.json'
SCRIPT = 'scripts/run_assistant_growth_development.py'
UPLOADED = '.units'
VERSIONS = {'separate_update': ('U1', 'U2'), 'separate_module': ('U1', 'L2'), 'shared': ('U2', 'U2')}
PROFILES = {f"assistant-growth-development-{version.replace('_', '-')}": version for version in VERSIONS}


def policies():
    """Each route's policy; the execution may name the scheduling route's, as round 5's free-slot policy."""
    growth = read(ROOT / read(ROOT / PLAN)['plan'])
    learning = read(ROOT / growth['cohort1']['learning'])
    scheduling = read(ROOT / EXECUTION).get('scheduling_policy', growth['cohort2']['policy'])
    return {'drafting': read(ROOT / learning['policy']), 'scheduling': read(ROOT / scheduling)}


def committed_sources(root=ROOT):
    execution = read(root / EXECUTION)
    for name, digest in execution['contracts'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'changed development contract: {name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in execution['sources']:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root) != (root / name).read_bytes():
            raise ValueError(f'uncommitted development source: {name}')
        sources[name] = sha256(root / name)
    return {'commit': commit, 'sources': sources}


def configure():
    if 'torch' in sys.modules:
        raise ValueError('configure the development runtime before importing torch')
    for key, value in read(ROOT / EXECUTION)['environment'].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def freeze():
    """The canonical parent's CPU runtime, byte for byte in packages and environment."""
    source = committed_sources()
    execution = read(ROOT / EXECUTION)
    pinned = read(ROOT / canonical.EXECUTION)
    if any(execution[key] != pinned[key] for key in ('packages', 'python', 'required_cpu_flags', 'environment')):
        raise ValueError('development runtime differs from the canonical parent runtime')
    packages = {key: importlib.metadata.version(key) for key in execution['packages']}
    if packages != execution['packages'] or platform.python_version() != execution['python']:
        raise ValueError('development runtime differs')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('development requires Linux x86_64')
    cpu = Path('/proc/cpuinfo').read_text().split()
    if any(flag not in cpu for flag in execution['required_cpu_flags']):
        raise ValueError('development CPU lacks required instructions')
    if any(os.environ.get(key) != value for key, value in execution['environment'].items()):
        raise ValueError('development numerical environment differs')
    return {**source, 'packages': packages, 'python': platform.python_version()}


def needed(version):
    """The checkpoints a version loads: its units, and U1 under L2, which is trained on top of it."""
    units = set(VERSIONS[version])
    return sorted(units | ({'U1'} if 'L2' in units else set()))


def verify_units(directory, execution, version):
    """The uploaded checkpoints and gates of one version, each matching its pinned digest."""
    directory = Path(directory)
    for unit in needed(version):
        pinned = execution['units'][unit]
        if read(directory / pinned['checkpoint'] / 'manifest.json')['trainable_sha256'] != pinned['trainable_sha256']:
            raise ValueError(f'{unit} differs from its pinned digest')
    gates = {}
    for name, pinned in execution['gates'].items():
        if sha256(directory / pinned['file']) != pinned['sha256']:
            raise ValueError(f'{name} gates differ from their pinned digest')
        gates[name] = read(directory / pinned['file'])
    turn = gates['router']['gate'] if 'router' in gates else gates['stage1']['gates'][version]
    return gates['a2']['arms']['update']['gate'], turn


def load_units(load_parent, spec, directory, execution, units):
    """Each unit on a fresh parent: U1 and U2 served as plain projections, L2 attached on top of U1."""
    from neuroshard.evolution import assistant_experience_train as trainer

    def load(unit):
        model = load_parent()
        if unit in ('U1', 'L2'):
            trainer.load_trainable(model, 'update', spec, Path(directory) / execution['units']['U1']['checkpoint'])
            trainer.serving(model, spec)
        if unit == 'U2':
            trainer.load_trainable(model, 'update', spec, Path(directory) / execution['units']['U2']['checkpoint'])
            trainer.serving(model, spec)
        if unit == 'L2':
            trainer.load_trainable(model, 'addition', spec, Path(directory) / execution['units']['L2']['checkpoint'])
        return model.eval()

    return {unit: load(unit) for unit in sorted(set(units))}


def refit_selector(parent, tokenizer, integration, version, rules):
    """The version's turn selector fitted from the pinned integration outcomes on this runtime's parent features.

    Features are computed here, on the serving runtime, before any development case is served.
    """
    from neuroshard.evolution import assistant_growth_run as growth
    from neuroshard.evolution import assistant_selector as selector
    from neuroshard.evolution.assistant_workflow_data import public_case

    plan = read(ROOT / PLAN)
    _, learning, _ = growth.contracts(plan)
    cases = growth.integration_cases(plan, learning)
    drafting_unit, scheduling_unit = VERSIONS[version]
    rows = routing.turn_targets(integration['outcomes'][f'{drafting_unit}-drafting'],
                                integration['outcomes'][f'{scheduling_unit}-scheduling'],
                                {case['id']: len(case['turns']) for case in cases}, failed_ties=rules['failed_ties'])
    drafting = policies()['drafting']
    features = {routing.turn_key(case['id'], turn): routing.turn_feature(parent, tokenizer, drafting, user, 'cpu')
                for case in cases for turn, user in enumerate(public_case(case)['user_turns'])
                if routing.turn_key(case['id'], turn) in rows}
    recipe = plan['integration']['recipe']
    if rules.get('rule') == 'centroid':
        gate = selector.fit_centroid(features, rows, recipe['epsilon'])
        extra = {'held_out_accuracy': held_out(features, rows, recipe['epsilon'])}
    else:
        gate, extra = selector.fit(features, rows, recipe), {}
    return gate, {**{k: v for k, v in gate.items() if k not in ('weight', 'mean', 'arm', 'parent')}, **extra,
                  'examples': len(rows), 'features_sha256': identity(features), 'gate_sha256': identity(gate)}


def held_out(features, rows, epsilon, folds=4):
    """Weighted accuracy of the centroid rule on each case's turns, fitted without that case's fold; reported only."""
    from neuroshard.evolution import assistant_selector as selector

    cases = sorted({key.rsplit('#', 1)[0] for key in rows})
    fold = {case: index % folds for index, case in enumerate(cases)}
    right = total = 0.0
    for k in range(folds):
        train = {key: row for key, row in rows.items() if fold[key.rsplit('#', 1)[0]] != k}
        gate = selector.fit_centroid({key: features[key] for key in train}, train, epsilon)
        for key, (target, weight) in rows.items():
            if fold[key.rsplit('#', 1)[0]] == k:
                right += weight * (selector.choose(gate, features[key]) == (target == 1.0))
                total += weight
    return right / total if total else None


def routed_episodes(parent, models, tokenizer, version, a2_gate, turn_gate, cases, *, units=None,
                    route_policies=None):
    """One version served turn by turn. The A2 gate chooses, once per episode, the drafting route's model.

    ``units`` and ``route_policies`` serve another system's drafting and scheduling units under its own policies.
    """
    from neuroshard.evolution import assistant_experience_run as accelerator
    from neuroshard.evolution import assistant_selector as selector
    from neuroshard.evolution.assistant_serving import cached_responder

    drafting_unit, scheduling_unit = units or VERSIONS[version]
    route_policies = route_policies or policies()
    drafting, scheduling = route_policies['drafting'], route_policies['scheduling']
    prefix = (routing.message_prefix(parent, tokenizer, drafting, 'cpu') if turn_gate.get('feature') == 'message-mean'
              else None)

    def select(turn, user):
        feature = (routing.message_feature(parent, tokenizer, drafting, user, 'cpu', prefix) if prefix
                   else routing.turn_feature(parent, tokenizer, drafting, user, 'cpu'))
        return 'scheduling' if selector.choose(turn_gate, feature) else 'drafting'

    rows = []
    for case in cases:
        chosen = time.monotonic()
        arm = selector.choose(a2_gate, accelerator.boundary_feature(parent, tokenizer, drafting, case, 'cpu'))
        a2_seconds = time.monotonic() - chosen
        routes = {'drafting': (cached_responder(models[drafting_unit] if arm else parent, tokenizer, drafting), drafting),
                  'scheduling': (cached_responder(models[scheduling_unit], tokenizer, scheduling), scheduling)}
        rows.append({**routing.execute(case, routes, select), 'a2_selected': 'arm' if arm else 'parent',
                     'a2_selection_seconds': a2_seconds})
    return rows


def worker(request_path):
    configure()
    request_path = Path(request_path)
    request = read(request_path)
    if freeze() != request['freeze']:
        raise ValueError('development worker differs from freeze')
    import torch
    from neuroshard.evolution import assistant_growth_run as growth

    execution = read(ROOT / EXECUTION)
    plan = read(ROOT / PLAN)
    version, phase = request['model'], request['phase']
    if (phase, version) != ('prepare', 'baseline') and (version not in VERSIONS or phase != 'development'):
        raise ValueError('unsupported development worker role')
    spec = growth.stage_spec(plan, read(ROOT / read(ROOT / plan['plan'])['cohort1']['learning']))
    torch.set_num_threads(execution['threads'])
    torch.set_num_interop_threads(1)
    reply = {'binding': request['binding'], 'version': version, 'execution_completed': False}
    started = time.monotonic()
    tokenizer = None
    try:
        inventory = read(ROOT / reference.ARTIFACTS)['models']['baseline']
        directory = Path(request['models']) / 'baseline'
        if phase == 'prepare':
            reply['file_state'] = verify_artifacts(directory, inventory, download=True)
            reply['execution_completed'] = True
            return
        sets = baseline.opened(read(ROOT / plan['plan']))
        state = verify_artifacts(directory, inventory, download=False)
        tokenizer, report = granite_tokenizer.load(directory)
        reply['tokenizer'] = report
        units = ROOT / UPLOADED
        a2_gate, turn_gate = verify_units(units, execution, version)
        parent, _ = reference.load_model(directory, 'baseline')
        if execution.get('selectors', {}).get('refit'):
            begun = time.monotonic()
            integration = read(units / execution['gates']['stage1']['file'])
            turn_gate, reply['selector'] = refit_selector(parent, tokenizer, integration, version, execution['selectors'])
            reply['selector']['seconds'] = time.monotonic() - begun
        else:
            reply['selector'] = {'rule': turn_gate['rule'], 'gate_sha256': identity(turn_gate)}
        models = load_units(lambda: reference.load_model(directory, 'baseline')[0], spec, units, execution,
                            VERSIONS[version])
        reply['units'] = {unit: execution['units'][unit]['trainable_sha256'] for unit in needed(version)}
        reply['episodes'] = {name: routed_episodes(parent, models, tokenizer, version, a2_gate, turn_gate, cases)
                             for name, cases in sets.items()}
        reply['serving'] = 'prefix-cache'
        if file_state(directory, inventory) != state:
            raise ValueError('parent checkpoint changed during development')
        reply['execution_completed'] = True
    except Exception as error:
        reply['error'] = f'{type(error).__name__}: {error}'
    finally:
        reply.update(checked_encodes=tokenizer.checked_encodes if tokenizer else 0,
                     wall_seconds=time.monotonic() - started, process_cpu_seconds=time.process_time(),
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        save(request_path.parent / 'reply.json', reply, exclusive=True)


def run(home, models, version):
    configure()
    source = freeze()
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    execution = read(ROOT / EXECUTION)
    if version not in VERSIONS:
        raise ValueError('unknown development version')
    binding = {'freeze': source, 'profile': f"assistant-growth-development-{version.replace('_', '-')}",
               'plan_sha256': sha256(ROOT / PLAN)}
    save(home / 'binding.json', binding, exclusive=True)
    result = {'binding': binding, 'version': version, 'execution_completed': False, 'checklist_credit': False,
              'admission_evidence': False, 'confirmation_opened': False}
    try:
        verify_units(ROOT / UPLOADED, execution, version)
        prepared = launch(home, models, binding, 'baseline', 'prepare', execution['prepare_seconds'],
                          execution['memory_bytes'], worker_script=SCRIPT)
        if not prepared['execution_completed']:
            raise ValueError(prepared.get('error', 'parent artifacts were not prepared'))
        reply = launch(home, models, binding, version, 'development', execution['worker_seconds'],
                       execution['memory_bytes'], worker_script=SCRIPT)
        result['reply'] = reply
        if not reply['execution_completed']:
            raise ValueError(reply.get('error', 'incomplete development'))
        result['execution_completed'] = True
    except Exception as error:
        result['error'] = str(error)
    save(home / 'result.json', result, exclusive=True)
    return result


def latency(row):
    """Episode wall time, which already holds each turn's selection pass, plus the A2 gate's pass before it."""
    return row['seconds'] + row['a2_selection_seconds']


def assess(plan, sets, replies, accepted):
    """The declared development gate: every episode rescored, the candidate chosen, drafting checked case by case.

    ``replies`` maps each version to its reply; ``accepted`` holds the accepted version's
    drafting development episodes under the drafting interface.
    """
    from neuroshard.evolution import assistant_experience_gate as gate
    from neuroshard.evolution import granite_context_reference as context

    route_policies = policies()
    accepted_passed = gate.passed(accepted)
    report = {'versions': {}}
    for version in VERSIONS:
        outcomes = {}
        rows_all = []
        for name, cases in sets.items():
            rows = replies[version]['episodes'][name]
            by_id = {case['id']: case for case in cases}
            if [row['id'] for row in rows] != [case['id'] for case in cases]:
                raise ValueError('every development episode must run, in order')
            for row in rows:
                if routing.score(by_id[row['id']], row, route_policies) != row['score']:
                    raise ValueError('development outcome rescore differs')
            passed = gate.passed(rows)
            outcomes[name] = {'correct': sum(passed.values()), 'cases': len(cases),
                              'routes': sorted({tuple(row['score']['routes']) for row in rows})}
            rows_all += rows
        drafting = gate.passed(replies[version]['episodes']['drafting'])
        lost = sorted(k for k in accepted_passed if accepted_passed[k] and not drafting.get(k))
        report['versions'][version] = {**outcomes, 'drafting_lost': lost,
                                       'p95_seconds': context.percentile([latency(r) for r in rows_all], .95)}
    rules = plan['development_gate']
    separate = [v for v in VERSIONS if v != 'shared']
    learned = {v: report['versions'][v]['scheduling']['correct'] + report['versions'][v]['cross']['correct']
               for v in separate}
    candidate = 'separate_module' if learned['separate_module'] >= learned['separate_update'] else 'separate_update'
    chosen = report['versions'][candidate]
    checks = {'scheduling': chosen['scheduling']['correct'] >= rules['minimum_scheduling'],
              'cross': chosen['cross']['correct'] >= rules['minimum_cross'],
              'drafting': not chosen['drafting_lost'],
              'p95': chosen['p95_seconds'] <= rules['p95_seconds']}
    report.update(candidate=candidate, checks=checks, passed=all(checks.values()), accepted_drafting=sum(accepted_passed.values()))
    return report
