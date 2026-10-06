"""A3 stage-1 confirmation on the sealed sets, opened once behind the pinned development pass.

Five CPU hosts on the canonical runtime with prefix-cache serving, one system and one set
each. The development candidate is routed turn by turn on scheduling and cross, and
separately on the drafting retention split; the shared version is routed on drafting.
The accepted version runs as accepted on drafting, and under the calendar interface on
scheduling and cross as the previous system; its round-4 gate chooses once per episode.
The gate and A3's comparison are assessed after every host finishes.
"""

import importlib.metadata
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from neuroshard.evolution import assistant_growth_eval as development
from neuroshard.evolution import assistant_routing as routing
from neuroshard.evolution import assistant_schedule_data as schedule
from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution import assistant_workflow_canonical as canonical
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution import granite_reference as reference
from neuroshard.evolution import granite_tokenizer
from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, identity, launch, read, save, sha256, verify_artifacts,
)

EXECUTION = 'config/experiments/assistant-growth-confirmation-execution.json'
SCRIPT = 'scripts/run_assistant_growth_confirmation.py'
UPLOADED = '.units'
# Each host's system and sets: the candidate or shared version routed, or the accepted version as accepted.
SYSTEMS = {'candidate-calendar': ('candidate', 'calendar'), 'candidate-drafting': ('candidate', 'drafting'),
           'shared-drafting': ('shared', 'drafting'), 'accepted-drafting': ('accepted', 'drafting'),
           'accepted-calendar': ('accepted', 'calendar')}
PROFILES = {f'assistant-growth-confirmation-{system}': system for system in SYSTEMS}
SETS = {'calendar': (('scheduling', 'confirmation'), ('cross', 'cross-confirmation')),
        'drafting': (('drafting', 'confirmation4'),)}


def development_pass(execution):
    """The candidate named by the pinned development report, only if that report records a pass."""
    pinned = execution['development_report']
    path = ROOT / pinned['path']
    report = read(path)
    if sha256(path) != pinned['sha256'] or not report['confirmation_may_open'] or not report['report']['passed']:
        raise ValueError('confirmation is sealed without a pinned development pass')
    return report['report']['candidate']


def sealed(execution, which):
    """A host's (name, split) pairs: stage 1's, or the fresh splits and manifest the execution names."""
    fresh = execution.get('sealed')
    if fresh is None:
        return SETS[which], read(ROOT / read(ROOT / read(ROOT / development.PLAN)['plan'])['cohort2']['data'])['splits']
    return tuple((name, fresh[name]) for name, _ in SETS[which]), read(ROOT / fresh['scheduling_data'])['splits']


def opened(execution, which):
    """One host's sealed sets behind the pinned development pass, each checked against its frozen manifest."""
    development_pass(execution)
    pairs, manifests = sealed(execution, which)
    sets = {}
    for name, split in pairs:
        if name == 'drafting':
            frozen = read(ROOT / execution['drafting_data'])
            if frozen['split'] != split:
                raise ValueError('drafting manifest is for another split')
            cases = data.cases(split)
        else:
            frozen, cases = manifests[split], schedule.cases(split)
        if identity(cases) != frozen['sha256'] or [c['id'] for c in cases] != frozen['case_ids']:
            raise ValueError(f'{name} cases differ from their frozen split')
        sets[name] = cases
    return sets


def units(system, candidate):
    """The units one host loads: a routed version's, or U1 for the accepted version."""
    role, _ = SYSTEMS[system]
    return ['U1'] if role == 'accepted' else development.needed(candidate if role == 'candidate' else 'shared')


def verify(directory, execution, names):
    """The uploaded units one host loads, and both gate files, each matching its pinned digest."""
    directory = Path(directory)
    for unit in names:
        pinned = execution['units'][unit]
        if read(directory / pinned['checkpoint'] / 'manifest.json')['trainable_sha256'] != pinned['trainable_sha256']:
            raise ValueError(f'{unit} differs from its pinned digest')
    gates = {}
    for name, pinned in execution['gates'].items():
        if sha256(directory / pinned['file']) != pinned['sha256']:
            raise ValueError(f'{name} gates differ from their pinned digest')
        gates[name] = read(directory / pinned['file'])
    return gates


def committed_sources(root=ROOT):
    execution = read(root / EXECUTION)
    for name, digest in execution['contracts'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'changed confirmation contract: {name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in execution['sources']:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root) != (root / name).read_bytes():
            raise ValueError(f'uncommitted confirmation source: {name}')
        sources[name] = sha256(root / name)
    return {'commit': commit, 'sources': sources}


def configure():
    if 'torch' in sys.modules:
        raise ValueError('configure the confirmation runtime before importing torch')
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
        raise ValueError('confirmation runtime differs from the canonical parent runtime')
    packages = {key: importlib.metadata.version(key) for key in execution['packages']}
    if packages != execution['packages'] or platform.python_version() != execution['python']:
        raise ValueError('confirmation runtime differs')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('confirmation requires Linux x86_64')
    cpu = Path('/proc/cpuinfo').read_text().split()
    if any(flag not in cpu for flag in execution['required_cpu_flags']):
        raise ValueError('confirmation CPU lacks required instructions')
    if any(os.environ.get(key) != value for key, value in execution['environment'].items()):
        raise ValueError('confirmation numerical environment differs')
    return {**source, 'packages': packages, 'python': platform.python_version()}


def worker(request_path):
    configure()
    request_path = Path(request_path)
    request = read(request_path)
    if freeze() != request['freeze']:
        raise ValueError('confirmation worker differs from freeze')
    import torch
    from neuroshard.evolution import assistant_experience_eval as accepted
    from neuroshard.evolution import assistant_experience_run as accelerator
    from neuroshard.evolution import assistant_growth_run as growth
    from neuroshard.evolution.assistant_serving import cached_responder

    execution = read(ROOT / EXECUTION)
    plan = read(ROOT / development.PLAN)
    system, phase = request['model'], request['phase']
    if (phase, system) != ('prepare', 'baseline') and (system not in SYSTEMS or phase != 'confirmation'):
        raise ValueError('unsupported confirmation worker role')
    spec = growth.stage_spec(plan, read(ROOT / read(ROOT / plan['plan'])['cohort1']['learning']))
    torch.set_num_threads(execution['threads'])
    torch.set_num_interop_threads(1)
    reply = {'binding': request['binding'], 'system': system, 'execution_completed': False}
    started = time.monotonic()
    tokenizer = None
    try:
        inventory = read(ROOT / reference.ARTIFACTS)['models']['baseline']
        directory = Path(request['models']) / 'baseline'
        if phase == 'prepare':
            reply['file_state'] = verify_artifacts(directory, inventory, download=True)
            reply['execution_completed'] = True
            return
        role, which = SYSTEMS[system]
        candidate = development_pass(execution)
        sets = opened(execution, which)
        state = verify_artifacts(directory, inventory, download=False)
        tokenizer, report = granite_tokenizer.load(directory)
        reply['tokenizer'] = report
        names = units(system, candidate)
        gates = verify(ROOT / UPLOADED, execution, names)
        a2_gate = gates['a2']['arms']['update']['gate']
        parent, _ = reference.load_model(directory, 'baseline')
        models = development.load_units(lambda: reference.load_model(directory, 'baseline')[0], spec, ROOT / UPLOADED,
                                        execution, names if role == 'accepted' else development.VERSIONS[
                                            candidate if role == 'candidate' else 'shared'])
        reply['units'] = {unit: execution['units'][unit]['trainable_sha256'] for unit in names}
        if role == 'accepted':
            policy = development.policies()['drafting' if which == 'drafting' else 'scheduling']

            def feature(case):
                return accelerator.boundary_feature(parent, tokenizer, policy, case, 'cpu')

            reply['episodes'] = {name: accepted.evaluate_arm(parent, models['U1'], tokenizer, a2_gate, feature, cases,
                                                             policy, {'tasks': []}, cached_responder)['episodes']
                                 for name, cases in sets.items()}
        else:
            version = candidate if role == 'candidate' else 'shared'
            reply['version'] = version
            turn_gate = gates['router']['gate'] if 'router' in gates else gates['stage1']['gates'][version]
            if execution.get('selectors', {}).get('refit'):
                turn_gate, reply['selector'] = development.refit_selector(parent, tokenizer, gates['stage1'], version,
                                                                          execution['selectors'])
                if reply['selector']['gate_sha256'] != execution['selectors']['gate_sha256'][version]:
                    raise ValueError('the refitted selector differs from the one development used')
            else:
                reply['selector'] = {'rule': turn_gate['rule'], 'gate_sha256': identity(turn_gate)}
            reply['episodes'] = {name: development.routed_episodes(parent, models, tokenizer, version, a2_gate,
                                                                   turn_gate, cases)
                                 for name, cases in sets.items()}
        reply['serving'] = 'prefix-cache'
        if file_state(directory, inventory) != state:
            raise ValueError('parent checkpoint changed during confirmation')
        reply['execution_completed'] = True
    except Exception as error:
        reply['error'] = f'{type(error).__name__}: {error}'
    finally:
        reply.update(checked_encodes=tokenizer.checked_encodes if tokenizer else 0,
                     wall_seconds=time.monotonic() - started, process_cpu_seconds=time.process_time(),
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        save(request_path.parent / 'reply.json', reply, exclusive=True)


def run(home, models, system):
    configure()
    source = freeze()
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    execution = read(ROOT / EXECUTION)
    if system not in SYSTEMS:
        raise ValueError('unknown confirmation system')
    binding = {'freeze': source, 'profile': f'assistant-growth-confirmation-{system}',
               'plan_sha256': sha256(ROOT / development.PLAN)}
    save(home / 'binding.json', binding, exclusive=True)
    result = {'binding': binding, 'system': system, 'execution_completed': False, 'checklist_credit': False,
              'admission_evidence': False}
    try:
        opened(execution, SYSTEMS[system][1])
        verify(ROOT / UPLOADED, execution, units(system, development_pass(execution)))
        prepared = launch(home, models, binding, 'baseline', 'prepare', execution['prepare_seconds'],
                          execution['memory_bytes'], worker_script=SCRIPT)
        if not prepared['execution_completed']:
            raise ValueError(prepared.get('error', 'parent artifacts were not prepared'))
        reply = launch(home, models, binding, system, 'confirmation', execution['worker_seconds'],
                       execution['memory_bytes'], worker_script=SCRIPT)
        result['reply'] = reply
        if not reply['execution_completed']:
            raise ValueError(reply.get('error', 'incomplete confirmation'))
        result['execution_completed'] = True
    except Exception as error:
        result['error'] = str(error)
    save(home / 'result.json', result, exclusive=True)
    return result


def assess(plan, sets, replies):
    """The declared confirmation gate and A3's comparison from the five completed hosts, every episode rescored.

    ``sets`` maps scheduling, cross and drafting to their sealed cases; ``replies`` maps each system to its reply.
    """
    from neuroshard.evolution import assistant_experience_gate as gate
    from neuroshard.evolution import granite_context_reference as context

    route_policies = development.policies()
    rows = {}
    for system, (role, which) in SYSTEMS.items():
        episodes = replies[system]['episodes']
        if set(episodes) != {name for name, _ in SETS[which]}:
            raise ValueError('a confirmation host ran other sets')
        for name, _ in SETS[which]:
            by_id = {case['id']: case for case in sets[name]}
            if [row['id'] for row in episodes[name]] != [case['id'] for case in sets[name]]:
                raise ValueError('every confirmation episode must run, in order')
            for row in episodes[name]:
                if role == 'accepted':
                    policy = route_policies['drafting' if which == 'drafting' else 'scheduling']
                    rescored = workflow.score(by_id[row['id']], row, policy)
                else:
                    rescored = routing.score(by_id[row['id']], row, route_policies)
                if rescored != row['score']:
                    raise ValueError('confirmation outcome rescore differs')
            rows[system, name] = episodes[name]
    rules = plan['confirmation_gate']
    new = gate.passed(rows['candidate-calendar', 'scheduling'] + rows['candidate-calendar', 'cross'])
    previous = gate.passed(rows['accepted-calendar', 'scheduling'] + rows['accepted-calendar', 'cross'])
    per_family = {f: sum(new[c['id']] for c in sets['scheduling'] if c['family'] == f) for f in schedule.FAMILIES}
    versus_previous = gate.paired(new, previous)
    lower = gate.family_bootstrap(sets['scheduling'] + sets['cross'], new, previous, rules['bootstrap_samples'],
                                  rules['bootstrap_seed'])
    retained = gate.passed(rows['accepted-drafting', 'drafting'])
    drafting = {system: gate.paired(gate.passed(rows[system, 'drafting']), retained)
                for system in ('candidate-drafting', 'shared-drafting')}
    candidate_rows = [row for key, value in rows.items() if key[0].startswith('candidate-') for row in value]
    p95 = context.percentile([development.latency(row) for row in candidate_rows], .95)
    correct = {'scheduling': sum(new[c['id']] for c in sets['scheduling']),
               'cross': sum(new[c['id']] for c in sets['cross'])}
    checks = {'scheduling': correct['scheduling'] >= rules['minimum_scheduling'],
              'per_family': min(per_family.values()) >= rules['minimum_per_scheduling_family'],
              'cross': correct['cross'] >= rules['minimum_cross'],
              'net_vs_previous': versus_previous['net'] >= rules['minimum_net_vs_previous'],
              'lower_vs_previous': lower > rules['lower_95_gain_vs_previous_strictly_above'],
              'drafting': not drafting['candidate-drafting']['lost'],
              'p95': p95 <= rules['p95_seconds']}
    comparison = len(drafting['candidate-drafting']['lost']) < len(drafting['shared-drafting']['lost'])
    return {'correct': {**correct, 'drafting': sum(gate.passed(rows['candidate-drafting', 'drafting']).values())},
            'previous': {'scheduling': sum(previous[c['id']] for c in sets['scheduling']),
                         'cross': sum(previous[c['id']] for c in sets['cross'])},
            'accepted_drafting': sum(retained.values()),
            'shared_drafting': sum(gate.passed(rows['shared-drafting', 'drafting']).values()),
            'per_family': per_family, 'versus_previous': versus_previous, 'lower_95_gain_vs_previous': lower,
            'drafting_versus_accepted': drafting, 'p95_seconds': p95, 'checks': checks,
            'passed': all(checks.values()), 'separate_retains_better': comparison}
