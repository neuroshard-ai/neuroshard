"""A3 cohort 3 confirmation on fresh sealed sets, opened once behind the pinned development pass.

Four CPU hosts on the canonical runtime with prefix-cache serving: the upgraded and the
previous system, each on the sealed drafting set and, on a second host, on the sealed
scheduling and cross sets, every episode routed turn by turn by the pinned router. The gate
and the resource-budget comparison are assessed after every host finishes.
"""

import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from neuroshard.evolution import assistant_growth_cohort3_eval as development
from neuroshard.evolution import assistant_growth_confirm as stage1_confirmation
from neuroshard.evolution import assistant_growth_eval as stage1_development
from neuroshard.evolution import assistant_schedule_data as schedule
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution import granite_reference as reference
from neuroshard.evolution import granite_tokenizer
from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, identity, launch, read, save, sha256, verify_artifacts,
)

DECLARATION = development.DECLARATION
EXECUTION = 'config/experiments/assistant-growth-cohort3-confirmation-execution.json'
SCRIPT = 'scripts/run_assistant_growth_cohort3_confirmation.py'
UPLOADED = '.units'
# Each host's system and sets.
SYSTEMS = {'upgrade-calendar': ('upgrade', 'calendar'), 'upgrade-drafting': ('upgrade', 'drafting'),
           'previous-calendar': ('previous', 'calendar'), 'previous-drafting': ('previous', 'drafting')}
PROFILES = {f'assistant-growth-cohort3-confirmation-{system}': system for system in SYSTEMS}
SETS = {'calendar': ('scheduling', 'cross'), 'drafting': ('drafting',)}


def development_pass(execution):
    """The pinned development report, only if it records a pass."""
    pinned = execution['development_report']
    path = ROOT / pinned['path']
    report = read(path)
    if sha256(path) != pinned['sha256'] or not report['confirmation_may_open'] or not report['report']['passed']:
        raise ValueError('confirmation is sealed without a pinned development pass')
    return report


def opened(execution, which):
    """One host's sealed sets behind the pinned development pass, each checked against its frozen manifest."""
    development_pass(execution)
    sealed = read(ROOT / DECLARATION)['sealed']
    sets = {}
    for name in SETS[which]:
        if name == 'drafting':
            frozen, cases = read(ROOT / sealed['drafting_data']), data.cases(sealed['drafting'])
            if frozen['split'] != sealed['drafting']:
                raise ValueError('drafting manifest is for another split')
        else:
            frozen, cases = read(ROOT / sealed['scheduling_data'])['splits'][sealed[name]], schedule.cases(sealed[name])
        if identity(cases) != frozen['sha256'] or [c['id'] for c in cases] != frozen['case_ids']:
            raise ValueError(f'{name} cases differ from their frozen split')
        sets[name] = cases
    return sets


def committed_sources(root=ROOT):
    execution = read(root / EXECUTION)
    for name, digest in execution['contracts'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'changed cohort-3 confirmation contract: {name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in execution['sources']:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root) != (root / name).read_bytes():
            raise ValueError(f'uncommitted cohort-3 confirmation source: {name}')
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
    return {**committed_sources(), **development.runtime(read(ROOT / EXECUTION))}


def worker(request_path):
    configure()
    request_path = Path(request_path)
    request = read(request_path)
    if freeze() != request['freeze']:
        raise ValueError('confirmation worker differs from freeze')
    import torch
    from neuroshard.evolution import assistant_growth_run as growth

    execution = read(ROOT / EXECUTION)
    plan = read(ROOT / growth.PLAN)
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
        sets = opened(execution, which)
        state = verify_artifacts(directory, inventory, download=False)
        tokenizer, report = granite_tokenizer.load(directory)
        reply['tokenizer'] = report
        reply.update(development.serve(directory, tokenizer, execution, role, sets, spec))
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
    binding = {'freeze': source, 'profile': f'assistant-growth-cohort3-confirmation-{system}',
               'declaration_sha256': sha256(ROOT / DECLARATION)}
    save(home / 'binding.json', binding, exclusive=True)
    result = {'binding': binding, 'system': system, 'execution_completed': False, 'checklist_credit': False,
              'admission_evidence': False}
    try:
        role, which = SYSTEMS[system]
        opened(execution, which)
        stage1_confirmation.verify(ROOT / UPLOADED, execution, development.needed(role))
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


def assess(declaration, sets, replies):
    """The declared confirmation gate and resource budget from the four completed hosts, every episode rescored.

    ``sets`` maps drafting, scheduling and cross to their sealed cases; ``replies`` maps each system to its reply.
    """
    from neuroshard.evolution import assistant_experience_gate as gate
    from neuroshard.evolution import granite_context_reference as context

    episodes = {role: {name: replies[f'{role}-{which}']['episodes'][name] for which in SETS for name in SETS[which]}
                for role in ('upgrade', 'previous')}
    for system, (role, which) in SYSTEMS.items():
        if set(replies[system]['episodes']) != set(SETS[which]):
            raise ValueError('a confirmation host ran other sets')
    compared = development.compare(sets, episodes, development.policies())
    rules = declaration['confirmation_gate']
    passed = {role: gate.passed(episodes[role]['drafting']) for role in episodes}
    lower = gate.family_bootstrap(sets['drafting'], passed['upgrade'], passed['previous'], rules['bootstrap_samples'],
                                  rules['bootstrap_seed'])
    p95 = {role: context.percentile([stage1_development.latency(row) for rows in episodes[role].values()
                                     for row in rows], .95) for role in episodes}
    budget = declaration['resource_budget']
    checks = {'net_drafting': compared['drafting']['net'] >= rules['minimum_net_drafting'],
              'lower_drafting': lower > rules['lower_95_drafting_gain_strictly_above'],
              'lost': not any(row['lost'] for row in compared.values()),
              'p95': p95['upgrade'] <= rules['p95_seconds'],
              'p95_ratio': p95['upgrade'] <= budget['serving_p95_ratio'] * p95['previous']}
    families = {family: {role: sum(passed[role][c['id']] for c in sets['drafting'] if c['family'] == family)
                         for role in passed} for family in data.FAMILIES}
    return {'sets': compared, 'lower_95_drafting_gain': lower, 'per_family': families, 'p95_seconds': p95,
            'checks': checks, 'passed': all(checks.values())}
