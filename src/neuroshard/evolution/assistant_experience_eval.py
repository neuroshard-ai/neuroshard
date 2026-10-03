"""Development evaluation of both trained systems on the canonical parent's CPU runtime.

Each system selects once per episode from the frozen parent feature, then runs
the whole episode with the chosen model. The parent control is the canonical
re-baseline, pinned by its result digest; protected successes come from it.
Each arm also answers every original anchor with selection forced on, which
measures forgetting directly; the routed system serves anchors with the parent.
Each arm runs alone in a fresh worker at the baseline's thread count, so numerics and latency match it.

An execution may evaluate only some arms (``systems``), name the arm that would be
served (``served``) and take its gate from a separate contract (``gate_plan``); a
gated arm with no separate control is judged against the parent alone.
"""

import importlib.metadata
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution import assistant_workflow_baseline as first
from neuroshard.evolution import assistant_workflow_canonical as canonical
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution import granite_reference as reference
from neuroshard.evolution import granite_tokenizer
from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, launch, read, save, sha256, verify_artifacts,
)

ARMS = ('update', 'addition')
SERVED = 'addition'
PLAN = 'config/experiments/assistant-experience-learning.json'
EXECUTION = 'config/experiments/assistant-experience-development-execution.json'
SCRIPT = 'scripts/run_assistant_experience_development.py'
PROFILE = 'assistant-experience-development'
UPLOADED = '.arms'


def systems(execution):
    """The arms an execution evaluates, and the one whose served version A1 judges."""
    arms = tuple(execution.get('systems', ARMS))
    served = execution.get('served', SERVED)
    if not arms or not set(arms) <= set(ARMS) or served not in arms:
        raise ValueError('an execution evaluates known arms, including the one it serves')
    return arms, served


def verify_arms(directory, pinned):
    """Uploaded checkpoints and gates must match the digests pinned before evaluation."""
    directory = Path(directory)
    arms = [name for name in ARMS if name in pinned]
    if not arms:
        raise ValueError('no arm is pinned')
    for arm in arms:
        manifest = read(directory / f'{arm}-checkpoint' / 'manifest.json')
        if manifest['trainable_sha256'] != pinned[arm]['trainable_sha256'] or manifest['arm'] != arm:
            raise ValueError(f'{arm} checkpoint differs from the pinned training result')
    if sha256(directory / 'integration.json') != pinned['integration_sha256']:
        raise ValueError('integration gates differ from the pinned result')
    return read(directory / 'integration.json')


def evaluate_arm(parent, model, tokenizer, gate, feature, cases, policy, anchor_plan, make_responder=None):
    """Selected complete episodes for one system, plus its forced-arm anchor answers.

    ``feature`` runs the parent at the first request; its time is served latency.
    ``make_responder`` builds a fresh responder per episode (default: uncached).
    """
    from neuroshard.evolution import assistant_selector as selector

    make = make_responder or first.native_responder
    models = {'parent': parent, 'arm': model}
    rows = []
    for case in cases:
        started = time.monotonic()
        chosen = 'arm' if selector.choose(gate, feature(case)) else 'parent'
        selection = time.monotonic() - started
        respond = make(models[chosen], tokenizer, policy)
        rows.append({**workflow.execute(case, respond, policy), 'selected': chosen,
                     'selection_seconds': selection})
    anchors = [reference.generate(model, tokenizer, anchor_plan, task, 'baseline') for task in anchor_plan['tasks']]
    return {'episodes': rows, 'forced_anchors': anchors}


def forgetting(anchor_rows, protected_ids):
    passed = {row['id'] for row in anchor_rows if row['passed']}
    return {'correct': len(passed), 'lost_protected': sorted(set(protected_ids) - passed)}


def committed_sources(root=ROOT):
    execution = read(root / EXECUTION)
    for name, digest in execution['contracts'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'changed evaluation contract: {name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in execution['sources']:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root) != (root / name).read_bytes():
            raise ValueError(f'uncommitted evaluation source: {name}')
        sources[name] = sha256(root / name)
    return {'commit': commit, 'sources': sources}


def configure():
    if 'torch' in sys.modules:
        raise ValueError('configure the evaluation runtime before importing torch')
    for key, value in read(ROOT / EXECUTION)['environment'].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def freeze():
    """The canonical parent's CPU runtime, byte for byte in packages and environment."""
    source = committed_sources()
    execution = read(ROOT / EXECUTION)
    baseline = read(ROOT / canonical.EXECUTION)
    if any(execution[key] != baseline[key] for key in ('packages', 'python', 'required_cpu_flags', 'environment')):
        raise ValueError('evaluation runtime differs from the canonical parent runtime')
    packages = {key: importlib.metadata.version(key) for key in execution['packages']}
    if packages != execution['packages'] or platform.python_version() != execution['python']:
        raise ValueError('evaluation runtime differs')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('evaluation requires Linux x86_64')
    cpu = Path('/proc/cpuinfo').read_text().split()
    if any(flag not in cpu for flag in execution['required_cpu_flags']):
        raise ValueError('evaluation CPU lacks required instructions')
    if any(os.environ.get(key) != value for key, value in execution['environment'].items()):
        raise ValueError('evaluation numerical environment differs')
    return {**source, 'packages': packages, 'python': platform.python_version()}


def worker(request_path):
    configure()
    request_path = Path(request_path)
    request = read(request_path)
    if freeze() != request['freeze']:
        raise ValueError('evaluation worker differs from freeze')
    import torch
    from neuroshard.evolution import assistant_experience_run as accelerator
    from neuroshard.evolution import assistant_experience_train as trainer

    execution = read(ROOT / EXECUTION)
    evaluated, served_arm = systems(execution)
    arm, phase = request['model'], request['phase']
    replay = (phase, arm) == ('replay', served_arm) and 'a1_served' in execution
    if (phase, arm) != ('prepare', 'baseline') and not replay and (arm not in evaluated or phase != 'development'):
        raise ValueError('unsupported evaluation worker role')
    plan = read(ROOT / PLAN)
    policy = read(ROOT / plan['policy'])
    torch.set_num_threads(execution['threads'])
    torch.set_num_interop_threads(1)
    reply = {'binding': request['binding'], 'arm': arm, 'execution_completed': False}
    started = time.monotonic()
    tokenizer = None
    try:
        inventory = read(ROOT / reference.ARTIFACTS)['models']['baseline']
        directory = Path(request['models']) / 'baseline'
        if phase == 'prepare':
            reply['file_state'] = verify_artifacts(directory, inventory, download=True)
            reply['execution_completed'] = True
            return
        arms = ROOT / UPLOADED
        gate = verify_arms(arms, execution['arms'])['arms'][arm]['gate']
        state = verify_artifacts(directory, inventory, download=False)
        tokenizer, report = granite_tokenizer.load(directory)
        reply['tokenizer'] = report
        parent, _ = reference.load_model(directory, 'baseline')
        cases = first.load_cases(read(ROOT / canonical.PLAN))
        anchor_plan = read(ROOT / reference.PLAN)
        if replay:
            replay_ids = read(ROOT / canonical.PLAN)['replay_ids']
            cases, anchor_plan = [c for c in cases if c['id'] in replay_ids], {**anchor_plan, 'tasks': []}
        model, _ = reference.load_model(directory, 'baseline')
        reply['checkpoint'] = trainer.load_trainable(model, arm, plan['training'], arms / f'{arm}-checkpoint')
        reply['served_projections_converted'] = trainer.serving(model, plan['training'])

        def feature(case):
            return accelerator.boundary_feature(parent, tokenizer, policy, case, 'cpu')

        make_responder = None
        if execution.get('serving') == 'prefix-cache':
            from neuroshard.evolution.assistant_serving import cached_responder as make_responder
        reply['serving'] = execution.get('serving', 'recompute')
        reply.update(evaluate_arm(parent, model, tokenizer, gate, feature, cases, policy, anchor_plan, make_responder))
        if file_state(directory, inventory) != state:
            raise ValueError('parent checkpoint changed during evaluation')
        reply['execution_completed'] = True
    except Exception as error:
        reply['error'] = f'{type(error).__name__}: {error}'
    finally:
        reply.update(checked_encodes=tokenizer.checked_encodes if tokenizer else 0,
                     wall_seconds=time.monotonic() - started, process_cpu_seconds=time.process_time(),
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        save(request_path.parent / 'reply.json', reply, exclusive=True)


def served(canonical_plan, cases, rows, canonical_report, replay_rows):
    """A1's usable-foundation check on the version that would be served.

    Workspace episodes run through the routed arm; anchors are served by the
    parent, so their outcomes are the canonical parent's.
    """
    from neuroshard.evolution import assistant_experience_gate as gate

    q = canonical_plan['qualification']
    primitive_families = data.FAMILIES[:4]
    family = {case['id']: case['family'] for case in cases}
    correct = {row['id'] for row in rows if row['score']['passed']}
    by_family = {name: sum(family[i] == name for i in correct) for name in primitive_families}
    originals = {row['id']: row for row in rows}
    replay_ids = canonical_plan['replay_ids']
    replayed = (replay_rows is not None and len(replay_rows) == len(replay_ids)
                and {row['id'] for row in replay_rows} == set(replay_ids)
                and all(workflow.replay_matches(originals[row['id']], row) for row in replay_rows))
    p95 = gate.p95(rows, routed=True)
    checks = {
        'primitive': sum(by_family.values()) >= q['minimum_primitive_successes']
        and all(count >= q['minimum_per_primitive_family'] for count in by_family.values()),
        'anchors': bool(canonical_report['canonical_anchor_gate'] and not canonical_report['prior_anchor_successes_lost']),
        'p95': p95 <= q['p95_episode_seconds'],
        'fresh_process_replay': replayed,
    }
    return {'passed': all(checks.values()), 'checks': checks, 'primitive_by_family': by_family,
            'primitive_correct': sum(by_family.values()), 'p95_episode_seconds': p95,
            'anchor_routing': 'parent', 'anchor_correct': canonical_report['anchor_correct'],
            'replayed_ids': sorted(replay_ids)}


def assess(plan, cases, canonical_result, replies, replay_rows=None, canonical_plan=None, served_arm=SERVED):
    """The development gate in ``plan`` over every evaluated arm in ``replies``, each episode rescored.

    With both arms, the addition is gated against the update control. With one arm,
    ``plan`` must declare a gate with no update comparison, and that arm is gated alone.
    """
    from neuroshard.evolution import assistant_experience_gate as gate

    policy = read(ROOT / plan['policy'])
    by_id = {case['id']: case for case in cases}
    arms = [arm for arm in ARMS if arm in replies]
    rows = {}
    for arm in arms:
        rows[arm] = replies[arm]['episodes']
        for row in rows[arm]:
            if workflow.score(by_id[row['id']], row, policy) != row['score']:
                raise ValueError('evaluation outcome rescore differs')
    parent, protected = canonical_result['primary']['episodes'], canonical_result['report']['protected_workflow_ids']
    if len(arms) == len(ARMS):
        report = gate.development(plan, cases, parent, rows['update'], rows['addition'], protected)
    else:
        (arm,) = arms
        report = gate.development(plan, cases, parent, None, rows[arm], protected, name=arm)
    anchors = canonical_result['report']['protected_anchor_ids']
    report['forced_anchor_forgetting'] = {arm: forgetting(replies[arm]['forced_anchors'], anchors) for arm in arms}
    report['selected_arm_episodes'] = {arm: sum(r['selected'] == 'arm' for r in rows[arm]) for arm in arms}
    if canonical_plan is not None:
        for row in replay_rows or []:
            if workflow.score(by_id[row['id']], row, policy) != row['score']:
                raise ValueError('replay outcome rescore differs')
        report['a1_served'] = served(canonical_plan, cases, rows[served_arm], canonical_result['report'], replay_rows)
        report['a1_served']['served'] = served_arm
        report['development_and_a1_passed'] = bool(report['passed'] and report['a1_served']['passed'])
    return report


def run(home, models):
    configure()
    source = freeze()
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    execution = read(ROOT / EXECUTION)
    plan = read(ROOT / PLAN)
    binding = {'freeze': source, 'profile': PROFILE, 'plan_sha256': sha256(ROOT / PLAN)}
    save(home / 'binding.json', binding, exclusive=True)
    result = {'binding': binding, 'execution_completed': False, 'checklist_credit': False,
              'admission_evidence': False, 'confirmation_opened': False}
    try:
        evaluated, served_arm = systems(execution)
        verify_arms(ROOT / UPLOADED, execution['arms'])
        prepared = launch(home, models, binding, 'baseline', 'prepare', execution['prepare_seconds'],
                          execution['memory_bytes'], worker_script=SCRIPT)
        if not prepared['execution_completed']:
            raise ValueError(prepared.get('error', 'parent artifacts were not prepared'))
        # One fresh worker at a time, as in the canonical baseline, so latency is comparable.
        replies = {arm: launch(home, models, binding, arm, 'development', execution['worker_seconds'],
                               execution['memory_bytes'], worker_script=SCRIPT) for arm in evaluated}
        result['replies'] = replies
        failed = [arm for arm in evaluated if not replies[arm]['execution_completed']]
        if failed:
            raise ValueError(f'incomplete evaluation for {failed}')
        replay_rows, canonical_plan = None, None
        if 'a1_served' in execution:
            canonical_plan = read(ROOT / canonical.PLAN)
            result['replay'] = launch(home, models, binding, served_arm, 'replay', execution['replay_seconds'],
                                      execution['memory_bytes'], worker_script=SCRIPT)
            if result['replay']['execution_completed']:
                replay_rows = result['replay']['episodes']
        canonical_result = read(ROOT / execution['canonical_result']['path'])
        if sha256(ROOT / execution['canonical_result']['path']) != execution['canonical_result']['sha256']:
            raise ValueError('canonical parent result changed')
        gated = read(ROOT / execution['gate_plan']) if 'gate_plan' in execution else plan
        result['report'] = assess(gated, first.load_cases(read(ROOT / canonical.PLAN)), canonical_result, replies,
                                  replay_rows, canonical_plan, served_arm)
        result['execution_completed'] = True
    except Exception as error:
        result['error'] = str(error)
    save(home / 'result.json', result, exclusive=True)
    return result
