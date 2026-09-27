"""Development evaluation of both trained systems on the canonical parent's CPU runtime.

Each system selects once per episode from the frozen parent feature, then runs
the whole episode with the chosen model. The parent control is the canonical
re-baseline, pinned by its result digest; protected successes come from it.
Each arm also answers every original anchor with selection forced on, which
measures forgetting directly; the routed system serves anchors with the parent.
"""

from pathlib import Path
import resource
import time

from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution import assistant_workflow_baseline as first
from neuroshard.evolution import assistant_workflow_canonical as canonical
from neuroshard.evolution import granite_reference as reference
from neuroshard.evolution import granite_tokenizer
from neuroshard.evolution.modular_reference_execution import ROOT, file_state, read, save, sha256, verify_artifacts

ARMS = ('update', 'addition')


def verify_arms(directory, pinned):
    """Uploaded checkpoints and gates must match the digests pinned before evaluation."""
    directory = Path(directory)
    for arm in ARMS:
        manifest = read(directory / f'{arm}-checkpoint' / 'manifest.json')
        if manifest['trainable_sha256'] != pinned[arm]['trainable_sha256'] or manifest['arm'] != arm:
            raise ValueError(f'{arm} checkpoint differs from the pinned training result')
    if sha256(directory / 'integration.json') != pinned['integration_sha256']:
        raise ValueError('integration gates differ from the pinned result')
    return read(directory / 'integration.json')


def evaluate(parent, arms, tokenizer, gates, features, cases, policy, anchor_plan):
    """Selected complete episodes per system, plus forced-arm anchor answers."""
    from neuroshard.evolution import assistant_selector as selector

    respond = {'parent': first.native_responder(parent, tokenizer, policy)}
    report = {}
    for arm, model in arms.items():
        respond[arm] = first.native_responder(model, tokenizer, policy)
        rows = []
        for case in cases:
            chosen = arm if selector.choose(gates[arm], features[case['id']]) else 'parent'
            row = workflow.execute(case, respond[chosen], policy)
            rows.append({**row, 'selected': chosen})
        anchors = [reference.generate(model, tokenizer, anchor_plan, task, 'baseline') for task in anchor_plan['tasks']]
        report[arm] = {'episodes': rows, 'forced_anchors': anchors}
    return report


def forgetting(anchor_rows, protected_ids):
    passed = {row['id'] for row in anchor_rows if row['passed']}
    return {'correct': len(passed), 'lost_protected': sorted(set(protected_ids) - passed)}


def worker(request_path, execution):
    """Runs inside the canonical CPU runtime; execution pins the canonical result and uploaded arms."""
    from neuroshard.evolution import assistant_experience_run as accelerator
    from neuroshard.evolution import assistant_experience_train as trainer
    import torch

    request_path = Path(request_path)
    request = read(request_path)
    plan = read(ROOT / accelerator.PLAN)
    policy = read(ROOT / plan['policy'])
    torch.set_num_threads(execution['threads'])
    torch.set_num_interop_threads(1)
    reply = {'binding': request['binding'], 'execution_completed': False}
    started = time.monotonic()
    tokenizer = None
    try:
        gates = {arm: value['gate'] for arm, value in verify_arms(request['arms'], execution['arms'])['arms'].items()}
        inventory = read(ROOT / reference.ARTIFACTS)['models']['baseline']
        directory = Path(request['models']) / 'baseline'
        state = verify_artifacts(directory, inventory, download=True)
        tokenizer, report = granite_tokenizer.load(directory)
        reply['tokenizer'] = report
        parent, _ = reference.load_model(directory, 'baseline')
        cases = first.load_cases(read(ROOT / canonical.PLAN))
        features = {c['id']: accelerator.boundary_feature(parent, tokenizer, policy, c, 'cpu') for c in cases}
        arms = {}
        for arm in ARMS:
            model, _ = reference.load_model(directory, 'baseline')
            trainer.load_trainable(model, arm, plan['training'], Path(request['arms']) / f'{arm}-checkpoint')
            arms[arm] = model
        reply['systems'] = evaluate(parent, arms, tokenizer, gates, features, cases, policy,
                                    read(ROOT / reference.PLAN))
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


def assess(plan, cases, canonical_result, reply):
    from neuroshard.evolution import assistant_experience_gate as gate

    policy = read(ROOT / plan['policy'])
    by_id = {case['id']: case for case in cases}
    parent = canonical_result['primary']['episodes']
    protected = canonical_result['report']['protected_workflow_ids']
    systems = {}
    for arm in ARMS:
        rows = reply['systems'][arm]['episodes']
        for row in rows:
            if workflow.score(by_id[row['id']], row, policy) != row['score']:
                raise ValueError('evaluation outcome rescore differs')
        systems[arm] = rows
    report = gate.development(plan, cases, parent, systems['update'], systems['addition'], protected)
    anchors = canonical_result['report']['protected_anchor_ids']
    report['forced_anchor_forgetting'] = {arm: forgetting(reply['systems'][arm]['forced_anchors'], anchors)
                                          for arm in ARMS}
    report['selected_arm_episodes'] = {arm: sum(r['selected'] == arm for r in systems[arm]) for arm in ARMS}
    return report
