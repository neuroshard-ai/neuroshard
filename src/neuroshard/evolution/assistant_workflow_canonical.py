"""Parent workspace re-baseline under the pinned Granite tokenizer; no learning or final access.

The first baseline encoded every prompt with transformers' GPT-2 pre-tokenizer
instead of the pinned tokenizer.json. Its failure remains published. This
execution repeats the same anchors and development workflows with every encode
checked, and records paired differences. It is a measurement, not admission.
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
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution import granite_context_reference as context
from neuroshard.evolution import granite_reference as reference
from neuroshard.evolution import granite_tokenizer
from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, launch, read, save, sha256, verify_artifacts,
)

PLAN = 'config/experiments/assistant-workflow-canonical.json'
EXECUTION = 'config/experiments/assistant-workflow-canonical-execution.json'
SCRIPT = 'scripts/run_assistant_workflow_canonical.py'
PROFILE = 'assistant-workflow-canonical'


def committed_sources(root=ROOT):
    execution = read(root / EXECUTION)
    for name, digest in execution['contracts'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'changed workflow contract: {name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in execution['sources']:
        contents = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root)
        if contents != (root / name).read_bytes():
            raise ValueError(f'uncommitted workflow source: {name}')
        sources[name] = sha256(root / name)
    return {'commit': commit, 'sources': sources}


def configure():
    if 'torch' in sys.modules:
        raise ValueError('configure the research runtime before importing torch')
    for key, value in read(ROOT / EXECUTION)['environment'].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def freeze():
    source = committed_sources()
    execution = read(ROOT / EXECUTION)
    packages = {key: importlib.metadata.version(key) for key in execution['packages']}
    if packages != execution['packages'] or platform.python_version() != execution['python']:
        raise ValueError('workflow runtime differs')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('workflow requires Linux x86_64')
    cpu = Path('/proc/cpuinfo').read_text().split()
    if any(flag not in cpu for flag in execution['required_cpu_flags']):
        raise ValueError('workflow CPU lacks required instructions')
    if any(os.environ.get(key) != value for key, value in execution['environment'].items()):
        raise ValueError('workflow numerical environment differs')
    reference.upstream_path()
    return {**source, 'packages': packages, 'python': platform.python_version(),
            'upstream': read(ROOT / reference.ARTIFACTS)['upstream']['commit']}


def expected_encodes(reply):
    """The responder encodes every request once, and again when it generates."""
    generations = [g for row in reply.get('episodes', []) for g in row['generations']]
    return len(reply.get('anchors', [])) + sum(1 + bool(g.get('executed', True)) for g in generations)


def check_tokenizer(plan, reply):
    report = reply.get('tokenizer') or {}
    pinned = plan['tokenizer']
    return (report.get('pipeline_sha256') == pinned['pipeline_sha256']
            and report.get('fixture_sha256') == pinned['fixture_sha256']
            and reply.get('checked_encodes') == expected_encodes(reply))


def assess(plan, primary):
    cases = {case['id']: case for case in first.load_cases(plan)}
    policy = read(ROOT / plan['policy'])
    rows = primary.get('episodes', [])
    if len(rows) != len({row['id'] for row in rows}) or any(row['id'] not in cases for row in rows):
        raise ValueError('duplicate or unknown workflow episode')
    for row in rows:
        if workflow.score(cases[row['id']], row, policy) != row['score']:
            raise ValueError('workflow outcome rescore differs')
    original = read(ROOT / reference.PLAN)
    tasks = {task['id']: task for task in original['tasks']}
    anchors = primary.get('anchors', [])
    if len(anchors) != len({row['id'] for row in anchors}) or any(row['id'] not in tasks for row in anchors):
        raise ValueError('duplicate or unknown assistant anchor')
    for row in anchors:
        if row['passed'] != reference.score(tasks[row['id']], row['text'], row['terminated']):
            raise ValueError('assistant anchor rescore differs')
    tokenizer = check_tokenizer(plan, primary)
    complete = (primary.get('execution_completed', False) and tokenizer
                and {row['id'] for row in rows} == set(cases) and {row['id'] for row in anchors} == set(tasks))
    anchor_gate = reference.baseline_gate(original, anchors)
    anchor_passed = {row['id'] for row in anchors if row['passed']}
    q = plan['qualification']
    correct = {row['id'] for row in rows if row['score']['passed']}
    by_family = {family: sum(c['id'] in correct for c in cases.values() if c['family'] == family)
                 for family in data.FAMILIES}
    p95 = context.percentile([row['seconds'] for row in rows], .95) if rows else None
    primitive = sum(by_family[family] for family in data.FAMILIES[:4])
    qualification = bool(complete and anchor_gate and primitive >= q['minimum_primitive_successes']
        and all(by_family[family] >= q['minimum_per_primitive_family'] for family in data.FAMILIES[:4])
        and p95 <= q['p95_episode_seconds'] and primary['peak_rss_bytes'] <= q['maximum_rss_bytes'])
    prior_anchors, prior_workflows = plan['prior_protected_anchor_ids'], plan['prior_protected_workflow_ids']
    return {'complete': complete, 'tokenizer_conforms': tokenizer, 'canonical_anchor_gate': anchor_gate,
            'anchor_correct': len(anchor_passed), 'anchor_total': len(tasks),
            'protected_anchor_ids': sorted(anchor_passed),
            'prior_anchor_successes_lost': [key for key in prior_anchors if key not in anchor_passed],
            'anchors_gained': sorted(anchor_passed - set(prior_anchors)),
            'baseline_qualified': qualification, 'correct': len(correct), 'total': len(cases),
            'by_family': by_family, 'primitive_correct': primitive, 'compound_correct': len(correct) - primitive,
            'protected_workflow_ids': sorted(correct),
            'prior_workflow_successes_lost': [key for key in prior_workflows if key not in correct],
            'workflows_gained': sorted(correct - set(prior_workflows)),
            'insufficient_development_headroom': len(correct) > 20, 'p95_episode_seconds': p95,
            'work': {key: sum(row[key] for row in rows) for key in ('model_calls', 'tool_calls', 'input_tokens', 'output_tokens')},
            'native_generation_calls': sum(g.get('executed', True) for row in rows for g in row['generations']),
            'prose_quality_evaluated': False, 'checklist_credit': False,
            'training_authorized': False, 'admission_evidence': False, 'decentralized_execution_proven': False}


def worker(request_path):
    configure()
    request_path = Path(request_path)
    request = read(request_path)
    if freeze() != request['freeze']:
        raise ValueError('workflow worker differs from freeze')
    import torch
    plan = read(ROOT / PLAN)
    torch.set_num_threads(plan['resources']['threads'])
    torch.set_num_interop_threads(1)
    phase = request['phase']
    if request['model'] != 'baseline' or phase not in ('primary', 'replay'):
        raise ValueError('unsupported workflow worker role')
    reply = {'binding': request['binding'], 'execution_completed': False, 'anchors': [], 'episodes': []}
    started = time.monotonic()
    tokenizer = None
    try:
        inventory = read(ROOT / reference.ARTIFACTS)['models']['baseline']
        directory = Path(request['models']) / 'baseline'
        state = verify_artifacts(directory, inventory, download=True)
        model, legacy = reference.load_model(directory, 'baseline')
        tokenizer, report = granite_tokenizer.load(directory)
        reply['tokenizer'] = {**report, 'legacy_autotokenizer_conforms': granite_tokenizer.agrees(legacy, tokenizer.reference)}
        if phase == 'primary':
            anchor_plan = read(ROOT / reference.PLAN)
            for task in anchor_plan['tasks']:
                row = reference.generate(model, tokenizer, anchor_plan, task, 'baseline')
                reply['anchors'].append(row)
                save(request_path.parent / f"anchor-{task['id']}.json", row, exclusive=True)
        policy = read(ROOT / plan['policy'])
        respond = first.native_responder(model, tokenizer, policy)
        for case in first.load_cases(plan):
            if phase == 'replay' and case['id'] not in plan['replay_ids']:
                continue
            row = workflow.execute(case, respond, policy)
            reply['episodes'].append(row)
            save(request_path.parent / f"episode-{case['id']}.json", row, exclusive=True)
            save(request_path.parents[2] / 'status.json', {'state': 'executing-workflows', 'phase': phase,
                 'completed': len(reply['episodes']), 'last_id': case['id'], 'last_passed': row['score']['passed']})
        if file_state(directory, inventory) != state:
            raise ValueError('parent checkpoint changed during workflow execution')
        reply.update(execution_completed=True, checkpoint_state=state,
                     checkpoint_bytes=sum((directory / name).stat().st_size for name in inventory['files']))
    except Exception as error:
        reply['error'] = str(error)
    finally:
        reply.update(checked_encodes=tokenizer.checked_encodes if tokenizer else 0,
                     wall_seconds=time.monotonic() - started, process_cpu_seconds=time.process_time(),
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        save(request_path.parent / 'reply.json', reply, exclusive=True)


def run(home, models):
    configure()
    source = freeze()
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    plan = read(ROOT / PLAN)
    binding = {'freeze': source, 'profile': PROFILE, 'plan_sha256': sha256(ROOT / PLAN)}
    save(home / 'binding.json', binding, exclusive=True)
    result = {'binding': binding, 'execution_completed': False, 'baseline_passed': False,
              'checklist_credit': False, 'training_authorized': False}
    try:
        primary = launch(home, models, binding, 'baseline', 'primary', plan['resources']['primary_worker_seconds'],
                         plan['resources']['memory_bytes'], worker_script=SCRIPT)
        result['primary'] = primary
        if not primary['execution_completed']:
            raise ValueError(primary.get('error', 'incomplete primary execution'))
        result['report'] = assess(plan, primary)
        if result['report']['baseline_qualified']:
            replay = launch(home, models, binding, 'baseline', 'replay', plan['resources']['replay_worker_seconds'],
                            plan['resources']['memory_bytes'], worker_script=SCRIPT)
            result['replay'] = replay
            originals = {row['id']: row for row in primary['episodes']}
            if (not replay['execution_completed'] or not check_tokenizer(plan, replay)
                    or {row['id'] for row in replay['episodes']} != set(plan['replay_ids'])
                    or len(replay['episodes']) != len(plan['replay_ids'])
                    or any(not workflow.replay_matches(originals[row['id']], row) for row in replay['episodes'])):
                raise ValueError('workflow fresh-process replay differs')
            result['baseline_passed'] = True
        result['execution_completed'] = True
    except Exception as error:
        result['error'] = str(error)
    save(home / 'result.json', result, exclusive=True)
    return result
