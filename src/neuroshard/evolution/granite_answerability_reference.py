"""Prospective published-module comparison through the exact-evidence interface.

No new weights, learned routing, promotion or checklist credit. Parent and
candidate run in isolated processes. Gold labels enter scoring only.
"""

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

from neuroshard.evolution import evidence_selection as evidence
from neuroshard.evolution import granite_context_reference as context
from neuroshard.evolution import granite_evidence_diagnostic as diagnostic
from neuroshard.evolution import granite_reference as reference
from neuroshard.evolution.finite_decision import FiniteDecision
from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, identity, launch, read, save, sha256, verify_artifacts,
)

PLAN = 'config/experiments/granite-answerability-reference.json'
EXECUTION = 'config/experiments/granite-answerability-reference-execution.json'
SCRIPT = 'scripts/run_granite_answerability_reference.py'
ARMS = ('selection', 'parent-check', 'module-check')
configure = reference.configure


def committed_sources(root=ROOT):
    execution = read(root / EXECUTION)
    for name, digest in execution['contracts'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'changed answerability contract: {name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in execution['sources']:
        data = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root)
        if data != (root / name).read_bytes():
            raise ValueError(f'uncommitted answerability source: {name}')
        sources[name] = sha256(root / name)
    return {'commit': commit, 'sources': sources}


def freeze():
    binding = committed_sources()
    execution = read(ROOT / EXECUTION)
    packages = {key: importlib.metadata.version(key) for key in execution['packages']}
    if packages != execution['packages'] or platform.python_version() != execution['python']:
        raise ValueError('answerability runtime differs')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('answerability requires Linux x86_64')
    cpu = Path('/proc/cpuinfo').read_text().split()
    if any(flag not in cpu for flag in execution['required_cpu_flags']):
        raise ValueError('answerability CPU lacks required instructions')
    if any(os.environ.get(key) != value for key, value in execution['environment'].items()):
        raise ValueError('answerability numerical environment differs')
    reference.upstream_path()
    return {**binding, 'packages': packages, 'python': platform.python_version(),
            'upstream': read(ROOT / reference.ARTIFACTS)['upstream']['commit']}


def task_sets(plan):
    fresh = read(ROOT / plan['cases'])['tasks']
    opened = read(ROOT / plan['opened_retention_plan'])
    retriever = context.Retriever(opened['corpus'], opened['retrieval'])
    retained = []
    for task in opened['tasks']:
        messages, sources, _ = diagnostic.prepare_request(task, retriever)
        retained.append({**task, 'messages': messages, 'sources': sources})
    return fresh, retained


def public_input(task):
    messages = context.public_input(task)
    # Explicitly project public fields; scoring metadata cannot become model input.
    sources = [{key: row[key] for key in ('id', 'text', 'spans')} for row in task['sources']]
    return messages, sources, evidence.build_menu(messages, sources)


def selection_task(plan, messages, menu):
    selector = read(ROOT / plan['selector_plan'])
    return context.stage_task(diagnostic.model_messages(selector, messages, menu, 'selection'), 'select')


def checker_task(plan, messages, sources, choice, menu, which):
    option = next(row for row in menu['choices'] if row['choice'] == choice)
    source = next(row for row in sources if row['id'] == option['source_id'])
    task = context.stage_task([{'role': 'system', 'content': plan['checker_instruction']}] + messages,
                              'check', plan['module'] if which == 'modular' else None)
    task['documents'] = [{'doc_id': source['id'], 'text': source['text']}]
    return task


def check_prompt(tokenizer, task, which):
    kwargs = {'tools': None, 'documents': task['documents'],
              'add_generation_prompt': True, 'tokenize': False}
    if which == 'modular':
        kwargs['adapter_name'] = task['adapter']
    return tokenizer.apply_chat_template(task['messages'], **kwargs)


def finish(messages, sources, menu, selection, check):
    choice = selection['text']
    if choice == evidence.ABSTAIN:
        if check is not None:
            raise ValueError('abstention must skip checking')
    else:
        if check is None or not check['terminated']:
            raise ValueError('selected record requires a completed check')
        verdict = reference.strict_json(check['text'])
        if verdict not in ('answerable', 'unanswerable'):
            raise ValueError('invalid answerability decision')
        if verdict == 'unanswerable':
            choice = evidence.ABSTAIN
    return evidence.resolve(messages, sources, {'invocation_root': menu['invocation_root'], 'choice': choice})


def is_correct(task, receipt):
    return reference.score({'kind': 'json', 'expected': task['expected']},
                           json.dumps(receipt['answer'], ensure_ascii=False), True)


def pipeline(model, tokenizer, plan, task, which):
    started = time.monotonic()
    messages, sources, menu = public_input(task)
    select_task = selection_task(plan, messages, menu)
    labels = evidence.label_token_ids(tokenizer, menu)
    generation = {'generation': {**plan['generation'], 'max_new_tokens': 1}}
    selection = reference.generate(model, tokenizer, generation, select_task, which,
        generation_kwargs={'prefix_allowed_tokens_fn': lambda batch, prefix: list(labels.values())})
    if selection['token_ids'] != [labels.get(selection['text'])]:
        raise ValueError('selection escaped one-token choice space')
    initial = evidence.resolve(messages, sources, {'invocation_root': menu['invocation_root'],
                                                  'choice': selection['text']})
    selection_seconds = time.monotonic() - started
    check = None
    if selection['text'] != evidence.ABSTAIN:
        check_task = checker_task(plan, messages, sources, selection['text'], menu, which)
        prompt = check_prompt(tokenizer, check_task, which)
        count = len(tokenizer(prompt, add_special_tokens=False)['input_ids'])
        decision = FiniteDecision(tokenizer, plan['checker_decoding']['choices'],
                                  plan['checker_decoding']['eos_token_id'])
        if decision.max_new_tokens > plan['checker_decoding']['max_new_tokens']:
            raise ValueError('finite decision exceeds declared cap')
        check = reference.generate(model, tokenizer, plan, check_task, which,
            generation_kwargs={'prefix_allowed_tokens_fn': decision.constraint(count)})
        if decision.decode(check['token_ids']) != check['text'] or check['prompt_sha256'] != identity(prompt):
            raise ValueError('checker differs from declared finite output or native prompt')
    receipt = finish(messages, sources, menu, selection, check)
    calls = [selection] + ([check] if check else [])
    return {'id': task['id'], 'model': which, 'task_sha256': identity(task), 'menu': menu,
            'selection': selection, 'check': check, 'selection_receipt': initial, 'receipt': receipt,
            'selection_seconds': selection_seconds, 'seconds': time.monotonic() - started,
            'generation_calls': len(calls), 'input_tokens': sum(len(c['input_token_ids']) for c in calls),
            'output_tokens': sum(len(c['token_ids']) for c in calls), 'passed': is_correct(task, receipt)}


def validate_row(plan, task, row, which):
    messages, sources, menu = public_input(task)
    tokens = read(ROOT / plan['preflight'])['models'][which]
    if row['model'] != which or row['task_sha256'] != identity(task) or row['menu'] != menu:
        raise ValueError('answerability task binding differs')
    selection, check = row['selection'], row['check']
    if (selection['task_sha256'] != identity(selection_task(plan, messages, menu))
            or selection['model'] != which
            or selection['token_ids'] != [tokens['choice_tokens'].get(selection['text'])]
            or selection['terminated']):
        raise ValueError('selection binding differs')
    for receipt in (row['selection_receipt'], row['receipt']):
        evidence.verify_receipt(messages, sources, receipt)
    if row['selection_receipt']['choice'] != selection['text']:
        raise ValueError('initial receipt differs from selected label')
    if check is not None:
        expected = checker_task(plan, messages, sources, selection['text'], menu, which)
        if check['task_sha256'] != identity(expected) or check['model'] != which:
            raise ValueError('checker binding differs')
        if check['token_ids'] != tokens['checker_paths'].get(check['text']):
            raise ValueError('checker tokens differ from finite decision')
    if row['receipt'] != finish(messages, sources, menu, selection, check):
        raise ValueError('final receipt does not implement the serving rule')
    if row['passed'] != is_correct(task, row['receipt']):
        raise ValueError('answerability rescore differs')
    calls = [selection] + ([check] if check else [])
    work = {'generation_calls': len(calls), 'input_tokens': sum(len(c['input_token_ids']) for c in calls),
            'output_tokens': sum(len(c['token_ids']) for c in calls)}
    if any(row[key] != value for key, value in work.items()):
        raise ValueError('answerability work accounting differs')
    if which == 'modular':
        if set(selection['route_counts']) != {'0'}:
            raise ValueError('selector must use only base weights')
        if check and (str(plan['module_route']) not in check['route_counts']
                      or set(check['route_counts']) - {'0', str(plan['module_route'])}):
            raise ValueError('answerability adapter did not activate correctly')
    elif any(call['route_counts'] or call['route_trace'] for call in calls):
        raise ValueError('parent unexpectedly routed to a module')


def anchor_gate(plan, rows):
    original = read(ROOT / reference.PLAN)
    tasks = {task['id']: task for task in original['tasks']}
    if len(rows) != len({r['id'] for r in rows}) or any(r['id'] not in tasks for r in rows):
        raise ValueError('duplicate or unknown anchor')
    for row in rows:
        if row['passed'] != reference.score(tasks[row['id']], row['text'], row['terminated']):
            raise ValueError('anchor rescore differs')
    by_id = {row['id']: row for row in rows}
    return (set(by_id) == set(tasks) and reference.baseline_gate(original, rows)
            and all(by_id[key]['passed'] for key in plan['protected_ids']))


def same_selection(left, right):
    return all(left['selection'][key] == right['selection'][key]
               for key in ('prompt_sha256', 'input_token_ids', 'token_ids', 'text', 'terminated'))


def assess(plan, workers):
    fresh, retained = task_sets(plan)
    tasks = {task['id']: task for task in fresh + retained}
    fresh_ids, retained_ids = {t['id'] for t in fresh}, {t['id'] for t in retained}
    report = {'complete': False, 'quality_gate': False, 'retention': {}, 'checklist_credit': False,
              'admission_evidence': False, 'new_learning_proven': False, 'training_authorized': False}
    groups = {}
    for which, worker in workers.items():
        rows = worker.get('rows', [])
        if len(rows) != len({r['id'] for r in rows}) or any(r['id'] not in tasks for r in rows):
            raise ValueError('duplicate or unknown answerability result')
        group = groups[which] = {row['id']: row for row in rows}
        for key, row in group.items():
            validate_row(plan, tasks[key], row, which)
        # The optional parent checker is a competing control, not accepted behavior.
        # Its failures must not prevent evaluating a module that might fix them.
        retained_correct = all(
            is_correct(tasks[key], group[key]['selection_receipt']) if which == 'baseline'
            else group[key]['passed'] for key in retained_ids if key in group)
        report['retention'][which] = (anchor_gate(plan, worker.get('anchors', []))
            and retained_ids <= set(group) and retained_correct)
    if 'baseline' not in groups or not fresh_ids <= set(groups['baseline']):
        return report
    parent = groups['baseline']
    selector = {key: {**row, 'passed': is_correct(tasks[key], row['selection_receipt']),
                      'seconds': row['selection_seconds'], 'generation_calls': 1,
                      'input_tokens': len(row['selection']['input_token_ids']),
                      'output_tokens': len(row['selection']['token_ids'])} for key, row in parent.items()}
    controls = {'selection': selector, 'parent-check': parent}
    report['control_correct'] = {arm: sum(group[key]['passed'] for key in fresh_ids)
                                 for arm, group in controls.items()}
    report['insufficient_headroom'] = any(count > len(fresh) - plan['quality']['minimum_net_gain']
                                          for count in report['control_correct'].values())
    if 'modular' not in groups or any(set(group) != set(tasks) for group in groups.values()):
        return report
    candidate = groups['modular']
    report['complete'] = all(w.get('execution_completed') for w in workers.values())
    report['selector_parity'] = all(same_selection(parent[key], candidate[key]) for key in tasks)
    arms = {**controls, 'module-check': candidate}
    report['correct'] = {arm: sum(group[key]['passed'] for key in fresh_ids) for arm, group in arms.items()}
    report['candidate_by_category'] = {category: sum(candidate[t['id']]['passed'] for t in fresh
                                                       if t['category'] == category)
                                      for category in plan['quality']['minimum_per_category']}
    report['comparisons'] = {arm: context.paired_gain(plan, fresh, candidate, group)
                             for arm, group in controls.items()}
    report['p95_seconds'] = {arm: context.percentile([group[key]['seconds'] for key in fresh_ids], .95)
                             for arm, group in arms.items()}
    report['work'] = {arm: {key: sum(group[t['id']][key] for t in fresh)
                            for key in ('input_tokens', 'output_tokens', 'generation_calls')}
                      for arm, group in arms.items()}
    report['retention_failures'] = {which: [key for key in retained_ids if not group[key]['passed']]
                                     for which, group in groups.items()}
    report['isolated_worker_peak_rss_bytes'] = {which: worker['peak_rss_bytes'] for which, worker in workers.items()}
    q = plan['quality']
    report['quality_gate'] = bool(report['complete'] and all(report['retention'].values())
        and report['selector_parity'] and report['correct']['module-check'] >= q['minimum_correct']
        and all(report['candidate_by_category'][key] >= count for key, count in q['minimum_per_category'].items())
        and all(c['net'] >= q['minimum_net_gain']
                and c['block_bootstrap_lower_95'] > q['minimum_lower_bound_exclusive']
                and len(c['lost_ids']) <= q['maximum_lost_control_answers'] for c in report['comparisons'].values())
        and report['p95_seconds']['module-check'] <= q['p95_seconds']
        and report['p95_seconds']['module-check'] <= q['p95_ratio_vs_parent_check'] * report['p95_seconds']['parent-check']
        and all(w['peak_rss_bytes'] <= q['maximum_worker_peak_rss_bytes'] for w in workers.values()))
    return report


def replay_matches(original, replay):
    for key in ('menu', 'selection_receipt', 'receipt', 'passed'):
        if original[key] != replay[key]:
            return False
    for key in ('selection', 'check'):
        left, right = original[key], replay[key]
        if left is None or right is None:
            if left != right:
                return False
        elif any(left[field] != right[field] for field in (
                'prompt_sha256', 'input_token_ids', 'token_ids', 'text', 'terminated', 'route_trace')):
            return False
    return True


def worker(request_path):
    configure()
    request_path = Path(request_path)
    request = read(request_path)
    if freeze() != request['freeze']:
        raise ValueError('answerability worker differs from freeze')
    import torch
    plan = read(ROOT / PLAN)
    torch.set_num_threads(plan['resources']['threads'])
    torch.set_num_interop_threads(1)
    which, phase = request['model'], request['phase']
    if which not in ('baseline', 'modular') or phase not in ('primary', 'replay'):
        raise ValueError('unknown answerability worker role')
    reply = {'binding': request['binding'], 'model': which, 'phase': phase, 'execution_completed': False,
             'rows': [], 'anchors': []}
    started = time.monotonic()
    try:
        inventory = read(ROOT / reference.ARTIFACTS)['models'][which]
        directory = Path(request['models']) / which
        state = verify_artifacts(directory, inventory, download=True)
        model, tokenizer = reference.load_model(directory, which)
        fresh, retained = task_sets(plan)
        if phase == 'primary':
            anchor_plan = read(ROOT / reference.PLAN)
            for task in anchor_plan['tasks']:
                row = reference.generate(model, tokenizer, anchor_plan, task, which)
                reply['anchors'].append(row)
                save(request_path.parent / f"anchor-{task['id']}.json", row, exclusive=True)
            if not anchor_gate(plan, reply['anchors']):
                raise ValueError('protected assistant retention failed; no source-task generation')
            tasks = retained + fresh
        else:
            tasks = [task for task in fresh if task['id'] in plan['replay_ids']]
        for task in tasks:
            row = pipeline(model, tokenizer, plan, task, which)
            validate_row(plan, task, row, which)
            reply['rows'].append(row)
            save(request_path.parent / f"answer-{task['id']}.json", row, exclusive=True)
            save(request_path.parents[2] / 'status.json', {'state': 'generating', 'model': which,
                 'phase': phase, 'completed': len(reply['rows']), 'last_id': task['id']})
        if file_state(directory, inventory) != state:
            raise ValueError('answerability checkpoint changed')
        reply.update(execution_completed=True, checkpoint_state=state,
                     checkpoint_bytes=sum((directory / name).stat().st_size for name in inventory['files']))
    except Exception as error:
        reply['error'] = str(error)
    finally:
        reply.update(wall_seconds=time.monotonic() - started, process_cpu_seconds=time.process_time(),
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        save(request_path.parent / 'reply.json', reply, exclusive=True)


def run(home, models):
    configure()
    source = freeze()
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    plan = read(ROOT / PLAN)
    binding = {'freeze': source, 'profile': 'granite-answerability-reference', 'plan_sha256': sha256(ROOT / PLAN)}
    save(home / 'binding.json', binding, exclusive=True)
    result = {'execution_completed': False, 'binding': binding, 'reference_passed': False,
              'checklist_credit': False, 'admission_evidence': False, 'workers': {}, 'replays': {}}
    started = time.monotonic()
    try:
        for which in ('baseline', 'modular'):
            outcome = launch(home, models, binding, which, 'primary', plan['resources']['primary_worker_seconds'],
                             plan['resources']['memory_bytes'], worker_script=SCRIPT)
            result['workers'][which] = outcome
            save(home / f'{which}-primary.json', outcome, exclusive=True)
            if not outcome['execution_completed']:
                raise ValueError(outcome.get('error', 'primary worker did not complete'))
            result['report'] = assess(plan, result['workers'])
            if not result['report']['retention'][which]:
                result['stop_reason'] = f'{which} failed protected retention'
                break
            if which == 'baseline' and result['report']['insufficient_headroom']:
                result['stop_reason'] = 'controls leave insufficient room for fixed gain; candidate not run'
                break
        if result['report']['quality_gate']:
            for which in ('baseline', 'modular'):
                replay = launch(home, models, binding, which, 'replay', plan['resources']['replay_worker_seconds'],
                                plan['resources']['memory_bytes'], worker_script=SCRIPT)
                result['replays'][which] = replay
                if not replay['execution_completed'] or len(replay['rows']) != len(plan['replay_ids']):
                    raise ValueError('incomplete conditional replay')
                primary = {row['id']: row for row in result['workers'][which]['rows']}
                if any(not replay_matches(primary[row['id']], row) for row in replay['rows']):
                    raise ValueError('answerability replay differs')
            result['reference_passed'] = True
        result['execution_completed'] = True
    except Exception as error:
        result['error'] = str(error)
    result['wall_seconds'] = time.monotonic() - started
    save(home / 'result.json', result, exclusive=True)
    return result
