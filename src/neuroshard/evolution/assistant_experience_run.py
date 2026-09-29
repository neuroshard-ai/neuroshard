"""One accelerator host: collect verified experience, record replay, train both arms, fit gates.

Development and confirmation evaluation are not here; they run on the CPU
runtime of the canonical parent baseline so device numerics cannot flip a
protected parent success. Every phase writes its inventory before the next.
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

from neuroshard.evolution import assistant_experience as experience
from neuroshard.evolution import assistant_replay as replay
from neuroshard.evolution import assistant_rollout as rollout
from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution import assistant_workspace as sandbox
from neuroshard.evolution import granite_tokenizer
from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, identity, launch, read, save, sha256, verify_artifacts,
)

PLAN = 'config/experiments/assistant-experience-learning.json'
EXECUTION = 'config/experiments/assistant-experience-execution.json'
ARTIFACTS = 'config/experiments/granite-reference-artifacts.json'
SCRIPT = 'scripts/run_assistant_experience.py'
PROFILE = 'assistant-experience-gpu'
UPLOADED = '.experience'
COLLECTION_FILES = ('experience.json', 'rollouts.jsonl.gz', 'trajectories.jsonl.gz', 'replay.json', 'replay.jsonl.gz')


def split_cases(plan, split):
    if split not in ('train', 'integration'):
        raise ValueError('accelerator phases may not access development or confirmation goals')
    manifest = read(ROOT / plan['data'])['splits'][split]
    cases = data.cases(split)
    if identity(cases) != manifest['sha256'] or [c['id'] for c in cases] != manifest['case_ids']:
        raise ValueError('workflow data differs from frozen split')
    return cases


def write_rows(path, rows):
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
    return identity(rows)


def trainer_module():
    from neuroshard.evolution import assistant_experience_train

    return assistant_experience_train


def release_accelerator():
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reporter(home, phase):
    def report(done, total):
        if done % 25 == 0 or done == total:
            save(home / 'progress.json', {'phase': phase, 'completed': done, 'total': total, 'unix': time.time()})
    return report


def sampling(policy, execution, temperature):
    return {**policy['generation'], 'temperature': temperature, 'top_p': execution['top_p']}


def collect(model, tokenizer, plan, policy, execution, home):
    """Uncoached rollouts, coached retries where needed, parent likelihood and selection."""
    from neuroshard.evolution import assistant_experience_train as trainer

    cases = split_cases(plan, 'train')
    by_id = {case['id']: case for case in cases}
    samples = execution['samples_per_case']
    card_policy = experience.coached(policy, plan['coaching']['card'])
    batcher = rollout.Batcher(model, tokenizer, sampling(policy, execution, execution['temperature']),
                              max_batch=execution['max_batch'], device=execution['device'], seed=execution['seed'])
    started = time.monotonic()
    try:
        natural = rollout.rollouts([(c, policy, s) for c in cases for s in range(samples)], batcher.respond,
                                   workers=execution['workers'], progress=reporter(home, 'collect-natural'))
        accepted = [t for row in natural if (t := experience.trajectory(
            by_id[row['case_id']], row['result'], policy, policy, sample=row['sample'])) is not None]
        pending = [c for c in cases if experience.needs_coaching(c, accepted)]
        coached_rows = rollout.rollouts([(c, card_policy, samples + s) for c in pending for s in range(samples)],
                                        batcher.respond, workers=execution['workers'],
                                        progress=reporter(home, 'collect-coached'))
        accepted += [t for row in coached_rows if (t := experience.trajectory(
            by_id[row['case_id']], row['result'], card_policy, policy, sample=row['sample'], coaching=True)) is not None]
    finally:
        batcher.close()
    save(home / 'progress.json', {'phase': 'parent-likelihood', 'accepted': len(accepted), 'unix': time.time()})
    sequences = {t['transcript_sha256']: trainer.encode(tokenizer, t, sandbox.TOOLS) for t in accepted}
    nll = {key: trainer.negative_log_likelihood(model, value, execution['device']) for key, value in sequences.items()}
    kept, ceiling = experience.near_policy(accepted, nll)
    chosen = experience.select(kept, plan['experience_selection_per_case'])
    report = {'summary': experience.summary(cases, len(natural) + len(coached_rows), chosen),
              'accepted_before_filter': len(accepted), 'near_policy_ceiling': ceiling,
              'coached_cases': [c['id'] for c in pending], 'seconds': time.monotonic() - started,
              'batches': len(batcher.batches), 'model_calls': sum(r['result']['model_calls'] for r in natural + coached_rows),
              'input_tokens': sum(r['result']['input_tokens'] for r in natural + coached_rows),
              'output_tokens': sum(r['result']['output_tokens'] for r in natural + coached_rows),
              'rollouts_sha256': write_rows(home / 'rollouts.jsonl.gz', natural + coached_rows),
              'trajectories_sha256': write_rows(home / 'trajectories.jsonl.gz', chosen),
              'nll': {t['transcript_sha256']: nll[t['transcript_sha256']] for t in chosen}}
    save(home / 'experience.json', report, exclusive=True)
    return [sequences[t['transcript_sha256']] for t in chosen], report


def record_replay(model, tokenizer, plan, execution, home):
    """Parent greedy answers to the generated replay prompts, before any training."""
    from neuroshard.evolution import assistant_experience_train as trainer

    anchors = read(ROOT / 'config/experiments/granite-reference.json')
    prompts = replay.prompts(execution['replay_seed'], anchors['tasks'] + anchors['reference_tasks'])
    greedy = {'max_input_tokens': execution['replay_max_input_tokens'],
              'max_new_tokens': execution['replay_max_new_tokens'], 'temperature': 0, 'top_p': 1.0}
    batcher = rollout.Batcher(model, tokenizer, greedy, max_batch=execution['max_batch'], device=execution['device'])
    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=execution['workers']) as pool:
            responses = list(pool.map(lambda p: batcher.respond(p['messages'], p['tools']), prompts))
    finally:
        batcher.close()
    items = [item for p, r in zip(prompts, responses) if (item := replay.replay_item(p, r)) is not None]
    sequences = [trainer.encode(tokenizer, item, item['tools']) for item in items]
    save(home / 'replay.json', {'prompts': len(prompts), 'terminated': len(items),
                                'items_sha256': write_rows(home / 'replay.jsonl.gz', items)}, exclusive=True)
    return sequences


def read_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle]


def load_collection(directory, pinned, tokenizer, plan, policy, execution, home):
    """Re-verify experience pinned from an earlier collection and encode it; no new rollouts.

    Every trajectory is rebuilt from its recorded rollout through the frozen scorer,
    so the files are evidence to re-check, not trusted training data.
    """
    from neuroshard.evolution import assistant_experience_train as trainer

    directory = Path(directory)
    for name, digest in pinned.items():
        if sha256(directory / name) != digest:
            raise ValueError(f'collected file differs from its pinned digest: {name}')
    report = read(directory / 'experience.json')
    rollouts, chosen = read_rows(directory / 'rollouts.jsonl.gz'), read_rows(directory / 'trajectories.jsonl.gz')
    if identity(rollouts) != report['rollouts_sha256'] or identity(chosen) != report['trajectories_sha256']:
        raise ValueError('collected rows differ from the collection report')
    cases = {case['id']: case for case in split_cases(plan, 'train')}
    card_policy = experience.coached(policy, plan['coaching']['card'])
    sources = {(row['case_id'], row['sample'], row['policy_sha256']): row for row in rollouts}
    if len(sources) != len(rollouts):
        raise ValueError('collected rollouts repeat a case, sample and policy')
    per_case = {}
    for item in chosen:
        executed = card_policy if item['coached'] else policy
        row = sources.get((item['case_id'], item['sample'], identity(executed)))
        if row is None or identity(row['result']['messages']) != item['transcript_sha256']:
            raise ValueError('trajectory has no matching rollout')
        rebuilt = experience.trajectory(cases[item['case_id']], row['result'], executed, policy,
                                        sample=item['sample'], coaching=item['coached'])
        if rebuilt != item:
            raise ValueError('trajectory does not re-verify from its rollout')
        if item['coached'] and report['nll'][item['transcript_sha256']] > report['near_policy_ceiling']:
            raise ValueError('coached trajectory exceeds the near-policy ceiling')
        per_case[item['case_id']] = per_case.get(item['case_id'], 0) + 1
    if (max(per_case.values()) > plan['experience_selection_per_case']
            or len({identity([t['case_id'], experience.assistant_texts(t)]) for t in chosen}) != len(chosen)):
        raise ValueError('collected selection exceeds the per-case cap or repeats a trajectory')
    items = read_rows(directory / 'replay.jsonl.gz')
    if identity(items) != read(directory / 'replay.json')['items_sha256']:
        raise ValueError('replay rows differ from the replay report')
    anchors = read(ROOT / 'config/experiments/granite-reference.json')
    prompts = {p['id']: p for p in replay.prompts(execution['replay_seed'], anchors['tasks'] + anchors['reference_tasks'])}
    for item in items:
        prompt = prompts.get(item['id'])
        if (prompt is None or identity(prompt) != item['prompt_sha256'] or item['messages'][:-1] != prompt['messages']
                or item['tools'] != prompt['tools'] or item['trainable'] != [False] * len(prompt['messages']) + [True]):
            raise ValueError('replay item differs from its generated prompt')
    save(home / 'collection.json', {'trajectories': len(chosen), 'replay_items': len(items), 'rollouts': len(rollouts),
                                    'cases_with_experience': len(per_case), 'files': pinned}, exclusive=True)
    # Stored rows sort JSON keys; the chat template renders tool schemas in their original key order.
    return ([trainer.encode(tokenizer, t, sandbox.TOOLS) for t in chosen],
            [trainer.encode(tokenizer, i, prompts[i['id']]['tools']) for i in items])


def train_arms(load_parent, plan, execution, experience_rows, replay_rows, home):
    from neuroshard.evolution import assistant_experience_train as trainer

    spec = {**plan['training'], 'seed': plan['training']['seed']}
    manifests = {}
    for arm in ('update', 'addition'):
        save(home / 'progress.json', {'phase': f'train-{arm}', 'unix': time.time()})
        model = load_parent()
        started = time.monotonic()
        trainable, receipt = trainer.train(model, arm, experience_rows, replay_rows, spec, device=execution['device'])
        roots = {'experience': identity([r['sha256'] for r in experience_rows]),
                 'replay': identity([r['sha256'] for r in replay_rows]), 'plan': identity(plan)}
        manifests[arm] = {**trainer.checkpoint(home / f'{arm}-checkpoint', trainable, receipt, roots),
                          'seconds': time.monotonic() - started, 'tokens_processed': receipt['tokens_processed']}
        del model, trainable, receipt
        release_accelerator()
    save(home / 'training.json', manifests, exclusive=True)
    return manifests


def build_pairs(tokenizer, plan, policy, rollouts):
    """Verified version-choice preferences from the pinned natural training rollouts."""
    from neuroshard.evolution import assistant_experience_train as trainer

    per_case = plan['decision_preferences']['per_case']
    grouped = {}
    for row in rollouts:
        grouped.setdefault(row['case_id'], []).append(row)
    pairs = [pair for case in split_cases(plan, 'train')
             for pair in experience.decision_pairs(case, grouped.get(case['id'], []), policy, per_case)]
    if not pairs:
        raise ValueError('no verified decision preferences in the pinned rollouts')
    return pairs, [trainer.encode_pair(tokenizer, pair, sandbox.TOOLS) for pair in pairs]


def collect_arm(model, tokenizer, plan, policy, execution, home):
    """Sampled training rollouts from a trained arm, for divergence preferences; every one is rescored."""
    cases = split_cases(plan, 'train')
    spec = plan['divergence_preferences']
    batcher = rollout.Batcher(model, tokenizer, sampling(policy, execution, spec['temperature']),
                              max_batch=execution['max_batch'], device=execution['device'], seed=spec['seed'])
    try:
        rows = rollout.rollouts([(c, policy, s) for c in cases for s in range(spec['samples_per_case'])],
                                batcher.respond, workers=execution['workers'], progress=reporter(home, 'collect-round3'))
    finally:
        batcher.close()
    save(home / 'rollouts-round3.json', {'rollouts': len(rows), 'passed': sum(r['result']['score']['passed'] for r in rows),
                                         'rows_sha256': write_rows(home / 'rollouts-round3.jsonl.gz', rows)}, exclusive=True)
    return rows


def repair_responder(recorded, at, text, live):
    """Replay the first ``at`` recorded generations, substitute the repaired call, then sample live."""
    count = [0]

    def respond(messages, tools):
        index = count[0]
        count[0] += 1
        if index < at:
            return {key: recorded[index][key] for key in
                    ('text', 'terminated', 'executed', 'input_token_ids', 'token_ids', 'prompt_sha256')
                    if key in recorded[index]}
        if index == at:
            return {'text': text, 'terminated': True, 'executed': False, 'input_token_ids': [], 'token_ids': [],
                    'prompt_sha256': None, 'repaired': True}
        return live(messages, tools)

    return respond


def collect_repairs(model, tokenizer, plan, policy, execution, home):
    """Sampled training rollouts of an arm, then goal-guided repairs of wrong reads, continued live."""
    from concurrent.futures import ThreadPoolExecutor

    spec = plan['goal_guided_repairs']
    cases = split_cases(plan, 'train')
    by_id = {case['id']: case for case in cases}
    batcher = rollout.Batcher(model, tokenizer, sampling(policy, execution, spec['temperature']),
                              max_batch=execution['max_batch'], device=execution['device'], seed=spec['seed'])
    try:
        natural = rollout.rollouts([(c, policy, s) for c in cases for s in range(spec['samples_per_case'])],
                                   batcher.respond, workers=execution['workers'], progress=reporter(home, 'collect-round4'))
        jobs = []
        for row in natural:
            if row['result']['score']['passed']:
                continue
            found = experience.wrong_read(by_id[row['case_id']], row['result'])
            if found:
                jobs += [(row, found, attempt) for attempt in range(spec['repairs_per_failure'])]
        report = reporter(home, 'repair-round4')
        done = [0]

        def repair(job):
            row, (index, at, text), attempt = job
            result = workflow.execute(by_id[row['case_id']],
                                      repair_responder(row['result']['generations'], at, text, batcher.respond), policy)
            done[0] += 1
            report(done[0], len(jobs))
            return {'case_id': row['case_id'], 'sample': row['sample'], 'attempt': attempt, 'index': index,
                    'policy_sha256': identity(policy), 'rejected': row['result']['messages'][index]['content'],
                    'result': result}

        with ThreadPoolExecutor(max_workers=execution['workers']) as pool:
            repaired = list(pool.map(repair, jobs))
    finally:
        batcher.close()
    save(home / 'rollouts-round4.json', {
        'rollouts': len(natural), 'passed': sum(r['result']['score']['passed'] for r in natural),
        'repairable_failures': len(jobs) // max(1, spec['repairs_per_failure']), 'repairs': len(repaired),
        'verified_repairs': sum(r['result']['score']['passed'] for r in repaired),
        'natural_sha256': write_rows(home / 'rollouts-round4.jsonl.gz', natural),
        'repairs_sha256': write_rows(home / 'repairs-round4.jsonl.gz', repaired)}, exclusive=True)
    return natural, repaired


def repair_data(plan, policy, repaired):
    """Preference pairs and trajectories from verified repairs; every repaired rollout is rescored."""
    spec = plan['goal_guided_repairs']
    by_id = {case['id']: case for case in split_cases(plan, 'train')}
    pairs, trajectories = {}, []
    for row in repaired:
        case, result = by_id[row['case_id']], row['result']
        if workflow.score(case, result, policy) != result['score']:
            raise ValueError('repaired rollout does not re-verify')
        if not result['score']['passed']:
            continue
        messages, index = result['messages'], row['index']
        chosen = messages[index]['content']
        if chosen == row['rejected']:
            raise ValueError('repair did not change the decision')
        key = identity([messages[:index], chosen, row['rejected']])
        pairs.setdefault(row['case_id'], {}).setdefault(key, {
            'case_id': row['case_id'], 'messages': messages[:index], 'chosen': chosen, 'rejected': row['rejected'],
            'samples': [row['sample'], row['attempt']]})
        trajectories.append(experience.trajectory(case, result, policy, policy,
                                                  sample=row['sample'] * 100 + row['attempt']))
    pair_list = [p for case_id in sorted(pairs) for p in list(pairs[case_id].values())[:spec['pairs_per_case']]]
    return pair_list, experience.select(trajectories, spec['experience_per_case'])


def build_divergence_pairs(tokenizer, plan, policy, rollouts):
    from neuroshard.evolution import assistant_experience_train as trainer

    per_case = plan['divergence_preferences']['per_case']
    grouped = {}
    for row in rollouts:
        grouped.setdefault(row['case_id'], []).append(row)
    pairs = [pair for case in split_cases(plan, 'train')
             for pair in experience.divergence_pairs(case, grouped.get(case['id'], []), policy, per_case)]
    if not pairs:
        raise ValueError('no verified divergence preferences in the arm rollouts')
    return pairs, [trainer.encode_pair(tokenizer, pair, sandbox.TOOLS) for pair in pairs]


def train_round2(load_parent, plan, execution, experience_rows, replay_rows, pair_rows, round1, pinned, home,
                 section='decision_preferences'):
    """Continue both prior arms on identical experience, replay and preference pairs."""
    from neuroshard.evolution import assistant_experience_train as trainer

    spec = {**plan['training'], **plan[section]['training']}
    manifests = {}
    for arm in ('update', 'addition'):
        save(home / 'progress.json', {'phase': f'train-{section}-{arm}', 'unix': time.time()})
        model = load_parent()
        started = time.monotonic()
        prior, trainable = trainer.resume(model, arm, plan['training'], Path(round1) / f'{arm}-checkpoint')
        if prior['trainable_sha256'] != pinned[arm]:
            raise ValueError(f'prior {arm} checkpoint differs from its pinned digest')
        trainable, receipt = trainer.train(model, arm, experience_rows, replay_rows, spec,
                                           device=execution['device'], trainable=trainable, pairs=pair_rows)
        roots = {'round1': prior['trainable_sha256'], 'experience': identity([r['sha256'] for r in experience_rows]),
                 'replay': identity([r['sha256'] for r in replay_rows]),
                 'pairs': identity([p['sha256'] for p in pair_rows]), 'plan': identity(plan)}
        manifests[arm] = {**trainer.checkpoint(home / f'{arm}-checkpoint', trainable, receipt, roots),
                          'seconds': time.monotonic() - started, 'tokens_processed': receipt['tokens_processed'],
                          'first_margins': receipt['preference_margins'][:8],
                          'last_margins': receipt['preference_margins'][-8:]}
        del model, trainable, receipt
        release_accelerator()
    save(home / 'training.json', manifests, exclusive=True)
    return manifests


def committee_member(case_id, members):
    """A fixed, content-independent slice of the training cases for each committee member."""
    import hashlib

    return int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16) % members


def study_data(tokenizer, plan, policy, runtime, home, inventory):
    """Every verified sequence and preference gathered in rounds 1-4, each tagged with its training case.

    ``inventory`` is the pinned execution file; ``runtime`` its merged runtime parameters.
    """
    from neuroshard.evolution import assistant_experience_train as trainer

    experience_rows, replay_rows = load_collection(ROOT / UPLOADED, inventory['collection']['files'], tokenizer,
                                                   plan, policy, runtime, home)
    cases = [t['case_id'] for t in read_rows(ROOT / UPLOADED / 'trajectories.jsonl.gz')]
    for name, digest in inventory['study']['files'].items():
        if sha256(ROOT / UPLOADED / name) != digest:
            raise ValueError(f'study input differs from its pinned digest: {name}')
    decisions, decision_rows = build_pairs(tokenizer, plan, policy, read_rows(ROOT / UPLOADED / 'rollouts.jsonl.gz'))
    divergences, divergence_rows = build_divergence_pairs(tokenizer, plan, policy,
                                                          read_rows(ROOT / UPLOADED / 'rollouts-round3.jsonl.gz'))
    repairs, repaired = repair_data(plan, policy, read_rows(ROOT / UPLOADED / 'repairs-round4.jsonl.gz'))
    counts = {'trajectories': len(experience_rows), 'repaired_trajectories': len(repaired), 'replay': len(replay_rows),
              'decision_pairs': len(decisions), 'divergence_pairs': len(divergences), 'repair_pairs': len(repairs)}
    if counts != plan['methodology_study']['counts']:
        raise ValueError(f'study data differs from its declaration: {counts}')
    experience_rows += [trainer.encode(tokenizer, t, sandbox.TOOLS) for t in repaired]
    cases += [t['case_id'] for t in repaired]
    pairs = decisions + divergences + repairs
    pair_rows = decision_rows + divergence_rows + [trainer.encode_pair(tokenizer, p, sandbox.TOOLS) for p in repairs]
    save(home / 'study-data.json', {**counts, 'experience_sha256': identity([r['sha256'] for r in experience_rows]),
                                    'pairs_sha256': identity([r['sha256'] for r in pair_rows])}, exclusive=True)
    return (list(zip(cases, experience_rows)), replay_rows, list(zip([p['case_id'] for p in pairs], pair_rows)))


def study_spec(plan, arm):
    """(first-phase spec, preference-phase spec) for one study arm; architecture keys come from the arm."""
    first = {**plan['training'], **arm.get('spec', {})}
    return first, {**first, **plan['goal_guided_repairs']['training'], 'seed': first['seed'] + 1}


def study_train(load_parent, plan, execution, experience, replay, pairs, home):
    """Every study arm with the same two phases: verified experience and replay, then all verified preferences."""
    from neuroshard.evolution import assistant_experience_train as trainer

    study, manifests = plan['methodology_study'], {}
    for name, arm in study['arms'].items():
        member = arm.get('member')
        keep = (lambda case: True) if member is None else (lambda case: committee_member(case, study['members']) == member)
        rows = [row for case, row in experience if keep(case)]
        pair_rows = [row for case, row in pairs if keep(case)]
        first, second = study_spec(plan, arm)
        save(home / 'progress.json', {'phase': f'study-train-{name}', 'unix': time.time()})
        model = load_parent()
        started = time.monotonic()
        trainable, receipt = trainer.train(model, arm['type'], rows, replay, first, device=execution['device'])
        trainable, receipt = trainer.train(model, arm['type'], rows, replay, second, device=execution['device'],
                                           trainable=trainable, pairs=pair_rows)
        roots = {'experience': identity([r['sha256'] for r in rows]), 'replay': identity([r['sha256'] for r in replay]),
                 'pairs': identity([p['sha256'] for p in pair_rows]), 'plan': identity(plan)}
        manifests[name] = {**trainer.checkpoint(home / f'{name}-checkpoint', trainable, receipt, roots),
                           'sequences': len(rows), 'pairs': len(pair_rows), 'seconds': time.monotonic() - started}
        del model, trainable, receipt
        release_accelerator()
    save(home / 'training.json', manifests, exclusive=True)
    return manifests


def system_outcomes(make, cases, policy, execution, samples, seed, progress):
    """One greedy and ``samples`` sampled episodes per case; ``make(temperature, seed)`` returns (respond, close)."""
    results = {case['id']: [] for case in cases}
    for temperature, count in ((0, 1), (execution['temperature'], samples)):
        if not count:
            continue
        respond, close = make(temperature, seed)
        try:
            rows = rollout.rollouts([(c, policy, s) for c in cases for s in range(count)], respond,
                                    workers=execution['workers'], progress=progress)
        finally:
            close()
        for row in rows:
            results[row['case_id']].append(row['result']['score']['passed'])
    return results


def study_evaluate(load_parent, tokenizer, plan, policy, execution, home):
    """Integration and development outcomes of the parent, every arm and the committee of members."""
    from neuroshard.evolution import assistant_committee as committee
    from neuroshard.evolution import assistant_experience_train as trainer

    study = plan['methodology_study']
    splits = {'integration': (split_cases(plan, 'integration'), study['integration_samples']),
              'development': (data.cases('development'), 0)}

    def single(model):
        def make(temperature, seed):
            batcher = rollout.Batcher(model, tokenizer, sampling(policy, execution, temperature),
                                      max_batch=execution['max_batch'], device=execution['device'], seed=seed)
            return batcher.respond, batcher.close
        return make

    def evaluate(name, make):
        return {split: system_outcomes(make, cases, policy, execution, samples, study['seed'],
                                       reporter(home, f'study-{name}-{split}'))
                for split, (cases, samples) in splits.items()}

    systems = {}
    parent = load_parent()
    systems['parent'] = evaluate('parent', single(parent))
    del parent
    release_accelerator()
    for name, arm in study['arms'].items():
        model = load_parent()
        trainer.load_trainable(model, arm['type'], study_spec(plan, arm)[0], home / f'{name}-checkpoint')
        trainer.serving(model, study_spec(plan, arm)[0])
        systems[name] = evaluate(name, single(model))
        del model
        release_accelerator()
    members = sorted((arm['member'], name) for name, arm in study['arms'].items() if arm.get('member') is not None)
    model, switch = load_parent(), committee.Switch()
    committee.attach(model, study_spec(plan, study['arms'][members[0][1]])[0],
                     [home / f'{name}-checkpoint' for _, name in members], switch)

    def voted(temperature, seed):
        batchers = [rollout.Batcher(model, tokenizer, sampling(policy, execution, temperature),
                                    max_batch=execution['max_batch'], device=execution['device'], seed=seed + index,
                                    context=lambda index=index: switch.using(index))
                    for index in range(len(members))]
        parent_batcher = rollout.Batcher(model, tokenizer, sampling(policy, execution, temperature),
                                         max_batch=execution['max_batch'], device=execution['device'],
                                         seed=seed + len(members), context=lambda: switch.using(None))
        everyone = batchers + [parent_batcher]
        return (committee.responder([b.respond for b in batchers], parent_batcher.respond),
                lambda: [b.close() for b in everyone])

    systems['committee'] = evaluate('committee', voted)
    del model
    release_accelerator()
    report = study_report(systems)
    save(home / 'study.json', {'systems': systems, 'report': report}, exclusive=True)
    return report


def study_report(systems):
    """Success rates, parent successes lost and the difference from the update, per system and split."""
    report = {}
    for name, splits in systems.items():
        report[name] = {}
        for split, outcomes_by_case in splits.items():
            greedy = {case: values[0] for case, values in outcomes_by_case.items()}
            sampled = [v for values in outcomes_by_case.values() for v in values[1:]]
            parent = {case: values[0] for case, values in systems['parent'][split].items()}
            update = {case: values[0] for case, values in systems['update'][split].items()}
            report[name][split] = {
                'greedy_correct': sum(greedy.values()), 'cases': len(greedy),
                'sampled_rate': sum(sampled) / len(sampled) if sampled else None,
                'lost_parent_successes': sorted(c for c in greedy if parent[c] and not greedy[c]),
                'versus_update': sum(greedy.values()) - sum(update.values())}
    return report


def feature_ids(tokenizer, policy, case):
    """Token IDs of the first assistant-generation boundary, the selection feature's input."""
    messages = [{'role': 'system', 'content': policy['system_instruction']},
                {'role': 'user', 'content': data.public_case(case)['user_turns'][0]}]
    prompt = tokenizer.apply_chat_template(messages, tools=sandbox.TOOLS, add_generation_prompt=True, tokenize=False)
    return tokenizer(prompt, add_special_tokens=False)['input_ids']


def boundary_feature(model, tokenizer, policy, case, device):
    """Frozen parent final-layer state at the first assistant-generation boundary."""
    import torch

    ids = torch.tensor([feature_ids(tokenizer, policy, case)], device=device)
    with torch.no_grad():
        return model.model(input_ids=ids).last_hidden_state[0, -1].float().cpu().tolist()


def outcomes(model, tokenizer, policy, execution, cases, seed, progress=None):
    """One greedy and the declared number of sampled complete episodes per integration case."""
    results = {case['id']: [] for case in cases}
    for temperature, count in ((0, 1), (execution['temperature'], execution['integration_samples'])):
        batcher = rollout.Batcher(model, tokenizer, sampling(policy, execution, temperature),
                                  max_batch=execution['max_batch'], device=execution['device'], seed=seed)
        try:
            rows = rollout.rollouts([(c, policy, s) for c in cases for s in range(count)], batcher.respond,
                                    workers=execution['workers'], progress=progress)
        finally:
            batcher.close()
        for row in rows:
            results[row['case_id']].append(row['result']['score']['passed'])
    return results


def integrate(load_parent, tokenizer, plan, policy, execution, home):
    from neuroshard.evolution import assistant_experience_train as trainer
    from neuroshard.evolution import assistant_selector as selector

    cases = split_cases(plan, 'integration')
    parent = load_parent()
    features = {c['id']: boundary_feature(parent, tokenizer, policy, c, execution['device']) for c in cases}
    parent_outcomes = outcomes(parent, tokenizer, policy, execution, cases, execution['seed'] + 1,
                               progress=reporter(home, 'integrate-parent'))
    del parent
    release_accelerator()
    recipe = plan['selection_recipe']
    gates = {}
    for arm in ('update', 'addition'):
        model = load_parent()
        trainer.load_trainable(model, arm, plan['training'], home / f'{arm}-checkpoint')
        arm_outcomes = outcomes(model, tokenizer, policy, execution, cases, execution['seed'] + 2,
                                progress=reporter(home, f'integrate-{arm}'))
        gates[arm] = {'gate': selector.fit(features, selector.targets(parent_outcomes, arm_outcomes), recipe),
                      'outcomes': arm_outcomes}
        del model
        release_accelerator()
    save(home / 'integration.json', {'parent_outcomes': parent_outcomes, 'arms': gates,
                                     'features_sha256': identity(features)}, exclusive=True)
    return gates


def committed_sources(root=ROOT):
    execution = read(root / EXECUTION)
    for name, digest in execution['contracts'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'changed experience contract: {name}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in execution['sources']:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root) != (root / name).read_bytes():
            raise ValueError(f'uncommitted experience source: {name}')
        sources[name] = sha256(root / name)
    return {'commit': commit, 'sources': sources}


def configure():
    if 'torch' in sys.modules:
        raise ValueError('configure the accelerator runtime before importing torch')
    for key, value in read(ROOT / EXECUTION)['environment'].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def gpu_names():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True)
    return [line.strip() for line in output.splitlines() if line.strip()]


def freeze():
    source = committed_sources()
    execution = read(ROOT / EXECUTION)
    packages = {key: importlib.metadata.version(key) for key in execution['packages']}
    if packages != execution['packages'] or platform.python_version() != execution['python']:
        raise ValueError('accelerator runtime differs')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('accelerator runtime requires Linux x86_64')
    gpus = gpu_names()
    if len(gpus) != 1 or gpus[0] not in execution['gpus']:
        raise ValueError('accelerator is not one declared GPU')
    if any(os.environ.get(key) != value for key, value in execution['environment'].items()):
        raise ValueError('accelerator environment differs')
    return {**source, 'packages': packages, 'python': platform.python_version(), 'gpus': gpus}


def worker(request_path):
    configure()
    request_path = Path(request_path)
    request = read(request_path)
    if freeze() != request['freeze']:
        raise ValueError('experience worker differs from freeze')
    import torch
    from transformers import AutoModelForCausalLM

    execution = read(ROOT / EXECUTION)
    gpu = request['freeze']['gpus'][0]
    parameters = {**execution['execution'], **execution['gpus'][gpu]}
    plan = read(ROOT / PLAN)
    policy = read(ROOT / plan['policy'])
    home = request_path.parent
    reply = {'binding': request['binding'], 'execution_completed': False, 'phases': {}, 'gpu': gpu,
             'max_batch': parameters['max_batch']}
    started = time.monotonic()
    tokenizer = None
    try:
        inventory = read(ROOT / ARTIFACTS)['models']['baseline']
        directory = Path(request['models']) / 'baseline'
        state = verify_artifacts(directory, inventory, download=True)
        tokenizer, report = granite_tokenizer.load(directory)
        if report['pipeline_sha256'] != execution['tokenizer_pipeline_sha256']:
            raise ValueError('tokenizer pipeline differs from the canonical baseline')
        reply['tokenizer'] = report

        def load_parent():
            model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.bfloat16, local_files_only=True,
                                                         attn_implementation=parameters['attention'])
            if sum(p.numel() for p in model.parameters()) != inventory['parameters']:
                raise ValueError('parent parameter inventory differs')
            return model.to(parameters['device']).eval()

        begun = time.monotonic()
        if 'study' in execution:
            experience_rows, replay_rows, pair_rows = study_data(tokenizer, plan, policy, parameters, home, execution)
            reply['phases']['verify_collection'] = time.monotonic() - begun
            begun = time.monotonic()
            reply['training'] = study_train(load_parent, plan, parameters, experience_rows, replay_rows, pair_rows, home)
            reply['phases']['train'] = time.monotonic() - begun
            begun = time.monotonic()
            reply['study'] = study_evaluate(load_parent, tokenizer, plan, policy, parameters, home)
            reply['phases']['evaluate'] = time.monotonic() - begun
        elif 'round4' in execution:
            trainer = trainer_module()
            experience_rows, replay_rows = load_collection(ROOT / UPLOADED, execution['collection']['files'],
                                                           tokenizer, plan, policy, parameters, home)
            prior = execution['round4']['prior']
            sampler = load_parent()
            manifest, _ = trainer.resume(sampler, 'addition', plan['training'], ROOT / UPLOADED / 'addition-checkpoint')
            if manifest['trainable_sha256'] != prior['addition']:
                raise ValueError('round-3 addition differs from its pinned digest')
            sampler.eval()
            reply['phases']['verify_collection'] = time.monotonic() - begun
            begun = time.monotonic()
            _, repaired = collect_repairs(sampler, tokenizer, plan, policy, parameters, home)
            del sampler
            release_accelerator()
            pairs, extra = repair_data(plan, policy, repaired)
            if not pairs:
                raise ValueError('no verified repairs')
            save(home / 'pairs.json', {'pairs': len(pairs), 'cases': len({p['case_id'] for p in pairs}),
                                       'repaired_trajectories': len(extra), 'pairs_sha256': identity(pairs),
                                       'trajectories_sha256': write_rows(home / 'repaired-trajectories.jsonl.gz', extra)},
                 exclusive=True)
            experience_rows = experience_rows + [trainer.encode(tokenizer, t, sandbox.TOOLS) for t in extra]
            pair_rows = [trainer.encode_pair(tokenizer, p, sandbox.TOOLS) for p in pairs]
            reply['phases']['collect_round4'] = time.monotonic() - begun
            begun = time.monotonic()
            reply['training'] = train_round2(load_parent, plan, parameters, experience_rows, replay_rows, pair_rows,
                                             ROOT / UPLOADED, prior, home, section='goal_guided_repairs')
            reply['phases']['train'] = time.monotonic() - begun
        elif 'round3' in execution:
            experience_rows, replay_rows = load_collection(ROOT / UPLOADED, execution['collection']['files'],
                                                           tokenizer, plan, policy, parameters, home)
            prior = execution['round3']['prior']
            sampler = load_parent()
            manifest, _ = trainer_module().resume(sampler, 'addition', plan['training'], ROOT / UPLOADED / 'addition-checkpoint')
            if manifest['trainable_sha256'] != prior['addition']:
                raise ValueError('round-2 addition differs from its pinned digest')
            sampler.eval()
            reply['phases']['verify_collection'] = time.monotonic() - begun
            begun = time.monotonic()
            rollouts = collect_arm(sampler, tokenizer, plan, policy, parameters, home)
            del sampler
            release_accelerator()
            pairs, pair_rows = build_divergence_pairs(tokenizer, plan, policy, rollouts)
            save(home / 'pairs.json', {'pairs': len(pairs), 'cases': len({p['case_id'] for p in pairs}),
                                       'turns': sorted({p['turn'] for p in pairs}), 'pairs_sha256': identity(pairs)},
                 exclusive=True)
            reply['phases']['collect_round3'] = time.monotonic() - begun
            begun = time.monotonic()
            reply['training'] = train_round2(load_parent, plan, parameters, experience_rows, replay_rows, pair_rows,
                                             ROOT / UPLOADED, prior, home, section='divergence_preferences')
            reply['phases']['train'] = time.monotonic() - begun
        elif 'round2' in execution:
            experience_rows, replay_rows = load_collection(ROOT / UPLOADED, execution['collection']['files'],
                                                           tokenizer, plan, policy, parameters, home)
            pairs, pair_rows = build_pairs(tokenizer, plan, policy, read_rows(ROOT / UPLOADED / 'rollouts.jsonl.gz'))
            save(home / 'pairs.json', {'pairs': len(pairs), 'cases': len({p['case_id'] for p in pairs}),
                                       'pairs_sha256': identity(pairs)}, exclusive=True)
            reply['phases']['verify_collection'] = time.monotonic() - begun
            begun = time.monotonic()
            reply['training'] = train_round2(load_parent, plan, parameters, experience_rows, replay_rows, pair_rows,
                                             ROOT / UPLOADED, execution['round2']['round1'], home)
            reply['phases']['train'] = time.monotonic() - begun
        elif 'collection' in execution:
            experience_rows, replay_rows = load_collection(ROOT / UPLOADED, execution['collection']['files'],
                                                           tokenizer, plan, policy, parameters, home)
            reply['phases']['verify_collection'] = time.monotonic() - begun
        else:
            parent = load_parent()
            experience_rows, _ = collect(parent, tokenizer, plan, policy, parameters, home)
            reply['phases']['collect'] = time.monotonic() - begun
            begun = time.monotonic()
            replay_rows = record_replay(parent, tokenizer, plan, parameters, home)
            reply['phases']['replay'] = time.monotonic() - begun
            del parent
            release_accelerator()
        if not {'round2', 'round3', 'round4', 'study'} & set(execution):
            begun = time.monotonic()
            reply['training'] = train_arms(load_parent, plan, parameters, experience_rows, replay_rows, home)
            reply['phases']['train'] = time.monotonic() - begun
        if 'study' not in execution:
            begun = time.monotonic()
            gates = integrate(load_parent, tokenizer, plan, policy, parameters, home)
            reply['gates'] = {arm: value['gate'] for arm, value in gates.items()}
            reply['phases']['integrate'] = time.monotonic() - begun
        if file_state(directory, inventory) != state:
            raise ValueError('parent checkpoint changed during experience execution')
        reply['execution_completed'] = True
    except Exception as error:
        reply['error'] = f'{type(error).__name__}: {error}'
    finally:
        reply.update(checked_encodes=tokenizer.checked_encodes if tokenizer else 0,
                     wall_seconds=time.monotonic() - started, process_cpu_seconds=time.process_time(),
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                     peak_gpu_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
        save(request_path.parent / 'reply.json', reply, exclusive=True)


def run(home, models):
    configure()
    source = freeze()
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    execution = read(ROOT / EXECUTION)
    binding = {'freeze': source, 'profile': PROFILE, 'plan_sha256': sha256(ROOT / PLAN)}
    save(home / 'binding.json', binding, exclusive=True)
    result = {'binding': binding, 'execution_completed': False, 'checklist_credit': False,
              'admission_evidence': False, 'confirmation_opened': False}
    try:
        reply = launch(home, models, binding, 'baseline', 'experience', execution['worker_seconds'],
                       execution['memory_bytes'], worker_script=SCRIPT, environment=execution['worker_environment'])
        result['reply'] = reply
        if not reply['execution_completed']:
            raise ValueError(reply.get('error', 'incomplete experience execution'))
        result['execution_completed'] = True
    except Exception as error:
        result['error'] = str(error)
    save(home / 'result.json', result, exclusive=True)
    return result
