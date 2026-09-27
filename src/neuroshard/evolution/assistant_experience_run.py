"""One accelerator host: collect verified experience, record replay, train both arms, fit gates.

Development and confirmation evaluation are not here; they run on the CPU
runtime of the canonical parent baseline so device numerics cannot flip a
protected parent success. Every phase writes its inventory before the next.
"""

import gzip
import json
from pathlib import Path
import time

from neuroshard.evolution import assistant_experience as experience
from neuroshard.evolution import assistant_experience_train as trainer
from neuroshard.evolution import assistant_replay as replay
from neuroshard.evolution import assistant_rollout as rollout
from neuroshard.evolution import assistant_selector as selector
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution import assistant_workspace as sandbox
from neuroshard.evolution.modular_reference_execution import ROOT, identity, read, save

PLAN = 'config/experiments/assistant-experience-learning.json'


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


def sampling(policy, execution, temperature):
    return {**policy['generation'], 'temperature': temperature, 'top_p': execution['top_p']}


def collect(model, tokenizer, plan, policy, execution, home):
    """Uncoached rollouts, coached retries where needed, parent likelihood and selection."""
    cases = split_cases(plan, 'train')
    by_id = {case['id']: case for case in cases}
    samples = execution['samples_per_case']
    card_policy = experience.coached(policy, plan['coaching']['card'])
    batcher = rollout.Batcher(model, tokenizer, sampling(policy, execution, execution['temperature']),
                              max_batch=execution['max_batch'], device=execution['device'], seed=execution['seed'])
    started = time.monotonic()
    try:
        natural = rollout.rollouts([(c, policy, s) for c in cases for s in range(samples)], batcher.respond,
                                   workers=execution['workers'])
        accepted = [t for row in natural if (t := experience.trajectory(
            by_id[row['case_id']], row['result'], policy, policy, sample=row['sample'])) is not None]
        pending = [c for c in cases if experience.needs_coaching(c, accepted)]
        coached_rows = rollout.rollouts([(c, card_policy, samples + s) for c in pending for s in range(samples)],
                                        batcher.respond, workers=execution['workers'])
        accepted += [t for row in coached_rows if (t := experience.trajectory(
            by_id[row['case_id']], row['result'], card_policy, policy, sample=row['sample'], coaching=True)) is not None]
    finally:
        batcher.close()
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


def train_arms(load_parent, plan, execution, experience_rows, replay_rows, home):
    spec = {**plan['training'], 'seed': plan['training']['seed']}
    manifests = {}
    for arm in ('update', 'addition'):
        model = load_parent()
        started = time.monotonic()
        trainable, receipt = trainer.train(model, arm, experience_rows, replay_rows, spec, device=execution['device'])
        roots = {'experience': identity([r['sha256'] for r in experience_rows]),
                 'replay': identity([r['sha256'] for r in replay_rows]), 'plan': identity(plan)}
        manifests[arm] = {**trainer.checkpoint(home / f'{arm}-checkpoint', trainable, receipt, roots),
                          'seconds': time.monotonic() - started, 'tokens_processed': receipt['tokens_processed']}
        del model, trainable
    save(home / 'training.json', manifests, exclusive=True)
    return manifests


def boundary_feature(model, tokenizer, policy, case, device):
    """Frozen parent final-layer state at the first assistant-generation boundary."""
    import torch

    messages = [{'role': 'system', 'content': policy['system_instruction']},
                {'role': 'user', 'content': data.public_case(case)['user_turns'][0]}]
    prompt = tokenizer.apply_chat_template(messages, tools=sandbox.TOOLS, add_generation_prompt=True, tokenize=False)
    ids = torch.tensor([tokenizer(prompt, add_special_tokens=False)['input_ids']], device=device)
    with torch.no_grad():
        return model.model(input_ids=ids).last_hidden_state[0, -1].float().cpu().tolist()


def outcomes(model, tokenizer, policy, execution, cases, seed):
    """One greedy and the declared number of sampled complete episodes per integration case."""
    results = {case['id']: [] for case in cases}
    for temperature, count in ((0, 1), (execution['temperature'], execution['integration_samples'])):
        batcher = rollout.Batcher(model, tokenizer, sampling(policy, execution, temperature),
                                  max_batch=execution['max_batch'], device=execution['device'], seed=seed)
        try:
            rows = rollout.rollouts([(c, policy, s) for c in cases for s in range(count)], batcher.respond,
                                    workers=execution['workers'])
        finally:
            batcher.close()
        for row in rows:
            results[row['case_id']].append(row['result']['score']['passed'])
    return results


def integrate(load_parent, tokenizer, plan, policy, execution, home):
    cases = split_cases(plan, 'integration')
    parent = load_parent()
    features = {c['id']: boundary_feature(parent, tokenizer, policy, c, execution['device']) for c in cases}
    parent_outcomes = outcomes(parent, tokenizer, policy, execution, cases, execution['seed'] + 1)
    del parent
    recipe = plan['selection_recipe']
    gates = {}
    for arm in ('update', 'addition'):
        model = load_parent()
        trainer.load_trainable(model, arm, plan['training'], home / f'{arm}-checkpoint')
        arm_outcomes = outcomes(model, tokenizer, policy, execution, cases, execution['seed'] + 2)
        gates[arm] = {'gate': selector.fit(features, selector.targets(parent_outcomes, arm_outcomes), recipe),
                      'outcomes': arm_outcomes}
        del model
    save(home / 'integration.json', {'parent_outcomes': parent_outcomes, 'arms': gates,
                                     'features_sha256': identity(features)}, exclusive=True)
    return gates
