#!/usr/bin/env python3
"""Freeze a C-only learning repair after measured ordinary access passes."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from neuroshard.evolution import cohort_questions, expert_curriculum, expert_data
from neuroshard.evolution.access_routing import ordinary_c_question
from neuroshard.evolution.reference_data import identity, save, sha256

ROOT = Path(__file__).resolve().parents[1]


def prepare(original, serving, home, *, scope_repair=False):
    read = lambda p: json.loads(p.read_bytes())
    access = read(serving/'result.json')
    if access.get('planning_passed') is not True or not all(access['planning_checks'].values()):
        raise ValueError('Measured automatic access must pass before neural repair starts')
    graph = read(serving/'inputs/graph.json')
    starting = ('f1aa762419334821aa65e2c3b3c020b18a66793e251b3663eef9cc5e2a7cdb86' if scope_repair
                else '618e3eb0ab52eb688cfa1d17961a124b940ba9267f3e0bc316e151748af7e14c')
    if graph['experts']['planner']['checkpoint'] != starting:
        raise ValueError('Repair the exact measured C checkpoint')
    if scope_repair and access['service'] != 'e1662f75df43257281f00cdb270bbca9b7f3f62928149ac757afab1bde8bebe9':
        raise ValueError('The scope repair starts from the verified 14/15 service')
    inputs = home/'inputs'
    inputs.mkdir(exist_ok=False)
    # The old opened final is only a historical bound in prepared metadata.
    # Its answer file is not copied into this learning allocation.
    for name in ('plan.json', 'prepared.json', 'parent.json', 'trial.json'):
        shutil.copyfile(original/'inputs'/name, inputs/name)
    for name in ('train.jsonl', 'train-questions.jsonl', 'provenance.json', 'augmentation.json'):
        shutil.copyfile(home/'augmented'/name, inputs/name)
    shutil.copytree(serving/'seed', home/'seed')
    annotations = [json.loads(line) for line in (inputs/'train-questions.jsonl').read_bytes().splitlines()]
    augmentation = read(inputs/'augmentation.json')
    if sha256(inputs/'train.jsonl') != augmentation['train']['sha256']:
        raise ValueError('Freeze the exact crossed training data')
    batches = expert_curriculum.balanced_batches(annotations, batch_size=16)
    if scope_repair:
        steps, schedule = 64, list(range(len(batches)))+list(range(5))
        if len(annotations) != 936 or len(batches) != 59 or len(schedule) != steps:
            raise ValueError('Scope repair covers 156 questions under six contracts, then five replay batches')
    else:
        steps, schedule = 2*len(batches), list(range(len(batches)))*2
        if steps != 108 or len(annotations) != 864:
            raise ValueError('This bounded intervention covers 144 questions under six contracts twice')
    plan, prepared = read(inputs/'plan.json'), read(inputs/'prepared.json')
    plan['seed_expert'] = {'name': 'planner', 'checkpoint': graph['experts']['planner']}
    plan['previous_graph'] = identity(graph['descriptor'])
    plan['training'] = {'steps': steps, 'learning_rate': 0.00005, 'warmup_steps': 4,
                        'weight_decay': 0.01, 'clip_norm': 1.0}
    prepared.update(plan=identity(plan), retention_cache=identity(graph), batches=batches,
                    schedule=schedule)
    prepared['roles']['train'] = augmentation['train']
    expert_data.validate_prepared(prepared, plan, graph['parent']['config']['vocab_size'])
    trial = read(inputs/'trial.json')
    trial.update(cohort='planner-contract-repair', training=plan['training'], max_seconds=3600,
        method='Warm-start only C with fresh Adam; response-only CE over crossed training contracts, two full passes.',
        final_rule='No final is opened. This is a development repair of the existing C expert, not a new cohort.',
        decision='Accept the research candidate only if all ordinary cases pass and every earlier correct C development answer is retained.',
        resources={'instance_types': ['g5.2xlarge', 'g5.2xlarge', 'g5.xlarge', 'g5.xlarge'],
                   'disk_gib': 200, 'max_hours': 2, 'planning_cap_usd': 25},
        intervention={'augmentation': identity(augmentation), 'provenance': identity(read(inputs/'provenance.json')),
                      'development_informed': True, 'final_opened': False},
        gates={'ordinary_cases': 'all', 'forced_gold_questions': 'all',
               'prior_C_development_correct': 'retain all', 'replay': 'exact'},
        evaluation={}, future_cohorts='Hold until the complete ordinary answering repair passes.')
    if scope_repair:
        trial.update(cohort='planner-scope-repair', method=(
            'Warm-start C with fresh Adam and response-only CE. Add twelve source-grounded claim/job '
            'scope variations; retain every earlier training topic. One complete pass plus five '
            'balanced replay batches; terminal checkpoint only.'))
    for name, value in (('plan', plan), ('prepared', prepared), ('trial', trial)):
        save(inputs/(name+'.json'), value)
    for name in ('graph.json', 'planned.json', 'profile.json', 'router.json', 'plan.json', 'access-trial.json'):
        shutil.copyfile(serving/'inputs'/name, inputs/('serving-'+name))
    shutil.copyfile(serving/'result.json', inputs/'before-access.json')
    dev = [json.loads(line) for line in (original/'inputs/dev-questions.jsonl').read_bytes().splitlines()]
    prior = read(original/'verified-results/owner-0/serving-dev/dev.json')
    executions = {row['id']: row['after'] for row in prior['executions']}
    retention = []
    for row in dev:
        if row['stratum'] != 'single':
            continue
        response = executions[row['id']]
        calls, outputs = response['request']['calls'], response['outputs']
        if len(calls) != 1 or calls[0]['model'] != 'planner' or len(outputs) != 1:
            raise ValueError('Earlier C retention must have actually reached C')
        retention.append({'row': row, 'prompt_root': outputs[0]['prompt_root'],
            'before_correct': cohort_questions.correct(row, response['text'], release_scope=False),
            'atom': {'expert': 'planner', 'gold_question': ordinary_c_question(calls[0]['question']),
                     'answer': row['answers'][0]}})
    if scope_repair:
        current = read(serving/'retention-result.json')
        outcomes = {row['id']: row for row in current['outcomes']}
        if len(outcomes) != 16 or current['correct'] != 15 or current['lost']:
            raise ValueError('Protect every newly correct C development answer')
        for item in retention:
            outcome = outcomes[item['row']['id']]
            control = outcome['control']
            if (control['error'] or control['forced_route'] != 'planner'
                    or identity(control['outputs'][0]['prompt_ids']) != item['prompt_root']
                    or cohort_questions.correct(item['row'], control['answer']['text'], release_scope=False)
                        != outcome['correct']):
                raise ValueError('The current C retention result changed prompts or scoring')
            item['before_correct'] = outcome['correct']
    if len(retention) != 16 or sum(row['before_correct'] for row in retention) != (15 if scope_repair else 13):
        raise ValueError('Preserve the complete published C development result')
    save(inputs/'retention.json', retention)
    # Reuse public per-owner inventories. C's weights are also needed by the
    # frozen feature reference on owner 2 and the sole learner on owner 3.
    catalog = read(serving/'inputs/objects-4.json')['objects']
    c_keys = {spec['sha256'] for spec in graph['experts']['planner']['tensors'].values()}
    c_objects = {key: catalog[key] for key in c_keys}
    for rank in range(4):
        objects = read(original/'inputs'/('objects-'+str(rank)+'.json'))
        if rank in (2, 3):
            for key, spec in c_objects.items():
                if key in objects['objects'] and objects['objects'][key] != spec:
                    raise ValueError('An immutable tensor changed provenance')
                objects['objects'][key] = copy.deepcopy(spec)
        save(inputs/('objects-'+str(rank)+'.json'), objects)
    files = ['scripts/run_continual_expert_trial.py', 'scripts/run_contract_learning_service.py',
             'scripts/prepare_contract_learning.py', 'scripts/prepare_semantic_expert_training.py',
             'scripts/run_ordinary_access_trial.py']
    files += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'src/neuroshard/evolution').rglob('*.py'))]
    freeze = {'trial': identity(trial), 'plan': identity(plan), 'prepared': identity(prepared),
              'driver': sha256(ROOT/files[0]), 'sources': {name: sha256(ROOT/name) for name in files},
              'retention': sha256(inputs/'retention.json')}
    save(inputs/'freeze.json', freeze)
    manifest = {'format': 'neuroshard-contract-learning-repair-v1', 'freeze': identity(freeze),
        'starting_checkpoint': graph['experts']['planner']['checkpoint'],
        'previous_service': access['service'], 'access_passed': True, 'training': plan['training'],
        'training_documents': augmentation['train'], 'source_facts_changed': False,
        'neural_changes': 'C tail only; earlier experts, backbone, selector and answering policy stay fixed',
        'acceptance': trial['gates'], 'resources': trial['resources'],
        'selection': 'Terminal checkpoint only. No development-based early stopping or checkpoint shopping.',
        'no_new_final': True, 'no_new_cohort': True, 'native_promotion': False}
    if scope_repair:
        manifest['semantic_intervention'] = {
            'questions': sha256(ROOT/'config/experiments/planner-scope-questions.json'),
            'additional_questions': 12, 'topics': ['planner-updates', 'planner-steps'],
            'schedule': '59 complete-coverage batches followed by the first five balanced batches',
            'prior_correct_to_preserve': {'ordinary': 14, 'C_standalone': 15},
            'diagnostic_is_exposed': True}
    filename = 'scope-learning-trial.json' if scope_repair else 'contract-learning-trial.json'
    save(ROOT/'config/experiments'/filename, manifest)
    print(json.dumps(manifest))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--serving', type=Path, required=True)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--scope-repair', action='store_true', help='Use the separately prescribed 14/15 C scope intervention')
    args = parser.parse_args()
    prepare(args.original, args.serving, args.home, scope_repair=args.scope_repair)
