#!/usr/bin/env python3
"""Assemble the fixed ordinary campaign after its metadata compiler finishes.

The runtime inventory is checked again on the actual partition owners before
initialization. A final source commitment is required before neural training.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess

from neuroshard.dataflow.store import canonical, LocalStore
from neuroshard.evolution import answering, expert_data, expert_source, ordinary_cohorts
from neuroshard.evolution import ordinary_operation
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.reference_data import identity, save, sha256
from neuroshard.evolution.sharded import graph_quality
from neuroshard.evolution.sharded import learned_graph, planned_graph

ROOT = Path(__file__).resolve().parents[1]


def freeze(home, revision, runtime):
    compiled = home/'compiled'
    read = lambda path: json.loads(path.read_bytes())
    inputs = read(compiled/'inputs.json')
    catalog, rules = read(compiled/'source-catalog.json'), read(compiled/'quality-rule.json')
    store = Objects(compiled/'objects')
    transport = LocalStore(home/'feed-objects')
    if len(revision) != 40 or subprocess.check_output(['git', 'rev-parse', revision], cwd=ROOT).decode().strip() != revision:
        raise ValueError('Use a real immutable source revision')
    profile = read(compiled/'baseline-profile.json')
    profile['runtime'] = copy.deepcopy(runtime)
    profile['numerical_profile'] = identity({'format': ordinary_cohorts.FORMAT+'/numerics',
        'runtime': runtime, 'recipe': 'frozen-partitions-bfloat16-forward-float32-adamw',
        'cross_hardware_equality_claimed': False})
    source_files = sorted(str(path.relative_to(ROOT)) for path in (ROOT/'src/neuroshard').rglob('*.py'))
    profile['sources'] = {name: sha256(ROOT/name) for name in source_files}
    core = read(compiled/'baseline-core.json')
    core['numerical_profile'], core['executor_root'] = profile['numerical_profile'], identity(profile)
    graph = ordinary_cohorts.bind_policy(core, store.json(inputs['policies']['baseline']), store)
    policy_objects = {name: store.json(key) for name, key in inputs['policies'].items()}
    for payload in policy_objects.values():
        for section in (payload['configuration'], payload['configuration']['learned']):
            if any(sha256(ROOT/path) != value for path, value in section['sources'].items()):
                raise ValueError('Recompile policies after changing their actual execution source')
    ordinary = policy_objects['admission']['configuration']
    route_models = {name: ('planner' if name == 'admission' else name)
                    for name in ordinary['learned']['router']['prototypes']}
    learned_control = learned_graph.configuration(core, ordinary['learned']['router'],
        ordinary['learned']['feature_profile'], ROOT, route_models, compose=True)
    prompts_control = {name: spec for name, spec in ordinary['expert_prompts'].items() if name in core['experts']}
    control_configuration = planned_graph.configuration(core, learned_control, ordinary['planner'], ROOT,
        prompts_control, ordinary['general_instruction'], request_policy=ordinary['request_policy'])
    control_graph = answering.attach(core, control_configuration, store)
    sources, windows = {}, []

    def source(name, role, records):
        spec = {'repo': 'neuroshard-ai/ordinary-campaign', 'revision': revision,
                'split': name, 'license': 'Apache-2.0', 'role': role}
        key = expert_source.publish_window(transport, spec, 0, records)
        windows.append(key)
        sources[name] = {'source': spec, 'records': store.put_json(records),
                         'window': key, 'count': len(records)}

    atoms = {fact['id']: fact for fact in catalog['cohorts']['admission']}
    rejection = read(ROOT/'config/experiments/ordinary-bootstrap-questions.json')
    bootstrap = [ordinary_cohorts.scored([{'role': 'user', 'content': row['question']}],
        [row['topic']], [atoms[row['topic']]['answer']]) for row in rejection['questions']]
    for index, left in enumerate(rejection['questions']):
        right = rejection['questions'][(index+1) % len(rejection['questions'])]
        bootstrap.append(ordinary_cohorts.scored([{'role': 'user', 'content': left['question']+' Also, '
            +right['question'][0].lower()+right['question'][1:]}], [left['topic'], right['topic']],
            [atoms[left['topic']]['answer'], atoms[right['topic']]['answer']]))
    source('bootstrap/test', 'heldout', bootstrap)
    for name in ordinary_cohorts.ORDER:
        for role, filename in (('train', 'training'), ('heldout', 'final')):
            values = [json.loads(line) for line in (compiled/name/(filename+'.jsonl')).read_bytes().splitlines()]
            source(name+('/train' if role == 'train' else '/test'), role, values)
    admission_rows = store.json(sources['admission/train']['records'])
    first = expert_source.publish_window(transport, sources['admission/train']['source'], 0, admission_rows[:16])
    remaining = expert_source.publish_window(transport, sources['admission/train']['source'], 16, admission_rows[16:])
    inventory = [first, sources['bootstrap/test']['window']]
    heads = [expert_source.append(transport, None, inventory)]
    for name in ordinary_cohorts.ORDER:
        additions = ([remaining] if name == 'admission' else [sources[name+'/train']['window']])
        additions.append(sources[name+'/test']['window'])
        heads.append(expert_source.append(transport, heads[-1], additions))
    head = heads[-1]
    feed = expert_source.Feed(transport, head)
    for spec in sources.values():
        if list(feed(spec['source'], 0, spec['count'])) != store.json(spec['records']):
            raise ValueError('Immutable feed read-back changed a complete source window')
    # All evaluation wording, including the failure exercise, stays out of
    # neural training and selector fitting. Context labels never enter serving.
    fitting = read(compiled/'selector-fitting.json')
    from neuroshard.evolution.access_routing import question_key
    trained = {question_key(row['question']) for row in fitting['rows']}
    if any(question_key(row['messages'][0]['content']) in trained for row in bootstrap):
        raise ValueError('The rejection exercise overlaps selector fitting')
    recipe = {'steps': 128, 'warmup_steps': 4, 'learning_rate': .00005,
              'weight_decay': .01, 'clip_norm': 1.}
    entries = [{'name': 'admission', 'label': 'bootstrap-rejection',
        'windows': {'admission/train': 16, 'bootstrap/test': 8},
        'replay_documents': 0, 'replay_sources': [],
        'recipe': {**recipe, 'steps': 1, 'warmup_steps': 0},
        'answering_policy': inputs['policies']['admission'], 'expected_promotion': False}]
    for index, name in enumerate(ordinary_cohorts.ORDER):
        previous = ordinary_cohorts.ORDER[max(0, index-1)]
        entries.append({'name': name, 'label': name, 'windows': {name+'/train': 368 if index == 0 else 384,
            name+'/test': 32}, 'replay_documents': 16, 'replay_sources': [previous+'/train'],
            'recipe': copy.deepcopy(recipe), 'answering_policy': inputs['policies'][name], 'expected_promotion': True})
    data_policy = {'format': expert_data.POLICY, 'tokenizer': graph['tokenizer']['root'],
        'max_length': 256, 'sources': {'neuroshard-ai/ordinary-campaign': [
            {'role': role, 'license': 'Apache-2.0'} for role in ('train', 'heldout')]},
        'near_duplicate_distance': 0,
        'quality_rule': identity({'gates': rules['gates'], 'generation': rules['generation'],
            'format': rules['format'], 'retention_gates': rules['retention_gates'],
            'retained_roles': rules['retention_anchors']})}
    kind, hours, allowance = {'NVIDIA A10G': ('g5.xlarge', 12, 1.1),
                              'NVIDIA L40S': ('g6e.xlarge', 8, 2.25)}[runtime['gpu']]
    value = {'format': ordinary_operation.FORMAT, 'source_revision': revision,
        'executor': store.put_json(profile), 'baseline_graph': store.put_json(graph),
        'data_policy': store.put_json(data_policy), 'quality_rule': store.put_json(rules),
        'source_evidence': store.put_json(read(compiled/'source-evidence.json')),
        'sources': sources, 'feed_head': heads[0], 'feed_heads': heads, 'entries': entries,
        'selection': 'Only each terminal checkpoint. No final-directed fitting, checkpoint choice, extra steps or altered gates.',
        'growth_rule': 'Add an isolated expert by default. Update or consolidate only after complete ordinary quality and retention pass.',
        'on_quality_failure': 'Persist and audit the measured failure; retain serving; stop later training under this prescription.',
        'on_unexpected_bootstrap_pass': 'Report it and stop; never manufacture a failed candidate.',
        'curation': 'Independently source-grounded fixed corpus, exact feed comparison, retokenization, native cursor/history and cross-role checks. Not arbitrary-web truth verification.',
        'resources': {'region': 'us-east-1', 'instance_types': [kind]*7, 'disk_gib': 300,
            'max_hours': hours, 'planning_cap_usd': 150, 'parallel_evaluations': 2,
            'planning_hourly_rate_usd': allowance,
            'rate_status': 'Conservative planning allowance; record actual EC2 price and allocated elapsed time separately.',
            'protected_instances': ['i-0d681a8ef83f72619', 'i-06bf7f1f01e6228bb', 'i-0ebe86ca07e97cf29']},
        'comparison': {'cohort': 'admission', 'arms': ['isolated-addition', 'replace-planner'],
            'answering_policy': control_graph['answering']['policy_root'],
            'provisioned_hosts_per_arm': 7, 'disk_gib_per_host': 300, 'seconds_per_arm': 5400,
            'new_experts_control': 0, 'new_experts_growth': 1,
            'training_and_audits': 'Repeat the identical numerical job and three fresh full audits in the control; shadow work cannot issue NEURO.',
            'quality': 'Identical frozen ordinary, accumulated knowledge and broader assistant questions; terminal only.',
            'serving': 'Two replicas of each complete service on the same seven hosts; a fixed general/specialist training-prompt queue fills the remaining interval.',
            'accounting': 'Count the entire equal provisioned interval, disks, actual training/audit/serving seconds, tokens, bytes transferred and retained object bytes. Preserve the same immutable checkpoint inventory for rollback in both arms.',
            'scope': 'Measured admission-cohort budget, not equal lifetime cost or an optimal fixed-capacity algorithm.',
            'interpretation': 'Prefer adding only if it passes complete quality and avoids retained-answer losses of the fixed-capacity control. Report throughput and costs even if neither wins.'},
        'recovery': {'publisher': 'Restart the actual operator process repeatedly from its durable journal.',
            'data_rejection': 'Offer a separately hashed source window with a substituted answer; the installed source-backed curator rejects it before funding.',
            'continuous_serving': 'Probe the currently accepted graph before and during numerical work, after rejection, restart and every promotion.'},
        'boundaries': 'Publish and hash-verify every committed numerical boundary before it can be pruned.',
        'claim_scope': 'One administrator, actual separate shard owners. No independent-operator or cross-hardware-equality claim.'}
    save(home/'operation.json', value)
    save(home/'feed-head.json', {'head': heads[0], 'entry': 0})
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({'operation': identity(freeze(args.home, args.revision,
        json.loads(args.runtime.read_bytes())))}))
