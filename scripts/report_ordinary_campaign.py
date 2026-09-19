#!/usr/bin/env python3
"""Reconcile a completed ordinary campaign with its full ledger and GPU audits.

This checks retained evidence, without signing transactions or opening any
unevaluated holdout. Earlier-genesis admissions are reported separately. It does
not turn a bounded experiment into a public-readiness or independence claim.
"""
import argparse
import json
from pathlib import Path

from neuroshard.evolution import ordinary_operation, settlement
from neuroshard.evolution.app import code_hash
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.reference_data import identity, save
from neuroshard.evolution.sharded import graph_quality


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def report(home):
    home = Path(home)
    state, replay = read(home/'final-state.json'), read(home/'ledger-replay.json')
    outcome, operation = read(home/'result.json'), read(home/'operation.json')
    freeze, native = read(home/'source-freeze.json'), read(home/'native.json')
    require(identity(operation) == freeze['operation'], 'The frozen operation changed')
    require(state['manifest'] == read(home/'native-manifest.json'), 'The native manifest changed')
    require(state['manifest']['code_hash'] == code_hash(), 'Use the numerical execution source')
    require(replay['passed'] is True and replay['final_state_root'] == identity(state)
            and replay['final_height'] == state['height']
            and replay['headers'] == state['height']+1
            and replay['issued_atoms'] == state['issued']
            and replay['genesis'] == native['genesis_root'],
            'Require a complete matching application replay from genesis')
    require(outcome['state_root'] == identity(state)
            and outcome['serving_root'] == state['serving_root']
            and outcome['issued_atoms'] == state['issued'], 'Outcome differs from the replayed state')
    settlement.invariant(state)
    sequence = ordinary_operation.outcome_sequence(state, operation['entries'])
    require(sequence == outcome['sequence'], 'Outcome differs from the native quality sequence')
    store = Objects(home/'compiled/objects')
    evidence = {}
    for path in (home/'jobs').glob('*/quality-0.json'):
        producer = read(path)
        require(producer['status'] == 'completed', 'An incomplete producer has no quality result')
        root = identity(producer['result'])
        require(root not in evidence, 'Ambiguous producer quality evidence')
        evidence[root] = (path.parent, producer)
    history = [row for row in state['expert_lifecycle']['history'] if row.get('kind') == 'quality']
    serving, cohorts = operation['baseline_graph'], []
    for index, record in enumerate(history):
        claim = record['report']
        folder, producer = evidence[claim['results_root']]
        result = producer['result']
        policy = store.json(claim['policy_root'])
        require(identity(policy) == claim['policy_root'] == result['policy'], 'Quality policy changed')
        require(claim['baseline_graph'] == serving, 'Quality did not extend the accepted graph')
        require(record['promoted'] is claim['passed'] is result['decision']['passed'],
                'Native promotion differs from the audited quality decision')
        require(read(folder/'job.json')['lifecycle']['quality']['policy_root'] == claim['policy_root'],
                'Quality evidence belongs to a different job')
        stages = graph_quality.stages(policy)
        expected = [{'stage': stage, 'valid': True} for stage in range(stages)]
        for actor in (1, 2, 3):
            audit = read(folder/f'quality-{actor}.json')
            require(audit['status'] == 'completed' and identity(audit['result']) == claim['results_root']
                    and audit['report']['stages'] == expected,
                    'Require all three complete, exactly matching numerical audits')
        retention = result['retention']
        retained = {role: {'count': len(rows),
            'correct_before': sum(row['before_correct'] for row in rows),
            'correct_after': sum(row['after_correct'] for row in rows),
            'lost_correct': sum(row['lost_correct'] for row in rows)}
            for role, rows in retention['roles'].items()}
        require(sum(row['lost_correct'] for row in retained.values()) == retention['lost_correct'],
                'Retained-answer totals disagree')
        if record['promoted']:
            serving = claim['candidate_graph']
        cohorts.append({'entry': operation['entries'][index]['label'], 'job': folder.name,
            'height': record['height'], 'claim': record['id'], 'promoted': record['promoted'],
            'quality': claim['results_root'], 'policy': claim['policy_root'],
            'decision': result['decision'], 'retention': retained,
            'exact_fresh_audits': 3, 'stages_per_audit': stages})
    require(serving == state['serving_root'], 'Serving moved outside the native quality history')
    inherited = None
    if 'accepted_history' in operation:
        inherited = read(home/'compiled/accepted-history.json')
        require(identity(inherited) == operation['accepted_history'], 'Imported admission provenance changed')
    comparison = read(home/'comparison/result.json') if (home/'comparison/result.json').exists() else None
    rejected = read(home/'rejected-source.json') if (home/'rejected-source.json').exists() else None
    return {'format': 'neuroshard-ordinary-campaign-evidence-v1',
        'operation': identity(operation), 'genesis': replay['genesis'],
        'state': identity(state), 'height': state['height'], 'issued_atoms': state['issued'],
        'sequence': sequence, 'cohorts': cohorts,
        'admissions_this_genesis': sum(row['promoted'] for row in cohorts),
        'prior_genesis_admissions': len(inherited['accepted']) if inherited else 0,
        'prior_genesis': inherited['genesis'] if inherited else None,
        'comparison': comparison,
        'source_substitution_rejected': bool(rejected and rejected['hashes_internally_valid'] is True
            and rejected['mechanical_checks_passed'] is False
            and rejected['source_correspondence_passed'] is False),
        'publisher_process_starts': outcome['publisher_process_starts'],
        'serving_probes': outcome['probes_during_work'],
        'ledger_replay': replay,
        'scope': 'Bounded learning and automatic admission under one administrator. '
                 'Prior-genesis admissions are separate; checklist completion needs review.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve the earlier evidence report')
    result = report(args.home)
    save(args.output, result)
    print(json.dumps({key: result[key] for key in ('sequence', 'admissions_this_genesis',
        'prior_genesis_admissions', 'source_substitution_rejected')}))
