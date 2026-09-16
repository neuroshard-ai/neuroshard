"""Re-execute new-answer quality and check unchanged retained computations.

The retained-input check is a structural argument about the exact same frozen
executor, parameters and inputs. It preserves old errors too. It does not replace
quality measurement on newly routed inputs or establish cross-hardware equality.
"""
from pathlib import Path
import math

from .. import cohort_questions, expert_lifecycle, serving_graph
from ..reference_data import identity, read_records
from ..schema import integer, root

FORMAT = 'neuroshard-expert-graph-quality-v1'
PROSPECTIVE = 'neuroshard-prospective-expert-graph-quality-v1'
GENERAL = 'neuroshard-general-expert-graph-quality-v1'
ROLES = ('test', 'retained-test-knowledge', 'retained-test-skills', 'retained-test-conversation')
POLICY_FIELDS = {'format', 'baseline_graph', 'candidate_graph', 'prepared', 'roles', 'generation', 'gates'}


def validate_policy(policy):
    prospective = isinstance(policy, dict) and policy.get('format') in (PROSPECTIVE, GENERAL)
    fields = POLICY_FIELDS - {'candidate_graph'} | {'candidate_template'} if prospective else POLICY_FIELDS
    serving_graph.fields(policy, fields, 'Invalid frozen graph quality policy')
    if policy['format'] not in (FORMAT, PROSPECTIVE, GENERAL):
        raise ValueError('Unsupported frozen graph quality policy')
    root(policy['baseline_graph'])
    root(policy['prepared'])
    if prospective:
        serving_graph.validate(policy['candidate_template'], allow_untrained=True)
        expert_lifecycle.training_expert(policy['candidate_template'])
    else:
        root(policy['candidate_graph'])
    serving_graph.fields(policy['roles'], ROLES, 'Require complete new and retained evaluation roles')
    for spec in policy['roles'].values():
        serving_graph.fields(spec, {'file', 'sha256', 'count', 'ids'}, 'Invalid quality role commitment')
        if not isinstance(spec['file'], str) or Path(spec['file']).name != spec['file']:
            raise ValueError('Quality inputs must be local committed filenames')
        root(spec['sha256'])
        root(spec['ids'])
        integer(spec['count'], 1, 4096)
    serving_graph.fields(policy['generation'], {'new', 'retained_knowledge', 'retained_skills'},
                         'Invalid frozen generation limits')
    for maximum in policy['generation'].values():
        integer(maximum, 1, 256)
    gates = policy['gates']
    serving_graph.fields(gates, {'single_accuracy', 'composed_accuracy', 'gain_lower',
                                'bootstrap_samples', 'bootstrap_seed', 'confidence'}, 'Invalid frozen quality gates')
    for key in ('single_accuracy', 'composed_accuracy', 'gain_lower', 'confidence'):
        if type(gates[key]) not in (float, int) or not math.isfinite(gates[key]) or not 0 <= gates[key] <= 1:
            raise ValueError('Quality thresholds must be finite fractions')
    if gates['confidence'] in (0, 1):
        raise ValueError('Declare a finite confidence interval')
    integer(gates['bootstrap_samples'], 100, 100000)
    integer(gates['bootstrap_seed'], 0, 2**32 - 1)
    return prospective


def rows(policy, inputs, role):
    spec = policy['roles'][role]
    if Path(spec['file']).name != spec['file']:
        raise ValueError('Quality inputs must be local committed filenames')
    result = read_records(Path(inputs) / spec['file'], spec['sha256'])
    if len(result) != spec['count'] or identity([row['id'] for row in result]) != spec['ids']:
        raise ValueError('Quality inputs changed their complete ordered identities')
    return result


def retention(policy, inputs, baseline, candidate):
    proofs = {}
    for role in ROLES[1:]:
        values = []
        for row in rows(policy, inputs, role):
            question = row['messages'][0]['content']
            if role.endswith('conversation'):
                def computation(graph):
                    if [call['model'] for call in serving_graph.calls(graph, question, 1)] != ['parent']:
                        return None
                    return identity({'format': FORMAT + '/forward', 'parent': identity(graph['parent']),
                        'input_ids': row['input_ids'], 'labels': row['labels'], 'targets': row['targets'],
                        'executor': graph['executor_root'], 'numerical_profile': graph['numerical_profile'],
                        'tokenizer': graph['tokenizer']})
            else:
                maximum = policy['generation']['retained_knowledge' if role.endswith('knowledge') else 'retained_skills']
                def computation(graph):
                    return serving_graph.execution_identity(graph, question, maximum)
            before, after = computation(baseline), computation(candidate)
            values.append({'id': row['id'], 'before': before, 'after': after,
                           'unchanged': before is not None and before == after})
        proofs[role] = values
    return {'format': FORMAT + '/retained-computations', 'roles': proofs,
            'passed': all(row['unchanged'] for values in proofs.values() for row in values)}


def evaluate(policy, inputs, baseline, candidate, network, progress=None):
    prospective = validate_policy(policy)
    expected = (identity(expert_lifecycle.materialize_graph(policy['candidate_template'],
                    candidate['experts'][expert_lifecycle.training_expert(policy['candidate_template'])]))
                if prospective else policy['candidate_graph'])
    if (policy['format'] not in (FORMAT, PROSPECTIVE, GENERAL) or policy['baseline_graph'] != identity(baseline)
            or expected != identity(candidate) or set(policy['roles']) != set(ROLES)):
        raise ValueError('Quality policy differs from its complete graphs or input cohorts')
    # Missing or corrupted bytes must fail before numerical execution starts.
    examples = rows(policy, inputs, 'test')
    cohort_questions.validate_rows(examples, release_scope=policy['format'] != GENERAL)
    retained = retention(policy, inputs, baseline, candidate)
    maximum = policy['generation']['new']
    before, after, executions = [], [], []
    for index, row in enumerate(examples):
        question = row['messages'][0]['content']
        old = network.answer(question, maximum, baseline)
        new = network.answer(question, maximum, candidate)
        before.append({'id': row['id'], 'text': old['text']})
        after.append({'id': row['id'], 'text': new['text']})
        executions.append({'id': row['id'], 'before': old, 'after': new})
        if progress:
            progress(index + 1, len(examples))
    decision = cohort_questions.decision(examples, before, after, policy['gates'],
                                         release_scope=policy['format'] != GENERAL)
    decision['checks']['unchanged_retained_computations'] = retained['passed']
    decision['passed'] = all(decision['checks'].values())
    return {'format': FORMAT + '/result', 'policy': identity(policy), 'executions': executions,
            'decision': decision, 'retention': retained}


def quality_transcript(claim, result):
    return {'format': expert_lifecycle.FORMAT + '/quality-transcript',
            'binding': expert_lifecycle.transcript_binding(claim), 'result': result}


def quality_report(claim, policy, inputs, network, progress=None):
    if (claim['kind'] != 'expert_quality' or claim['model_root'] != identity(claim['graph'])
            or claim['executor_root'] != claim['graph']['executor_root']
            or claim['report']['policy_root'] != identity(policy)
            or claim['report']['prepared'] != policy['prepared']
            or claim['stages'] != policy['roles']['test']['count']):
        raise ValueError('Quality audit changed its graph, executor, policy or coverage')
    result = evaluate(policy, inputs, claim['baseline_graph'], claim['graph'], network, progress)
    expected = {'format': expert_lifecycle.FORMAT + '/quality', 'policy_root': identity(policy),
        'baseline_graph': identity(claim['baseline_graph']), 'candidate_graph': identity(claim['graph']),
        'prepared': policy['prepared'], 'passed': result['decision']['passed'], 'results_root': identity(result)}
    valid = (claim['report'] == expected and identity(quality_transcript(claim, result)) == claim['record_root'])
    report = {'format': expert_lifecycle.FORMAT + '/replay',
        'statement': identity(expert_lifecycle.service_statement(claim)),
        'stages': [{'stage': index, 'valid': valid} for index in range(claim['stages'])]}
    expert_lifecycle.replay_report(claim, report)
    return report, result
