"""Re-execute new-answer quality and check unchanged retained computations.

The retained-input check is a structural argument about the exact same frozen
executor, parameters and inputs. It preserves old errors too. It does not replace
quality measurement on newly routed inputs or establish cross-hardware equality.
"""
from pathlib import Path
import math

from .. import cohort_questions, expert_lifecycle, ordinary_quality, serving_graph
from ..reference_data import identity, read_records
from ..schema import integer, root

FORMAT = 'neuroshard-expert-graph-quality-v1'
PROSPECTIVE = 'neuroshard-prospective-expert-graph-quality-v1'
GENERAL = 'neuroshard-general-expert-graph-quality-v1'
CONTINUAL = 'neuroshard-continual-expert-graph-quality-v1'
ORDINARY = ordinary_quality.FORMAT
MEASURED = (CONTINUAL, ORDINARY)
ROLES = ('test', 'retained-test-knowledge', 'retained-test-skills', 'retained-test-conversation')
POLICY_FIELDS = {'format', 'baseline_graph', 'candidate_graph', 'prepared', 'roles', 'generation', 'gates'}


def validate_policy(policy):
    prospective = isinstance(policy, dict) and policy.get('format') in (PROSPECTIVE, GENERAL, *MEASURED)
    fields = POLICY_FIELDS - {'candidate_graph'} | {'candidate_template'} if prospective else POLICY_FIELDS
    if policy.get('format') in MEASURED:
        fields = fields | {'retention_gates', 'retention_anchors'}
    serving_graph.fields(policy, fields, 'Invalid frozen graph quality policy')
    if policy['format'] not in (FORMAT, PROSPECTIVE, GENERAL, *MEASURED):
        raise ValueError('Unsupported frozen graph quality policy')
    root(policy['baseline_graph'])
    root(policy['prepared'])
    if prospective:
        serving_graph.validate(policy['candidate_template'], allow_untrained=True)
        expert_lifecycle.training_expert(policy['candidate_template'])
        if (policy['format'] == ORDINARY) != ('answering' in policy['candidate_template']):
            raise ValueError('Complete answering graphs require ordinary response quality')
    else:
        root(policy['candidate_graph'])
    serving_graph.fields(policy['roles'], ROLES, 'Require complete new and retained evaluation roles')
    specs = list(policy['roles'].values())
    if policy['format'] in MEASURED:
        retention_fields = {'max_lost_correct'} | ({'minimum_accuracy'} if policy['format'] == ORDINARY else set())
        serving_graph.fields(policy['retention_gates'], retention_fields, 'Invalid measured retention gate')
        integer(policy['retention_gates']['max_lost_correct'], 0, 0)
        if policy['format'] == ORDINARY:
            minimum = policy['retention_gates']['minimum_accuracy']
            serving_graph.fields(minimum, ROLES[1:], 'Declare useful measured retention floors')
            if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1
                   for value in minimum.values()):
                raise ValueError('Retention floors must require some correct actual answers')
        serving_graph.fields(policy['retention_anchors'], ROLES[1:], 'Pin the initial retention anchors')
        specs.extend(policy['retention_anchors'].values())
        for role in ROLES[2:]:
            if policy['roles'][role] != policy['retention_anchors'][role]:
                raise ValueError('Broader assistant retention anchors cannot change between cohorts')
    for spec in specs:
        serving_graph.fields(spec, {'file', 'sha256', 'count', 'ids'}, 'Invalid quality role commitment')
        if not isinstance(spec['file'], str) or Path(spec['file']).name != spec['file']:
            raise ValueError('Quality inputs must be local committed filenames')
        root(spec['sha256'])
        root(spec['ids'])
        integer(spec['count'], 1, 4096)
    generation = {'new', 'retained_knowledge', 'retained_skills'}
    if policy['format'] == ORDINARY:
        generation.add('retained_conversation')
    serving_graph.fields(policy['generation'], generation,
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


def stages(policy):
    """Fund every new and retained question pair that the auditor executes."""
    roles = ROLES if policy['format'] == ORDINARY else ROLES[:3] if policy['format'] == CONTINUAL else ('test',)
    return sum(policy['roles'][role]['count'] for role in roles)


def admission_rule(policy):
    """Keep acceptance rules fixed while prior admitted evaluations accumulate."""
    rule = {key: policy[key] for key in ('gates', 'generation')}
    if policy['format'] in MEASURED:
        rule.update(format=policy['format'], retention_gates=policy['retention_gates'],
                    retained_roles=policy['retention_anchors'])
    else:
        rule['retained_roles'] = {key: value for key, value in policy['roles'].items() if key != 'test'}
    return rule


def validate_rows(policy, values):
    if policy['format'] == ORDINARY:
        return ordinary_quality.validate_rows(values)
    return cohort_questions.validate_rows(values, release_scope=False)


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


def measured_retention(policy, inputs, baseline, candidate, network):
    """Allow changed expert weights only when previously correct answers survive.

    The committed initial anchors and all earlier admitted evaluations are
    enforced by data admission. Scores use the same declared short-answer
    normalization as the new-data gate, with no learned judge or loss proxy.
    """
    executions, seen = {}, set()
    ordinary = policy['format'] == ORDINARY
    for role in (ROLES[1:] if ordinary else ROLES[1:3]):
        examples = rows(policy, inputs, role)
        validate_rows(policy, examples)
        maximum = policy['generation']['retained_' + role.rsplit('-', 1)[1]]
        executions[role] = []
        for row in examples:
            if row['id'] in seen:
                raise ValueError('Retained questions must have distinct identities')
            seen.add(row['id'])
            question = row['messages'][:-1] if ordinary else row['messages'][0]['content']
            old = network.answer(question, maximum, baseline)
            new = network.answer(question, maximum, candidate)
            before = (ordinary_quality.correct(row, old) if ordinary else
                      cohort_questions.correct(row, old['text'], release_scope=False))
            after = (ordinary_quality.correct(row, new) if ordinary else
                     cohort_questions.correct(row, new['text'], release_scope=False))
            executions[role].append({'id': row['id'], 'before': old, 'after': new,
                'before_correct': before, 'after_correct': after, 'lost_correct': before and not after})
    # The unchanged parent forward path remains an additional check, rather
    # than a substitute for generating the old expert and assistant answers.
    structural = [] if ordinary else retention(policy, inputs, baseline, candidate)['roles'][ROLES[-1]]
    lost = sum(row['lost_correct'] for values in executions.values() for row in values)
    accuracy = {role: sum(row['after_correct'] for row in values)/len(values)
                for role, values in executions.items()}
    floors = (all(accuracy[role] >= threshold for role, threshold in
                  policy['retention_gates']['minimum_accuracy'].items()) if ordinary else True)
    return {'format': policy['format'] + '/retained-answers', 'roles': executions,
        **({'accuracy': accuracy, 'minimum_accuracy_passed': floors} if ordinary else {}),
        'conversation_computations': structural, 'lost_correct': lost,
        'passed': lost <= policy['retention_gates']['max_lost_correct']
                  and floors and all(row['unchanged'] for row in structural)}


def evaluate(policy, inputs, baseline, candidate, network, progress=None):
    prospective = validate_policy(policy)
    expected = (identity(expert_lifecycle.materialize_graph(policy['candidate_template'],
                    candidate['experts'][expert_lifecycle.training_expert(policy['candidate_template'])]))
                if prospective else policy['candidate_graph'])
    if (policy['format'] not in (FORMAT, PROSPECTIVE, GENERAL, *MEASURED) or policy['baseline_graph'] != identity(baseline)
            or expected != identity(candidate) or set(policy['roles']) != set(ROLES)):
        raise ValueError('Quality policy differs from its complete graphs or input cohorts')
    ordinary = policy['format'] == ORDINARY
    if ordinary != ('answering' in baseline) or ordinary != ('answering' in candidate):
        raise ValueError('Evaluate the complete answering system on both sides of promotion')
    # Missing or corrupted bytes must fail before numerical execution starts.
    examples = rows(policy, inputs, 'test')
    release_scope = policy['format'] not in (GENERAL, *MEASURED)
    if ordinary:
        ordinary_quality.validate_rows(examples)
    else:
        cohort_questions.validate_rows(examples, release_scope=release_scope)
    # Verify every retained file before the first forward pass, too.
    for role in ROLES[1:]:
        values = rows(policy, inputs, role)
        if ordinary:
            ordinary_quality.validate_rows(values)
    retained = (measured_retention(policy, inputs, baseline, candidate, network)
                if policy['format'] in MEASURED else retention(policy, inputs, baseline, candidate))
    maximum = policy['generation']['new']
    before, after, executions = [], [], []
    for index, row in enumerate(examples):
        question = row['messages'][:-1] if ordinary else row['messages'][0]['content']
        old = network.answer(question, maximum, baseline)
        new = network.answer(question, maximum, candidate)
        before.append(old if ordinary else {'id': row['id'], 'text': old['text']})
        after.append(new if ordinary else {'id': row['id'], 'text': new['text']})
        executions.append({'id': row['id'], 'before': old, 'after': new})
        if progress:
            progress(index + 1, len(examples))
    decision = (ordinary_quality.decision(examples, before, after, policy['gates']) if ordinary else
                cohort_questions.decision(examples, before, after, policy['gates'], release_scope=release_scope))
    check = 'retained_answers' if policy['format'] in MEASURED else 'unchanged_retained_computations'
    decision['checks'][check] = retained['passed']
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
            or claim['stages'] != stages(policy)):
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
