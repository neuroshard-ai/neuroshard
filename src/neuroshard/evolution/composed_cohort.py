"""Read-only composition of the already trained second expert."""
import json

from . import interpreted_cohort as previous, reference_data as data
from .sharded import composition, incremental_state
from .sharded.interpretation import example_messages

base, ROOT = previous.base, previous.ROOT
FORMAT = 'neuroshard-composed-cohort-v1'
PLAN = ROOT / 'config/experiments/composed-cohort.json'
PREPARED = ROOT / 'config/experiments/composed-cohort-prepared.json'
SELECTION = ROOT / 'config/experiments/composed-cohort-selection.json'
SOURCES = sorted(set(previous.SOURCES) | {
    'src/neuroshard/evolution/composed_cohort.py',
    'src/neuroshard/evolution/sharded/composition.py', 'scripts/run_composed_cohort.py'})


def validate():
    plan, prepared = (json.loads(base.committed(path)) for path in (PLAN, PREPARED))
    if (plan['format'] != FORMAT or plan['operation'] != 'read-only'
            or plan['composition'] != composition.FORMAT
            or plan['second_checkpoint'] != '698d9ae2ba3a89de5b1bf19ec42e6680fe2d408a96f92efa84b383ac5d95b530'
            or plan['training']['steps'] != 560 or prepared['format'] != FORMAT + '/prepared'
            or prepared['plan'] != data.identity(plan)
            or prepared['sources'] != {name: data.sha256(ROOT / name) for name in SOURCES}
            or json.loads(base.committed(PLAN, prepared['source_commit'])) != plan):
        raise ValueError('Require the fixed read-only composition and actual learned checkpoint')
    for name in SOURCES:
        if base.committed(ROOT / name) != base.committed(ROOT / name, prepared['source_commit']):
            raise ValueError('Composition source changed after preparation')
    training_plan = json.loads(base.committed(previous.PLAN))
    training_prepared = json.loads(base.committed(previous.PREPARED))
    if (plan['second_training_job'] != previous.job(training_plan, training_prepared)
            or plan['second_training_prepared'] != data.identity(training_prepared)
            or prepared['roles'] != training_prepared['roles']
            or prepared['retention_cache'] != training_prepared['retention_cache']
            or any(plan[key] != training_plan[key] for key in
                   ('parent', 'expert', 'previous_graph', 'previous_prepared', 'previous_plan',
                    'interpreter', 'interpretation', 'rules', 'training', 'gate', 'generation',
                    'parent_layout', 'expert_layout', 'split', 'tokenizer', 'tokenizer_files',
                    'runtime', 'resident_parameter_limit', 'parameter_limit'))):
        raise ValueError('Preserve learned weights, quality gates, inputs and numerical runtime')
    previous.prerequisite(plan, prepared, base.committed(previous.PREVIOUS_RESULT))
    result_path = ROOT / 'config/experiments/interpreted-cohort-results.json'
    result = json.loads(base.committed(result_path))
    expected = {'prepared': plan['second_training_prepared'], 'job': plan['second_training_job'],
                'checkpoint': plan['second_checkpoint'], 'evidence': result['evidence']['sha256'],
                'inventory': result['weights']['inventory_sha256'], 'result_sha256': data.sha256(result_path)}
    if (prepared['second_training'] != expected or result['partial_observations']['steps'] != 560
            or result['partial_observations']['checkpoint'] != plan['second_checkpoint']
            or result['partial_observations']['finals_opened'] is not False
            or result['partial_observations']['earlier_paths_served_during_updates'] is not True
            or result['evidence']['readback_verified'] is not True
            or result['weights']['readback_verified'] is not True or result['resources']['terminated'] is not True):
        raise ValueError('Bind the actual completed training, durable weights and unopened final')
    return plan, prepared


def job(plan, prepared):
    # This evaluation must not rename or repay the original training work.
    return plan['second_training_job']


def require_checkpoint(plan, prepared, selected, parent, first):
    if (selected is None or data.identity(selected) != plan['second_checkpoint']
            or selected['job'] != job(plan, prepared) or selected['step'] != plan['training']['steps']):
        raise ValueError('Restore the archived terminal expert without any further learning')
    incremental_state.validate(selected, parent)
    previous.graph(plan, parent, first, selected)


def graph(plan, parent, first, second):
    require_checkpoint(plan, {}, second, parent, first)
    value = previous.graph(plan, parent, first, second)
    value.update(format=FORMAT + '/graph', composition=plan['composition'],
                 interpreter_prompt=plan['interpreter_prompt'])
    return value


def network(args, plan, prepared, net):
    tokenizer = next(iter(net.networks.values())).tokenizer
    spec = plan['interpretation']
    prefix = example_messages(spec['instruction'], spec['examples'])
    tokens = tokenizer.apply_chat_template(prefix, tokenize=True, add_generation_prompt=False)
    actual = {'format': 'name-field-json-v1', 'messages': data.identity(prefix), 'tokens': data.identity(tokens)}
    if actual != plan['interpreter_prompt']:
        raise ValueError('The exact interpreter prompt changed')
    data.save(args.home / 'interpreter-prompt.json', {'binding': actual, 'ids': tokens})
    fixed = previous.network(args, plan, prepared, net)
    if 'protocol' in net.answer_paths:
        net.answer_paths['protocol'] = composition.ComposedAnswers(net.answer_paths['protocol'], tokenizer)
    return fixed


rows = previous.rows


def final_selection(plan, prepared):
    selection = json.loads(base.committed(SELECTION))
    if (selection['format'] != FORMAT + '/selection' or selection['eligible'] is not True
            or selection['plan'] != data.identity(plan) or selection['prepared'] != data.identity(prepared)
            or selection['checkpoint'] != plan['second_checkpoint']
            or selection['step'] != plan['training']['steps']):
        raise ValueError('Only archived passing development may open the untouched final')
    return selection
