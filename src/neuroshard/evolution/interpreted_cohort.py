"""Add a separately learning expert while preserving neural interpretation."""
import hashlib
import json

from . import cohort_experiment as cohort, preserved_interpreter as previous
from . import reference_data as data
from .sharded.branch_groups import OrderedRoutes

base, ROOT = cohort.base, cohort.ROOT
FORMAT = 'neuroshard-interpreted-cohort-v1'
PLAN = ROOT / 'config/experiments/interpreted-cohort.json'
PREPARED = ROOT / 'config/experiments/interpreted-cohort-prepared.json'
SELECTION = ROOT / 'config/experiments/interpreted-cohort-selection.json'
PREVIOUS_RESULT = ROOT / 'config/experiments/preserved-interpreter-results.json'
SOURCES = sorted(set(cohort.SOURCES) | set(previous.SOURCES) | {
    'src/neuroshard/evolution/interpreted_cohort.py',
    'scripts/run_interpreted_cohort.py',
    'config/experiments/branch-cohort-questions.json'})


def prerequisite(plan, prepared, result_bytes):
    """Check the published final, not an operator-supplied eligibility flag."""
    result = json.loads(result_bytes)
    expected = prepared['prerequisite']
    checks = ('knowledge_accuracy', 'knowledge_gain', 'prior_correct_answers_retained',
              'conversation_retention', 'exact_parent_answers', 'exact_parent_losses',
              'question_only_selector', 'parent_survives_expert_exit')
    if (result['format'] != 'neuroshard-preserved-interpreter-results-v1'
            or result['completed'] is not True or result['passed'] is not True
            or result['failure'] is not None or result['numerical']['passed'] is not True
            or result['numerical']['finals_opened'] is not True
            or result['numerical']['final']['passed'] is not True
            or set(result['numerical']['final']['checks']) != set(checks)
            or any(value is not True for value in result['numerical']['final']['checks'].values())
            or data.identity(result['graph']) != plan['previous_graph']
            or result['source_freeze']['prepared'] != plan['previous_prepared']
            or result['evidence']['readback_verified'] is not True
            or result['resources']['terminated'] is not True
            or expected != {'graph': plan['previous_graph'],
                            'result_sha256': hashlib.sha256(result_bytes).hexdigest(),
                            'evidence_sha256': result['evidence']['sha256']}):
        raise ValueError('Require the passing, archived final of this exact preserved graph')
    return result


def validate():
    plan, prepared = (json.loads(base.committed(path)) for path in (PLAN, PREPARED))
    if (plan['format'] != FORMAT or plan['split'] != 22
            or plan['parent_layout'] != [0, 6, 15, 24]
            or plan['expert_layout'] != [0, 6, 15, 22, 24]
            or plan['training']['steps'] != 560 or plan['checkpoints'] != [0, 280, 560]
            or plan['batch_documents'] != 32 or plan['microbatch'] != 8
            or prepared['plan'] != data.identity(plan)
            or prepared['schedule'] != list(range(28)) * 20
            or prepared['sources'] != {name: data.sha256(ROOT / name) for name in SOURCES}):
        raise ValueError('Require the complete frozen interpreted-cohort recipe')
    rules = OrderedRoutes(plan['rules'])
    expected_rules = [{'id': 'directory', 'needle': 'fictional luma directory', 'owner': 3},
                      {'id': 'protocol', 'needle': 'neuroshard 0.4.0', 'owner': 4}]
    if list(rules.rules) != expected_rules:
        raise ValueError('Preserve the earlier domain and append one new learning owner')
    if json.loads(base.committed(PLAN, prepared['source_commit'])) != plan:
        raise ValueError('Freeze this learning recipe before preparing inputs')
    for name in SOURCES:
        if base.committed(ROOT / name) != base.committed(ROOT / name, prepared['source_commit']):
            raise ValueError('Numerical source changed after cohort preparation')
    result = prerequisite(plan, prepared, base.committed(PREVIOUS_RESULT))
    prior = result['graph']
    prior_plan = json.loads(base.committed(previous.PLAN))
    if (data.identity(prior_plan) != plan['previous_plan']
            or any(prior[key] != plan[key] for key in ('interpreter', 'interpretation'))
            or any(prior_plan[key] != plan[key] for key in
                   ('parent', 'expert', 'parent_layout', 'expert_layout', 'split', 'selector', 'tokenizer', 'tokenizer_files'))
            or data.identity(prepared['interpreter_assets']) != plan['interpreter']['partitioned_assets']):
        raise ValueError('Retain the exact successful interpreter and instruction')
    return plan, prepared


def job(plan, prepared):
    return data.identity({'format': FORMAT, 'plan': data.identity(plan), 'prepared': data.identity(prepared)})


def graph(plan, parent, first, second):
    value = cohort.graph(plan, parent, first, second)
    original = previous.graph({**plan, 'format': 'neuroshard-preserved-interpreter-v1'}, parent, first)
    if data.identity(original) != plan['previous_graph']:
        raise ValueError('Growth must extend the exact established serving graph')
    value.update(format=FORMAT + '/graph', previous_graph=plan['previous_graph'],
                 interpreter=plan['interpreter'], interpretation=plan['interpretation'],
                 total_parameters=original['total_parameters'] + original['added_parameters'])
    return value


def network(args, plan, prepared, net):
    """The old four-owner group serves while the fifth owner trains alone."""
    if net.rank == 4:
        return None
    from transformers import LlamaConfig
    from .sharded.branch import Network
    from .sharded.interpretation import InterpretedNetwork
    from .sharded.model import Partition
    trained = net.networks['directory']
    preserved = None
    if net.rank < 3:
        manifest = prepared['interpreter_assets']['partitions'][str(net.rank)]
        config = LlamaConfig(**trained.shard.config.to_dict())
        config._attn_implementation = 'sdpa'
        shard = Partition(config, plan['parent_layout'], net.rank, trained.device, plan['parameter_limit'])
        if (shard.resident_parameters != manifest['parameters']
                or shard.resident_parameters + trained.shard.resident_parameters > plan['resident_parameter_limit']):
            raise ValueError('Combined original and trained partitions exceed the owner limit')
        shard.load_weights(args.interpreter, manifest)
        shard.eval().requires_grad_(False)
        preserved = Network(shard, trained.wire, trained.parent_wire, trained.tokenizer, plan['split'])
    def record(value):
        with (args.home / 'interpretations.jsonl').open('a') as destination:
            destination.write(json.dumps(value) + '\n')
    path = InterpretedNetwork(trained, preserved, plan['interpretation']['instruction'],
        plan['interpretation']['examples'], plan['interpretation']['max_tokens'], record)
    net.answer_paths['directory'] = path.answer
    return path


rows = cohort.rows


def final_selection(plan, prepared):
    selection = json.loads(base.committed(SELECTION))
    if (selection['format'] != FORMAT + '/selection' or selection['eligible'] is not True
            or selection['plan'] != data.identity(plan) or selection['prepared'] != data.identity(prepared)
            or selection['step'] != plan['training']['steps']):
        raise ValueError('Only the committed eligible terminal expert may open finals')
    return selection
