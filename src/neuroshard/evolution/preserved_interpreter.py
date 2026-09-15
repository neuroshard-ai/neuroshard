"""Freeze a preserved interpreter plus the previously trained neural expert."""
import json
import math
from pathlib import Path

from . import branch_experiment as original, incremental_capacity as base, reference_data as data

ROOT = base.ROOT
PLAN = ROOT / 'config/experiments/preserved-interpreter.json'
PREPARED = ROOT / 'config/experiments/preserved-interpreter-prepared.json'
SELECTION = ROOT / 'config/experiments/preserved-interpreter-selection.json'
SOURCES = sorted(set(original.SOURCES) | {
    'src/neuroshard/evolution/preserved_interpreter.py',
    'src/neuroshard/evolution/sharded/interpretation.py',
    'scripts/run_preserved_interpreter.py',
    'config/experiments/preserved-interpreter-questions.json'})


def graph(plan, parent, expert):
    value = original.graph(plan, parent, expert)
    value.update(interpreter=plan['interpreter'], interpretation=plan['interpretation'],
                 interpreter_parameters=plan['interpreter']['parameters'])
    value['total_parameters'] = value['parent_parameters'] + value['added_parameters'] + value['interpreter_parameters']
    return value


def validate():
    plan, prepared = (json.loads(base.committed(path)) for path in (PLAN, PREPARED))
    if (plan['format'] != 'neuroshard-preserved-interpreter-v1' or plan['split'] != 22
            or plan['parent_layout'] != [0, 6, 15, 24] or plan['expert_layout'] != [0, 6, 15, 22, 24]
            or plan['interpretation']['instruction_placement'] != 'after-quoted-question'
            or plan['gate']['knowledge_accuracy'] != .75
            or prepared['plan'] != data.identity(plan)
            or prepared['sources'] != {name: data.sha256(ROOT / name) for name in SOURCES}
            or data.identity(prepared['interpreter_assets']) != plan['interpreter']['partitioned_assets']):
        raise ValueError('Preserved interpretation differs from its frozen plan or assets')
    assets = prepared['interpreter_assets']
    if (assets['source_weight_sha256'] != plan['interpreter']['weight_sha256']
            or any(assets['source'][key] != plan['interpreter'][key] for key in ('repo', 'revision', 'parameters'))
            or set(assets['partitions']) != {'0', '1', '2'}
            or sum(math.prod(spec['shape']) for part in assets['partitions'].values()
                   for spec in part['tensors'].values()) != plan['interpreter']['parameters']):
        raise ValueError('Original-model provenance or parameter inventory differs')
    if json.loads(base.committed(PLAN, prepared['source_commit'])) != plan:
        raise ValueError('Commit the interpretation, new final wording and gates before evaluation')
    for name in SOURCES:
        current = base.committed(ROOT / name)
        if current != base.committed(ROOT / name, prepared['source_commit']):
            raise ValueError('Numerical source differs from the prepared commit')
    return plan, prepared


def rows(prepared, home, role):
    return original.rows(prepared, home, role)


def network(args, plan, prepared, shard, wire, parent_wire, tokenizer):
    from transformers import LlamaConfig
    from .sharded.branch import Network
    from .sharded.interpretation import InterpretedNetwork
    from .sharded.model import Partition
    trained = Network(shard, wire, parent_wire, tokenizer, plan['split'])
    preserved = None
    if wire.rank < 3:
        manifest = prepared['interpreter_assets']['partitions'][str(wire.rank)]
        config = LlamaConfig(**shard.config.to_dict())
        config._attn_implementation = 'sdpa'
        owned = Partition(config, plan['parent_layout'], wire.rank, 'cuda', plan['parameter_limit'])
        if (owned.resident_parameters != manifest['parameters']
                or owned.resident_parameters + shard.resident_parameters > plan['resident_parameter_limit']):
            raise ValueError('Combined original and trained partitions exceed the owner limit')
        owned.load_weights(args.interpreter, manifest)
        owned.eval().requires_grad_(False)
        preserved = Network(owned, wire, parent_wire, tokenizer, plan['split'])
    def record(value):
        with (args.home / 'interpretations.jsonl').open('a') as destination:
            destination.write(json.dumps(value) + '\n')
    return InterpretedNetwork(trained, preserved, plan['interpretation']['instruction'],
        plan['interpretation']['examples'], plan['interpretation']['max_tokens'], record)


def final_questions(previous, forms, tokenizer, max_length):
    """Use trained facts with new committed wording and new record identities."""
    from collections import Counter
    counts, result = Counter(), []
    old_questions = {row['messages'][0]['content'] for row in previous}
    for row in previous:
        task = row['task']
        key = (task['entity'], task['attribute'])
        variant = counts[key]
        if variant >= 2:
            raise ValueError('Require exactly two prior forms per entity and field')
        counts[key] += 1
        question = forms['final'][task['attribute']][variant].format(name=task['name']) + forms['suffix']
        if question in old_questions:
            raise ValueError('The independent final cannot repeat an exposed question')
        messages = [{'role': 'user', 'content': question},
                    {'role': 'assistant', 'content': json.dumps({'answer': task['expected']}, separators=(',', ':'))}]
        identity = data.identity({'format': forms['format'], 'source_id': row['id'], 'question': question})
        result.append({'id': identity, 'task': dict(task), 'messages': messages,
                       **data.conversation(tokenizer, messages, max_length)})
    if not counts or any(count != 2 for count in counts.values()):
        raise ValueError('Incomplete entity and field coverage')
    if len({row['id'] for row in result}) != len(result) or {row['id'] for row in previous} & {row['id'] for row in result}:
        raise ValueError('New final identities must be unique and disjoint')
    return result
