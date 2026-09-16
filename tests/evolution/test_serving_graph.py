"""Use the published 3.691B graph metadata without allocating neural tensors."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from neuroshard.evolution import serving_graph as graph
from neuroshard.evolution.reference_data import identity

FIXTURE = Path(__file__).with_name('fixtures') / 'native_expert_graph.json'


@pytest.fixture
def graphs():
    data = json.loads(FIXTURE.read_bytes())
    candidate = data['candidate']
    previous = copy.deepcopy(candidate)
    previous['descriptor'] = data['previous_descriptor']
    del previous['experts']['protocol']
    return previous, candidate


def output(model, tokens):
    return {'model': model, 'prompt_root': 'a'*64, 'token_ids': tokens}


def test_published_graph_reconstructs_exact_roots_ages_and_ownership(graphs):
    previous, candidate = graphs
    assert graph.validate(previous) is previous and graph.validate(candidate) is candidate
    assert identity(candidate['descriptor']) == '8c274134954f28f1a35114f74a314b659ccd1a267a0d0d1125325a393d64da81'
    for name in ('parent', 'interpreter', 'directory', 'protocol'):
        counts = graph.ownership(candidate, name)
        assert sum(counts.values()) == 1711376384
        assert max(counts.values()) < sum(counts.values())
    assert graph.ownership(candidate, 'protocol')['4'] == 134225920
    assert candidate['experts']['directory']['step'] == 1024
    assert candidate['experts']['protocol']['step'] == 560
    assert candidate['parent']['step'] == 448


def test_old_computation_is_unchanged_but_new_protocol_questions_are_rerouted(graphs):
    previous, candidate = graphs
    for question in ('Explain why the sky is blue.', 'In the fictional Luma directory, where does Ada Lane live?'):
        assert graph.execution_identity(previous, question, 64) == graph.execution_identity(candidate, question, 64)
    question = 'Regarding NeuroShard 0.4.0, what is its peer port?'
    assert graph.execution_identity(previous, question, 64) != graph.execution_identity(candidate, question, 64)
    changed = copy.deepcopy(candidate)
    changed['tokenizer']['files']['tokenizer.json'] = 'b'*64
    assert graph.execution_identity(previous, 'Hello!', 64) != graph.execution_identity(changed, 'Hello!', 64)
    changed = copy.deepcopy(candidate)
    changed['descriptor']['interpretation']['instruction'] += ' Changed'
    question = 'In the fictional Luma directory, where does Ada Lane live?'
    assert graph.execution_identity(previous, question, 64) != graph.execution_identity(changed, question, 64)


def test_composed_billing_pays_both_neural_calls_and_never_the_unused_expert(graphs):
    _, candidate = graphs
    question = ('NeuroShard 0.4.0: First: What is its peer port? Second: What is its token? '
        'Reply with the two short answers in order. Separate the two answers with a semicolon.')
    plan = graph.calls(candidate, question, 64)
    assert len(plan) == 2 and all(call['model'] == 'protocol' for call in plan)
    assert graph.maximum_price(candidate, plan, 7) == 128 * 7
    payments = graph.payments(candidate, plan, [output('protocol', [3, 2]), output('protocol', [4, 5, 2])], 7)
    assert sum(payments.values()) == 35 and set(payments) == {'0', '1', '2', '4'}
    assert '3' not in payments
    with pytest.raises(ValueError, match='every planned'):
        graph.payments(candidate, plan, [output('protocol', [3, 2])], 7)


def test_directory_billing_includes_interpretation_and_parameter_weighted_shares(graphs):
    _, candidate = graphs
    plan = graph.calls(candidate, 'In the fictional Luma directory, who is Ada Lane?', 64)
    assert [call['model'] for call in plan] == ['interpreter', 'directory']
    assert graph.maximum_price(candidate, plan, 13) == (40 + 64) * 13
    payments = graph.payments(candidate, plan, [output('interpreter', [3, 4, 2]), output('directory', [5, 2])], 13)
    assert sum(payments.values()) == 65 and set(payments) == {'0', '1', '2', '3'}
    assert payments['3'] <= 2 * 13 * 134225920 // 1711376384 + 1


@pytest.mark.parametrize('tokens', [[], [2, 3], [3], [True, 2], [49152, 2]])
def test_every_neural_call_requires_valid_greedy_stopping(graphs, tokens):
    candidate = graphs[1]
    with pytest.raises(ValueError):
        graph.payments(candidate, graph.calls(candidate, 'Hello!', 64), [output('parent', tokens)], 1)


@pytest.mark.parametrize('change', ['age', 'owner', 'interpreter', 'shape', 'size', 'prompt'])
def test_substituted_graph_components_cannot_be_served(graphs, change):
    value = graphs[1]
    if change == 'age':
        next(iter(value['experts']['protocol']['tensors'].values()))['optimizer_step'] = 559
    elif change == 'owner':
        value['descriptor']['rules'][1]['owner'] = 3
    elif change == 'interpreter':
        value['interpreter_assets']['partitions']['0']['tensors'].pop('model.embed_tokens.weight')
    elif change == 'shape':
        next(iter(value['experts']['directory']['tensors'].values()))['shape'][0] += 1
    elif change == 'size':
        value['descriptor']['total_parameters'] += 1
    else:
        value['interpreter_prompt']['tokens'] = 'b'*64
    with pytest.raises(ValueError):
        graph.validate(value)


def test_graph_validation_import_does_not_load_pytorch():
    subprocess.run([sys.executable, '-c',
        'import sys; from neuroshard.evolution import serving_graph; assert "torch" not in sys.modules'],
        check=True, timeout=15)
