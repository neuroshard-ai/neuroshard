"""Prompt processing, planner work and all selected sources share a bound."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from neuroshard.evolution import planned_metering as meter, serving_graph
from neuroshard.evolution.reference_data import identity


@pytest.fixture
def execution():
    graph = json.loads((Path(__file__).with_name('fixtures')/'native_expert_graph.json').read_bytes())['candidate']
    service = {'graph': identity(graph), 'planner': {'max_tokens': 128},
               'composer': {'max_tokens': 128}, 'planner_weights': {'fusion': 'a'*64,
               'binding': {'layout': {'rank': 8, 'projections': {'15.mlp.up_proj': [2048, 8192]}}}}}
    tariff = {'prompt_atom_price': 2, 'output_atom_price': 7, 'context': 1024}
    def call(model, purpose):
        value = {'model': model, 'purpose': purpose, 'prompt_ids': [1, 3, 4],
                 'token_ids': [9, 2], 'owners': sorted(map(int, serving_graph.ownership(graph, model)))}
        if purpose == 'planning':
            value['planner_adapter'] = 'a'*64
        return value
    outputs = [call('interpreter', 'planning'), call('interpreter', 'directory_arguments'),
               call('directory', 'answer'), call('protocol', 'answer')]
    response = {'format': 'neuroshard-planned-graph-service-v1/response', 'service': identity(service),
                'request': {'service': identity(service), 'max_tokens': 128, 'messages': []},
                'outputs': outputs, 'generated_tokens': 8, 'status': 'completed',
                'rendering': {'token_ids': [3]*50}}
    return graph, service, tariff, response


def test_complete_meter_covers_prompts_planning_and_sources_without_billing_rendered_values(execution):
    graph, service, tariff, response = execution
    report = meter.meter(service, graph, response, tariff)
    assert report['prompt_tokens'] == 12 and report['output_tokens'] == 8
    assert report['total_atoms'] == 12*2+8*7 == sum(report['payments'].values())
    assert set(report['payments']) == {'0', '1', '2', '3', '4'}
    assert report['requires_complete_neural_replay']
    quote = meter.quote(service, graph, 128, tariff)
    assert report['unused_reserved_atoms']+report['total_atoms'] == quote['maximum_atoms']
    prompt_heavy = copy.deepcopy(response)
    prompt_heavy['outputs'][0]['prompt_ids'] = [3]*896
    changed = meter.meter(service, graph, prompt_heavy, tariff)
    assert changed['total_atoms']-report['total_atoms'] == (896-3)*2
    baseline_sizes = serving_graph.ownership(graph, 'interpreter')
    adapted_sizes = meter.ownership(graph, service, response['outputs'][0])
    assert adapted_sizes['2']-baseline_sizes['2'] == 8*(2048+8192)
    assert adapted_sizes['0'] == baseline_sizes['0']


@pytest.mark.parametrize('attack', ['owner', 'owner_bool', 'adapter', 'tokens', 'stopping', 'context', 'extra_call', 'purpose', 'total', 'total_bool', 'service'])
def test_changed_work_or_unbounded_price_inputs_are_rejected(execution, attack):
    graph, service, tariff, response = execution
    if attack == 'owner': response['outputs'][2]['owners'].append(4)
    elif attack == 'owner_bool': response['outputs'][0]['owners'][0] = False
    elif attack == 'adapter': response['outputs'][0]['planner_adapter'] = 'b'*64
    elif attack == 'tokens': response['outputs'][1]['token_ids'] = [True, 2]
    elif attack == 'stopping': response['outputs'][1]['token_ids'] = [2, 3]
    elif attack == 'context': response['outputs'][0]['prompt_ids'] = [3]*897
    elif attack == 'extra_call': response['outputs'].append(copy.deepcopy(response['outputs'][-1]))
    elif attack == 'purpose': response['outputs'][2]['purpose'] = 'free_work'
    elif attack == 'total': response['generated_tokens'] -= 1
    elif attack == 'total_bool': response['generated_tokens'] = True
    elif attack == 'service': response['request']['service'] = 'b'*64
    with pytest.raises(ValueError):
        meter.meter(service, graph, response, tariff)


def test_metering_import_does_not_load_neural_runtime():
    subprocess.run([sys.executable, '-c',
        'import sys; from neuroshard.evolution import planned_metering; assert "torch" not in sys.modules'],
        check=True, timeout=15)


def test_preserved_requests_pay_actual_calls_and_reserve_one_reference_repair(execution):
    from neuroshard.evolution.request_planning import FORMAT
    graph, service, tariff, response = execution
    service['request_policy'] = FORMAT
    response['service'] = response['request']['service'] = identity(service)
    response['request']['messages'] = [{'role': 'user', 'content': 'What command starts the renderer?'}]
    response['outputs'] = response['outputs'][1:]
    response['generated_tokens'] = 6
    report = meter.meter(service, graph, response, tariff)
    assert len(report['calls']) == 3
    assert meter.quote(service, graph, 128, tariff)['call_limits']['planning_repair'] == 1
    response['request']['messages'][0]['content'] = 'What does the renderer do and how does it work?'
    with pytest.raises(ValueError, match='order'):
        meter.meter(service, graph, response, tariff)


def test_reference_repair_is_metered_once_before_expert_execution(execution):
    from neuroshard.evolution.request_planning import FORMAT
    graph, service, tariff, response = execution
    service['request_policy'] = FORMAT
    response['service'] = response['request']['service'] = identity(service)
    response['request']['messages'] = [{'role': 'user', 'content': 'What does the renderer do and how does it work?'}]
    repair = copy.deepcopy(response['outputs'][0])
    repair['purpose'] = 'planning_repair'
    repair.pop('planner_adapter')
    response['outputs'].insert(1, repair)
    response['generated_tokens'] = 10
    assert meter.meter(service, graph, response, tariff)['total_atoms'] == 15*2+10*7
    response['outputs'].insert(2, copy.deepcopy(repair))
    response['generated_tokens'] = 12
    with pytest.raises(ValueError):
        meter.meter(service, graph, response, tariff)
