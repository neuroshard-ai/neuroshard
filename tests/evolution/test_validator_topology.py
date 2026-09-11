import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('check_validator_topology',Path(__file__).resolve().parents[2]/'scripts/check_validator_topology.py')
topology = importlib.util.module_from_spec(spec)
spec.loader.exec_module(topology)


def inventory(count, hosts=None, operators=None):
    return {'validators':[{'consensus_key':f'{i+1:064x}','host':(hosts or [f'h{i}' for i in range(count)])[i],
                          'operator':(operators or [f'o{i}' for i in range(count)])[i]} for i in range(count)]}


def test_four_equal_independent_domains_survive_one_failure():
    value = inventory(4)
    result = topology.assess({row['consensus_key']:10 for row in value['validators']},value)
    assert result['tolerates_each_single_declared_domain_failure']


@pytest.mark.parametrize('value',[inventory(3),inventory(4,hosts=['a','a','b','b']),inventory(4,operators=['one']*4)])
def test_two_thirds_exact_or_collocated_keys_cannot_fake_fault_tolerance(value):
    result = topology.assess({row['consensus_key']:10 for row in value['validators']},value)
    assert not result['tolerates_each_single_declared_domain_failure']
    assert any(not row['can_finalize'] for row in result['failure_cases'])


def test_inventory_cannot_omit_voting_power_or_repeat_keys():
    value = inventory(4)
    power = {row['consensus_key']:10 for row in value['validators']}
    with pytest.raises(ValueError,match='exactly the current'):
        topology.assess(power,{'validators':value['validators'][:-1]})
    with pytest.raises(ValueError,match='duplicate'):
        topology.assess(power,{'validators':value['validators']+[value['validators'][0]]})
