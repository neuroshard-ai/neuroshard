import importlib.util
from pathlib import Path

import pytest

from neuroshard.evolution import auditing, settlement


def operator_module():
    path = Path(__file__).resolve().parents[2]/'scripts/run_lifecycle_operator.py'
    spec = importlib.util.spec_from_file_location('funded_operator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inference_price_covers_auditors_when_first_token_stops_generation():
    module = operator_module()
    manifest = {'auditing':auditing.PROFILE, 'params':settlement.PARAMS}
    floor = module.inference_price_floor(manifest, 4, 2)
    assert floor == 1_002_000
    assert floor > 1000  # The old fixture price requires a subsidy.


@pytest.mark.parametrize('price,tokens,subsidy,phase', [
    (1000, 1, False, 'inference_requires_explicit_subsidy_or_new_price_profile'),
    (1000, 8, True, 'inference_exceeds_operator_capacity_policy'),
    (1_000_000, 8, False, 'inference_exceeds_operator_capacity_policy'),
    (1000, 1, True, 'waiting_for_auditor_acceptance'),
    (1_000_000, 1, False, 'waiting_for_auditor_acceptance'),
])
def test_operator_bounds_inference_before_escrowing_service_fees(seed, price, tokens, subsidy, phase):
    module = operator_module()
    store, root, _ = seed
    operator = module.Operator.__new__(module.Operator)
    from neuroshard.demo.protocol import Identity
    operator.owner = Identity('operator-price-test')
    operator.chain_id = 'operator-tests'
    operator.store = store
    operator.capacities = [6000, 6000]
    operator.auditors = ['a'*66]
    operator.manifest = {'auditing':auditing.PROFILE, 'params':settlement.PARAMS}
    operator.config = {'budget':{'minimum_balance':0, 'allow_inference_subsidy':subsidy}}
    class Outbox:
        def pending(self):return None
    operator.outbox = Outbox()
    status = {'chain_id':operator.chain_id, 'candidate':None, 'assignment':None, 'height':1, 'training_round':0}
    job = {'provider':operator.owner.public_key,'claim_id':None,'expires':100,'model_root':root,'unit_price':price,'max_tokens':tokens}
    def query(path='/status', options=None):
        return {'/status':status,'/account':{'balance':1_000_000}, '/data':None,
                '/evaluation':None,'/inference':{'jobs':{'job':job}}}[path]
    operator.query = query
    purchases = []
    def budget(action, count):
        purchases.append((action, count))
        return None
    operator.budget = budget
    result = operator.tick()
    assert result['phase'] == phase
    if phase == 'waiting_for_auditor_acceptance':
        assert purchases == [('respond:job', 3)]
    else:
        assert not purchases
    if phase == 'inference_requires_explicit_subsidy_or_new_price_profile':
        assert result['jobs'][0]['minimum_cost_per_token'] == 302_000
