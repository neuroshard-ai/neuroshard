"""Independent hosting soak for checklist item 4.

The operated GPU alpha served the accepted graph under one administrator and
is retired. This module pins the next experiment: a CPU protocol preflight of
equal-power genesis and stranger-provider join, then an independent-operator
soak that this operator's AWS account cannot satisfy. It does not train. It
does not authorize a GPU launch. It does not upgrade the public 0.4.0 chain.
It does not reopen learned-integration confirmation.
"""
import json
from pathlib import Path

from neuroshard.lab import state as ledger
from neuroshard.evolution.reference_data import identity, sha256

FORMAT = 'neuroshard-independent-hosting-v1'
CONTRACT_IDENTITY = '8213fdfd1a6f8713adc0356c99714e2587867f55f1b61cc16540175b75891e28'
PREVIOUS_CONTRACT = '5dc22aab5901cfb7a778c5897d50cf79e4c57013aab9ceb3c1368334e38e2141'
LEARNED_INTEGRATION = '6801d1a260e622948fa90714d41be11223d36d6a5d7a000c7d173f74f174186f'
ACCEPTED_GRAPH = '54f2361bf99e092e7fe9eb10597cbb31b19a509bc85608d35391553eb5ccfafa'
ITEM = 4
VALIDATORS = 4
OPERATORS = 4
SHARE_NUMERATOR = 1
SHARE_DENOMINATOR = 3
METHOD_FORMAT = FORMAT + '/method'
METHOD_SOURCES = (
    'config/experiments/independent-hosting.json',
    'docs/INDEPENDENT_HOSTING.md',
    'scripts/prepare_independent_hosting.py',
    'src/neuroshard/evolution/independent_hosting.py',
)


def spec_path():
    marker = Path('config/experiments/independent-hosting.json')
    for parent in Path(__file__).resolve().parents:
        candidate = parent / marker
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('independent-hosting.json is not next to this source tree')


def load_spec(path=None):
    return json.loads(Path(path or spec_path()).read_text())


def bind_spec(spec):
    if identity(spec) != CONTRACT_IDENTITY:
        raise ValueError('Independent-hosting contract does not bind this measurement')
    if spec.get('format') != FORMAT:
        raise ValueError('Independent-hosting format changed')
    if spec.get('item') != ITEM or spec.get('checklist_complete') is not False:
        raise ValueError('Item 4 remains open until the independent soak passes')
    if spec.get('gpu_launch_authorized') is not False:
        raise ValueError('This specification does not authorize a GPU launch')
    if spec.get('admission_evidence') is not False:
        raise ValueError('Independent hosting may not count as learning admission')
    if spec.get('new_training') is not False or spec.get('new_final') is not False:
        raise ValueError('Independent hosting does not train or open a new final')
    if spec.get('upgrade_public_0_4_0') is not False:
        raise ValueError('Independent hosting does not upgrade the public 0.4.0 chain')
    if spec.get('reopen_learned_integration_confirmation') is not False:
        raise ValueError('Learned-integration confirmation stays closed')
    if spec.get('aws_under_one_account_does_not_satisfy') is not True:
        raise ValueError('AWS machines under one account do not satisfy item 4')
    if spec.get('keys_are_not_operators') is not True:
        raise ValueError('Separate keys are not independent operators')
    if spec.get('ssh_controller_does_not_satisfy') is not True:
        raise ValueError('An SSH controller does not satisfy permissionless join')
    if spec.get('accepted_graph') != ACCEPTED_GRAPH:
        raise ValueError('Independent hosting is bound to the accepted operated-alpha graph')
    if spec.get('parent', {}).get('learned_integration') != LEARNED_INTEGRATION:
        raise ValueError('Independent hosting is bound to the failed learned-integration contract')
    if spec.get('revision', {}).get('replaces') != PREVIOUS_CONTRACT:
        raise ValueError('Independent-hosting revision must replace the three-operator contract')
    if spec.get('parent', {}).get('independent_hosting') != PREVIOUS_CONTRACT:
        raise ValueError('Independent-hosting parent must record the three-operator contract')
    protocol = spec.get('cpu_protocol') or {}
    if protocol.get('authorized') is not True or protocol.get('neural_execution') is not False:
        raise ValueError('The CPU protocol preflight is authorized without neural execution')
    if protocol.get('independent_administration') is not False:
        raise ValueError('The CPU preflight does not demonstrate independent administration')
    if protocol.get('validators') != VALIDATORS or protocol.get('equal_genesis_bond') is not True:
        raise ValueError('The CPU preflight uses four equal genesis bonds')
    bound = protocol.get('max_validator_share') or {}
    if (bound.get('numerator') != SHARE_NUMERATOR or bound.get('denominator') != SHARE_DENOMINATOR
            or bound.get('inclusive') is not False):
        raise ValueError('No validator may hold one third or more of voting power')
    soak = spec.get('independent_soak') or {}
    if soak.get('authorized') is not False:
        raise ValueError('The independent-operator soak is not authorized')
    if soak.get('minimum_independent_operators') != OPERATORS:
        raise ValueError('The soak requires four independently administered operators')
    if soak.get('voting_share_aggregates_by') != 'administrator':
        raise ValueError('Soak voting concentration is by administrator, not key')
    if soak.get('this_operator_may_hold_at_most_validators') != 1:
        raise ValueError('This operator may hold at most one soak validator')
    return {
        'independent_hosting': CONTRACT_IDENTITY,
        'gpu_launch_authorized': False,
        'admission_evidence': False,
        'independent_administration': False,
    }


def method_freeze():
    return {
        'format': METHOD_FORMAT,
        'spec': CONTRACT_IDENTITY,
        'gpu_launch_authorized': False,
        'admission_evidence': False,
        'independent_soak_authorized': False,
        'neural_execution': False,
        'files': {name: sha256(name) for name in METHOD_SOURCES},
    }


def bind_method_freeze(freeze, spec):
    bind_spec(spec)
    expected = method_freeze()
    if freeze != expected:
        raise ValueError('Method freeze does not match the committed independent-hosting method')
    if freeze.get('gpu_launch_authorized') is not False:
        raise ValueError('Method freeze does not authorize a GPU launch')
    if freeze.get('independent_soak_authorized') is not False:
        raise ValueError('Method freeze does not authorize the independent-operator soak')
    return identity(freeze)


def refuse_launch(spec, freeze):
    bind_method_freeze(freeze, spec)
    raise ValueError('This specification does not authorize a GPU launch or independent-operator soak')


def validator_powers(state):
    return ledger.voting_power(state, max(1, state['height']))


def share_is_concentrated(powers, spec=None):
    spec = spec or load_spec()
    bound = spec['cpu_protocol']['max_validator_share']
    total = sum(powers.values())
    if total <= 0:
        raise ValueError('Soak genesis requires bonded voting weight')
    return any(power * bound['denominator'] >= total * bound['numerator']
               for power in powers.values())


def administrator_powers(powers, ownership):
    if set(ownership) != set(powers):
        raise ValueError('Administrator ownership must name every bonded validator')
    if any(not administrator for administrator in ownership.values()):
        raise ValueError('Administrator identity is required')
    aggregated = {}
    for key, power in powers.items():
        administrator = ownership[key]
        aggregated[administrator] = aggregated.get(administrator, 0) + power
    return aggregated


def bind_soak_administration(powers, ownership, spec=None):
    spec = spec or load_spec()
    bind_spec(spec)
    admins = administrator_powers(powers, ownership)
    if len(admins) < spec['independent_soak']['minimum_independent_operators']:
        raise ValueError('Soak requires four independently administered operators')
    if share_is_concentrated(admins, spec):
        raise ValueError('No administrator may hold one third or more of voting power')
    return admins


def bind_cpu_genesis(state, spec=None):
    spec = spec or load_spec()
    bind_spec(spec)
    powers = validator_powers(state)
    if len(powers) != spec['cpu_protocol']['validators']:
        raise ValueError('Soak genesis requires four validators')
    if spec['cpu_protocol']['equal_genesis_bond'] and len(set(powers.values())) != 1:
        raise ValueError('Soak genesis requires equal validator bonds')
    if share_is_concentrated(powers, spec):
        raise ValueError('Soak genesis may not give any validator one third or more of voting power')
    return powers
