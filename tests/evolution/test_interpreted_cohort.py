"""The next learner cannot silently waive a failed or different first graph."""
import copy
import hashlib
import json

import pytest

from neuroshard.evolution import interpreted_cohort as contract, reference_data as data
from test_expert_cohort_job import inputs


def published():
    checks = ('knowledge_accuracy', 'knowledge_gain', 'prior_correct_answers_retained',
              'conversation_retention', 'exact_parent_answers', 'exact_parent_losses',
              'question_only_selector', 'parent_survives_expert_exit')
    result = {'format': 'neuroshard-preserved-interpreter-results-v1', 'completed': True,
        'passed': True, 'failure': None, 'graph': {'bound': 'actual four-owner graph'},
        'source_freeze': {'prepared': 'a' * 64},
        'numerical': {'passed': True, 'finals_opened': True,
                      'final': {'passed': True, 'checks': dict.fromkeys(checks, True)}},
        'evidence': {'readback_verified': True, 'sha256': 'b' * 64},
        'resources': {'terminated': True}}
    raw = json.dumps(result).encode()
    plan = {'previous_graph': data.identity(result['graph']), 'previous_prepared': 'a' * 64}
    prepared = {'prerequisite': {'graph': plan['previous_graph'],
        'result_sha256': hashlib.sha256(raw).hexdigest(), 'evidence_sha256': 'b' * 64}}
    return plan, prepared, result, raw


def test_only_exact_successful_archived_final_opens_second_learning():
    plan, prepared, result, raw = published()
    assert contract.prerequisite(plan, prepared, raw) == result
    mutations = [(['completed'], False), (['passed'], False), (['passed'], 1),
        (['failure'], {'reason': 'expired'}), (['numerical', 'finals_opened'], False),
        (['numerical', 'final', 'passed'], False), (['numerical', 'final', 'checks'], {}),
        (['numerical', 'final', 'checks', 'knowledge_accuracy'], False),
        (['numerical', 'final', 'checks', 'knowledge_accuracy'], 1),
        (['graph'], {'bound': 'earlier failed graph'}), (['source_freeze', 'prepared'], '0' * 64),
        (['evidence', 'readback_verified'], False), (['resources', 'terminated'], False)]
    for path, value in mutations:
        bad = copy.deepcopy(result)
        current = bad
        for key in path[:-1]:
            current = current[key]
        current[path[-1]] = value
        altered = json.dumps(bad).encode()
        claimed = copy.deepcopy(prepared)
        # Updating the operator's digest cannot make a failed result eligible.
        claimed['prerequisite']['result_sha256'] = hashlib.sha256(altered).hexdigest()
        with pytest.raises(ValueError, match='exact preserved graph'):
            contract.prerequisite(plan, claimed, altered)
    with pytest.raises(ValueError, match='exact preserved graph'):
        contract.prerequisite(plan, prepared, raw + b'\n')


def test_expansion_binds_unchanged_interpretation_and_actual_expert_ages(tmp_path):
    plan, _ = inputs(tmp_path)
    parent = json.loads((tmp_path / 'parent.json').read_bytes())
    first = json.loads((tmp_path / 'first.json').read_bytes())
    plan.update(format=contract.FORMAT, selector='the fixed first domain',
                interpreter={'parameters': 100}, interpretation={'instruction': 'fixed'})
    original = contract.previous.graph({**plan, 'format': 'neuroshard-preserved-interpreter-v1'}, parent, first)
    plan['previous_graph'] = data.identity(original)
    second = copy.deepcopy(first)
    second['job'] = 'd' * 64
    from neuroshard.evolution.sharded import incremental_state
    second['state_root'] = incremental_state.state_root(second)
    result = contract.graph(plan, parent, first, second)
    assert result['previous_graph'] == data.identity(original)
    assert result['experts'][1]['checkpoint'] == data.identity(second)
    assert result['total_parameters'] == original['total_parameters'] + original['added_parameters']
    changed = copy.deepcopy(plan)
    changed['interpretation']['instruction'] = 'different prompt'
    with pytest.raises(ValueError, match='exact established'):
        contract.graph(changed, parent, first, second)
    changed = copy.deepcopy(second)
    changed['tensors']['model.norm.weight']['optimizer_step'] = 1
    changed['state_root'] = incremental_state.state_root(changed)
    with pytest.raises(ValueError):
        contract.graph(plan, parent, first, changed)
