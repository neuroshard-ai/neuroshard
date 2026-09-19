"""Changed weights may pass retention; lost correct answers may not."""
import json
from types import SimpleNamespace

import pytest

from neuroshard.evolution import expert_data
from neuroshard.evolution.reference_data import identity, sha256
from neuroshard.evolution.sharded import graph_quality


def test_measured_retention_rejects_forgetting_even_with_new_answer_gains(tmp_path, monkeypatch):
    rows = []
    for question, answer in [('old fact', '16'), ('old skill', '5')]:
        messages = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]
        rows.append({'id': expert_data.document_identity(messages), 'stratum': 'single',
                     'topics': [question], 'answers': [answer], 'messages': messages})
    roles = {}
    for role, row in zip(graph_quality.ROLES[1:3], rows):
        path = tmp_path/(role+'.jsonl')
        path.write_text(json.dumps(row)+'\n')
        roles[role] = {'file': path.name, 'sha256': sha256(path), 'count': 1, 'ids': identity([row['id']])}
    policy = {'format': graph_quality.CONTINUAL, 'roles': roles,
              'generation': {'retained_knowledge': 16, 'retained_skills': 16},
              'retention_gates': {'max_lost_correct': 0}}
    monkeypatch.setattr(graph_quality, 'retention', lambda *args: {
        'roles': {graph_quality.ROLES[-1]: [{'id': 'conversation', 'unchanged': True}]}})
    calls = []
    outputs = {'baseline': {'old fact': '16', 'old skill': 'wrong'},
               'candidate': {'old fact': '16.', 'old skill': '5'}}
    def answer(question, cap, graph):
        calls.append((question, cap, graph))
        return {'text': outputs[graph][question], 'graph': graph}
    network = SimpleNamespace(answer=answer)
    result = graph_quality.measured_retention(policy, tmp_path, 'baseline', 'candidate', network)
    assert result['passed'] and result['lost_correct'] == 0 and len(calls) == 4
    outputs['candidate']['old fact'] = '17'
    result = graph_quality.measured_retention(policy, tmp_path, 'baseline', 'candidate', network)
    assert not result['passed'] and result['lost_correct'] == 1
    # The candidate's improvement on the second task cannot offset forgetting.
    assert result['roles']['retained-test-skills'][0]['after_correct']
    policy['roles']['test'] = {'count': 3}
    assert graph_quality.stages(policy) == 5
    policy['format'] = graph_quality.GENERAL
    assert graph_quality.stages(policy) == 3
