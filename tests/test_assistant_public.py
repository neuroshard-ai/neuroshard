import copy
import json

import pytest

from neuroshard.assistant.public import CURRENT, PREVIOUS, Session, bind, default_consent
from neuroshard.evolution import assistant_workflow_baseline as study
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution import assistant_workspace as workspace


def policy():
    return study.read(study.ROOT / 'config/experiments/assistant-workflow-policy.json')


def reply(text, terminated=True):
    return {'text': text, 'terminated': terminated, 'executed': True,
            'input_token_ids': [1], 'token_ids': list(text.encode()),
            'prompt_sha256': 'test-only', 'seconds': .1}


def envelope(name, args):
    return '<tool_call>' + json.dumps({'name': name, 'arguments': args}) + '</tool_call>'


def test_version_binds_policy_and_keeps_rollback_target():
    bound = bind(policy())
    assert bound['name'] == CURRENT and bound['previous'] == PREVIOUS
    assert bound['version_sha256'] == bind(policy())['version_sha256']
    other = copy.deepcopy(policy())
    other['limits']['tool_calls_per_user_turn'] += 1
    assert bind(other)['version_sha256'] != bound['version_sha256']


def test_default_consent_keeps_session_off_the_ledger_and_out_of_training():
    case = data.make_case('development', 'copy', 0)
    session = Session(policy(), case['world'])
    session.note('owner', 'private')
    assert session.consent == default_consent()
    exported = session.export()
    assert exported['ledger'] is None and exported['training'] is None
    assert exported['messages'] == [] and exported['memory'] == {}
    with pytest.raises(ValueError, match='opt-in'):
        session.training_export()
    with pytest.raises(ValueError, match='external'):
        Session(policy(), case['world'], consent={**default_consent(), 'external_actions': True})
    forbidden = copy.deepcopy(policy())
    forbidden['external_effects'] = True
    with pytest.raises(ValueError, match='external'):
        bind(forbidden)


def test_turn_uses_workspace_and_shared_export_is_opt_in():
    case = data.make_case('development', 'copy', 0)
    goal = case['turns'][0]['expected']
    texts = iter((
        envelope('list_documents', {'project': goal['project']}),
        envelope('read_document', {'document_id': goal['source_ids'][0]}),
        envelope('save_draft', goal),
        'Draft saved locally.',
    ))
    session = Session(policy(), case['world'],
                      consent={**default_consent(), 'share_outside_session': True})
    result = session.turn(case['turns'][0]['user'], lambda messages, tools: reply(next(texts)))
    assert result['completed'] and result['version'] == CURRENT
    assert result['snapshot']['drafts'][goal['project']]['total'] == goal['total']
    session.note('hint', 'keep private unless shared')
    exported = session.export()
    assert any(row['role'] == 'user' for row in exported['messages'])
    assert exported['memory']['hint'] == 'keep private unless shared'
    assert session.rollback_target() == PREVIOUS
    replayed = workspace.replay_transcript(case['world'], session.calls)
    assert replayed['drafts'][goal['project']]['total'] == goal['total']
