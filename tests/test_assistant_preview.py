import hashlib
import io
import json

import pytest

from neuroshard.assistant import preview
from neuroshard.evolution import assistant_growth_cohort3_eval as development
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution.modular_reference_execution import ROOT, read


def reply(text, terminated=True):
    return {'text': text, 'terminated': terminated, 'executed': True, 'input_token_ids': [1],
            'token_ids': list(text.encode()), 'prompt_sha256': 'test-only', 'seconds': .1}


def envelope(name, args):
    return '<tool_call>' + json.dumps({'name': name, 'arguments': args}) + '</tool_call>'


def test_release_pins_are_the_confirmed_units_and_gates():
    release = preview.pins()
    execution = read(ROOT / release['execution'])
    units = {name.split('/')[0] for name in release['files'] if '/' in name}
    assert units == set(development.needed('upgrade'))
    for unit in units:
        assert release['files'][f'{unit}/trainable.safetensors']['sha256'] == execution['units'][unit]['trainable_sha256']
    assert {gate['file']: gate['sha256'] for gate in execution['gates'].values()} == {
        name: pin['sha256'] for name, pin in release['files'].items() if '/' not in name}
    assert len({pin['asset'] for pin in release['files'].values()}) == len(release['files'])
    assert release['base_url'].endswith(f"/releases/download/{release['release']}/")
    assert read(ROOT / release['published']['summary'])['report']['sets']['drafting']['upgrade'] == 192


def test_downloads_are_kept_only_when_they_match_their_pin(tmp_path, monkeypatch):
    good = b'module bytes'
    pin = {'asset': 'U1-trainable.safetensors', 'bytes': len(good), 'sha256': hashlib.sha256(good).hexdigest()}
    monkeypatch.setattr(preview, 'pins', lambda: {'base_url': 'https://example.invalid/',
                                                   'files': {'U1/trainable.safetensors': pin}})
    calls = []

    def serving(payload):
        def opener(url, timeout):
            calls.append(url)
            return io.BytesIO(payload)
        return opener

    for payload, error in ((b'tampered byt', 'pinned digest'), (good + b'!', 'larger')):
        with pytest.raises(ValueError, match=error):
            preview.fetch_modules(tmp_path, opener=serving(payload), progress=lambda _: None)
        assert not (tmp_path / 'units' / 'U1').exists() or not any((tmp_path / 'units' / 'U1').iterdir())
    preview.fetch_modules(tmp_path, opener=serving(good), progress=lambda _: None)
    preview.fetch_modules(tmp_path, opener=serving(b''), progress=lambda _: None)
    assert (tmp_path / 'units/U1/trainable.safetensors').read_bytes() == good
    assert calls == ['https://example.invalid/U1-trainable.safetensors'] * 3
    with pytest.raises(ValueError, match='unsafe'):
        preview.local_path(tmp_path, '../outside')


def test_conversation_replays_earlier_turns_and_sends_only_the_new_one_to_the_model():
    case = data.make_case('development', 'copy', 0)
    goal = case['turns'][0]['expected']
    policies = development.policies()
    texts = iter([envelope('list_documents', {'project': goal['project']}),
                  envelope('read_document', {'document_id': goal['source_ids'][0]}),
                  envelope('save_draft', goal), 'Draft saved locally.', 'You are welcome.'])
    live = []

    def respond(messages, tools):
        live.append(messages[-1]['content'])
        return reply(next(texts))

    conversation = preview.Conversation(case['world'], lambda first: {name: (respond, policy)
                                                                      for name, policy in policies.items()},
                                        lambda turn, user: 'drafting')
    first = conversation.say(case['turns'][0]['user'])
    assert first['completed'] and first['route'] == 'drafting' and len(first['calls']) == 3
    assert first['drafts'][goal['project']]['total'] == goal['total']
    second = conversation.say('Thank you.')
    assert second['final_text'] == 'You are welcome.' and second['calls'] == []
    assert second['drafts'] == first['drafts'] and len(live) == 5


def test_a_failed_turn_ends_the_conversation():
    case = data.make_case('development', 'copy', 0)
    policies = development.policies()
    conversation = preview.Conversation(case['world'], lambda first: {
        name: (lambda messages, tools: reply('', terminated=False), policy) for name, policy in policies.items()},
        lambda turn, user: 'drafting')
    assert conversation.say('Create the draft.')['failure'] and conversation.ended
    with pytest.raises(ValueError, match='ended'):
        conversation.say('Again.')


def test_chat_shows_a_sample_workspace_and_its_own_requests():
    policies = development.policies()

    class Fake:
        def __init__(self):
            self.policies = policies

        def routes(self, case):
            return {name: (lambda messages, tools: reply('Done.'), policy) for name, policy in policies.items()}

        def select(self, turn, user):
            return 'scheduling'

    lines, typed = [], iter(['Hello', '/new', '/quit'])
    preview.chat(Fake(), 0, None, read_line=lambda prompt: next(typed), write=lines.append)
    assert lines[0].startswith('Workspace: ') and any(line.startswith('Try: ') for line in lines)
    assert any(line.startswith('assistant (scheduling, ') and line.endswith('Done.') for line in lines)
    assert 'New conversation in the same workspace.' in lines


def test_cli_offers_the_preview(capsys):
    from neuroshard.client import cli

    with pytest.raises(SystemExit) as stopped:
        cli.main(['assistant', 'chat', '--help'])
    assert stopped.value.code == 0 and '--world' in capsys.readouterr().out
