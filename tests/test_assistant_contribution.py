import copy
import hashlib
import json
from pathlib import Path

import pytest

from neuroshard.client import cli, contribution, wire


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def example():
    return contribution.load(ROOT / 'examples/assistant-contribution.json')


@pytest.fixture
def wallet(tmp_path):
    return wire.Wallet(tmp_path / 'identity/account.key', create=True)


def packet(example, wallet):
    return contribution.create(example, wallet, 'original fictional demonstration', 'v1', True)


def review(packet, directory):
    raw = wire.canonical(packet) + b'\n'
    (directory / 'packet.json').write_bytes(raw)
    return {'format': 'neuroshard-assistant-review/v1', 'purpose': 'training', 'entries': [{
        'file': 'packet.json', 'sha256': hashlib.sha256(raw).hexdigest(),
        'example_root': packet['body']['example_root'], 'accepted_for_training': True}]}


def test_complete_correction_produces_targets_without_evaluation_goals(example, wallet):
    signed = packet(example, wallet)
    result = contribution.verify(signed)
    assert result['author'] == wallet.public_key
    assert result['turns'] == 2 and result['assistant_decisions'] == 8 and result['tool_calls'] == 7
    assert result['neural_execution'] is result['admitted'] is result['rewarded'] is False
    for row in result['records']:
        assert set(row) == {'messages', 'tools', 'example_root'}
        assert row['messages'][-1]['role'] == 'assistant'
        assert 'expected' not in json.dumps(row)
    assert result['records'][0]['messages'][-2] == {'role': 'user', 'content': example['turns'][0]['user']}
    assert not any(m['role'] == 'tool' for m in result['records'][0]['messages'])
    assert any(m['role'] == 'tool' for m in result['records'][-1]['messages'])
    assert contribution.policy() == contribution.load(ROOT / 'config/experiments/assistant-workflow-policy.json')


def test_signature_consent_and_content_root_are_enforced(example, wallet):
    with pytest.raises(ValueError, match='consent'):
        contribution.create(example, wallet, 'source', 'v1')
    signed = packet(example, wallet)
    tampered = copy.deepcopy(signed)
    tampered['body']['example']['turns'][0]['user'] += ' Changed'
    with pytest.raises(ValueError, match='Signature'):
        contribution.verify(tampered)
    for value in (False, 1):
        body = copy.deepcopy(signed['body'])
        body['consent']['training'] = value
        with pytest.raises(ValueError, match='consent'):
            contribution.verify(wallet.sign(body))
    body = copy.deepcopy(signed['body'])
    body['example_root'] = '0' * 64
    with pytest.raises(ValueError, match='root differs'):
        contribution.verify(wallet.sign(body))


def test_wrong_outcomes_invalid_actions_and_cross_turn_responses_are_rejected(example):
    changed = copy.deepcopy(example)
    changed['turns'][0]['expected']['total'] += 1
    with pytest.raises(ValueError, match='outcome'):
        contribution.validate_example(changed)
    changed = copy.deepcopy(example)
    changed['turns'][0]['assistant'].insert(0, '<tool_call>{"name":"shell","arguments":{"command":"true"}}</tool_call>')
    with pytest.raises(ValueError, match='rejected tool'):
        contribution.validate_example(changed)
    # Identical flattened responses must not disguise misassigned conversation turns.
    changed = copy.deepcopy(example)
    changed['turns'][0]['assistant'].append(changed['turns'][1]['assistant'].pop(0))
    with pytest.raises(ValueError, match='cross declared'):
        contribution.validate_example(changed)
    changed = copy.deepcopy(example)
    changed['world']['documents'][0]['project'] = []
    with pytest.raises(ValueError, match='metadata'):
        contribution.validate_example(changed)


def test_export_requires_exact_reviewed_bytes_and_never_partially_writes(example, wallet, tmp_path):
    manifest = review(packet(example, wallet), tmp_path)
    output = tmp_path / 'training.jsonl'
    bad = copy.deepcopy(manifest)
    bad['entries'].append({**bad['entries'][0], 'sha256': '0' * 64})
    with pytest.raises(ValueError, match='bytes changed'):
        contribution.export_reviewed(bad, tmp_path, output)
    assert not output.exists()
    bad = copy.deepcopy(manifest)
    bad['entries'].append(bad['entries'][0])
    with pytest.raises(ValueError, match='Duplicate'):
        contribution.export_reviewed(bad, tmp_path, output)
    bad = copy.deepcopy(manifest)
    bad['entries'][0]['accepted_for_training'] = False
    with pytest.raises(ValueError, match='approved'):
        contribution.export_reviewed(bad, tmp_path, output)
    result = contribution.export_reviewed(manifest, tmp_path, output)
    assert result['examples'] == 1 and result['assistant_decisions'] == 8
    assert result['training_started'] is result['admitted'] is False
    assert hashlib.sha256(output.read_bytes()).hexdigest() == result['sha256']
    assert len(output.read_text().splitlines()) == 8
    with pytest.raises(FileExistsError):
        contribution.export_reviewed(manifest, tmp_path, output)


def test_reserved_prompts_cannot_be_exported_even_with_a_valid_signature(example, wallet, tmp_path):
    from neuroshard.evolution.assistant_workflow_data import cases
    # Matching outcome alone does not prove the prose requests it: human review is required.
    example['turns'][0]['user'] = cases('confirmation')[0]['turns'][0]['user']
    manifest = review(packet(example, wallet), tmp_path)
    with pytest.raises(ValueError, match='Reserved workflow'):
        contribution.export_reviewed(manifest, tmp_path, tmp_path / 'never.jsonl')
    assert not (tmp_path / 'never.jsonl').exists()


def test_cli_packages_locally_without_network_or_neural_runtime(tmp_path, monkeypatch, capsys):
    import subprocess
    def forbidden(*args, **kwargs):
        raise AssertionError('Contributing must not send data, download models or execute a subprocess')
    monkeypatch.setattr(wire, 'http', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    output = tmp_path / 'packet.json'
    cli.main(['contribute', 'package', '--home', str(tmp_path / 'home'),
              '--input', str(ROOT / 'examples/assistant-contribution.json'), '--output', str(output),
              '--origin', 'original demonstration', '--revision', 'v1', '--public-training'])
    report = json.loads(capsys.readouterr().out)
    assert not report['uploaded'] and not report['admitted']
    assert (tmp_path / 'home/account.key').read_text().strip() not in output.read_text()
    cli.main(['contribute', 'verify', str(output)])
    assert json.loads(capsys.readouterr().out)['example_root'] == report['example_root']
