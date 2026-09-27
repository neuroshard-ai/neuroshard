import copy
import importlib.util
import json
import subprocess
import sys

import pytest

from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution import assistant_workflow_baseline as first
from neuroshard.evolution import assistant_workflow_canonical as study
from neuroshard.evolution import assistant_workflow_data as data

from test_assistant_workflow import execute_fixture, policy
from test_granite_tokenizer import granite_like, load_tiny


def plan():
    return study.read(study.ROOT / study.PLAN)


def cloud_module():
    spec = importlib.util.spec_from_file_location('canonical_cloud', study.ROOT / 'scripts/modular_reference_cloud.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def anchor_rows(failing=()):
    rows = []
    for task in study.read(study.ROOT / study.reference.PLAN)['tasks']:
        text = task['accept'][0] if task['kind'] == 'exact' else json.dumps(task['expected'])
        if task['kind'] == 'tool':
            text = '<tool_call>' + text + '</tool_call>'
        passed = task['id'] not in failing
        rows.append({'id': task['id'], 'text': text if passed else 'wrong', 'terminated': True, 'passed': passed})
    return rows


def primary(**changes):
    reply = {'execution_completed': True, 'peak_rss_bytes': 1024, 'anchors': anchor_rows(),
             'episodes': [execute_fixture(case) for case in data.cases('development')],
             'tokenizer': {key: plan()['tokenizer'][key] for key in ('pipeline_sha256', 'fixture_sha256')}}
    reply.update(changes)
    reply.setdefault('checked_encodes', study.expected_encodes(reply))
    return reply


def test_inventory_pins_contracts_and_every_imported_source():
    execution = study.read(study.ROOT / study.EXECUTION)
    for name, digest in execution['contracts'].items():
        assert study.sha256(study.ROOT / name) == digest
    assert set(execution['contracts']) <= set(execution['sources'])
    assert {study.PLAN, study.EXECUTION, study.SCRIPT, 'docs/granite-reference-requirements.txt',
            'src/neuroshard/evolution/granite_tokenizer.py'} <= set(execution['sources'])
    imported = subprocess.check_output([sys.executable, '-c', (
        'import os, sys; import neuroshard.evolution.assistant_workflow_canonical; '
        'root = os.path.abspath("src"); '
        'print("\\n".join(sorted(os.path.relpath(m.__file__) for m in list(sys.modules.values()) '
        'if getattr(m, "__file__", None) and os.path.abspath(m.__file__).startswith(root))))')],
        cwd=study.ROOT, text=True, env={'PYTHONPATH': str(study.ROOT / 'src')}).split()
    assert set(imported) <= set(execution['sources'])
    assert execution['packages'] == study.read(study.ROOT / first.EXECUTION)['packages']
    assert not execution['training_authorized'] and not execution['gpu_launch_authorized']


def test_only_tokenizer_and_anchor_stop_differ_from_the_failed_baseline():
    old, new = study.read(study.ROOT / first.PLAN), plan()
    for key in ('policy', 'data', 'split', 'model', 'replay_ids', 'resources', 'learning_contract'):
        assert new[key] == old[key]
    assert new['prior_protected_anchor_ids'] == old['protected_anchor_ids']
    kept = {k: v for k, v in old['qualification'].items()
            if k not in ('original_anchor_gate_required', 'all_18_original_successes_required')}
    assert kept == {k: v for k, v in new['qualification'].items() if k != 'canonical_anchor_gate_required'}
    failed = new['failed_baseline']
    assert study.sha256(study.ROOT / failed['result']) == failed['result_sha256']
    assert study.sha256(study.ROOT / failed['report']) == failed['report_sha256']
    report = study.read(study.ROOT / failed['report'])
    assert new['prior_protected_workflow_ids'] == report['report']['protected_workflow_ids']
    assert not report['baseline_passed']
    assert not new['training_authorized'] and not new['gpu_launch_authorized'] and not new['checklist_credit']


def test_cloud_profile_is_one_bounded_cpu_host():
    cloud = cloud_module()
    resources = cloud.resources(study.PROFILE)
    assert resources['hours'] == 2 and resources['planning_cap_usd'] == 6 and not resources['gpu']
    assert resources['purpose'] == study.PROFILE and resources['instance_type'] == 'r7i.4xlarge'
    assert cloud.remote_command(study.PROFILE)[1].endswith(study.SCRIPT)
    assert cloud.GRANITE_PROFILES[study.PROFILE][0] == 'assistant_workflow_canonical'


def test_qualification_needs_conforming_tokenizer_and_consistent_encode_count():
    report = study.assess(plan(), primary())
    assert report['baseline_qualified'] and report['correct'] == 24 and report['tokenizer_conforms']
    assert report['insufficient_development_headroom']
    assert report['prior_workflow_successes_lost'] == [] and len(report['workflows_gained']) == 22
    assert not report['training_authorized'] and not report['admission_evidence']
    legacy = primary(tokenizer={'pipeline_sha256': 'gpt2-regex', 'fixture_sha256': plan()['tokenizer']['fixture_sha256']})
    assert not study.assess(plan(), legacy)['baseline_qualified']
    skipped = primary()
    skipped['checked_encodes'] -= 1
    assert not study.assess(plan(), skipped)['tokenizer_conforms']
    missing = primary()
    missing['anchors'] = missing['anchors'][:-1]
    missing['checked_encodes'] = study.expected_encodes(missing)
    assert not study.assess(plan(), missing)['baseline_qualified']
    forged = primary()
    forged['episodes'][0]['score']['passed'] = not forged['episodes'][0]['score']['passed']
    with pytest.raises(ValueError, match='outcome rescore'):
        study.assess(plan(), forged)


def test_lost_prior_anchor_is_published_without_stopping_and_gate_still_applies():
    lost = 'granite-chat-booking'
    report = study.assess(plan(), primary(anchors=anchor_rows([lost])))
    assert report['prior_anchor_successes_lost'] == [lost] and report['canonical_anchor_gate']
    assert report['baseline_qualified'] and lost not in report['protected_anchor_ids']
    tools = [key for key in plan()['prior_protected_anchor_ids'] if key.startswith('granite-tool')]
    collapsed = study.assess(plan(), primary(anchors=anchor_rows(tools)))
    assert not collapsed['canonical_anchor_gate'] and not collapsed['baseline_qualified']


def test_native_worker_path_checks_every_encode_through_a_tiny_granite(tmp_path):
    torch = pytest.importorskip('torch')
    from transformers import GraniteConfig, GraniteForCausalLM

    tokenizer, _ = load_tiny(granite_like(tmp_path / 'granite'))
    torch.manual_seed(0)
    config = GraniteConfig(vocab_size=len(tokenizer.runtime), hidden_size=32, intermediate_size=64,
                           num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                           tie_word_embeddings=True, eos_token_id=tokenizer.eos_token_id,
                           pad_token_id=tokenizer.pad_token_id)
    model = GraniteForCausalLM(config).eval()
    bounded = copy.deepcopy(policy())
    bounded['generation'].update(max_input_tokens=100000, max_new_tokens=3)
    respond = first.native_responder(model, tokenizer, bounded)
    row = workflow.execute(data.make_case('development', 'copy', 0), respond, bounded)
    assert row['generations'] and all(g['executed'] for g in row['generations'])
    assert tokenizer.checked_encodes == study.expected_encodes({'episodes': [row], 'anchors': []})
    capped = copy.deepcopy(bounded)
    capped['generation']['max_input_tokens'] = 10
    before = tokenizer.checked_encodes
    row = workflow.execute(data.make_case('development', 'copy', 0), first.native_responder(model, tokenizer, capped), capped)
    assert not row['generations'][0]['executed']
    assert tokenizer.checked_encodes - before == study.expected_encodes({'episodes': [row]})
