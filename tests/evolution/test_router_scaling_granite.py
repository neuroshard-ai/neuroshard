import subprocess
import sys

import pytest

from neuroshard.evolution import assistant_routing as routing
from neuroshard.evolution import router_scaling_granite as granite
from neuroshard.evolution import router_scaling_study as study
from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256

from test_assistant_growth_baseline import cloud_module

torch = pytest.importorskip('torch')


@pytest.fixture(scope='module')
def tiny(tmp_path_factory):
    from test_assistant_experience import TEMPLATE, tiny_model
    from test_granite_tokenizer import granite_like, load_tiny

    directory = granite_like(tmp_path_factory.mktemp('granite'))
    (directory / 'chat_template.jinja').write_text(TEMPLATE)
    tokenizer = load_tiny(directory)[0]
    return tiny_model(tokenizer, seed=3), tokenizer


def policy():
    return read(ROOT / 'config/experiments/assistant-workflow-policy.json')


def test_layer_features_equal_the_accepted_router_feature_bit_for_bit(tiny):
    model, tokenizer = tiny
    texts = ['Create a draft for Cedar 4101.', 'Book room Atlas on 2027-01-02 at 10:00.', 'Move that 3 days later.']
    features, prefix = granite.layer_features(model, tokenizer, policy(), texts, layers=(1,))
    assert sorted(features) == ['layer-1', 'layer1'] and all(len(v) == len(texts) for v in features.values())
    accepted = [routing.message_feature(model, tokenizer, policy(), text, 'cpu', prefix) for text in texts]
    assert accepted == features['layer-1']
    assert features['layer1'] != features['layer-1']
    again, _ = granite.layer_features(model, tokenizer, policy(), list(reversed(texts)), layers=(1,))
    assert again['layer-1'] == list(reversed(features['layer-1']))


def test_the_declaration_pins_the_study_texts():
    declaration = read(ROOT / granite.DECLARATION)
    texts = study.texts()
    assert declaration['texts'] == len(texts) and declaration['texts_sha256'] == granite.sha256_texts(texts)
    assert declaration['no_training'] and declaration['no_serving'] and not declaration['checklist_credit']
    assert not declaration['opens_sealed_or_development_data']
    assert not any(case['id'].startswith(('real-development', 'real-confirmation'))
                   for case in study.everything()[0])


def test_execution_pins_contracts_sources_and_the_canonical_runtime():
    execution = read(ROOT / granite.EXECUTION)
    for name, digest in execution['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert set(execution['contracts']) <= set(execution['sources']) and granite.SCRIPT in execution['sources']
    assert not execution['training_authorized'] and not execution['gpu_launch_authorized']
    probe = ('import os, sys; import neuroshard.evolution.router_scaling_granite; assert "torch" not in sys.modules; '
             'import neuroshard.evolution.assistant_growth_round4, neuroshard.evolution.assistant_growth_run, '
             'neuroshard.evolution.assistant_routing, neuroshard.evolution.granite_reference, '
             'neuroshard.evolution.granite_tokenizer, neuroshard.evolution.router_scaling_study, '
             'neuroshard.evolution.assistant_workflow_canonical; '
             'import neuroshard.evolution.router_scaling_study as s; s.texts(); '
             'root = os.path.abspath("src"); print("\\n".join(sorted(os.path.relpath(x.__file__) '
             'for x in list(sys.modules.values()) if getattr(x, "__file__", None) and '
             'os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(execution['sources']), sorted(set(imported) - set(execution['sources']))
    canonical = read(ROOT / 'config/experiments/assistant-workflow-canonical-execution.json')
    assert all(execution[key] == canonical[key] for key in ('packages', 'python', 'required_cpu_flags', 'environment'))


def test_one_cpu_host_within_its_allowance():
    cloud = cloud_module()
    resources = cloud.resources(granite.PROFILE)
    execution = read(ROOT / granite.EXECUTION)
    assert not resources['gpu'] and 'upload' not in resources and resources['instance_type'] == 'r7i.4xlarge'
    assert resources['hours'] * resources['price']['usd_per_hour'] + 3 <= resources['planning_cap_usd'] <= 5.5
    assert (execution['prepare_seconds'] + execution['worker_seconds'] + resources['setup_seconds']
            + resources['copy_seconds'] + 600 <= resources['hours'] * 3600)
    assert granite.PROFILE not in cloud.GPU_PROFILES and granite.PROFILE not in cloud.UPLOAD_PROFILES
    assert cloud.GRANITE_PROFILES[granite.PROFILE][0] == 'router_scaling_granite'
    assert cloud.remote_command(granite.PROFILE)[1].endswith(granite.SCRIPT)


def test_the_granite_run_is_recorded_retired_and_within_its_allowance():
    report = read(ROOT / 'config/experiments/router-scaling-granite-report.json')
    result = read(ROOT / 'config/experiments/router-scaling-granite-result.json')
    declaration = read(ROOT / granite.DECLARATION)
    assert report['execution_completed'] and result['execution_completed'] and not report['error']
    assert report['accepted_feature_identical'] and report['accepted_feature_checked'] == granite.CHECKED
    assert report['texts'] == declaration['texts'] and report['layers'] == sorted(declaration['layers'])
    assert report['features_sha256'] == report['features_file_sha256'] == result['reply']['features_sha256']
    assert report['source_commit'] == result['binding']['freeze']['commit']
    finished = report['resources_finished']
    assert not finished['remaining_instances'] and not finished['remaining_volumes']
    assert finished['security_group_retired'] and finished['instance_type'] == 'r7i.4xlarge'
    resources = read(ROOT / 'config/experiments/router-scaling-granite-resources.json')
    assert finished['conservative_compute_usd'] <= resources['planning_cap_usd']
    assert not report['checklist_credit'] and not report['sealed_opened']
