import json

import pytest

from neuroshard.evolution import assistant_experience_run as run
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution.modular_reference_execution import ROOT, read

from test_assistant_experience import SPEC, TEMPLATE, tiny_model
from test_assistant_workflow import envelope, policy, reference_texts, reply
from test_granite_tokenizer import granite_like, load_tiny

torch = pytest.importorskip('torch')
CASES = [data.make_case('train', 'copy', 0), data.make_case('train', 'latest', 1), data.make_case('train', 'sum', 2)]
STUBBORN = CASES[1]['id']
EXECUTION = {'device': 'cpu', 'max_batch': 4, 'workers': 3, 'samples_per_case': 2, 'temperature': .8,
             'top_p': .95, 'seed': 5, 'integration_samples': 2, 'replay_seed': 27092029,
             'replay_max_input_tokens': 100000, 'replay_max_new_tokens': 4}


class ScriptedBatcher:
    """Goal-directed stand-in for sampling; one case succeeds only with the coaching card."""
    card = read(ROOT / run.PLAN)['coaching']['card']

    def __init__(self, model, tokenizer, generation, **options):
        self.batches = [1]

    def close(self):
        pass

    def respond(self, messages, tools):
        first = next(m['content'] for m in messages if m['role'] == 'user')
        case = next((c for c in CASES if c['turns'][0]['user'] == first), None)
        if case is None:
            return reply('Okay.')
        texts = reference_texts(case)
        if case['id'] == STUBBORN and self.card not in messages[0]['content']:
            texts[2] = envelope('save_draft', {**case['turns'][0]['expected'], 'total': 0})
        return reply(texts[sum(m['role'] == 'assistant' for m in messages)])


@pytest.fixture
def setup(tmp_path, monkeypatch):
    directory = granite_like(tmp_path / 'granite')
    (directory / 'chat_template.jinja').write_text(TEMPLATE)
    tokenizer = load_tiny(directory)[0]
    monkeypatch.setattr(run.rollout, 'Batcher', ScriptedBatcher)
    monkeypatch.setattr(run, 'split_cases', lambda plan, split: CASES)
    plan = {**read(ROOT / run.PLAN), 'training': {**SPEC, 'steps': 3}}
    return tokenizer, plan, tmp_path / 'home'


def test_gpu_inventory_pins_contracts_sources_packages_and_matching_authority():
    import importlib.util
    import subprocess
    import sys

    execution = read(ROOT / run.EXECUTION)
    for name, digest in execution['contracts'].items():
        assert run.sha256(ROOT / name) == digest
    assert set(execution['contracts']) <= set(execution['sources'])
    probe = ('import os, sys; import neuroshard.evolution.assistant_experience_run as m; '
             'assert "torch" not in sys.modules; '
             'import neuroshard.evolution.assistant_experience_train, neuroshard.evolution.assistant_selector; '
             'root = os.path.abspath("src"); '
             'print("\\n".join(sorted(os.path.relpath(x.__file__) for x in list(sys.modules.values()) '
             'if getattr(x, "__file__", None) and os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(execution['sources'])
    pinned = [line.split('==')[0] for line in (ROOT / 'docs/assistant-experience-requirements.txt').read_text().splitlines()
              if line and not line.startswith('#') and '==' in line]
    assert set(pinned) | {'torch'} == set(execution['packages'])
    assert execution['environment'] == execution['worker_environment']
    plan = read(ROOT / run.PLAN)
    assert execution['gpu_launch_authorized'] == plan['gpu_launch_authorized'] == plan['budget']['training_spend_currently_authorized']
    assert execution['attempts'] == 1 and not plan['native_promotion_authorized'] and not plan['checklist_credit']
    spec = importlib.util.spec_from_file_location('experience_cloud', ROOT / 'scripts/modular_reference_cloud.py')
    cloud = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cloud)
    resources = cloud.resources(run.PROFILE)
    candidates = resources['candidates']
    assert resources['gpu'] and candidates[0]['instance_type'] == 'g6e.xlarge' and resources['planning_cap_usd'] <= 25
    assert resources['hours'] * max(c['price']['usd_per_hour'] for c in candidates) + 3 <= resources['planning_cap_usd']
    assert {c['gpu_name'] for c in candidates} == set(execution['gpus'])
    assert execution['worker_seconds'] + resources['setup_seconds'] + 600 <= resources['hours'] * 3600
    assert cloud.remote_command(run.PROFILE)[1].endswith(run.SCRIPT)
    assert cloud.GRANITE_PROFILES[run.PROFILE] == ('assistant_experience_run', 'docs/assistant-experience-requirements.txt')


def test_gpu_placement_leaves_the_controller_zone_only_when_it_lacks_the_instance_type():
    import importlib.util

    spec = importlib.util.spec_from_file_location('placement_cloud', ROOT / 'scripts/modular_reference_cloud.py')
    cloud = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cloud)

    class EC2:
        def __init__(self, zones):
            self.zones = zones

        def describe_instance_type_offerings(self, **_):
            return {'InstanceTypeOfferings': [{'Location': zone} for zone in self.zones]}

        def describe_subnets(self, **_):
            return {'Subnets': [{'SubnetId': f'subnet-{z[-1]}', 'AvailabilityZone': z, 'State': 'available'}
                                for z in ('us-east-1f', 'us-east-1c', 'us-east-1a', 'us-east-1e')]}

    source = {'SubnetId': 'subnet-f', 'VpcId': 'vpc-1', 'Placement': {'AvailabilityZone': 'us-east-1f'}}
    assert cloud.placement_subnets(EC2({'us-east-1f', 'us-east-1a'}), source, 'r7i.4xlarge') == ['subnet-f']
    assert cloud.placement_subnets(EC2({'us-east-1a', 'us-east-1c'}), source, 'g6e.xlarge') == ['subnet-a', 'subnet-c']
    with pytest.raises(ValueError, match='not offered'):
        cloud.placement_subnets(EC2({'us-west-2a'}), source, 'g6e.xlarge')


def test_gpu_allocation_falls_back_through_declared_types_and_charges_the_placed_one(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    from botocore.exceptions import ClientError

    spec = importlib.util.spec_from_file_location('fallback_cloud', ROOT / 'scripts/modular_reference_cloud.py')
    cloud = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cloud)
    limits = cloud.resources(run.PROFILE)
    attempts = []

    class EC2:
        def describe_instances(self, **kw):
            if 'InstanceIds' in kw:
                return {'Reservations': [{'Instances': [{'VpcId': 'vpc-1', 'PrivateIpAddress': '10.0.0.2', 'KeyName': 'k',
                                                          'SubnetId': 'subnet-f', 'Placement': {'AvailabilityZone': 'us-east-1f'}}]}]}
            return {'Reservations': [{'Instances': [{'InstanceId': 'gpu', 'PrivateIpAddress': '10.0.1.3',
                                                       'State': {'Name': 'running'}}]}]}

        def describe_images(self, **kw):
            return {'Images': [{**limits['image'], 'State': 'available'}]}

        def describe_instance_type_offerings(self, **kw):
            return {'InstanceTypeOfferings': [{'Location': 'us-east-1a'}, {'Location': 'us-east-1b'}]}

        def describe_subnets(self, **kw):
            return {'Subnets': [{'SubnetId': f'subnet-{z}', 'AvailabilityZone': f'us-east-1{z}', 'State': 'available'}
                                for z in 'abf']}

        def create_security_group(self, **kw):
            return {'GroupId': 'sg-1'}

        def authorize_security_group_ingress(self, **kw):
            pass

        def run_instances(self, **kw):
            attempts.append((kw['InstanceType'], kw['NetworkInterfaces'][0]['SubnetId'], kw['ClientToken']))
            if kw['InstanceType'].startswith('g6e'):
                raise ClientError({'Error': {'Code': 'InsufficientInstanceCapacity', 'Message': 'none'}}, 'RunInstances')
            return {'Instances': [{'InstanceId': 'gpu'}]}

    monkeypatch.setattr(cloud.boto3, 'client', lambda *a, **kw: EC2())
    monkeypatch.setattr(cloud.subprocess, 'run', lambda *a, **kw: None)
    old_read = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda p, *a, **kw: 'ssh-ed25519 key' if p.name == 'id_ed25519.pub'
                        else old_read(p, *a, **kw))
    allocation = cloud.allocate(tmp_path, 'commit', run.PROFILE)
    assert [a[:2] for a in attempts] == [('g6e.xlarge', 'subnet-a'), ('g6e.xlarge', 'subnet-b'),
                                          ('g6e.2xlarge', 'subnet-a'), ('g6e.2xlarge', 'subnet-b'),
                                          ('g5.2xlarge', 'subnet-a')]
    assert len({a[2] for a in attempts}) == len(attempts)
    assert allocation['resources']['instance_type'] == 'g5.2xlarge'
    assert allocation['resources']['price']['usd_per_hour'] == 1.212
    assert len(allocation['placement_refusals']) == 4 and allocation['subnet'] == 'subnet-a'


def test_accelerator_phases_never_open_evaluation_goals():
    plan = read(ROOT / run.PLAN)
    assert len(run.split_cases(plan, 'train')) == 256 and len(run.split_cases(plan, 'integration')) == 64
    for split in ('development', 'confirmation'):
        with pytest.raises(ValueError, match='may not access'):
            run.split_cases(plan, split)


def test_collection_coaches_only_unsolved_cases_then_trains_and_fits_gates(setup):
    tokenizer, plan, home = setup
    home.mkdir()
    model = tiny_model(tokenizer)
    rows, report = run.collect(model, tokenizer, plan, policy(), EXECUTION, home)
    assert report['coached_cases'] == [STUBBORN]
    summary = report['summary']
    assert summary['cases_without_experience'] == [] and summary['rollouts'] == 3 * 2 + 2
    assert summary['by_family']['latest'] == {'cases': 1, 'complete': 1, 'coached_only': 1}
    # Identical scripted texts deduplicate to one trajectory per case.
    assert len(rows) == 3 and report['near_policy_ceiling'] is not None
    replay_rows = run.record_replay(model, tokenizer, plan, EXECUTION, home)
    assert len(replay_rows) == 256 and read(home / 'replay.json')['terminated'] == 256
    manifests = run.train_arms(lambda: tiny_model(tokenizer), plan, EXECUTION, rows, replay_rows, home)
    assert set(manifests) == {'update', 'addition'}
    assert manifests['addition']['trainable_parameters'] < manifests['update']['trainable_parameters']
    assert manifests['addition']['schedule_sha256'] == manifests['update']['schedule_sha256']
    gates = run.integrate(lambda: tiny_model(tokenizer), tokenizer, plan, policy(), EXECUTION, home)
    # Scripted parent and arms both succeed except the stubborn case, so every rate ties.
    assert {arm: value['gate']['rule'] for arm, value in gates.items()} == {
        'update': 'constant-parent', 'addition': 'constant-parent'}
    integration = read(home / 'integration.json')
    assert integration['parent_outcomes'][STUBBORN] == [False] * 3
    assert json.dumps(integration).count('"gate"') == 2
