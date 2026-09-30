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


def test_pinned_collection_reverifies_every_trajectory_before_training(setup):
    import gzip
    from neuroshard.evolution.modular_reference_execution import identity, save, sha256

    tokenizer, plan, home = setup
    home.mkdir()
    model = tiny_model(tokenizer)
    original, report = run.collect(model, tokenizer, plan, policy(), EXECUTION, home)
    replay_rows = run.record_replay(model, tokenizer, plan, EXECUTION, home)
    pinned = {name: sha256(home / name) for name in run.COLLECTION_FILES}
    again = home / 'again'
    again.mkdir()
    experience_rows, reloaded_replay = run.load_collection(home, pinned, tokenizer, plan, policy(), EXECUTION, again)
    assert experience_rows == original and reloaded_replay == replay_rows
    assert read(again / 'collection.json')['trajectories'] == len(original)

    rows = run.read_rows(home / 'trajectories.jsonl.gz')
    rows[0]['messages'][-1]['content'] += ' Extra.'
    run.write_rows(home / 'trajectories.jsonl.gz', rows)
    with pytest.raises(ValueError, match='pinned digest'):
        run.load_collection(home, pinned, tokenizer, plan, policy(), EXECUTION, home / 'tampered')
    forged = read(home / 'experience.json')
    forged['trajectories_sha256'] = identity(rows)
    (home / 'experience.json').unlink()
    save(home / 'experience.json', forged)
    repinned = {name: sha256(home / name) for name in run.COLLECTION_FILES}
    with pytest.raises(ValueError, match='re-verify'):
        run.load_collection(home, repinned, tokenizer, plan, policy(), EXECUTION, home / 'forged')


def test_round_two_continues_both_pinned_arms_on_identical_verified_preferences(tmp_path, monkeypatch):
    from test_assistant_experience import sequences, version_rollouts
    from neuroshard.evolution import assistant_experience_train as trainer

    directory = granite_like(tmp_path / 'granite')
    (directory / 'chat_template.jinja').write_text(TEMPLATE)
    tokenizer = load_tiny(directory)[0]
    cases = [data.make_case('train', 'copy', i) for i in range(3, 6)]
    monkeypatch.setattr(run, 'split_cases', lambda plan, split: cases)
    rollouts = [row for case in cases for row in version_rollouts(case)[0]]
    base = read(ROOT / run.PLAN)
    plan = {**base, 'training': SPEC, 'decision_preferences': {**base['decision_preferences'], 'training': {
        **base['decision_preferences']['training'], 'steps': 3, 'gradient_accumulation': 4,
        'preference_per_update': 2, 'learning_rates': {'update': 1e-3, 'addition': 1e-2}}}}
    pairs, pair_rows = run.build_pairs(tokenizer, plan, policy(), rollouts)
    assert len(pairs) == 3 and {p['case_id'] for p in pairs} == {c['id'] for c in cases}
    experience_rows, replay_rows = sequences(tokenizer)
    round1, pinned = tmp_path / 'round1', {}
    for arm in ('update', 'addition'):
        trainable, receipt = trainer.train(tiny_model(tokenizer), arm, experience_rows, replay_rows, SPEC)
        pinned[arm] = trainer.checkpoint(round1 / f'{arm}-checkpoint', trainable, receipt, {})['trainable_sha256']
    home = tmp_path / 'home'
    home.mkdir()
    # The worker passes runtime parameters, not the inventory; pinned digests travel separately.
    manifests = run.train_round2(lambda: tiny_model(tokenizer), plan, {'device': 'cpu'},
                                 experience_rows, replay_rows, pair_rows, round1, pinned, home)
    assert {arm: m['roots']['round1'] for arm, m in manifests.items()} == pinned
    assert manifests['update']['schedule_sha256'] == manifests['addition']['schedule_sha256']
    assert all(m['trainable_sha256'] != pinned[arm] for arm, m in manifests.items())
    with pytest.raises(ValueError, match='pinned digest'):
        run.train_round2(lambda: tiny_model(tokenizer), plan, {'device': 'cpu'},
                         experience_rows, replay_rows, pair_rows, round1, {**pinned, 'update': 'x'}, tmp_path / 'other')
    with pytest.raises(ValueError, match='no verified decision'):
        run.build_pairs(tokenizer, plan, policy(), [r for r in rollouts if r['sample'] == 0])


def test_round_three_collects_from_an_arm_and_continues_on_divergence_pairs(setup, tmp_path):
    from test_assistant_experience import followup_rollouts, sequences
    from neuroshard.evolution import assistant_experience_train as trainer

    tokenizer, plan, home = setup
    home.mkdir()
    plan = {**plan, 'divergence_preferences': {**plan['divergence_preferences'], 'samples_per_case': 2, 'training': {
        **plan['divergence_preferences']['training'], 'steps': 2, 'gradient_accumulation': 4, 'preference_per_update': 2}}}
    rows = run.collect_arm(tiny_model(tokenizer), tokenizer, plan, policy(), EXECUTION, home)
    assert len(rows) == 2 * len(CASES) and read(home / 'rollouts-round3.json')['rollouts'] == len(rows)
    cases = [data.make_case('train', 'recipient', i) for i in range(4, 6)]
    monkey = pytest.MonkeyPatch()
    monkey.setattr(run, 'split_cases', lambda plan, split: cases)
    try:
        pairs, pair_rows = run.build_divergence_pairs(tokenizer, plan, policy(),
                                                      [r for c in cases for r in followup_rollouts(c)])
    finally:
        monkey.undo()
    assert len(pairs) == 4 and {p['turn'] for p in pairs} == {0, 1}
    experience_rows, replay_rows = sequences(tokenizer)
    prior, pinned = tmp_path / 'round2', {}
    for arm in ('update', 'addition'):
        trainable, receipt = trainer.train(tiny_model(tokenizer), arm, experience_rows, replay_rows, SPEC)
        pinned[arm] = trainer.checkpoint(prior / f'{arm}-checkpoint', trainable, receipt, {})['trainable_sha256']
    out = tmp_path / 'round3'
    out.mkdir()
    manifests = run.train_round2(lambda: tiny_model(tokenizer), plan, {'device': 'cpu'}, experience_rows, replay_rows,
                                 pair_rows, prior, pinned, out, section='divergence_preferences')
    assert manifests['update']['steps'] == manifests['addition']['steps'] == 2
    assert {arm: m['roots']['round1'] for arm, m in manifests.items()} == pinned


def test_goal_guided_repairs_replay_the_prefix_and_keep_only_verified_continuations(tmp_path, monkeypatch):
    from test_assistant_experience import draft_failure, documents

    difference, copy_case = data.make_case('train', 'difference', 7), data.make_case('train', 'copy', 8)
    cases = [difference, copy_case]
    monkeypatch.setattr(run, 'split_cases', lambda plan, split: cases)

    class Sampler:
        """Makes the draft-read mistake on the difference case unless the read was repaired."""
        def __init__(self, *args, **kwargs):
            self.batches = [1]

        def close(self):
            pass

        def respond(self, messages, tools):
            first = next(m['content'] for m in messages if m['role'] == 'user')
            case = next(c for c in cases if c['turns'][0]['user'] == first)
            position = sum(m['role'] == 'assistant' for m in messages)
            texts = draft_failure(case) if case is difference and position < 2 else reference_texts(case)
            return reply(texts[position])

    monkeypatch.setattr(run.rollout, 'Batcher', Sampler)
    plan = {**read(ROOT / run.PLAN)}
    plan['goal_guided_repairs'] = {**plan['goal_guided_repairs'], 'samples_per_case': 2}
    home = tmp_path / 'home'
    home.mkdir()
    natural, repaired = run.collect_repairs(None, None, plan, policy(), EXECUTION, home)
    summary = read(home / 'rollouts-round4.json')
    assert summary['rollouts'] == 4 and summary['passed'] == 2 and summary['repairable_failures'] == 2
    assert summary['repairs'] == 4 and summary['verified_repairs'] == 4
    pairs, trajectories = run.repair_data(plan, policy(), repaired)
    docs = documents(difference)
    assert len(pairs) == 1 and docs[(1, 'approved')] in pairs[0]['chosen'] and docs[(3, 'draft')] in pairs[0]['rejected']
    assert len(trajectories) == 1 and trajectories[0]['complete']
    repaired_index = len(pairs[0]['messages'])
    assert trajectories[0]['trainable'][repaired_index] and trajectories[0]['messages'][repaired_index]['content'] == pairs[0]['chosen']
    forged = [dict(repaired[0], result={**repaired[0]['result'], 'score': {**repaired[0]['result']['score'], 'passed': False}})]
    with pytest.raises(ValueError, match='re-verify'):
        run.repair_data(plan, policy(), forged)


def study_plan(plan):
    study = {'members': 3, 'integration_samples': 1, 'seed': 11,
             'counts': {'trajectories': 3, 'repaired_trajectories': 1, 'replay': 2, 'decision_pairs': 1,
                        'divergence_pairs': 1, 'repair_pairs': 1},
             'arms': {'update': {'type': 'update'}, 'small': {'type': 'addition'},
                      'large': {'type': 'addition', 'spec': {'rank': 8, 'alpha': 16,
                                                             'projections': ['self_attn.q_proj', 'self_attn.v_proj',
                                                                             'mlp.up_proj']}},
                      **{f'member-{k}': {'type': 'addition', 'member': k, 'spec': {'seed': 100 + k}} for k in range(3)}}}
    return {**plan, 'methodology_study': study,
            'goal_guided_repairs': {**plan['goal_guided_repairs'],
                                    'training': {**plan['goal_guided_repairs']['training'], 'steps': 2,
                                                 'preference_per_update': 1}}}


def sequence(tokenizer, seed):
    import random
    rng = random.Random(seed)
    ids = [rng.randrange(3, len(tokenizer.runtime)) for _ in range(12)]
    return {'input_ids': ids, 'labels': [-100] * 6 + ids[6:], 'sha256': str(seed)}


def test_study_data_tags_every_sequence_with_its_case_and_checks_declared_counts(setup, monkeypatch):
    tokenizer, plan, home = setup
    home.mkdir()
    plan = study_plan(plan)
    monkeypatch.setattr(run, 'load_collection', lambda *a: ([sequence(tokenizer, i) for i in range(3)],
                                                            [sequence(tokenizer, 9), sequence(tokenizer, 10)]))
    monkeypatch.setattr(run, 'read_rows', lambda path: [{'case_id': c['id']} for c in CASES])
    monkeypatch.setattr(run, 'sha256', lambda path: 'pinned')
    pair = {'case_id': CASES[0]['id'], 'messages': [{'role': 'user', 'content': 'x'}], 'chosen': 'a', 'rejected': 'b'}
    encoded = {'chosen': sequence(tokenizer, 20), 'rejected': sequence(tokenizer, 21), 'sha256': 'p'}
    monkeypatch.setattr(run, 'build_pairs', lambda *a: ([pair], [encoded]))
    monkeypatch.setattr(run, 'build_divergence_pairs', lambda *a: ([{**pair, 'case_id': CASES[1]['id']}], [encoded]))
    trajectory = {'case_id': CASES[2]['id'], 'messages': [{'role': 'user', 'content': 'hello'},
                                                          {'role': 'assistant', 'content': 'Okay.'}],
                  'trainable': [False, True]}
    monkeypatch.setattr(run, 'repair_data', lambda plan, policy, rows: ([{**pair, 'case_id': CASES[2]['id']}], [trajectory]))
    inventory = {'collection': {'files': {}}, 'study': {'files': {'rollouts-round3.jsonl.gz': 'pinned'}}}
    assert 'collection' not in EXECUTION and 'study' not in EXECUTION
    experience, replay, pairs = run.study_data(tokenizer, plan, policy(), EXECUTION, home, inventory)
    assert [case for case, _ in experience] == [c['id'] for c in CASES] + [CASES[2]['id']]
    assert [case for case, _ in pairs] == [c['id'] for c in CASES] and len(replay) == 2
    with pytest.raises(ValueError, match='declaration'):
        run.study_data(tokenizer, {**plan, 'methodology_study': {**plan['methodology_study'],
                                                                 'counts': {'trajectories': 99}}}, policy(), EXECUTION,
                       home.parent / 'other', inventory)


def test_study_trains_every_arm_on_its_slice_and_evaluates_the_committee(setup, monkeypatch):
    tokenizer, plan, home = setup
    home.mkdir()
    plan = study_plan(plan)
    monkeypatch.setattr(run.data, 'cases', lambda split: CASES[:2])
    slices = {}
    for index in range(100):
        slices.setdefault(run.committee_member(f'case-{index}', 3), f'case-{index}')
    owners = [slices[k] for k in range(3)]
    experience = [(owners[i % 3], sequence(tokenizer, i)) for i in range(9)]
    pairs = [(owners[i % 3], {'chosen': sequence(tokenizer, 40 + i), 'rejected': sequence(tokenizer, 50 + i),
                              'sha256': str(i)}) for i in range(6)]
    replay = [sequence(tokenizer, 90), sequence(tokenizer, 91)]
    manifests = run.study_train(lambda: tiny_model(tokenizer), plan, EXECUTION, experience, replay, pairs, home)
    assert set(manifests) == {'update', 'small', 'large', 'member-0', 'member-1', 'member-2'}
    assert manifests['small']['sequences'] == 9 and manifests['small']['pairs'] == 6
    assert [manifests[f'member-{k}']['sequences'] for k in range(3)] == [3, 3, 3]
    assert [manifests[f'member-{k}']['pairs'] for k in range(3)] == [2, 2, 2]
    assert manifests['large']['trainable_parameters'] > manifests['small']['trainable_parameters']
    report = run.study_evaluate(lambda: tiny_model(tokenizer), tokenizer, plan, policy(), EXECUTION, home)
    assert set(report) == {'parent', 'update', 'small', 'large', 'member-0', 'member-1', 'member-2', 'committee'}
    committee = report['committee']['integration']
    assert committee['cases'] == 3 and committee['lost_parent_successes'] == [] and committee['versus_update'] == 0
    assert report['committee']['development']['cases'] == 2
    saved = read(home / 'study.json')
    assert all(len(values) == 2 for values in saved['systems']['committee']['integration'].values())


def test_accelerator_phases_never_open_evaluation_goals():
    plan = read(ROOT / run.PLAN)
    assert len(run.split_cases(plan, 'train')) == 256 and len(run.split_cases(plan, 'integration')) == 64
    assert [len(run.split_cases(plan, split)) for split in data.GROWTH] == [256, 256]
    for split in ('development', 'confirmation', 'confirmation3'):
        with pytest.raises(ValueError, match='may not access'):
            run.split_cases(plan, split)


def test_growth_splits_are_frozen_training_cases_disjoint_from_every_other_split():
    from neuroshard.evolution.modular_reference_execution import identity

    growth = read(ROOT / run.GROWTH_PLAN)
    manifest = read(ROOT / growth['data'])
    assert list(manifest['splits']) == list(data.GROWTH) == growth['collection']['splits']
    assert sorted(manifest['disjoint_from']) == sorted(s for s in data.SPLITS if s not in data.GROWTH)
    for split, frozen in manifest['splits'].items():
        cases = data.cases(split)
        assert identity(cases) == frozen['sha256'] and [c['id'] for c in cases] == frozen['case_ids']
        assert all(sum(c['family'] == f for c in cases) == 32 for f in data.FAMILIES)
        projects = {t['expected']['project'] for c in cases for t in c['turns']}
        for other in manifest['disjoint_from'] + [s for s in data.GROWTH if s != split]:
            earlier = data.cases(other)
            assert not {c['id'] for c in cases} & {c['id'] for c in earlier}
            assert not projects & {t['expected']['project'] for c in earlier for t in c['turns']}
        # Training grammar: the second turn keeps the unchanged fields named, unlike held-out splits.
        assert any('Keep the date, total and cited source unchanged.' in c['turns'][1]['user']
                   for c in cases if c['family'] == 'recipient')


def test_growth_profiles_run_collection_hosts_with_their_declared_split():
    import importlib.util

    spec = importlib.util.spec_from_file_location('growth_cloud', ROOT / 'scripts/modular_reference_cloud.py')
    cloud = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cloud)
    growth = read(ROOT / run.GROWTH_PLAN)
    assert set(growth['collection']['profiles'].values()) == set(data.GROWTH)
    for profile in growth['collection']['profiles']:
        assert profile in cloud.GPU_PROFILES and profile not in cloud.UPLOAD_PROFILES
        assert cloud.remote_command(profile)[-2:] == ['--profile', profile]
        assert cloud.GRANITE_PROFILES[profile] == cloud.GRANITE_PROFILES[run.PROFILE]
        resources = cloud.resources(profile)
        assert resources['purpose'] == 'assistant-experience-growth' and 'upload' not in resources
    assert len(set(growth['collection']['seeds'].values())) == len(data.GROWTH)
    execution = read(ROOT / run.EXECUTION)
    assert 'growth' not in execution
    assert {run.GROWTH_PLAN, growth['data']} <= set(execution['contracts'])


def test_growth_collection_reverifies_without_replay_and_the_pool_checks_pinned_counts(setup, monkeypatch):
    from neuroshard.evolution.modular_reference_execution import sha256

    tokenizer, plan, home = setup
    home.mkdir()
    grown = [data.make_case('train2', 'copy', 0), data.make_case('train2', 'latest', 1)]

    class Growth(ScriptedBatcher):
        def respond(self, messages, tools):
            first = next(m['content'] for m in messages if m['role'] == 'user')
            case = next(c for c in grown if c['turns'][0]['user'] == first)
            return reply(reference_texts(case)[sum(m['role'] == 'assistant' for m in messages)])

    monkeypatch.setattr(run.rollout, 'Batcher', Growth)
    monkeypatch.setattr(run, 'split_cases', lambda plan, split: grown if split == 'train2' else CASES)
    rows, report = run.collect(tiny_model(tokenizer), tokenizer, plan, policy(), EXECUTION, home, split='train2')
    assert report['summary']['cases'] == len(grown) and len(rows) == len(grown) and not report['coached_cases']
    pinned = {name: sha256(home / name) for name in run.GROWTH_FILES}
    again = home / 'again'
    again.mkdir()
    reloaded, replay = run.load_collection(home, pinned, tokenizer, plan, policy(), EXECUTION, again, split='train2')
    assert reloaded == rows and replay == []
    assert read(again / 'collection-train2.json')['trajectories'] == len(rows)

    base = ([('old', sequence(tokenizer, 1))], [sequence(tokenizer, 2)], [])
    monkeypatch.setattr(run, 'study_data', lambda *a: (list(base[0]), base[1], base[2]))
    monkeypatch.setattr(run, 'load_collection', lambda *a, **k: ([sequence(tokenizer, 3)], []))
    monkeypatch.setattr(run, 'read_rows', lambda path: [{'case_id': 'grown'}])
    inventory = {'growth': {'collections': {s: {} for s in data.GROWTH}, 'counts': {s: 1 for s in data.GROWTH}}}
    experience, replay, pairs = run.growth_data(tokenizer, plan, policy(), EXECUTION, home, inventory)
    assert [case for case, _ in experience] == ['old', 'grown', 'grown'] and len(replay) == 1
    with pytest.raises(ValueError, match='pinned counts'):
        run.growth_data(tokenizer, plan, policy(), EXECUTION, home,
                        {'growth': {**inventory['growth'], 'counts': {s: 2 for s in data.GROWTH}}})


def test_one_pass_first_phase_draws_each_experience_sequence_once(setup):
    tokenizer, plan, home = setup
    home.mkdir()
    plan = study_plan(plan)
    growth = {**plan['methodology_study'], 'first_phase_steps': 'one-pass',
              'arms': {'small': {'type': 'addition'}, 'member-0': {'type': 'addition', 'member': 0}}}
    slices = {}
    for index in range(100):
        slices.setdefault(run.committee_member(f'case-{index}', 3), f'case-{index}')
    experience = [(slices[i % 3], sequence(tokenizer, i)) for i in range(13)]
    replay = [sequence(tokenizer, 90)]
    manifests = run.study_train(lambda: tiny_model(tokenizer), plan, EXECUTION, experience, replay, [], home, study=growth)
    per_update = SPEC['gradient_accumulation'] - SPEC['gradient_accumulation'] // (SPEC['experience_per_replay'] + 1)
    assert manifests['small']['first_phase_steps'] == -(-13 // per_update)
    assert manifests['member-0']['first_phase_steps'] == -(-5 // per_update) and manifests['member-0']['sequences'] == 5
    assert run.one_pass_steps(read(ROOT / run.PLAN)['training'], 740) == 124


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
