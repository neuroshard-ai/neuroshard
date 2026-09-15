"""Compare partition-at-a-time replay with the actual five-process producer."""
import copy
import gc
import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import LlamaConfig

from neuroshard.evolution import cohort_experiment, reference_data as data
from neuroshard.evolution.sharded import portable, prefix_audit
from neuroshard.evolution.sharded.model import Partition
from test_expert_cohort_job import test_actual_training_job_serves_and_survives_learner_exit as train_fixture


def test_complete_partition_replay_matches_and_rejects_rehashed_false_features(tmp_path):
    train_fixture(tmp_path, False)
    read = lambda path: json.loads(path.read_bytes())
    plan, prepared = read(tmp_path/'plan.json'), read(tmp_path/'prepared.json')
    parent = read(tmp_path/'parent.json')
    original = read(tmp_path/'rank-4/features/index.json')
    config = LlamaConfig(**parent['config'])
    config._attn_implementation = 'sdpa'
    records = cohort_experiment.rows(prepared, tmp_path/'inputs', 'train')
    torch.set_num_threads(1)

    def check(target, directory):
        reports, incoming = [], None
        for rank in range(3):
            shard = Partition(config, parent['boundaries'], rank)
            objects = directory/f'objects-{rank}'
            objects.mkdir(parents=True)
            for name, _ in shard.named_owned_parameters():
                digest = parent['tensors'][name]['sha256']
                path = portable.tensor_path(objects, digest)
                if not path.exists(): path.hardlink_to(portable.tensor_path(tmp_path/'objects', digest))
            # No complete model is available in any stage's object directory.
            assert len(list(objects.iterdir())) < len(list((tmp_path/'objects').iterdir()))
            home = directory/f'rank-{rank}'
            report = prefix_audit.replay_stage(shard, parent, objects, records, prepared['batches'],
                original['binding'], plan['split'], plan['microbatch'], target, home, incoming, 60)
            reports.append(report)
            incoming = (home/'features', report)
            del shard
            gc.collect()
        return reports

    expected = data.identity(original)
    reports = check(expected, tmp_path/'good')
    result = prefix_audit.complete(reports, expected)
    assert result['passed'] and result['microbatches_per_stage'] == 112
    assert read(tmp_path/'good/rank-2/features/index.json') == original
    with pytest.raises(ValueError, match='all three'):
        prefix_audit.complete(reports[1:], expected)
    broken = copy.deepcopy(reports)
    broken[1]['input_root'] = 'a'*64
    with pytest.raises(ValueError, match='disconnected'):
        prefix_audit.complete(broken, expected)

    forged = copy.deepcopy(original)
    spec = forged['batches'][0]['files'][0]
    payload = load_file(tmp_path/'rank-4/features'/spec['file'])
    payload['prefix'][0, 0, 0] += .25
    altered = tmp_path/'forged.safetensors'
    save_file(payload, altered)
    spec['sha256'], spec['bytes'] = data.sha256(altered), altered.stat().st_size
    target = data.identity(forged)
    assert target != expected
    with pytest.raises(ValueError, match='claimed feature bank'):
        check(target, tmp_path/'bad')
    failed = read(tmp_path/'bad/rank-2/result.json')
    assert failed['valid'] is False and failed['output_root'] == expected
