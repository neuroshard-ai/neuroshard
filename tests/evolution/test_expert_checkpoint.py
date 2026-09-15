"""The native codec must preserve real numerical checkpoints and optimizer ages."""
from copy import deepcopy
import subprocess
import sys

import pytest
from transformers import LlamaConfig

from neuroshard.evolution import expert_checkpoint as codec, reference_data as data
from neuroshard.evolution.sharded import cohort_state, portable
from test_expert_cohort_state import prepare, update, RECIPE


def examples(home):
    parent, objects, shard, optimizer = prepare(home)
    # The shared tail-only fixture omits parent process manifests. Supply opaque
    # references here; actual published parent metadata is checked separately.
    parent['shards'] = [data.identity({'fixture_parent_owner': i}) for i in range(3)]
    zero = cohort_state.commit_tail(home / 'expert', shard, optimizer, parent, objects,
                                    'c' * 64, 0, RECIPE, 5)
    update(shard, optimizer, 0)
    one = cohort_state.commit_tail(home / 'expert', shard, optimizer, parent, objects,
                                   'c' * 64, 1, RECIPE, 5)
    return parent, zero, one


def test_actual_tensor_checkpoint_roundtrip_and_frozen_age_preservation(tmp_path):
    parent, zero, one = examples(tmp_path)
    a, b = codec.pack(parent, zero), codec.pack(parent, one)
    assert codec.shapes(parent['config']) == portable.shapes(LlamaConfig(**parent['config']))
    assert codec.unpack(parent, a) == zero and codec.unpack(parent, b) == one
    assert len(b['tensors']) < len(one['tensors'])
    assert codec.transition(parent, a, b) == 1
    assert {spec['optimizer_step'] for name, spec in one['tensors'].items() if name not in b['tensors']} == {7}
    assert {spec['optimizer_step'] for spec in b['tensors'].values()} == {1}
    for kind in ('age', 'frozen', 'missing', 'shape', 'shard'):
        wrong = deepcopy(b)
        name = next(iter(wrong['tensors']))
        if kind == 'age':
            wrong['tensors'][name]['optimizer_step'] = 0
        elif kind == 'frozen':
            wrong['tensors']['model.norm.weight'] = one['tensors']['model.norm.weight']
        elif kind == 'missing':
            del wrong['tensors'][name]
        elif kind == 'shape':
            wrong['tensors'][name]['shape'] = [1]
        else:
            wrong['checkpoint'] = 'f' * 64
        with pytest.raises(ValueError):
            codec.unpack(parent, wrong)


def test_renaming_job_cannot_create_another_work_identity(tmp_path):
    parent, zero, one = examples(tmp_path)
    a = codec.pack(parent, zero)
    alias = deepcopy(a)
    alias['job'] = 'd' * 64
    common = codec.reconstruct(parent, alias)
    alias['checkpoint'], alias['state_root'] = data.identity(common), common['state_root']
    assert codec.work_identity(parent, a, 'a' * 64, 'b' * 64) == codec.work_identity(parent, alias, 'a' * 64, 'b' * 64)
    assert codec.work_identity(parent, a, 'a' * 64, 'b' * 64) != codec.work_identity(parent, codec.pack(parent, one), 'a' * 64, 'b' * 64)
    with pytest.raises(ValueError):
        codec.transition(parent, alias, codec.pack(parent, one))


def test_consensus_codec_imports_no_neural_runtime():
    subprocess.run([sys.executable, '-c',
        'import sys; import neuroshard.evolution.expert_checkpoint; '
        'assert not ({"torch", "transformers", "safetensors"} & set(sys.modules))'], check=True)
