import copy
import json

import pytest
import torch
from transformers import LlamaConfig

from neuroshard.evolution import reference_data as data
from neuroshard.evolution.sharded import feature_bank as bank
from neuroshard.evolution.sharded.model import batch_tensors


def fixture(tmp_path, different_reference=False):
    config = LlamaConfig(vocab_size=47, hidden_size=24, intermediate_size=48,
                         num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
                         tie_word_embeddings=True)
    rows = [{'id': str(index), 'input_ids': [1, 3, 9, 2], 'labels': [-100, -100, 9, 2],
             'targets': 2, 'loss_weight': .5, 'distill': True} for index in range(3)]
    packets = []
    for offset in range(0, len(rows), 2):
        ids, labels, mask, weights = batch_tensors(rows[offset:offset + 2], 'cpu')
        prefix = torch.randn(*ids.shape, 24)
        packets.append({'prefix': prefix, 'reference': prefix + 1 if different_reference else prefix,
                        'ids': ids, 'labels': labels, 'mask': mask, 'weights': weights})
    binding = {'parent': 'parent', 'prepared': 'inputs', 'cut_layer': 4}
    writer = bank.Writer(tmp_path / 'bank', binding, config, 2)
    writer.batch(rows, packets)
    root = writer.finish(1)
    reader = bank.Reader(tmp_path / 'bank', root, binding, config, 2, 1)
    return reader, root, binding, config, rows, packets


@pytest.mark.parametrize('different_reference', [False, True])
def test_durable_bank_preserves_exact_microbatches_and_deduplicates_only_identical_reference(tmp_path, different_reference):
    reader, root, binding, config, rows, original = fixture(tmp_path, different_reference)
    for left, right in zip(original, reader.batch(0, rows, 'cpu')):
        assert all(torch.equal(value, right[key]) for key, value in left.items())
        assert (right['prefix'].data_ptr() == right['reference'].data_ptr()) == (not different_reference)
    changed = copy.deepcopy(rows)
    changed[0]['labels'][-1] = 3
    with pytest.raises(ValueError, match='ordered training records'):
        reader.batch(0, changed, 'cpu')
    with pytest.raises(ValueError, match='production commitment'):
        bank.Reader(reader.home, root, {**binding, 'parent': 'other'}, config, 2, 1)


def test_altered_feature_bytes_and_rehashed_index_cannot_replace_committed_bank(tmp_path):
    reader, root, binding, config, rows, _ = fixture(tmp_path)
    path = reader.home / reader.manifest['batches'][0]['files'][0]['file']
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match='tensor bytes'):
        reader.batch(0, rows, 'cpu')
    altered = copy.deepcopy(reader.manifest)
    altered['batches'][0]['files'][0]['sha256'] = data.sha256(path)
    (reader.home / 'index.json').write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='production commitment'):
        bank.Reader(reader.home, root, binding, config, 2, 1)
