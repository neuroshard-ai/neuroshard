"""A retained checkpoint must remain usable by the next training allocation."""
import copy
import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[2]/'scripts/prepare_contract_learning.py'
spec = importlib.util.spec_from_file_location('contract_learning_preparation', path)
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


def test_publication_receipts_become_fetchable_without_changing_provenance():
    key = 'a'*64
    checkpoint = {'tensors': {'weight': {'sha256': key, 'bytes': 128}}}
    receipts = {key: {'sha256': key, 'bytes': 128, 'url': 'https://example.org/'+key,
                      'readback_verified': True}}
    before = copy.deepcopy(receipts)
    fetched = preparation.training_objects(checkpoint, receipts)
    assert fetched[key] == {**receipts[key], 'folder': 'objects'}
    assert receipts == before
    assert preparation.training_objects(checkpoint, fetched) == fetched


@pytest.mark.parametrize('field,value', [('sha256', 'b'*64), ('bytes', 129), ('folder', 'interpreter')])
def test_wrong_checkpoint_inventory_is_rejected(field, value):
    key = 'a'*64
    checkpoint = {'tensors': {'weight': {'sha256': key, 'bytes': 128}}}
    receipts = {key: {'sha256': key, 'bytes': 128, field: value}}
    with pytest.raises(ValueError, match='exact owned tensor'):
        preparation.training_objects(checkpoint, receipts)
