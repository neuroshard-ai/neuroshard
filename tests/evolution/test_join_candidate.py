import importlib.util
import json
from pathlib import Path

import pytest

from neuroshard.dataflow.store import canonical
from neuroshard.evolution.app import code_hash
from neuroshard.evolution.objects import digest


def module():
    spec = importlib.util.spec_from_file_location('join_candidate', Path(__file__).resolve().parents[2]/'scripts/join_funded_candidate.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def agreed(tmp_path):
    genesis = {'chain_id':'candidate-bootstrap-test', 'initial_height':'1',
               'app_state':{'manifest':{'code_hash':code_hash(), 'auditing':{}, 'lifecycle':{}}}}
    path = tmp_path/'agreed.json'
    path.write_bytes(canonical(genesis))
    return path, digest(canonical(genesis))


def test_join_refuses_wrong_genesis_before_creating_keys(agreed, tmp_path):
    path, _ = agreed
    home = tmp_path/'node'
    with pytest.raises(ValueError, match='Genesis hash'):
        module().prepare(home, path, 'a'*64, Path('must-not-run'), ['peer'], 55150)
    assert not home.exists()


def test_join_refuses_other_source_even_with_matching_genesis_checksum(agreed, tmp_path):
    path, _ = agreed
    genesis = json.loads(path.read_bytes())
    genesis['app_state']['manifest']['code_hash'] = 'f'*64
    path.write_bytes(canonical(genesis))
    home = tmp_path/'node'
    with pytest.raises(ValueError, match='exact source'):
        module().prepare(home, path, digest(canonical(genesis)), Path('must-not-run'), ['peer'], 55150)
    assert not home.exists()


def test_join_never_overwrites_an_unrecognized_home(agreed, tmp_path):
    path, sha = agreed
    home = tmp_path/'node'
    home.mkdir()
    sentinel = home/'private-key'
    sentinel.write_bytes(b'preserve existing state')
    with pytest.raises(ValueError, match='will not be overwritten'):
        module().prepare(home, path, sha, Path('must-not-run'), ['peer'], 55150)
    assert sentinel.read_bytes() == b'preserve existing state'
    assert list(home.iterdir()) == [sentinel]


def test_resume_preserves_keys_database_and_configuration(agreed, tmp_path):
    path, sha = agreed
    home = tmp_path/'node'
    (home/'config').mkdir(parents=True)
    (home/'config/genesis.json').write_bytes(path.read_bytes())
    identity = {'genesis_sha256':sha, 'chain_id':'candidate-bootstrap-test', 'base_port':55150}
    (home/'candidate-bootstrap.json').write_bytes(canonical(identity))
    (home/'evolution.sqlite').write_bytes(b'retained database marker')
    before = {p.relative_to(home):p.read_bytes() for p in home.rglob('*') if p.is_file()}
    assert module().prepare(home, path, sha, Path('must-not-run'), ['peer'], 55150) == identity
    assert before == {p.relative_to(home):p.read_bytes() for p in home.rglob('*') if p.is_file()}
