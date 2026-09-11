import copy
import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

from neuroshard.dataflow.store import canonical
from neuroshard.evolution.objects import digest
from test_cohort_review import proposal, reviewer


spec = importlib.util.spec_from_file_location('native_cohort_state',
    Path(__file__).resolve().parents[2]/'scripts/native_cohort_state.py')
history = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = history
spec.loader.exec_module(history)


def node(tmp_path, policy):
    home = tmp_path/'node'
    (home/'config').mkdir(parents=True)
    manifest = {'lifecycle':{'tokenizer_root':policy['tokenizer_root'],
                            'vocabulary':64,'steps_per_cohort':4}}
    genesis = canonical({'chain_id':'native-history-test','app_state':{'manifest':manifest}})
    (home/'config/genesis.json').write_bytes(genesis)
    state = {'chain_id':'native-history-test','manifest':manifest,'height':9,
             'data_root':'c'*64,'lifecycle':{'active':None,'cursors':{},
                                           'seen_documents':{},'seen_batches':{}}}
    db = sqlite3.connect(home/'evolution.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE state (id INTEGER PRIMARY KEY,value BLOB)')
    save(db,state)
    return home,digest(genesis),state,db


def save(db,state):
    with db:
        db.execute('INSERT OR REPLACE INTO state VALUES (1,?)',(canonical(state),))


def test_committed_wal_history_is_complete_and_stable_during_pending_writes(tmp_path,proposal):
    _,prepared,policy,_ = proposal
    home,pin,state,db = node(tmp_path,policy)
    document = prepared['metadata'][prepared['data_root']]['documents'][0]
    state['lifecycle']['seen_documents'][document['id']] = 'earlier admission'
    state['lifecycle']['seen_batches'][document['batches'][0]] = 'earlier admission'
    save(db,state)
    committed = history.read(home,pin)
    assert committed.report()['consumed_documents'] == 1
    assert committed.report()['consumed_batches'] == 1
    # A writer has uncommitted changes in WAL. Readers must still see exactly
    # the previously committed version without forcing a checkpoint or commit.
    state['height'] += 1
    state['data_root'] = prepared['data_root']
    db.execute('INSERT OR REPLACE INTO state VALUES (1,?)',(canonical(state),))
    assert history.read(home,pin).report() == committed.report()
    db.commit()
    with pytest.raises(ValueError,match='history changed'):
        committed.ensure_current()
    assert history.read(home,pin).state['data_root'] == prepared['data_root']
    db.close()


def test_read_refuses_wrong_network_missing_or_oversized_state_without_initializing(tmp_path,proposal,monkeypatch):
    _,_,policy,_ = proposal
    home,pin,state,db = node(tmp_path,policy)
    with pytest.raises(ValueError,match='independently pinned'):
        history.read(home,'f'*64)
    state['chain_id'] = 'unrelated-chain'
    save(db,state)
    with pytest.raises(ValueError,match='differs from the pinned genesis'):
        history.read(home,pin)
    monkeypatch.setattr(history,'MAX_STATE_BYTES',1)
    with pytest.raises(ValueError,match='oversized'):
        history.read(home,pin)
    db.close()
    database = home/'evolution.sqlite'
    database.unlink()
    with pytest.raises(sqlite3.OperationalError):
        history.read(home,pin)
    assert not database.exists()


def test_snapshot_detects_history_change_or_rollback_but_allows_ordinary_blocks(tmp_path,proposal):
    _,_,policy,_ = proposal
    home,pin,state,db = node(tmp_path,policy)
    before = history.read(home,pin)
    state['height'] += 1
    save(db,state)
    assert before.ensure_current().state['height'] == state['height']
    state['lifecycle']['cursors']['a'*64] = 96
    save(db,state)
    with pytest.raises(ValueError,match='history changed'):
        before.ensure_current()
    state['lifecycle']['cursors'] = {}
    state['height'] = before.state['height']-1
    save(db,state)
    with pytest.raises(ValueError,match='history changed'):
        before.ensure_current()
    db.close()


def test_native_review_rejects_stale_parent_cursor_history_and_wrong_step_budget(tmp_path,proposal):
    store,prepared,policy,upstream = proposal
    home,pin,state,db = node(tmp_path,policy)
    snap = history.read(home,pin)
    report = reviewer.review(store,prepared,policy,upstream,native_state=snap.state)
    assert report['native_history_checked'] and report['upstream_documents_checked'] == 68
    cohort = prepared['metadata'][prepared['data_root']]
    for change,reason in (
        (lambda s:s.update(data_root='a'*64),'Stale dataset parent'),
        (lambda s:s['lifecycle']['cursors'].update({cohort['windows'][0]['source']:1}),'nonconsecutive'),
        (lambda s:s['lifecycle']['seen_batches'].update({cohort['documents'][0]['batches'][0]:True}),'repeat an existing token batch'),
        (lambda s:s['manifest']['lifecycle'].update(steps_per_cohort=64),'Insufficient unique fresh'),
    ):
        changed = copy.deepcopy(snap.state)
        change(changed)
        with pytest.raises(ValueError,match=reason):
            reviewer.review(store,prepared,policy,native_state=changed)
    db.close()
