import importlib.util
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from neuroshard.evolution.batches import from_windows
from neuroshard.evolution.pipeline import LocalEndpoint, Pipeline
from neuroshard.evolution.worker import Worker

SPEC = importlib.util.spec_from_file_location('learning_driver',
    Path(__file__).resolve().parents[2]/'scripts/run_learning_milestone.py')
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def test_completed_step_is_recovered_when_outer_receipt_was_never_written(seed, tmp_path):
    store, root, _ = seed
    homes = [tmp_path/f'worker{i}' for i in range(2)]
    windows = [store.put_json({'tokens': row, 'labels': [-100, -100, *row[2:]]})
               for row in [[1, 12, 9, 18, 6, 2], [1, 13, 7, 17, 5, 2]]]
    endpoints = [LocalEndpoint(Worker(home, store, 6000)) for home in homes]
    pipe = Pipeline(store, root, endpoints, [6000]*2, 'receipt-recovery', journal=tmp_path/'coordinator.json')
    accepted = pipe.train(from_windows(store, windows))
    pipe.close()
    before = [e.worker.db.execute('SELECT COUNT(*) FROM operations').fetchone()[0] for e in endpoints]
    for e in endpoints:
        e.worker.db.close()
    # Simulate a coordinator crash after its accepted journal, before run.json.
    state = {'session': 'receipt-recovery', 'steps': []}
    workers = SimpleNamespace(home=tmp_path, homes=homes)
    selection = {'baseline': root, 'training_batches': [windows], 'tokenizer_root': None}
    driver.reconcile_steps(store, workers, selection, state, 1.)
    assert state['steps'][0]['record_root'] == accepted['record_root']
    driver.reconcile_steps(store, workers, selection, state, 1.)
    assert len(state['steps']) == 1
    endpoints = [LocalEndpoint(Worker(home, store, 6000)) for home in homes]
    recovered = Pipeline(store, root, endpoints, [6000]*2, 'receipt-recovery', journal=tmp_path/'coordinator.json')
    assert recovered.step == 1 and recovered.model_root == accepted['model_root']
    assert before == [e.worker.db.execute('SELECT COUNT(*) FROM operations').fetchone()[0] for e in endpoints]
    recovered.close()
    for e in endpoints:
        e.worker.db.close()


def test_document_scores_weight_targets_and_resume_only_missing_windows(seed):
    store, _, _ = seed
    windows = [store.put_json({'tokens': [1, 2, 3, 4], 'labels': labels, 'tokenizer_root': 'a'*64})
               for labels in [[-100, -100, -100, 4], [-100, 2, 3, 4]]]
    measurements, calls = {}, []
    def evaluate(batch):
        calls.append(batch)
        return {'loss_hex': float(1 if len(calls) == 1 else 3).hex()}
    pipe = SimpleNamespace(evaluate=evaluate, model={'tokenizer_root': 'a'*64})
    budget = SimpleNamespace(check=lambda: None)
    def interrupted():
        raise ConnectionError('Stop after first durable measurement')
    docs = [{'windows': windows}]
    with pytest.raises(ConnectionError):
        driver.score(pipe, store, docs, measurements, budget, interrupted)
    assert len(measurements) == len(calls) == 1
    values = driver.score(pipe, store, docs, measurements, budget, lambda: None)
    assert values == [2.5] and len(calls) == 2
    assert driver.score(pipe, store, docs, measurements, budget, lambda: None) == values
    assert len(calls) == 2


def test_restart_cannot_reset_elapsed_budget(tmp_path):
    state = {'deadline': 100., 'started': 1.}
    plan = {'budget': {'disk_gib': 256}}
    driver.save(tmp_path/'run.json', state)
    restarted = json.loads((tmp_path/'run.json').read_bytes())
    with pytest.raises(ValueError, match='original wall-clock budget'):
        driver.Budget(tmp_path, restarted, plan, clock=lambda: 101.).check()


def test_selection_excludes_incomplete_documents_and_cross_role_token_duplicates(seed):
    store, _, _ = seed
    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE documents (id TEXT, source TEXT, row INTEGER, object TEXT, role TEXT)')
    prepared = {}
    for role_index, (role, quota) in enumerate((('test', 64), ('retention', 64), ('fresh', 64), ('train', 256))):
        for i in range(quota+2):
            identity = f'{10000*role_index+i:064x}'
            # Every role's second row has identical numerical input/targets.
            # The incomplete first row must never fill a quota.
            token = 7 if i == 1 else 10000*(role_index+1)+i
            prepared[identity] = {'truncated': i == 0,
                                 'windows': [{'tokens': [1, 2, token], 'labels': [-100, -100, token]}]}
            root = store.put_json({'messages': [{'role': 'user', 'content': identity}]})
            db.execute('INSERT INTO documents VALUES (?,?,?,?,?)', (identity, str(role_index), i, root, role))
    codec = SimpleNamespace(response_windows=lambda messages, *_: prepared[messages[0]['content']])
    corpus = SimpleNamespace(db=db, store=store, codec=codec)
    plan = {'data': {'context_tokens': 64, 'response_tokens': 64, 'max_windows': 4}}
    selected, rejected = driver.select_documents(corpus, plan)
    assert {role: len(items) for role, items in selected.items()} == {
        'test': 64, 'retention': 64, 'fresh': 64, 'train': 256}
    assert rejected == {'incomplete': 4, 'repeated_tokens': 3}
    assert selected['test'][0]['row'] == 1
    assert all(selected[role][0]['row'] == 2 for role in ('retention', 'fresh', 'train'))
    # Removing one eligible row cannot silently shorten the training pool.
    db.execute('DELETE FROM documents WHERE role="train" AND row=257')
    with pytest.raises(ValueError, match='255/256; do not lower n'):
        driver.select_documents(corpus, plan)
    db.close()
