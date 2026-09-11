import importlib.util
import fcntl
from pathlib import Path

import pytest

from neuroshard.evolution import cohorts
from neuroshard.evolution.data import TextCorpus
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.text import TextCodec
from neuroshard.evolution.verification import Metadata
from test_text import tokenizer, conversation, source


spec = importlib.util.spec_from_file_location('prepare_native_cohort',Path(__file__).resolve().parents[2]/'scripts/prepare_native_cohort.py')
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


def test_prepare_retokenizes_complete_documents_and_mirrors_all_evidence(tmp_path):
    store = Objects(tmp_path/'objects')
    codec = TextCodec(tokenizer(),store)
    corpus = TextCorpus(tmp_path/'corpus',store,codec)
    sources = []
    for group, role in enumerate(('train','retention','fresh')):
        key = corpus.register(source(role))
        sources.append(key)
        corpus.collect(key,96,rows=[conversation(1000*group+i,answer=8) for i in range(96)])
    prepared = preparation.prepare(corpus,'c'*64,{},sources,4)
    assert prepared['status'] == 'prepared_for_review'
    assert prepared['report']['selected'] == {'train':4,'retention':32,'fresh':32}
    s = {'data_root':'c'*64,'manifest':{'lifecycle':{'tokenizer_root':codec.root,'vocabulary':64,'steps_per_cohort':4}},
         'lifecycle':{'cursors':{},'seen_documents':{},'seen_batches':{},'active':None}}
    value, roles = cohorts.validate(cohorts.metadata(prepared['metadata']),prepared['data_root'],s)
    assert all(document['omitted_targets'] == 0 for document in value['documents'])
    replica = Objects(tmp_path/'replica')
    report = preparation.publish(store,prepared,replica)
    assert report['read_back_verified']
    assert replica.json(codec.root) == codec.profile
    assert all(replica.json(document['object'])['id'] == document['id'] for document in value['documents'])
    # Native cursors allow an interrupted exporter to derive the same proposal
    # from durable collected rows without re-downloading or advancing them.
    assert preparation.prepare(corpus,'c'*64,{},sources,4)['data_root'] == prepared['data_root']
    advanced = {key:96 for key in sources}
    assert preparation.prepare(corpus,prepared['data_root'],advanced,sources,4)['status'] == 'needs_more_data'
    corpus.db.close()


def test_concurrent_collection_cannot_reuse_or_regress_the_source_cursor(tmp_path):
    store = Objects(tmp_path/'objects')
    codec = TextCodec(tokenizer(),store)
    corpus = TextCorpus(tmp_path/'corpus',store,codec)
    key = corpus.register(source('train'))
    with (corpus.home/'collection.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(ValueError,match='Another collector owns'):
            corpus.collect(key,1,rows=[conversation(1,answer=8)])
        assert corpus.db.execute('SELECT cursor FROM sources WHERE id=?',(key,)).fetchone()[0] == 0
    assert corpus.collect(key,1,rows=[conversation(1,answer=8)])['end'] == 1
    corpus.db.close()


def test_preparation_preserves_multiwindow_data_without_silent_truncation(tmp_path):
    store = Objects(tmp_path/'objects')
    codec = TextCodec(tokenizer(),store)
    corpus = TextCorpus(tmp_path/'corpus',store,codec)
    key = corpus.register(source('train'))
    corpus.collect(key,4,rows=[conversation(i,answer=100) for i in range(4)])
    result = preparation.prepare(corpus,'c'*64,{},[key],1)
    assert result['status'] == 'needs_more_data'
    assert result['report']['selected']['train'] == 1
    assert result['report']['rejected']['multi_window'] == 0
    assert 'metadata' not in result
    with pytest.raises(ValueError,match='complete reviewed proposal'):
        preparation.publish(store,result,Objects(tmp_path/'replica'))
    corpus.db.close()
