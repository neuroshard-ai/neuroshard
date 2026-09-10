import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from neuroshard.dataflow.ingest import Ingestor, verify_snapshot
from neuroshard.dataflow.store import LocalStore, S3Store, digest


PROVENANCE = {"origin":"test fixture","revision":"fixture-v1","license":"CC0-1.0"}


def corpus(tmp_path, count=150):
    path=tmp_path/'input.jsonl'
    path.write_text(''.join(json.dumps({'text':f'Document {i}: a reproducible training example.'})+'\n' for i in range(count)))
    return path


def test_concurrent_identical_uploads_are_idempotent_and_corruption_is_detected(tmp_path):
    store=LocalStore(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        roots=list(pool.map(store.put,[b'same bytes']*24))
    assert len(set(roots))==1
    store.path(roots[0]).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='SHA-256'):
        store.put(b'same bytes')


def test_resume_publishes_every_document_once_with_disjoint_splits(tmp_path):
    store=LocalStore(tmp_path/'objects');path=corpus(tmp_path)
    ingest=Ingestor(tmp_path/'journal.sqlite',store)
    first=ingest.ingest(path,PROVENANCE,batch_records=10,max_shards=2)
    assert verify_snapshot(store,first['sha256'])['unique_documents']==20
    ingest.close()
    ingest=Ingestor(tmp_path/'journal.sqlite',store)
    final=ingest.ingest(path,PROVENANCE,batch_records=10,max_shards=100)
    assert verify_snapshot(store,final['sha256'])=={'dataset_root':final['sha256'],'shards':15,
        'unique_documents':150,'train':143,'validation':7,'duplicates':0}
    again=ingest.ingest(path,PROVENANCE)
    assert again==final
    ingest.close()


def test_crash_after_upload_before_cursor_commit_is_recoverable(tmp_path):
    class InterruptedStore(LocalStore):
        interrupted=False
        def put(self,data):
            root=super().put(data)
            if not self.interrupted:
                self.interrupted=True
                raise OSError('simulated crash after durable object write')
            return root
    store=InterruptedStore(tmp_path/'objects');path=corpus(tmp_path,12)
    ingest=Ingestor(tmp_path/'journal.sqlite',store)
    with pytest.raises(OSError,match='simulated crash'):
        ingest.ingest(path,PROVENANCE,batch_records=5)
    assert ingest.db.execute('SELECT cursor FROM sources').fetchone()[0]==0
    with pytest.raises(ValueError,match='pending'):
        ingest.snapshot()
    ingest.close()
    ingest=Ingestor(tmp_path/'journal.sqlite',store)
    result=ingest.ingest(path,PROVENANCE,batch_records=5)
    assert verify_snapshot(store,result['sha256'])['unique_documents']==12
    assert ingest.db.execute('SELECT count(*) FROM pending').fetchone()[0]==0
    ingest.close()


def test_changed_input_is_a_new_source_and_old_snapshot_remains_valid(tmp_path):
    store=LocalStore(tmp_path/'objects');path=corpus(tmp_path,2)
    ingest=Ingestor(tmp_path/'journal.sqlite',store)
    old=ingest.ingest(path,PROVENANCE)
    path.write_text(json.dumps({'text':'A new source revision'})+'\n')
    new=ingest.ingest(path,PROVENANCE)
    assert old['sha256']!=new['sha256']
    assert verify_snapshot(store,old['sha256'])['unique_documents']==2
    assert verify_snapshot(store,new['sha256'])['unique_documents']==3
    ingest.close()


def test_s3_uses_conditional_creation_and_fails_closed_on_auth_errors():
    boto=pytest.importorskip('botocore.exceptions')
    class Fake:
        def put_object(self,**kwargs):
            assert kwargs['IfNoneMatch']=='*'
            raise boto.ClientError({'Error':{'Code':'AccessDenied'},'ResponseMetadata':{'HTTPStatusCode':403}},'PutObject')
    with pytest.raises(boto.ClientError):
        S3Store('bucket',client=Fake()).put(b'new data')


def test_collection_does_not_advance_cursor_when_publication_fails(tmp_path):
    from neuroshard.dataflow.collect import collect
    class InterruptedStore(LocalStore):
        fail=True
        def put(self,data):
            if self.fail:raise OSError('temporary S3 failure')
            return super().put(data)
    config={'repo':'fixture','revision':'a'*40,'split':'train','license':'CC0-1.0','tokenizer_revision':'b'*40,
            'batch_documents':5,'max_documents':10}
    store=InterruptedStore(tmp_path/'objects');home=tmp_path/'collection'
    with pytest.raises(OSError):collect(config,home,store,rows=iter(range(5)),render=lambda i:f'Document {i}')
    assert not (home/'progress.json').exists() and (home/'pending.json').exists()
    store.fail=False
    result=collect(config,home,store,rows=iter(()),render=lambda i:pytest.fail('Must recover existing input'))
    assert result['cursor']==5
    assert verify_snapshot(store,result['snapshot'])['unique_documents']==5
    config['max_documents']=5
    assert collect(config,home,store)['status']=='collection_budget_complete'


def test_parquet_resume_crosses_file_boundaries_without_duplicate_rows(tmp_path):
    arrow=pytest.importorskip('pyarrow')
    import pyarrow.parquet as parquet
    from neuroshard.dataflow.collect import parquet_rows
    paths=[]
    for part,values in enumerate(([0,1,2],[3,4,5,6])):
        path=tmp_path/f'{part}.parquet'
        parquet.write_table(arrow.Table.from_pylist([{'messages':[{'role':'user','content':str(i)}]} for i in values]),path,row_group_size=2)
        paths.append(path)
    result=list(parquet_rows(paths,2,4))
    assert [r['messages'][0]['content'] for r in result]==['2','3','4','5']
    assert list(parquet_rows(paths,7,2))==[]
