#!/usr/bin/env python3
"""Collect bounded pinned data and prepare a reviewable native cohort proposal.

This command never votes, submits a transaction, trains, or changes a live model.
Native source cursors anchor retries even if collection finished before a crash.
"""
import argparse
import json
from pathlib import Path

from neuroshard.dataflow.store import canonical
from neuroshard.evolution import cohorts
from neuroshard.evolution.data import TextCorpus, normalized
from neuroshard.evolution.objects import Objects, digest
from neuroshard.evolution.schema import integer, root
from neuroshard.evolution.text import TextCodec
from neuroshard.evolution.verification import Metadata


def prepare(corpus, previous, cursors, source_ids, training_documents=32):
    root(previous)
    integer(training_documents,1,128)
    if not 1 <= len(source_ids) <= 8 or len(set(source_ids)) != len(source_ids):
        raise ValueError('Select 1–8 distinct sources')
    store = corpus.store
    values, documents, windows = {}, [], []
    quotas = {'train':training_documents,'retention':32,'fresh':32}
    selected = dict.fromkeys(quotas,0)
    rejected = {'multi_window':0,'incomplete':0,'duplicate_tokens':0,'protected_test':0,'outside_selected_roles':0}
    seen = set()
    def add(value):
        key = store.put_json(value)
        values[key] = value
        return key
    for source_id in source_ids:
        row = corpus.db.execute('SELECT spec,cursor FROM sources WHERE id=?',(source_id,)).fetchone()
        if row is None:raise ValueError('Source is not registered in this corpus')
        spec, end = json.loads(row[0]), row[1]
        cohorts.source(spec)
        start = integer(cursors.get(source_id,0),0,2**53-1)
        if end < start or end-start > 4096:
            raise ValueError('Native cursor and collector differ by more than the bounded admission range')
        if end == start:continue
        windows.append({'source':add(spec),'start':start,'end':end})
        rows = corpus.db.execute('SELECT id,role,row,object FROM documents WHERE source=? AND row>=? AND row<? ORDER BY row,id',
                                 (source_id,start,end)).fetchall()
        for identity,role,position,document_root in rows:
            if role == 'test':
                rejected['protected_test'] += 1
                continue
            if role not in quotas:
                rejected['outside_selected_roles'] += 1
                continue
            if selected[role] >= quotas[role]:continue
            # Cached windows can only exclude; inclusion always re-tokenizes
            # original bytes. Incomplete protected documents are never admitted.
            cached = corpus.db.execute('SELECT object FROM sequences WHERE document=?',(identity,)).fetchall()
            if not 1 <= len(cached) <= 4:
                rejected['multi_window'] += 1
                continue
            document = store.json(document_root)
            if (document['id'] != identity or document['source'] != source_id or document['row'] != position
                    or document['license'] != spec['license']):
                raise ValueError('Document bytes differ from durable provenance')
            if digest(normalized('\n'.join(message['content'] for message in document['messages'])).encode()) != identity:
                raise ValueError('Document identity differs from its original text')
            # Re-tokenize the original message bytes. Never trust a stale cache's
            # claim that a response was complete or used this tokenizer.
            prepared = corpus.codec.response_windows(document['messages'],corpus.sequence_length//2,
                corpus.sequence_length-corpus.sequence_length//2,corpus.max_windows)
            if prepared['truncated'] and role != 'train':
                rejected['incomplete'] += 1
                continue
            if not 1 <= len(prepared['windows']) <= 4:
                rejected['multi_window'] += 1
                continue
            batches = [{'input_ids':[window['tokens']],'labels':[window['labels']]} for window in prepared['windows']]
            keys = [digest(canonical(batch)) for batch in batches]
            if len(set(keys))!=len(keys) or any(key in seen for key in keys):
                rejected['duplicate_tokens'] += 1
                continue
            seen.update(keys)
            for batch in batches:add(batch)
            documents.append({'id':identity,'source':source_id,'row':position,'object':document_root,
                              'batches':keys,'role':role,'omitted_targets':prepared['omitted_tokens']})
            selected[role] += 1
    report = {'selected':selected,'required':quotas,'rejected':rejected,
              'windows':windows,'tokenizer_root':corpus.tokenizer_root,
              'scope':'1–4 windows per document; evaluation requires complete response targets; training omissions are explicit'}
    if selected != quotas:
        return {'status':'needs_more_data','report':report}
    value = {'format':cohorts.FORMAT,'previous':previous,'tokenizer_root':corpus.tokenizer_root,
             'windows':windows,'documents':documents}
    key = add(value)
    cohorts.metadata(values)
    return {'status':'prepared_for_review','data_root':key,'metadata':values,'report':report}


def publish(store, prepared, destination):
    """Explicitly mirror selected public cohort evidence and read every object back."""
    if prepared['status'] != 'prepared_for_review':
        raise ValueError('Only a complete reviewed proposal can be mirrored')
    metadata = cohorts.metadata(prepared['metadata'])
    value = metadata.json(prepared['data_root'])
    codec = TextCodec.load(store,value['tokenizer_root'])
    keys = set(metadata.values) | {codec.root,codec.profile['backend']}
    keys.update(document['object'] for document in value['documents'])
    size = 0
    for key in sorted(keys):
        raw = store.get(key)
        if destination.put(raw) != key or destination.get(key) != raw:
            raise ValueError('Cohort replica read-back mismatch')
        size += len(raw)
    return {'data_root':prepared['data_root'],'objects':len(keys),'bytes':size,'read_back_verified':True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--previous-data-root',required=True)
    parser.add_argument('--cursors',type=Path,required=True,help='JSON source->cursor mapping from native /lifecycle')
    parser.add_argument('--output',type=Path,required=True,help='New local proposal/report file')
    parser.add_argument('--collect-records',type=int,default=0,help='Fetch at most this many records per configured source (0–1024)')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output path before collecting additional records')
    count = integer(args.collect_records,0,1024)
    config = json.loads(args.config.read_bytes())
    base = args.config.resolve().parent
    def local(value):return base/Path(value).expanduser()
    store = Objects(local(config['objects']))
    codec = TextCodec.load(store,config['tokenizer_root'])
    corpus = TextCorpus(local(config['corpus']),store,codec,**config.get('text',{}))
    try:
        sources = [corpus.register(spec) for spec in config['sources']]
        cursors = json.loads(args.cursors.read_bytes())
        if not isinstance(cursors,dict):raise ValueError('Expected a native cursor mapping')
        for source in sources:
            current = corpus.db.execute('SELECT cursor FROM sources WHERE id=?',(source,)).fetchone()[0]
            start = integer(cursors.get(source,0),0,2**53-1)
            if current < start or current-start > 4096:
                raise ValueError('Collector cursor is outside the native admission window')
            remaining = 4096-(current-start)
            if count and remaining:
                collected = corpus.collect(source,min(count,remaining))
                print(json.dumps({'phase':'collected','source':source,'start':collected['start'],
                                  'end':collected['end'],'rejected':collected['rejected']}),flush=True)
        result = prepare(corpus,args.previous_data_root,cursors,sources,config.get('training_documents',32))
        with args.output.open('x') as file:file.write(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'status':result['status'],'data_root':result.get('data_root'),'report':result['report'],'output':str(args.output)},indent=2))
    finally:
        corpus.db.close()


if __name__ == '__main__':main()
