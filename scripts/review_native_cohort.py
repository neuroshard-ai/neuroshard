#!/usr/bin/env python3
"""Check cohort evidence before an explicit native curation vote.

Uses the reviewer's source/tokenizer policy and independently checks pinned
upstream rows by default. This command never signs, votes or activates data.
"""
import argparse
import json
from pathlib import Path

from neuroshard.dataflow.store import canonical
from neuroshard.evolution import cohorts
from neuroshard.evolution.data import normalized, fingerprint
from neuroshard.evolution.objects import Objects, digest
from neuroshard.evolution.schema import integer, root
from neuroshard.evolution.text import TextCodec


def review(store, prepared, policy, upstream=None, *, consumed_documents=(), consumed_batches=()):
    consumed_documents,consumed_batches = set(consumed_documents),set(consumed_batches)
    metadata = cohorts.metadata(prepared['metadata'])
    key = root(prepared['data_root'])
    value = metadata.json(key)
    root(value['previous'])
    codec = TextCodec.load(store, root(policy['tokenizer_root']))
    sources = policy['sources']
    if not isinstance(sources, list) or not 1 <= len(sources) <= 8:
        raise ValueError('Reviewer policy needs 1–8 explicitly approved sources')
    allowed = {digest(canonical(cohorts.source(spec))): spec for spec in sources}
    if len(allowed) != len(sources):
        raise ValueError('Reviewer policy repeats a source')
    length = integer(policy.get('text', {}).get('sequence_length', 128), 16, 256)
    maximum = integer(policy.get('text', {}).get('max_windows', 4), 1, 64)
    # Check the bounded cohort schema without pretending to know native history.
    # Consensus still checks the actual parent, cursors and consumed identities.
    state = {'data_root':value['previous'], 'manifest':{'lifecycle':{
        'tokenizer_root':codec.root, 'vocabulary':codec.profile['vocabulary'], 'steps_per_cohort':1}},
        'lifecycle':{'cursors':{w['source']:w['start'] for w in value['windows']},
                     'seen_documents':dict.fromkeys(consumed_documents), 'seen_batches':dict.fromkeys(consumed_batches), 'active':None}}
    value, roles = cohorts.validate(metadata, key, state)
    for window in value['windows']:
        if window['source'] not in allowed:
            raise ValueError('Source revision, license or role is outside reviewer policy')
    documents, signatures, evidence_bytes = {}, [], 0
    for document in value['documents']:
        raw = store.get(document['object'])
        if len(raw) > 512*1024:
            raise ValueError('Raw document evidence exceeds the review bound')
        evidence_bytes += len(raw)
        original = json.loads(raw)
        if set(original) != {'id','source','row','messages','license'} or any(
                original[field] != document[field] for field in ('id','source','row')):
            raise ValueError('Original document differs from committed provenance')
        spec = allowed[document['source']]
        if original['license'] != spec['license']:
            raise ValueError('Document license differs from reviewer source policy')
        windows = codec.response_windows(original['messages'], length//2, length-length//2, maximum)
        text = '\n'.join(message['content'] for message in original['messages'])
        if digest(normalized(text).encode()) != document['id']:
            raise ValueError('Document identity differs from original content')
        role = ('retention','fresh','test')[int(document['id'],16)%3] if spec['role']=='heldout' else spec['role']
        if role != document['role']:
            raise ValueError('Document violates the reviewer held-out partition')
        expected = [{'input_ids':[w['tokens']], 'labels':[w['labels']]} for w in windows['windows']]
        if expected != [metadata.json(batch) for batch in document['batches']]:
            raise ValueError('Token windows or response targets differ from original content')
        if windows['omitted_tokens'] != document['omitted_targets'] or (role != 'train' and windows['truncated']):
            raise ValueError('Document misstates response coverage')
        signature = fingerprint(text)
        if any((signature^old).bit_count() <= 3 for old in signatures):
            raise ValueError('Cohort contains heuristic near-duplicate documents')
        signatures.append(signature)
        documents[(document['source'],document['row'])] = original
    checked = 0
    if upstream is not None:
        for window in value['windows']:
            source, start, end = window['source'], window['start'], window['end']
            found = set()
            for position, row in enumerate(upstream(allowed[source],start,end-start), start):
                if position >= end:
                    raise ValueError('Upstream reader exceeded the admitted source window')
                original = documents.get((source,position))
                if original is not None:
                    if not isinstance(row,dict) or canonical(row.get('messages')) != canonical(original['messages']):
                        raise ValueError('Committed messages differ from the pinned upstream row')
                    found.add(position)
            expected = {position for identity,position in documents if identity==source}
            if found != expected:
                raise ValueError('Pinned upstream is missing committed source rows')
            checked += len(found)
    return {'data_root':key, 'tokenizer_root':codec.root, 'mechanical_evidence_verified':True,
        'documents':{role:len(rows) for role,rows in roles.items()},
        'windows':{role:sum(len(d['batches']) for d in rows) for role,rows in roles.items()},
        'raw_evidence_bytes':evidence_bytes, 'upstream_documents_checked':checked,
        'excluded_prior_documents':len(consumed_documents), 'excluded_prior_windows':len(consumed_batches),
        'upstream_checked':upstream is not None, 'curation_decision_required':True,
        'scope':'local policy, original bytes and tokenizer correspondence; upstream checking trusts the pinned repository service',
        'not_checked':['completeness of supplied admission history; live parent/cursors and quorum',
            'semantic contamination beyond the within-cohort heuristic',
            'license rights, truth, harmful content, usefulness or model quality']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True,help='Reviewer-controlled native data policy; never accept a submitter policy blindly')
    parser.add_argument('--cache',type=Path,help='Private directory for independently verified upstream Parquet files')
    parser.add_argument('--offline',action='store_true',help='Skip upstream source checks; report is explicitly incomplete')
    parser.add_argument('--exclude-cohort',type=Path,action='append',default=[],help='Previously admitted proposal file; repeat for retained admission history')
    args = parser.parse_args()
    if args.proposal.stat().st_size > 8*1024*1024 or args.config.stat().st_size > 65536:
        raise ValueError('Review input exceeds its bounded size')
    policy = json.loads(args.config.read_bytes())
    from prepare_native_cohort import exclusions
    if any(path.stat().st_size > 8*1024*1024 for path in args.exclude_cohort):
        raise ValueError('Prior cohort file exceeds review bounds')
    consumed_documents,consumed_batches = exclusions(json.loads(path.read_bytes()) for path in args.exclude_cohort)
    store = Objects(args.config.resolve().parent/Path(policy['objects']).expanduser())
    prepared = json.loads(args.proposal.read_bytes())
    upstream = None
    if not args.offline:
        if args.cache is None:
            parser.error('--cache is required unless --offline is selected')
        from neuroshard.dataflow.collect import upstream_rows
        args.cache.mkdir(parents=True,exist_ok=True)
        upstream = lambda spec,start,count: upstream_rows(spec,args.cache,start,count)
    print(json.dumps(review(store,prepared,policy,upstream,
        consumed_documents=consumed_documents,consumed_batches=consumed_batches),indent=2))


if __name__ == '__main__':main()
