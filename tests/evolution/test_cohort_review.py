import copy
import importlib.util
from pathlib import Path

import pytest

from neuroshard.evolution.data import TextCorpus
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.text import TextCodec
from test_cohort_preparation import preparation
from test_text import tokenizer, conversation, source


spec = importlib.util.spec_from_file_location('review_native_cohort',Path(__file__).resolve().parents[2]/'scripts/review_native_cohort.py')
reviewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reviewer)


@pytest.fixture
def proposal(tmp_path):
    store = Objects(tmp_path/'objects')
    codec = TextCodec(tokenizer(),store)
    corpus = TextCorpus(tmp_path/'corpus',store,codec)
    sources, rows = [], {}
    policy = {'tokenizer_root':codec.root,'sources':[]}
    for group,role in enumerate(('train','retention','fresh')):
        value = source(role)
        key = corpus.register(value)
        sources.append(key)
        policy['sources'].append(value)
        rows[role] = [conversation(1000*group+i,answer=8) for i in range(96)]
        corpus.collect(key,96,rows=rows[role])
    prepared = preparation.prepare(corpus,'c'*64,{},sources,4)
    corpus.db.close()
    assert prepared['status'] == 'prepared_for_review'
    return store,prepared,policy,lambda spec,start,count:rows[spec['role']][start:start+count]


def test_reviewer_checks_original_windows_and_every_pinned_upstream_row(proposal):
    store,prepared,policy,upstream = proposal
    result = reviewer.review(store,prepared,policy,upstream)
    assert result['mechanical_evidence_verified'] and result['curation_decision_required']
    assert result['upstream_checked'] and result['upstream_documents_checked'] == 68
    offline = reviewer.review(store,prepared,policy)
    assert not offline['upstream_checked'] and offline['upstream_documents_checked'] == 0


def test_valid_hashes_cannot_hide_substituted_tokens(proposal):
    store,prepared,policy,_ = proposal
    altered = copy.deepcopy(prepared)
    cohort = altered['metadata'].pop(altered['data_root'])
    document = cohort['documents'][0]
    batch = copy.deepcopy(altered['metadata'][document['batches'][0]])
    index = next(i for i,label in enumerate(batch['labels'][0]) if label != -100)
    batch['input_ids'][0][index] = batch['labels'][0][index] = (batch['labels'][0][index]+1)%64
    key = store.put_json(batch)
    altered['metadata'][key] = batch
    document['batches'][0] = key
    altered['data_root'] = store.put_json(cohort)
    altered['metadata'][altered['data_root']] = cohort
    with pytest.raises(ValueError,match='Token windows or response targets'):
        reviewer.review(store,altered,policy)


def test_review_uses_own_source_policy_and_rejects_wrong_upstream_bytes(proposal):
    store,prepared,policy,upstream = proposal
    changed = copy.deepcopy(policy)
    changed['sources'][0]['revision'] = 'b'*40
    with pytest.raises(ValueError,match='outside reviewer policy'):
        reviewer.review(store,prepared,changed)
    with pytest.raises(ValueError,match='pinned upstream row'):
        reviewer.review(store,prepared,policy,lambda spec,start,count:[conversation(999999)]*count)
    with pytest.raises(ValueError,match='missing committed source rows'):
        reviewer.review(store,prepared,policy,lambda spec,start,count:[])


def test_public_hash_partition_cannot_be_relabeled_by_a_proposer(proposal):
    store,prepared,policy,_ = proposal
    changed = copy.deepcopy(prepared)
    policy = copy.deepcopy(policy)
    cohort = changed['metadata'].pop(changed['data_root'])
    document = next(d for d in cohort['documents'] if d['role']=='fresh' and int(d['id'],16)%3!=1)
    heldout = source('heldout')
    heldout_key = store.put_json(heldout)
    policy['sources'].append(heldout)
    changed['metadata'][heldout_key] = heldout
    original = store.json(document['object'])
    original['source'] = document['source'] = heldout_key
    document['object'] = store.put_json(original)
    cohort['windows'].append({'source':heldout_key,'start':0,'end':96})
    changed['data_root'] = store.put_json(cohort)
    changed['metadata'][changed['data_root']] = cohort
    with pytest.raises(ValueError,match='held-out partition'):
        reviewer.review(store,changed,policy)
