"""Retokenize actual source conversations, then produce and replay new work."""
import copy
import json
import subprocess
import sys

import pytest
from transformers import PreTrainedTokenizerFast

from neuroshard.dataflow.store import canonical
from neuroshard.evolution import auditing, expert_data, expert_work, expert_lifecycle as life
from neuroshard.evolution import expert_admission as admission
from neuroshard.evolution.expert_history import HistoryIndex
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.reference_data import identity, save, sha256, tokenizer_identity
from neuroshard.evolution.sharded import expert_execution, graph_quality, prefix_execution
from neuroshard.evolution.sharded.interpretation import example_messages
from test_graph_execution import prepare_extension
from test_expert_admission import renamed


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv('PYTORCH_CUDA_ALLOC_CONF', 'test')
    prepare_extension(tmp_path)
    read = lambda name: json.loads((tmp_path / name).read_bytes())
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tmp_path/'seed', local_files_only=True)
    tokenizer.chat_template = ("{% for m in messages %}{{ m['role'] }} {{ m['content'] }} "
        "{% if m['role'] == 'assistant' %}</s> {% endif %}{% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}")
    tokenizer.save_pretrained(tmp_path/'seed')
    baseline, template = read('pre-growth.json'), read('extension-template.json')
    for graph in (baseline, template):
        graph['tokenizer'] = {'root': tokenizer_identity(tokenizer), 'eos_id': 2, 'max_context': 64,
            'files': {p.name: sha256(p) for p in (tmp_path/'seed').iterdir()}}
        interpretation = graph['descriptor']['interpretation']
        prefix = example_messages(interpretation['instruction'], interpretation['examples'])
        graph['interpreter_prompt'] = {'format': 'name-field-json-v1', 'messages': identity(prefix),
            'tokens': identity(tokenizer.apply_chat_template(prefix, tokenize=True, add_generation_prompt=False))}
        graph['descriptor'].update(tokenizer=graph['tokenizer']['root'], interpreter_prompt=graph['interpreter_prompt'])
    template['descriptor']['previous_graph'] = identity(baseline['descriptor'])
    quality = read('quality-policy.json')
    rule = {key: quality[key] for key in ('gates', 'generation')}
    rule['retained_roles'] = {key: value for key, value in quality['roles'].items() if key != 'test'}
    policy = {'format': expert_data.POLICY, 'tokenizer': tokenizer_identity(tokenizer), 'max_length': 32,
              'sources': {'fixture/messages': [{'role': 'train', 'license': 'Apache-2.0'},
                                              {'role': 'heldout', 'license': 'Apache-2.0'}]},
              'near_duplicate_distance': 0, 'quality_rule': identity(rule)}
    plan = {'format': expert_data.FORMAT, 'parent': identity(baseline['parent']), 'split': 5,
        'parent_layout': baseline['parent']['boundaries'], 'expert_layout': template['experts']['astronomy']['boundaries'],
        'training': template['experts']['astronomy']['recipe'], 'max_length': 32, 'microbatch': 1,
        'runtime': read('profile.json')['runtime'], 'threads': 1, 'parameter_limit': 200000,
        'previous_graph': identity(baseline['descriptor']), 'data_policy': identity(policy),
        'objective': {'kl_strength': 0., 'margin_strength': 0., 'margin_min': .5, 'margin_max': 2.}}
    store = Objects(tmp_path/'data-objects')
    store.put_json(plan)
    sources, documents, windows, roles, conversations = {}, [], [], {}, {}
    inputs = tmp_path/'general-inputs'
    inputs.mkdir()
    for role in ('train', 'test'):
        source = {'repo': 'fixture/messages', 'revision': 'b'*40, 'split': role,
                  'license': 'Apache-2.0', 'role': 'train' if role == 'train' else 'heldout'}
        key = identity(source)
        sources[key] = source
        values = []
        pairs = ([('word3 word7', 'word10'), ('word4 word8', 'word11')] if role == 'train' else
                 [('word5 word9', 'word12'), ('word5 word6', 'word12; word13')])
        for index, (question, answer) in enumerate(pairs):
            messages = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]
            row = expert_data.encode(tokenizer, messages, 32, source=key, position=index)
            values.append(row)
            original = {'source': key, 'row': index, 'messages': messages, 'license': 'Apache-2.0'}
            documents.append({'id': row['id'], 'source': key, 'row': index, 'object': store.put_json(original),
                              'tokens': expert_data.token_identity(row), 'role': 'train' if role == 'train' else 'evaluation'})
        conversations[role] = [{'messages': row['messages']} for row in values]
        raw = b''.join(canonical(row) + b'\n' for row in values)
        (inputs/(role+'.jsonl')).write_bytes(raw)
        roles[role] = {'sha256': store.put(raw), 'count': len(values), 'ids': identity([row['id'] for row in values])}
        windows.append({'source': key, 'start': 0, 'end': len(values)})
    prepared = {'format': expert_data.INPUTS, 'plan': identity(plan), 'roles': roles,
                'batches': [[0], [1]], 'schedule': [0, 1], 'retention_cache': identity(baseline)}
    store.put_json(prepared)
    initial = renamed(baseline['parent'], template['experts']['astronomy'], expert_data.job_identity(plan, prepared))
    template['experts']['astronomy'] = initial
    template['descriptor']['experts'][-1]['checkpoint'] = initial['checkpoint']
    quality_rows = []
    for index, row in enumerate(conversations['test']):
        messages = row['messages']
        quality_rows.append({'id': expert_data.document_identity(messages),
            'stratum': 'single' if index == 0 else 'composed',
            'topics': ['fact-a'] if index == 0 else ['fact-a', 'fact-b'],
            'answers': messages[-1]['content'].split('; '), 'messages': messages})
    quality.pop('candidate_graph')
    quality.update(format=graph_quality.GENERAL, candidate_template=template,
                   baseline_graph=identity(baseline), prepared=identity(prepared))
    quality['roles']['test'] = {'file': 'quality-test.jsonl',
        'sha256': store.put(b''.join(canonical(row) + b'\n' for row in quality_rows)),
        'count': len(quality_rows), 'ids': identity([row['id'] for row in quality_rows])}
    quality_root = store.put_json(quality)
    work = {'format': expert_work.PROSPECTIVE, 'parent': baseline['parent'], 'checkpoint': initial,
        'prepared': identity(prepared), 'feature_stages': 6, 'batch_count': 2, 'schedule': [0, 1],
        'numerical_profile': baseline['numerical_profile']}
    previous_data = identity({'previous-data': True})
    dataset = {'format': admission.DATA, 'previous': previous_data, 'prepared': identity(prepared),
        'policy': identity(policy), 'sources': sources, 'windows': windows, 'documents': documents,
        'batches': [[document['id']] for document in documents if document['role'] == 'train']}
    lifecycle = {'format': life.PROSPECTIVE, 'serving_graph': baseline, 'candidate_template': template,
        'quality': {'policy_root': quality_root, 'prepared': identity(prepared), 'stages': len(quality_rows)},
        'price_per_token': 1, 'max_tokens': 64}
    # This snapshot isolates source-review mechanics; it is not a live ledger.
    state = {'data_root': previous_data,
        'manifest': {'auditing': auditing.QUORUM_PROFILE, 'expert_admission': {'data_policy': identity(policy)},
                     'expert_lifecycle': lifecycle},
        'expert_lifecycle': {'serving_graph': baseline, 'admission': {
            'seen_jobs': {}, 'cursors': {}, 'seen_documents': {}, 'trained_documents': {}}}}
    job = {'work': work, 'lifecycle': lifecycle, 'data': dataset}
    upstream = lambda source, start, count: conversations[source['split']][start:start+count]
    return tmp_path, state, job, policy, store, tokenizer, upstream, plan, prepared


def test_review_retokenizes_and_checks_actual_immutable_upstream(prepared):
    _, state, job, policy, store, tokenizer, upstream, _, _ = prepared
    result = expert_data.review(state, job, policy, store, tokenizer, upstream)
    assert result['mechanical_checks_passed'] and result['upstream_documents'] == 4
    assert result['semantic_curation_required']
    with pytest.raises(ValueError, match='pinned upstream'):
        expert_data.review(state, job, policy, store, tokenizer,
                           lambda source, start, count: [{'messages': []}] * count)
    changed = copy.deepcopy(job)
    changed['data']['documents'][0]['tokens'] = '0'*64
    with pytest.raises(ValueError, match='token commitment'):
        expert_data.review(state, changed, policy, store, tokenizer, upstream)
    with pytest.raises(ValueError, match='independent pinned upstream'):
        expert_data.review(state, job, policy, store, tokenizer, None)
    changed = copy.deepcopy(job)
    quality = store.json(job['lifecycle']['quality']['policy_root'])
    quality['gates']['gain_lower'] = 0
    changed['lifecycle']['quality']['policy_root'] = store.put_json(quality)
    with pytest.raises(ValueError, match='scoring rules'):
        expert_data.review(state, changed, policy, store, tokenizer, upstream)


def test_generic_source_data_drives_new_prefix_and_independently_replayed_training(prepared):
    home, state, job, policy, store, tokenizer, upstream, plan, prepared = prepared
    assert expert_data.review(state, job, policy, store, tokenizer, upstream)['mechanical_checks_passed']
    paths = {'inputs': str(home/'general-inputs'), 'objects': str(home/'objects'),
             'bank_home': str(home/'unused-bank'), 'checkpoint_store': str(home/'production')}
    result = prefix_execution.produce_features(job['work'], plan, prepared, **paths, max_seconds=60)
    profile = expert_work.resolve_prefix(job['work'], result['feature_root'], result['batch_roots'])
    paths['bank_home'] = str(home/'production/prefix'/result['transcript_root']/'rank-2/features')
    produced = expert_execution.produce_training(profile['checkpoint'], 2, profile, plan, prepared,
                                                **paths, max_seconds=60)
    window = produced['window']
    assert window['output']['step'] == 2 and window['output']['state_root'] != profile['checkpoint']['state_root']
    graph = life.materialize_graph(job['lifecycle']['candidate_template'], window['output'])
    assert graph['experts']['astronomy'] == window['output']
    claim = {'kind': 'expert_training', 'id': 'a'*64, 'input_checkpoint': window['input'],
        'output_checkpoint': window['output'], 'parent_checkpoint': profile['parent'],
        'prepared': profile['prepared'], 'feature_root': profile['feature_root'], 'numerical_profile': profile['numerical_profile'],
        'feature_claim': 'b'*64, 'stages': 2, 'record_root': identity(window),
        'work_ids': [step['work_identity'] for step in window['steps']], **produced}
    paths['checkpoint_store'] = str(home/'independent-auditor')
    config = {'format': 'neuroshard-expert-executor-v1', 'profile': profile, 'plan': plan,
              'prepared': prepared, 'paths': paths, 'max_seconds': 60}
    save(home/'generic-executor.json', config)
    replay = subprocess.run([sys.executable, '-m', 'neuroshard.evolution.sharded.expert_execution',
        '--config', str(home/'generic-executor.json')], input=json.dumps(claim), capture_output=True, text=True, timeout=90)
    assert replay.returncode == 0, replay.stderr
    assert expert_work.replay_report(claim, json.loads(replay.stdout))['valid']


def test_prior_evaluation_reworded_as_fresh_training_cannot_pass_review(prepared):
    home, state, job, policy, store, tokenizer, upstream, _, _ = prepared
    original = copy.deepcopy(store.json(job['data']['documents'][0]['object']))
    original['source'], original['row'] = 'c'*64, 19
    original['messages'][0]['content'] += '!'
    key = expert_data.document_identity(original['messages'])
    assert key != job['data']['documents'][0]['id']
    historical = {'id': key, 'source': original['source'], 'row': original['row'],
        'object': store.put_json(original), 'tokens': 'd'*64, 'role': 'evaluation'}
    state['expert_lifecycle']['admission']['seen_documents'][key] = historical
    index = HistoryIndex(home/'history.sqlite')
    try:
        with pytest.raises(ValueError, match='Historical near-duplicate'):
            expert_data.review(state, job, policy, store, tokenizer, upstream, history_index=index)
    finally:
        index.close()
    # A restart reuses verified historical fingerprints. No past source
    # redownload is needed to keep detecting the same contaminated proposal.
    index = HistoryIndex(home/'history.sqlite')
    try:
        summary = index.synchronize(state['expert_lifecycle']['admission']['seen_documents'], store)
        assert summary['documents'] == 1 and summary['indexed_now'] == 0
        with pytest.raises(ValueError, match='Historical near-duplicate'):
            expert_data.review(state, job, policy, store, tokenizer, upstream, history_index=index)
        # Cache membership follows the exact canonical snapshot after recovery.
        state['expert_lifecycle']['admission']['seen_documents'] = {}
        result = expert_data.review(state, job, policy, store, tokenizer, upstream, history_index=index)
        assert result['historical_documents'] == 0 and result['mechanical_checks_passed']
    finally:
        index.close()


def test_history_requires_original_objects_and_preserves_explicit_training_replay(prepared):
    _, state, job, _, store, _, _, _, _ = prepared
    document = job['data']['documents'][0]
    history = {document['id']: document}
    index = HistoryIndex()
    try:
        original = store.json(document['object'])
        from neuroshard.evolution.data import fingerprint
        signature = fingerprint('\n'.join(m['content'] for m in original['messages']))
        index.synchronize(history, store)
        assert index.match(signature, 0) == document['id']
        assert index.match(signature, 0, replay=True) is None
        history[document['id']] = {**document, 'role': 'evaluation'}
        index.synchronize(history, store)
        assert index.match(signature, 0, replay=True) == document['id']
        # Failed synchronization rolls back; it cannot erase the last complete
        # membership and leave an apparently empty, successful history check.
        history[document['id']] = {**document, 'object': 'e'*64}
        with pytest.raises(FileNotFoundError):
            index.synchronize(history, store)
        assert index.match(signature, 0, replay=True) == document['id']
    finally:
        index.close()


def test_history_bands_never_omit_a_candidate_within_supported_distance():
    import random
    from neuroshard.evolution.expert_history import bands
    randomizer = random.Random(61029)
    for distance in range(9):
        for _ in range(100):
            original = randomizer.getrandbits(64)
            changed = original
            for bit in randomizer.sample(range(64), distance):
                changed ^= 1 << bit
            assert set(bands(original)) & set(bands(changed))
