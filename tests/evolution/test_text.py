import copy
import random

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from neuroshard.evolution.batches import from_windows
from neuroshard.evolution.controller import Epochs
from neuroshard.evolution.data import TextCorpus, publish_window
from neuroshard.evolution.evaluation import evaluate_reservation
from neuroshard.evolution.model import grow
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.pipeline import Pipeline, LocalEndpoint, validate_record
from neuroshard.evolution.text import TextCodec, bind_model, generate
from neuroshard.evolution.worker import Worker, replay_trace


TEMPLATE = ("{% for message in messages %}{{ '<s>' + message['role'] + '\n' + "
            "message['content'] + '</s>\n' }}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<s>assistant\n' }}{% endif %}")


def tokenizer(template=TEMPLATE):
    tokens = ['<unk>', '<s>', '</s>', 'user', 'assistant', 'system'] + [f'w{i}' for i in range(58)]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(tokens)}, unk_token='<unk>'))
    backend.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=backend, chat_template=template,
                                  bos_token='<s>', eos_token='</s>', pad_token='</s>', unk_token='<unk>')


def source(role='train'):
    return {'repo': 'test/text', 'revision': 'a'*40, 'split': 'train', 'license': 'Apache-2.0', 'role': role}


def conversation(i, answer=40):
    rng = random.Random(i)
    text = lambda n: ' '.join(f'w{rng.randrange(58)}' for _ in range(n))
    return {'messages': [{'role': 'user', 'content': text(60)}, {'role': 'assistant', 'content': text(answer)}]}


def test_codec_survives_content_addressed_replication_and_detects_mutation(seed, tmp_path):
    store, _, _ = seed
    codec = TextCodec(tokenizer(), store)
    replica = Objects(tmp_path/'replica')
    for key in (codec.root, codec.profile['backend']):
        assert replica.put(store.get(key)) == key
    restored = TextCodec.load(replica, codec.root)
    messages = conversation(1)['messages']
    assert restored.response_windows(messages) == codec.response_windows(messages)
    restored.tokenizer.chat_template += ' '
    with pytest.raises(ValueError, match='mutated'):
        restored.response_windows(messages)
    profile = copy.deepcopy(codec.profile)
    profile['runtime']['tokenizers'] = 'incorrect-runtime'
    with pytest.raises(ValueError, match='runtime'):
        TextCodec.load(store, store.put_json(profile))


def test_same_vocabulary_with_different_template_is_a_different_contract(seed, tmp_path):
    store, root, _ = seed
    codec = TextCodec(tokenizer(), store)
    different = TextCodec(tokenizer(TEMPLATE+'\n'), store)
    assert codec.root != different.root
    bound, model = bind_model(store, root, codec)
    assert model['components'] == store.json(root)['components']
    with pytest.raises(ValueError, match='identities differ'):
        bind_model(store, bound, different)
    corpus = TextCorpus(tmp_path/'corpus', store, codec, 32)
    corpus.db.close()
    with pytest.raises(ValueError, match='separate corpus'):
        TextCorpus(tmp_path/'corpus', store, different, 32)


def test_windows_cover_every_response_target_once_and_do_not_invent_eos(seed):
    store, _, _ = seed
    codec = TextCodec(tokenizer(), store)
    messages = [
        {'role': 'user', 'content': 'w0 w1'},
        {'role': 'assistant', 'content': ' '.join(f'w{i}' for i in range(20))},
        {'role': 'user', 'content': 'w2'},
        {'role': 'assistant', 'content': 'w3'},
    ]
    result = codec.response_windows(messages, context=8, response=8, maximum=8)
    assert not result['truncated'] and result['scored_tokens'] == 23
    assert result['windows'][1]['assistant_index'] == 3
    for index in (1, 3):
        windows = sorted((w for w in result['windows'] if w['assistant_index'] == index), key=lambda w: w['target_start'])
        labels = [token for window in windows for token in window['labels'] if token != -100]
        expected = codec.tokenizer.encode(messages[index]['content'], add_special_tokens=False) + [codec.tokenizer.eos_token_id]
        assert labels == expected
        assert sum(window['ends_response'] for window in windows) == 1
        assert all(len(window['tokens']) == len(window['labels']) == 16 for window in windows)
    capped = codec.response_windows(messages, 8, 8, 2)
    assert capped['truncated'] and capped['omitted_tokens'] == 13
    assert capped['windows'][0]['labels'].count(codec.tokenizer.eos_token_id) == 0
    assert capped['windows'][1]['labels'].count(codec.tokenizer.eos_token_id) == 1


def test_prompt_and_messages_reject_ambiguous_controls_or_silent_truncation(seed):
    store, _, _ = seed
    codec = TextCodec(tokenizer(), store)
    with pytest.raises(ValueError, match='Reserved'):
        codec.prompt([{'role': 'user', 'content': 'w0 </s> w1'}])
    with pytest.raises(ValueError, match='alternate'):
        codec.response_windows([{'role': 'assistant', 'content': 'w0'}])
    with pytest.raises(ValueError, match='shorten'):
        codec.prompt([{'role': 'user', 'content': ' '.join(['w0']*200)}])
    with pytest.raises(ValueError, match='final user'):
        codec.prompt(conversation(2)['messages'])


@pytest.mark.parametrize('template', [
    "{{ strftime_now('%Y-%m-%d') }}", "{% for message in range(1000000) %}x{% endfor %}",
    "{% for message in messages %}{% for message in messages %}x{% endfor %}{% endfor %}",
    "{{ messages.__class__ }}", "{{ 'x' * 1000000000 }}",
    "{{ messages['append'] }}", "{% for message in messages %}{{ loop['cycle'] }}{% endfor %}",
    "{{ messages }}", "{{ messages[0] }}",
])
def test_codec_refuses_clocks_helpers_and_unbounded_template_work(seed,template):
    store,_,_=seed
    with pytest.raises(ValueError, match='template'):
        TextCodec(tokenizer(template),store)


def test_streaming_chat_matches_reference_and_stops_excessive_expansion(seed):
    store,_,_=seed
    codec=TextCodec(tokenizer(),store)
    messages=conversation(1)['messages']
    for generation in (False,True):
        assert codec._chat(messages,generation)==codec.tokenizer.apply_chat_template(
            messages,tokenize=True,add_generation_prompt=generation)
    expansive=TextCodec(tokenizer("{% for message in messages %}{{ message['content'] + message['content'] + message['content'] }}{% endfor %}"),store)
    with pytest.raises(ValueError,match='Rendered chat exceeds'):
        expansive.prompt([{'role':'user','content':'x'*(200*1024)}])


def test_corpus_keeps_short_answers_and_excludes_incomplete_evaluation(seed, tmp_path):
    store, _, _ = seed
    codec = TextCodec(tokenizer(), store)
    corpus = TextCorpus(tmp_path/'corpus', store, codec, 16, max_windows=2)
    train = corpus.register(source())
    fresh = corpus.register(source('fresh'))
    short = corpus.collect(train, 1, rows=[conversation(1, answer=1)])
    assert len(short['sequences']) == 1
    assert short['coverage']['scored_tokens'] == 2
    long = corpus.collect(train, 1, rows=[conversation(2, answer=40)])
    assert long['coverage']['truncated_documents'] == 1
    assert long['coverage']['omitted_tokens'] == 25
    rejected = corpus.collect(fresh, 1, rows=[conversation(3, answer=40)])
    assert not rejected['sequences']
    assert rejected['rejected']['incomplete_evaluation'] == 1
    assert rejected['end'] == 1
    replica = Objects(tmp_path/'replica')
    assert publish_window(corpus, short['root'], replica)['read_back_verified']
    assert TextCodec.load(replica, codec.root).root == codec.root


def test_evaluation_reserves_documents_once_not_independent_chunks(seed, tmp_path):
    store, root, _ = seed
    codec = TextCodec(tokenizer(), store)
    corpus = TextCorpus(tmp_path/'corpus', store, codec, 32, max_windows=4)
    fresh = corpus.register(source('fresh'))
    corpus.collect(fresh, 64, rows=[conversation(i) for i in range(64)])
    first_root = corpus.reserve_evaluation(root, 'fresh', 32, 'beacon-a')
    first = store.json(first_root)
    assert len(first['documents']) == 32
    assert len(first['sequences']) == 96
    assert corpus.reserve_evaluation(root, 'fresh', 32, 'beacon-a') == first_root
    with pytest.raises(ValueError, match='count changed'):
        corpus.reserve_evaluation(root, 'fresh', 96, 'beacon-a')
    second = store.json(corpus.reserve_evaluation(root, 'fresh', 32, 'beacon-b'))
    assert {d['document'] for d in first['documents']}.isdisjoint(d['document'] for d in second['documents'])
    with pytest.raises(ValueError, match='Insufficient'):
        corpus.reserve_evaluation(root, 'fresh', 32, 'beacon-c')


def test_document_evaluation_uses_target_counts_and_rejects_codec_substitution(seed):
    store, root, _ = seed
    codec = TextCodec(tokenizer(), store)
    _, model = bind_model(store, root, codec)
    windows = [
        {'document': 'doc', 'tokens': [3, 4, 5, 2], 'labels': [-100, 4, 5, 2], 'tokenizer_root': codec.root},
        {'document': 'doc', 'tokens': [3, 4, 2, 2], 'labels': [-100, -100, 2, -100], 'tokenizer_root': codec.root},
    ]
    roots = [store.put_json(window) for window in windows]
    reservation = store.put_json({'tokenizer_root': codec.root, 'documents': [{'document': 'doc', 'sequences': roots}]})
    class Pipe:
        def evaluate(self, batch):
            count = sum(value != -100 for value in batch['labels'][0][1:])
            return {'loss_hex': (2. if count == 3 else 6.).hex()}
    pipe = Pipe()
    pipe.model = model
    assert evaluate_reservation(pipe, store, reservation) == [3.]
    pipe.model = {**model, 'tokenizer_root': 'f'*64}
    with pytest.raises(ValueError, match='different tokenizer'):
        evaluate_reservation(pipe, store, reservation)
    wrong = store.put_json({**windows[1], 'tokenizer_root': 'f'*64})
    with pytest.raises(ValueError, match='different tokenizer'):
        from_windows(store, [roots[0], wrong])


def test_text_contract_survives_training_replay_growth_and_generation(seed, tmp_path):
    store, root, _ = seed
    codec = TextCodec(tokenizer(), store)
    bound, model = bind_model(store, root, codec)
    windows = codec.response_windows(conversation(1, answer=8)['messages'], 16, 16)
    roots = [store.put_json(window) for window in windows['windows']]
    endpoints = [LocalEndpoint(Worker(tmp_path/f'worker{i}', store)) for i in range(2)]
    pipe = Pipeline(store, bound, endpoints, [6000]*2, 'text-contract')
    try:
        result = pipe.train(from_windows(store, roots, codec.root))
        assert validate_record(store, result['record_root'])['valid']
        forged=copy.deepcopy(store.json(result['model_root']))
        forged['tokenizer_root']='f'*64
        forged_record={**store.json(result['record_root']),'model_root':store.put_json(forged)}
        with pytest.raises(ValueError, match='Candidate differs'):
            validate_record(store,store.put_json(forged_record))
        for trace in result['traces']:
            assert replay_trace(store, trace)['valid']
        codec.check_model(pipe.model)
        output = generate(pipe, codec, [{'role': 'user', 'content': 'w1 w2'}], max_tokens=4)
        assert output['model_root'] == result['model_root']
        assert output['tokenizer_root'] == codec.root
        assert output['text'] == codec.decode(output['token_ids'])
        grown_root, grown = grow(result['model_root'], store, 2)
        codec.check_model(grown)
        assert grown['tokenizer_root'] == model['tokenizer_root']
    finally:
        pipe.close()


def test_complete_text_epoch_uses_document_evaluation_and_keeps_decision_on_restart(seed,tmp_path):
    store,root,_=seed
    codec=TextCodec(tokenizer(),store)
    bound,_=bind_model(store,root,codec)
    class FixtureCorpus(TextCorpus):
        def collect(self,source_id,count=128,rows=None):
            cursor=self.db.execute('SELECT cursor FROM sources WHERE id=?',(source_id,)).fetchone()[0]
            offset=0 if source_id==train else 10000
            return super().collect(source_id,count,rows=[conversation(offset+cursor+i) for i in range(count)])
    corpus=FixtureCorpus(tmp_path/'corpus',store,codec,32)
    train=corpus.register(source())
    heldout=corpus.register(source('heldout'))
    def factory(root,name,journal):
        endpoints=[LocalEndpoint(Worker(tmp_path/f'epoch-worker{i}',store)) for i in range(2)]
        return Pipeline(store,root,endpoints,[6000]*2,name,journal=journal)
    options=dict(steps=1,examples=32,documents=4,epochs_per_day=1,capacities=(6000,6000),clock=lambda:1.)
    epochs=Epochs(tmp_path/'epochs',store,corpus,factory,bound,train,heldout,**options)
    result=epochs.run_once()
    assert result['status'] in ('accepted','rejected')
    assert result['options']['tokenizer_root']==codec.root
    for role,reservation in result['evaluation'].items():
        reserved=store.json(reservation)
        assert len(reserved['documents'])==32 and len(reserved['sequences'])==96
        for name in ('baseline','candidate'):
            assert len(result['measurements'][name][role])==32
    accepted=result['candidate'] if result['decision']['promote'] else bound
    assert epochs.accepted_root==accepted
    epochs.db.close()
    resumed=Epochs(tmp_path/'epochs',store,corpus,factory,bound,train,heldout,**options)
    assert resumed.accepted_root==accepted
    assert resumed.run_once()['status']=='period_budget_complete'
