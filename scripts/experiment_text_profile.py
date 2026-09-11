#!/usr/bin/env python3
"""Exercise a model-bound text codec, a real training step, replay and generation.

The manually written conversations are conformance fixtures, not a quality
benchmark. This run does not promote a model or change the public testnet.
Use --workers-config for separately running workers, including another host.
"""
import argparse
import json
import secrets
import time
from pathlib import Path

from neuroshard.evolution.model import from_pretrained, grow, place
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.pipeline import Pipeline, LocalEndpoint, validate_record
from neuroshard.evolution.text import TextCodec, bind_model, generate
from neuroshard.evolution.batches import from_windows, response_window
from neuroshard.evolution.worker import Worker, replay_trace
from neuroshard.evolution.runtime import check
from transformers import AutoTokenizer


PROBES = [
    'Hello, world!', 'שלום עולם', 'مرحبا بالعالم', '你好，世界',
    '🙂 café e\u0301', 'def f(x):\n    return x + 1',
    'Spaces  and\ttabs\nare preserved.', '\u200fעברית 123 English\u200e',
]
CONVERSATION = [
    {'role': 'user', 'content': 'Why should a decentralized language model version its tokenizer?'},
    {'role': 'assistant', 'content': (
        'Every participant must map the same text to the same token IDs. The model embedding rows '
        'depend on that mapping. Vocabulary, normalization rules, special tokens, and the chat '
        'template therefore belong to one versioned model definition. A dataset should record '
        'which definition produced its tokens. Otherwise two workers can agree on a numerical '
        'array while interpreting it as different text. An upgrade must check compatibility '
        'before activating new model and tokenizer versions together.')},
    {'role': 'user', 'content': 'How should a long response be split for training?'},
    {'role': 'assistant', 'content': (
        'Retain earlier text as context and train each response token once. Mark the end of '
        'a response only at its actual end. A fixed window boundary is not an end of response.')},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--workers-config', type=Path)
    parser.add_argument('--codec-only', action='store_true', help='Only check text semantics; do not load or train model weights')
    args = parser.parse_args()
    args.home.mkdir(parents=True, exist_ok=False)
    store = Objects(args.home/'objects')
    original = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    codec = TextCodec(original, store)
    restored = TextCodec.load(store, codec.root)
    probes = []
    for text in PROBES:
        ids = original.encode(text, add_special_tokens=False)
        restored_ids = restored.tokenizer.encode(text, add_special_tokens=False)
        decoded = restored.tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        assert ids == restored_ids and decoded == text
        probes.append({'text': text, 'token_ids': ids, 'utf8_bytes': len(text.encode()),
                       'tokens': len(ids), 'exact_roundtrip': True})
    current = codec.response_windows(CONVERSATION, maximum=8)
    for messages,generation in ((CONVERSATION,False),(CONVERSATION[:1],True)):
        assert codec._chat(messages,generation)==original.apply_chat_template(
            messages,tokenize=True,add_generation_prompt=generation)
    legacy = response_window(CONVERSATION, original)
    assert not current['truncated']
    assert len({window['assistant_index'] for window in current['windows']}) == 2
    result = {
        'tokenizer_root': codec.root, 'vocabulary': codec.profile['vocabulary'],
        'codec_profile': codec.profile, 'probes': probes,
        'coverage': {k: v for k, v in current.items() if k != 'windows'},
        'legacy_scored_tokens': legacy['response_tokens'],
        'windows': len(current['windows']), 'quality_improvement_claimed': False,
    }
    (args.home/'text.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('probes', 'codec_profile')}) , flush=True)
    if args.codec_only:
        return
    result['numerical_runtime'] = check()
    numerical_root, _ = from_pretrained(args.model_dir, store)
    model_root, model = bind_model(store, numerical_root, codec)
    capacities = [48000000]*3
    count = len(place(model, capacities))
    if args.workers_config:
        from neuroshard.evolution.transport import Endpoint
        config = json.loads(args.workers_config.read_bytes())['workers'][:count]
        endpoints = [Endpoint(w['url'], (args.workers_config.resolve().parent/Path(w['token_file']).expanduser()).read_text().strip(), store) for w in config]
    else:
        endpoints = [LocalEndpoint(Worker(args.home/f'worker{i}', store)) for i in range(count)]
    pipe = Pipeline(store, model_root, endpoints, capacities, 'text-profile-'+secrets.token_hex(8))
    try:
        # Two bounded windows form one numerical batch. Other windows are used
        # for coverage checks; the result never claims they were all trained.
        windows = [store.put_json(window) for window in current['windows'][:2]]
        record = pipe.train(from_windows(store, windows, codec.root))
        result['training'] = record
        result['training_window_roots'] = windows
        result['graph'] = validate_record(store, record['record_root'])
        codec.check_model(pipe.model)
        print(json.dumps({'phase': 'trained', 'seconds': record['elapsed_seconds'], 'model_root': pipe.model_root}), flush=True)
        audits = []
        for trace in record['traces']:
            started = time.monotonic()
            checked = replay_trace(store, trace)
            assert checked['valid']
            audits.append({**checked, 'elapsed_seconds': time.monotonic()-started})
            print(json.dumps({'phase': 'replayed', **audits[-1]}), flush=True)
        result['audits'] = audits
        result['generation'] = generate(pipe, codec, [{'role': 'user', 'content': 'What is the capital of France?'}], max_tokens=8)
        grown_root, grown = grow(pipe.model_root, store, 4)
        codec.check_model(grown)
        result.update(model_root=model_root, numerical_root=numerical_root, parameters=model['parameters'],
                      grown_model_root=grown_root, grown_parameters=grown['parameters'],
                      worker_transport='http' if args.workers_config else 'local-single-process')
    finally:
        pipe.close()
    (args.home/'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'phase': 'complete', 'result': str(args.home/'result.json'), 'generation': result['generation']}), flush=True)


if __name__ == '__main__':
    main()
