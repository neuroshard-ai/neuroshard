import gzip
import hashlib
import json

import pytest

from neuroshard.evolution import assistant_chat_data as chat
from neuroshard.evolution import assistant_documents_data as docs
from neuroshard.evolution import assistant_documents_demonstration as demo
from neuroshard.evolution import assistant_general_l4 as l4
from neuroshard.evolution import assistant_phrasings as phrasings
from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256


def committed():
    return read(ROOT / phrasings.FILE)


def test_every_committed_template_is_valid_and_judged():
    data = committed()
    raw = gzip.decompress((ROOT / phrasings.FILE).with_suffix('.responses.json.gz').read_bytes()).decode().rstrip('\n')
    assert hashlib.sha256(raw.encode()).hexdigest() == data['responses_sha256']
    record = json.loads(raw)
    kept = {(j['route'], j['split'], j['template']) for j in record['judging'] if j['kept']}
    for route, by_split in data['sets'].items():
        for split, by_kind in by_split.items():
            source = 'train' if split == 'development' else split
            assert set(by_kind) == set(phrasings.table(route))
            for kind, rows in by_kind.items():
                assert rows and len(rows) == len(set(rows))
                names = phrasings.table(route)[kind][1]
                assert all(phrasings.valid(t, names) and (route, source, t) in kept for t in rows)


def test_training_development_and_confirmation4_templates_are_disjoint():
    for route, sets in committed()['sets'].items():
        train, development, test = sets['train'], sets['development'], sets['confirmation4']
        assert phrasings.disjoint(train, test)[0]
        for kind in train:
            assert not set(train[kind]) & set(development[kind])
        assert not {t for rows in development.values() for t in rows} & {t for rows in test.values() for t in rows}


def test_confirmation4_shares_no_span_with_any_authored_split():
    authored = set()
    for split in ('train', 'development', 'confirmation', 'confirmation2', 'confirmation3'):
        authored |= {t['user'] for c in docs.cases(split) + chat.cases(split) for t in c['turns']}
    seen = set().union(*(phrasings.spans(t) for t in authored))
    for route in ('documents', 'chat'):
        for rows in committed()['sets'][route]['confirmation4'].values():
            assert all(not phrasings.spans(t) & seen for t in rows)


@pytest.mark.parametrize('split,docs_n,chat_n', [('generated-train', 660, 320), ('generated-development', 66, 48),
                                                ('confirmation4', 132, 96)])
def test_generated_splits_have_distinct_requests_and_our_own_expected_outcomes(split, docs_n, chat_n):
    documents, chats = docs.cases(split), chat.cases(split)
    assert len(documents) == docs_n and len(chats) == chat_n
    assert len({c['turns'][0]['user'] for c in documents}) == docs_n
    assert len({c['turns'][0]['user'] for c in chats}) == chat_n
    for case in documents:
        assert '{' not in case['turns'][0]['user']
        expected = case['turns'][0]['expected']
        assert expected.get('not_found') or (expected['facts'] and expected['source_ids'])
    for case in chats:
        assert '{' not in case['turns'][0]['user'] and case['turns'][0]['expected']['constraints']


def test_generated_arithmetic_values_follow_the_declared_formula():
    for case in chat.cases('confirmation4') + chat.cases('generated-train'):
        if case['family'] == 'arithmetic':
            value = case['turns'][0]['expected']['constraints'][0]['value']
            assert isinstance(value, int) and value > 0


def test_ending_strips_emphasis_only_where_declared():
    strip = {'kind': 'ends_with', 'phrase': 'Go for it', 'markup': 'strip'}
    plain = {'kind': 'ends_with', 'phrase': 'Go for it'}
    assert chat.check(strip, 'Try it today. **Go for it**.') and not chat.check(plain, 'Try it today. **Go for it**.')
    assert chat.check(strip, 'Try it. "Go for it!"') and not chat.check(strip, 'Go for it, then rest.')


def test_every_generated_training_case_has_a_verified_demonstration():
    rows = demo.all_demonstrations(l4.policies(l4.declared('assistant-general-l5-gpu'))['documents'], 'generated-train')
    assert len(rows) == 660 and all(row['passed'] for _, row in rows)
    for split in ('generated-development', 'confirmation4'):
        with pytest.raises(ValueError):
            demo.texts(docs.cases(split)[0])


def test_l5_declares_generated_splits_and_pins_its_contract():
    profile = 'assistant-general-l5-gpu'
    declaration_path, execution_path = l4.PROFILES[profile]
    declaration, execution = read(ROOT / declaration_path), read(ROOT / execution_path)
    assert l4.splits(declaration) == ('generated-train', 'generated-train',
                                      {'documents': 'generated-development', 'chat': 'generated-development'})
    assert l4.splits(l4.declared()) == ('train', 'train', {'documents': 'development', 'chat': 'development'})
    assert all(sha256(ROOT / name) == digest for name, digest in execution['contracts'].items())
    assert phrasings.FILE in execution['sources'] and 'src/neuroshard/evolution/assistant_phrasings.py' in execution['sources']
    assert not declaration['opens_confirmation'] and declaration['budget_usd'] <= 10
    pinned = read(ROOT / 'config/experiments/assistant-growth-execution.json')
    keys = ('packages', 'python', 'gpus', 'environment', 'worker_environment', 'tokenizer_pipeline_sha256', 'execution')
    assert all(execution[key] == pinned[key] for key in keys)


def test_parse_and_judge_reject_malformed_or_wrong_answers():
    names = ['city']
    text = 'Sure:\n["When do I fly back from {city}?", "Flight on day 3 from {city}?", "no placeholder", 4]'
    assert phrasings.parse(text, names) == ['When do I fly back from {city}?']
    assert phrasings.parse('not json', names) == []

    class Fake:
        def __init__(self, replies):
            self.replies = iter(replies)

        def converse(self, **_):
            return {'output': {'message': {'content': [{'text': next(self.replies)}]}}, 'usage': {}}

    template = 'I bought {a} at {b} each and returned {c}; total spent? End with the number.'
    ok, _ = phrasings.judge('chat', 'arith-refund', template, Fake(['fine\nVERDICT: YES', '0'] * 2), 1)
    assert not ok
    ok, _ = phrasings.judge('chat', 'ending', 'Cheer me on about {topic}, ending with "{phrase}".',
                            Fake(['ok\nVERDICT: YES']), 1)
    assert ok
    empty = type('E', (), {'converse': lambda self, **_: {'output': {'message': {'content': []}}, 'usage': {}}})()
    assert not phrasings.judge('chat', 'ending', 'Cheer me on about {topic}, ending with "{phrase}".', empty, 1)[0]
