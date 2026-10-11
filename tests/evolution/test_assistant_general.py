import json
import subprocess
import sys

import pytest

from neuroshard.evolution import assistant_chat_data as chat
from neuroshard.evolution import assistant_documents as documents
from neuroshard.evolution import assistant_documents_data as docs
from neuroshard.evolution import assistant_general as general
from neuroshard.evolution import assistant_general_baseline as baseline
from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256

from test_assistant_growth_baseline import cloud_module


def policy(name):
    return read(ROOT / baseline.POLICIES[name])


def reply(text):
    return {'text': text, 'terminated': True, 'executed': True, 'input_token_ids': [1], 'token_ids': [2]}


def solve(case):
    """A scripted correct episode: search, read the cited documents, submit or report."""
    expected = case['turns'][0]['expected']
    texts = ['<tool_call>{"name": "search_documents", "arguments": {"query": %s}}</tool_call>'
             % json.dumps(case['turns'][0]['user'])]
    if expected.get('not_found'):
        texts.append('<tool_call>{"name": "report_not_found", "arguments": {"reason": "not in the documents"}}'
                     '</tool_call>')
    else:
        for key in expected['source_ids']:
            texts.append('<tool_call>{"name": "read_document", "arguments": {"document_id": "%s"}}</tool_call>' % key)
        texts.append('<tool_call>{"name": "submit_answer", "arguments": {"answer": %s, "source_ids": %s}}</tool_call>'
                     % (json.dumps(' and '.join(expected['facts'])), json.dumps(expected['source_ids'])))
    return texts + ['Done.']


def test_every_documents_case_is_solvable_and_scored_from_its_transcript():
    for split in docs.SPLITS:
        for case in docs.cases(split):
            texts = iter(solve(case))
            row = general.documents_episode(case, lambda m, t: reply(next(texts)), policy('documents'))
            assert row['passed'], case['id']
            assert general.rescore([case], [row], 'documents') == 1


def test_documents_scoring_rejects_wrong_facts_sources_and_guesses():
    case = next(c for c in docs.cases('development') if c['family'] == 'invoice-sum')
    expected = case['turns'][0]['expected']
    world = documents.Workspace(case['world'])
    for key in expected['source_ids']:
        world.execute({'name': 'read_document', 'arguments': {'document_id': key}})
    world.execute({'name': 'submit_answer', 'arguments': {'answer': 'You owe 1.', 'source_ids': expected['source_ids']}})
    assert not documents.score_round(world.snapshot(), expected, True, 'ok')
    world.execute({'name': 'submit_answer', 'arguments': {'answer': expected['facts'][0],
                                                         'source_ids': expected['source_ids'][:1]}})
    assert not documents.score_round(world.snapshot(), expected, True, 'ok')
    unread = documents.Workspace(case['world'])
    assert 'error' in unread.execute({'name': 'submit_answer', 'arguments': {'answer': 'x', 'source_ids': ['doc-x']}})
    missing = next(c for c in docs.cases('development') if c['family'] == 'not-found-insurance')
    guess = documents.Workspace(missing['world'])
    guess.execute({'name': 'read_document', 'arguments': {'document_id': missing['world']['documents'][0]['id']}})
    guess.execute({'name': 'submit_answer', 'arguments': {'answer': 'Policy 123',
                                                         'source_ids': [missing['world']['documents'][0]['id']]}})
    assert not documents.score_round(guess.snapshot(), missing['turns'][0]['expected'], True, 'ok')


def test_search_is_deterministic_and_finds_the_current_document_first_for_its_title_words():
    case = next(c for c in docs.cases('development') if c['family'] == 'flight-time')
    world = documents.Workspace(case['world'])
    city = next(d['title'] for d in case['world']['documents'] if d['title'].startswith('Itinerary ')
                and 'superseded' not in d['title']).split()[1]
    first, second = world.search(f'current itinerary {city}'), world.search(f'current itinerary {city}')
    assert first == second and first['results'][0]['id'] == case['turns'][0]['expected']['source_ids'][0]
    with pytest.raises(ValueError):
        world.search('zzzz qqqq')


def test_chat_constraints_are_exact():
    assert chat.check({'kind': 'bullets', 'n': 3}, '- a\n- b\n- c') and not chat.check({'kind': 'bullets', 'n': 3}, '- a')
    assert chat.check({'kind': 'json_keys', 'keys': ['a']}, '{"a": 1}') and not chat.check(
        {'kind': 'json_keys', 'keys': ['a']}, 'Here: {"a": 1}')
    assert chat.check({'kind': 'max_words', 'n': 3}, 'one two three') and not chat.check(
        {'kind': 'max_words', 'n': 3}, 'one two three four')
    assert chat.check({'kind': 'number', 'value': 126}, 'So the total is 126.')
    assert not chat.check({'kind': 'number', 'value': 126}, '126 minus 6 is 120.')
    assert chat.check({'kind': 'exclude', 'word': 'the'}, 'There they go.') and not chat.check(
        {'kind': 'exclude', 'word': 'the'}, 'See the cat.')
    assert chat.check({'kind': 'lowercase'}, 'all small') and not chat.check({'kind': 'lowercase'}, 'Not small')


def test_splits_are_disjoint_and_the_confirmation_wording_is_held_out():
    for module in (docs, chat):
        texts = {s: {t['user'] for c in module.cases(s) for t in c['turns']} for s in module.SPLITS}
        assert not texts['confirmation'] & (texts['train'] | texts['development'])
        assert len({c['id'] for s in module.SPLITS for c in module.cases(s)}) == sum(
            len(module.cases(s)) for s in module.SPLITS)
    assert set(baseline.feature_texts()) == {t['user'] for m in (docs, chat) for s in ('train', 'development')
                                             for c in m.cases(s) for t in c['turns']}


def test_the_general_route_runs_through_the_granite_chat_template(tmp_path):
    torch = pytest.importorskip('torch')
    from test_assistant_experience import TEMPLATE, tiny_model
    from test_granite_tokenizer import granite_like, load_tiny
    from neuroshard.evolution.assistant_serving import cached_responder

    directory = granite_like(tmp_path)
    (directory / 'chat_template.jinja').write_text(TEMPLATE)
    tokenizer = load_tiny(directory)[0]
    model = tiny_model(tokenizer, seed=1)
    small = {**policy('documents'), 'generation': {'max_input_tokens': 6144, 'max_new_tokens': 4}}
    row = general.documents_episode(docs.cases('development')[0], cached_responder(model, tokenizer, small), small)
    assert not row['passed'] and row['generations'] and general.rescore(docs.cases('development')[:1], [row],
                                                                         'documents') == 0
    talk = {**policy('chat'), 'generation': {'max_input_tokens': 4096, 'max_new_tokens': 4}}
    answer = general.chat_episode(chat.cases('development')[0], cached_responder(model, tokenizer, talk), talk)
    assert general.rescore(chat.cases('development')[:1], [answer], 'chat') == int(answer['passed'])
    del torch


def test_execution_pins_sources_and_one_cpu_host_within_its_allowance():
    execution = read(ROOT / baseline.EXECUTION)
    for name, digest in execution['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert baseline.SCRIPT in execution['sources'] and not execution['training_authorized']
    probe = ('import os, sys; import neuroshard.evolution.assistant_general_baseline as b; assert "torch" not in sys.modules; '
             'import neuroshard.evolution.assistant_general, neuroshard.evolution.assistant_growth_round4, '
             'neuroshard.evolution.assistant_growth_run, neuroshard.evolution.granite_reference, '
             'neuroshard.evolution.granite_tokenizer, neuroshard.evolution.router_scaling_granite, '
             'neuroshard.evolution.assistant_serving, neuroshard.evolution.assistant_workflow_canonical; b.feature_texts(); '
             'root = os.path.abspath("src"); print("\\n".join(sorted(os.path.relpath(x.__file__) '
             'for x in list(sys.modules.values()) if getattr(x, "__file__", None) and '
             'os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(execution['sources']), sorted(set(imported) - set(execution['sources']))
    for path in baseline.POLICIES.values():
        assert path in execution['sources']
    cloud = cloud_module()
    resources = cloud.resources(baseline.PROFILE)
    assert not resources['gpu'] and 'upload' not in resources
    assert (execution['prepare_seconds'] + execution['worker_seconds'] + resources['setup_seconds']
            + resources['copy_seconds'] + 600 <= resources['hours'] * 3600)
    assert cloud.remote_command(baseline.PROFILE)[1].endswith(baseline.SCRIPT)
