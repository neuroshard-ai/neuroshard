"""Actual structured feed bytes drive native preparation and independent review."""
import copy
import io

import pytest

from neuroshard.dataflow.store import LocalStore, S3Store, digest
from neuroshard.evolution import expert_preparation, expert_source
from neuroshard.evolution.reference_data import identity
from test_expert_data import prepared
from test_expert_admission import renamed
from test_expert_preparation import arguments


def source():
    return {'repo': 'example/conversations', 'revision': '1'*40, 'split': 'train',
            'license': 'Apache-2.0', 'role': 'train'}


def rows():
    return [{'messages': [{'role': 'user', 'content': 'Question '+str(i)},
                          {'role': 'assistant', 'content': 'Answer '+str(i)}]} for i in range(4)]


def test_append_restart_and_cross_window_read_preserve_original_conversations(tmp_path):
    store = LocalStore(tmp_path/'feed')
    values, spec = rows(), source()
    first = expert_source.publish_window(store, spec, 0, values[:2])
    initial = expert_source.append(store, None, [first])
    reader = expert_source.Feed(store, initial)
    assert reader(spec, 0, 2) == values[:2]
    with pytest.raises(OSError, match='entire requested'):
        reader(spec, 1, 2)
    second = expert_source.publish_window(store, spec, 2, values[2:])
    head = expert_source.append(store, initial, [second])
    reader.advance(head)
    assert reader(spec, 1, 3) == values[1:]
    assert expert_source.Feed(store, head)(spec, 0, 4) == values
    before = copy.deepcopy(reader.manifest)
    with pytest.raises(ValueError, match='pinned previous'):
        reader.advance(initial)
    assert reader.manifest == before
    duplicate = expert_source.publish_window(store, spec, 1, values[:1])
    with pytest.raises(ValueError, match='without overlap'):
        expert_source.append(store, head, [duplicate])


def test_payload_mutation_missing_bytes_and_oversized_s3_reads_never_become_rows(tmp_path):
    store = LocalStore(tmp_path/'feed')
    window = expert_source.publish_window(store, source(), 0, rows())
    head = expert_source.append(store, None, [window])
    reader = expert_source.Feed(store, head)
    key = reader.windows[0]['records']['sha256']
    store.path(key).write_bytes(b'{"messages": []}\n')
    with pytest.raises(ValueError, match='bounded commitment'):
        reader(source(), 0, 1)
    store.path(key).unlink()
    with pytest.raises(FileNotFoundError):
        reader(source(), 0, 1)

    class Client:
        body = io.BytesIO(b'never read')
        def get_object(self, **kwargs):
            return {'ContentLength': 10**9, 'Body': self.body}
    client = Client()
    remote = S3Store('unused', client=client)
    with pytest.raises(ValueError, match='transport bound'):
        expert_source.checked(remote, digest(b'never read'), 4096)
    assert client.body.closed


def test_feed_bytes_drive_preparation_and_a_second_read_checks_the_sealed_job(prepared):
    args, windows, quality = arguments(prepared)
    state, plan, policy, objects, tokenizer, original_reader = args
    home, _, original, _, _, _, _, _, _ = prepared
    published = LocalStore(home/'published-feed')
    roots = []
    for selection in windows:
        spec = selection['source']
        fields = ('messages',) if spec['role'] == 'train' else ('messages', 'stratum', 'topics', 'answers')
        values = [{key: row[key] for key in fields} for row in original_reader(spec, 0, 2)]
        roots.append(expert_source.publish_window(published, spec, 0, values))
    head = expert_source.append(published, None, roots)
    reader = expert_source.Feed(published, head)
    bundle = expert_preparation.prepare(*args[:-1], reader, windows=windows, batch_size=1)
    independent = expert_source.Feed(LocalStore(home/'published-feed'), head)
    inputs = objects.json(bundle['prepared'])
    initial = renamed(original['work']['parent'], original['work']['checkpoint'],
                      identity({'format': plan['format'], 'plan': identity(plan), 'prepared': identity(inputs)}))
    job, report = expert_preparation.seal(state, bundle, initial, original['lifecycle']['candidate_template'],
        quality, policy, objects, tokenizer, independent)
    assert report['mechanical_checks_passed'] and report['upstream_documents'] == 4
    assert job['data'] == bundle['data']
    assert expert_preparation.prepare(*args[:-1], independent, windows=windows, batch_size=1) == bundle


def test_collector_restart_recovers_uploaded_bytes_before_advancing_any_cursor(tmp_path, monkeypatch):
    store, home = LocalStore(tmp_path/'objects'), tmp_path/'publisher'
    values, reads = rows(), []
    def upstream(spec, start, count):
        reads.append((start, count))
        return values[start:start+count]
    original = expert_source.save
    attempted = []
    def die_after_upload(path, value):
        attempted.append(value['head'])
        raise OSError('Publisher process exited before journal acknowledgement')
    monkeypatch.setattr(expert_source, 'save', die_after_upload)
    with pytest.raises(OSError, match='acknowledgement'):
        expert_source.collect(home, store, source(), upstream, count=2)
    monkeypatch.setattr(expert_source, 'save', original)
    first = expert_source.collect(home, store, source(), upstream, count=2)
    assert first['head'] == attempted[0] and first['cursor'] == 2
    second = expert_source.collect(home, store, source(), upstream, count=2)
    assert second['cursor'] == 4 and not second['native_admitted']
    reader = expert_source.Feed(store, first['head'])
    reader.advance(second['head'])
    assert reader(source(), 0, 4) == values
    empty = expert_source.collect(home, store, source(), upstream, count=2)
    assert empty['head'] == second['head'] and empty['status'] == 'no_new_rows'
    assert reads == [(0, 2), (0, 2), (2, 2), (4, 2)]


def test_oversized_upstream_is_closed_without_publishing_a_partial_window(tmp_path, monkeypatch):
    store, home = LocalStore(tmp_path/'objects'), tmp_path/'publisher'
    closed = []
    def upstream(*_):
        try:
            yield from rows()
        finally:
            closed.append(True)
    monkeypatch.setattr(expert_source, 'MAX_BYTES', 128)
    with pytest.raises(ValueError, match='publication bound'):
        expert_source.collect(home, store, source(), upstream, count=4)
    assert closed and not (home/'conversation-feed.json').exists()
    assert not list(store.root.iterdir())
