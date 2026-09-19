"""Immutable structured-conversation windows for the native cohort preparer.

A feed head is a discovery hint, never admission approval. Operators pin its
previous head and allowed source publishers; curators still retokenize, review
provenance/contamination and vote. Objects use the existing local or S3 store.
No renderer, tokenizer or network-supplied program runs in this transport.
"""
import copy
import fcntl
from pathlib import Path

from neuroshard.dataflow.store import LocalStore, S3Store, canonical, digest
from neuroshard.demo.protocol import parse_json
from . import cohorts
from .schema import integer, root
from .reference_data import save

FORMAT = 'neuroshard-expert-conversation-feed-v1'
WINDOW = FORMAT+'/window'
MAX_BYTES = 8*1024**2


def checked(store, key, maximum):
    root(key)
    # Bound transport reads before allocating the payload, including a malicious
    # object stored under a perfectly valid hash for much larger bytes.
    if isinstance(store, LocalStore):
        with store.path(key).open('rb') as source:
            raw = source.read(maximum+1)
    elif isinstance(store, S3Store):
        response = store.client.get_object(Bucket=store.bucket, Key=store.key(key))
        body = response['Body']
        try:
            if response.get('ContentLength', 0) > maximum:
                raise ValueError('Conversation object exceeds its transport bound')
            raw = body.read(maximum+1)
        finally:
            body.close()
    else:
        raise TypeError('Require the bounded local or S3 conversation transport')
    if not isinstance(raw, bytes) or len(raw) > maximum or digest(raw) != key:
        raise ValueError('Conversation object differs from its bounded commitment')
    return raw


def record(value, role):
    fields = {'messages'} | ({'stratum', 'topics', 'answers'} if role == 'heldout' else set())
    ordinary = isinstance(value, dict) and role == 'heldout' and 'quality_format' in value
    if ordinary:
        fields |= {'quality_format'} | ({'answer_aliases', 'case_sensitive'} & set(value))
    if not isinstance(value, dict) or set(value) != fields or len(canonical(value)) > 256*1024:
        raise ValueError('Require bounded structured source conversations and declared scoring metadata')
    messages = value['messages']
    if not isinstance(messages, list) or not 2 <= len(messages) <= 128:
        raise ValueError('Require complete structured conversations')
    expected = 'user'
    for index, message in enumerate(messages):
        if (not isinstance(message, dict) or set(message) != {'role', 'content'}
                or not isinstance(message['content'], str) or not message['content'].strip()):
            raise ValueError('Require explicit roles and nonempty conversation text')
        if index == 0 and message['role'] == 'system':
            continue
        if message['role'] != expected:
            raise ValueError('Conversation roles must alternate')
        expected = 'assistant' if expected == 'user' else 'user'
    if messages[-1]['role'] != 'assistant':
        raise ValueError('Complete the final assistant turn')
    if role == 'heldout':
        from . import cohort_questions, ordinary_quality
        from .data import document_identity
        rows = [{**value, 'id': document_identity(messages)}]
        if ordinary:
            ordinary_quality.validate_rows(rows)
        else:
            cohort_questions.validate_rows(rows, release_scope=False)
    return value


def publish_window(store, source, start, rows):
    source = copy.deepcopy(cohorts.source(source))
    if source['role'] not in ('train', 'heldout'):
        raise ValueError('Publish explicit training or held-out source roles')
    integer(start, 0, 2**53-1)
    if not isinstance(rows, list) or not 1 <= len(rows) <= 2048:
        raise ValueError('Publish a bounded complete source window')
    integer(start+len(rows), 1, 2**53-1)
    payload = bytearray()
    for row in rows:
        raw = canonical(record(row, source['role']))+b'\n'
        if len(payload)+len(raw) > MAX_BYTES:
            raise ValueError('Source window exceeds the publication bound')
        payload.extend(raw)
    raw = bytes(payload)
    window = {'format': WINDOW, 'source': source, 'start': start, 'end': start+len(rows),
              'records': {'sha256': store.put(raw), 'bytes': len(raw), 'count': len(rows)}}
    return store.put(canonical(window))


def window(store, key):
    value = parse_json(checked(store, key, 4096))
    if set(value) != {'format', 'source', 'start', 'end', 'records'} or value['format'] != WINDOW:
        raise ValueError('Invalid immutable conversation window')
    cohorts.source(value['source'])
    if value['source']['role'] not in ('train', 'heldout'):
        raise ValueError('Invalid source role')
    integer(value['start'], 0, 2**53-1)
    integer(value['end'], value['start']+1, min(2**53-1, value['start']+2048))
    spec = value['records']
    if set(spec) != {'sha256', 'bytes', 'count'}:
        raise ValueError('Commit complete record bytes and count')
    root(spec['sha256'])
    integer(spec['bytes'], 1, MAX_BYTES)
    integer(spec['count'], value['end']-value['start'], value['end']-value['start'])
    return value


class Feed:
    def __init__(self, store, head):
        self.store = store
        self.head, self.manifest, self.windows = None, None, None
        self._load(head)

    def _load(self, head):
        manifest = parse_json(checked(self.store, head, 128*1024))
        if (set(manifest) != {'format', 'previous', 'windows'} or manifest['format'] != FORMAT
                or not isinstance(manifest['windows'], list) or not 1 <= len(manifest['windows']) <= 1024
                or len(set(manifest['windows'])) != len(manifest['windows'])):
            raise ValueError('Require a bounded distinct immutable window inventory')
        if manifest['previous'] is not None:
            root(manifest['previous'])
        if self.head is not None and (manifest['previous'] != self.head
                or manifest['windows'][:len(self.manifest['windows'])] != self.manifest['windows']
                or len(manifest['windows']) <= len(self.manifest['windows'])):
            raise ValueError('A feed update must extend the pinned previous inventory')
        windows = [window(self.store, key) for key in manifest['windows']]
        ends = {}
        for value in windows:
            source = digest(canonical(value['source']))
            if value['start'] != ends.get(source, 0):
                raise ValueError('Source windows must cover consecutive rows without overlap')
            ends[source] = value['end']
        self.head, self.manifest, self.windows = head, manifest, windows

    def advance(self, head):
        if head != self.head:
            self._load(head)

    def __call__(self, source, start, count):
        """Read the exact native source cursor; missing bytes never skip rows."""
        cohorts.source(source)
        integer(start, 0, 2**53-1)
        integer(count, 1, 2048)
        end = integer(start+count, 1, 2**53-1)
        result = []
        for value in self.windows:
            if value['source'] != source or value['end'] <= start or value['start'] >= end:
                continue
            spec = value['records']
            raw = checked(self.store, spec['sha256'], MAX_BYTES)
            if len(raw) != spec['bytes']:
                raise ValueError('Source window byte count changed')
            rows = [parse_json(line) for line in raw.splitlines()]
            if len(rows) != spec['count']:
                raise ValueError('Source window row count changed')
            for row in rows:
                record(row, source['role'])
            result.extend(rows[max(0, start-value['start']):min(len(rows), end-value['start'])])
        if len(result) != count:
            raise OSError('The pinned feed does not yet supply the entire requested source window')
        return result


def append(store, previous, additions):
    if not isinstance(additions, list) or not additions:
        raise ValueError('Append at least one immutable window')
    windows = Feed(store, previous).manifest['windows'] if previous is not None else []
    manifest = {'format': FORMAT, 'previous': previous, 'windows': windows+additions}
    key = store.put(canonical(manifest))
    Feed(store, key)  # Do not announce malformed, overlapping or missing windows.
    return key


def collect(home, store, source, upstream, *, count=128):
    """Publish the next source window, recovering upload-before-journal crashes.

    ``upstream`` is an operator-installed reader of the pinned source revision.
    The journal advances only after both immutable objects are available. Its
    cursor records publication; native admission keeps its separate cursor.
    """
    source = copy.deepcopy(cohorts.source(source))
    integer(count, 1, 2048)
    if not callable(upstream):
        raise ValueError('Require a locally configured pinned upstream reader')
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    with (home/'conversation-feed.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = home/'conversation-feed.json'
        journal = parse_json(path.read_bytes()) if path.exists() else {'format': FORMAT, 'head': None}
        if set(journal) != {'format', 'head'} or journal['format'] != FORMAT:
            raise ValueError('Conversation publisher journal changed')
        previous = journal['head']
        feed = Feed(store, previous) if previous is not None else None
        start = max((value['end'] for value in feed.windows if value['source'] == source), default=0) if feed else 0
        rows, buffered = [], 0
        iterator = iter(upstream(source, start, count))
        try:
            for row in iterator:
                if len(rows) >= count:
                    raise ValueError('Upstream exceeded its requested publication window')
                record(row, source['role'])
                buffered += len(canonical(row))+1
                if buffered > MAX_BYTES:
                    raise ValueError('Source window exceeds the publication bound')
                rows.append(row)
        finally:
            close = getattr(iterator, 'close', None)
            if close:
                close()
        if not rows:
            return {'head': previous, 'source': digest(canonical(source)), 'cursor': start,
                    'status': 'no_new_rows', 'native_admitted': False}
        key = publish_window(store, source, start, rows)
        head = append(store, previous, [key])
        save(path, {'format': FORMAT, 'head': head})
        return {'head': head, 'source': digest(canonical(source)), 'cursor': start+len(rows),
                'window': key, 'status': 'window_published', 'native_admitted': False}
