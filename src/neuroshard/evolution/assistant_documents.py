"""A workspace over the user's own documents: search, read, compute, and a cited answer or an explicit not-found.

This is the general route's interface. Unlike the drafting and scheduling grammars, the documents
are free text with no project/revision schema, and the outcome is a structured answer the user
can check: the answer text and the IDs of the documents it relies on, or a statement that the
documents do not contain the answer. Nothing leaves the in-memory workspace.

Only this module and its scorer know the interface; release-pinned modules are not changed.
"""

import copy
import math
import re
import unicodedata

from neuroshard.evolution import assistant_workspace as base
from neuroshard.evolution.modular_reference_execution import identity

INTERFACES = ('workspace-documents/1', 'workspace-documents/2')
INTERFACE = INTERFACES[-1]
STRING, INTEGER = base.STRING, base.INTEGER
MAX_DOCUMENTS, MAX_BYTES, TOP = 64, 16384, 5
TOOLS = [
    base.tool('search_documents', 'Search the user\'s documents by keywords; returns up to five IDs, titles and '
              'snippets. Read a document before relying on it.', {'query': STRING}),
    base.tool('read_document', 'Read the full text of one document by its ID.', {'document_id': STRING}),
    base.TOOLS[2],  # calculate
    base.TOOLS[3],  # shift_date
    base.tool('submit_answer', 'Give the final answer to the user\'s question, citing the IDs of every document '
              'it relies on; each must have been found or read in this conversation.',
              {'answer': STRING, 'source_ids': {'type': 'array', 'items': STRING}}),
    base.tool('report_not_found', 'Say that the documents do not contain the answer, after searching them.',
              {'reason': STRING}),
]
REGISTRY = {item['function']['name']: item['function']['parameters'] for item in TOOLS}
WORD = re.compile(r'[a-z0-9]+')
CITATION = re.compile(r'doc-[0-9a-f]{8}')
HYPHENS = dict.fromkeys(map(ord, '\u2010\u2011\u2012\u2013\u2014\u2212'), '-')


def parse_calls(text):
    return base.parse_calls(text, REGISTRY)


def words(text):
    return WORD.findall(text.lower())


class Workspace:
    registry = REGISTRY

    def __init__(self, public_world, version=2):
        if version not in (1, 2):
            raise ValueError('unknown documents workspace version')
        self.version = version
        if set(public_world) != {'documents'}:
            raise ValueError('workspace accepts public documents only, not scoring metadata')
        self.documents = {}
        for document in copy.deepcopy(public_world['documents']):
            if set(document) != {'id', 'title', 'content'}:
                raise ValueError('a document has exactly an id, a title and content')
            if (not isinstance(document['id'], str) or not document['id'] or document['id'] in self.documents
                    or not isinstance(document['title'], str) or not isinstance(document['content'], str)
                    or len(document['content'].encode()) > MAX_BYTES):
                raise ValueError('invalid document identity or content')
            self.documents[document['id']] = document
        if not 1 <= len(self.documents) <= MAX_DOCUMENTS:
            raise ValueError('workspace document bound exceeded')
        terms = {key: words(d['title'] + ' ' + d['content']) for key, d in self.documents.items()}
        self.lengths = {key: len(value) for key, value in terms.items()}
        self.counts = {key: {} for key in terms}
        frequency = {}
        for key, value in terms.items():
            for term in value:
                self.counts[key][term] = self.counts[key].get(term, 0) + 1
            for term in set(value):
                frequency[term] = frequency.get(term, 0) + 1
        self.idf = {term: math.log(1 + (len(terms) - n + 0.5) / (n + 0.5)) for term, n in frequency.items()}
        self.average = sum(self.lengths.values()) / len(self.lengths)
        self.read_ids, self.found_ids, self.answer, self.not_found, self.events = set(), set(), None, None, []
        self.world_root = identity(public_world)

    def snapshot(self):
        return {'world_root': self.world_root, 'read_ids': sorted(self.read_ids),
                'found_ids': sorted(self.found_ids | self.read_ids), 'answer': copy.deepcopy(self.answer),
                'not_found': copy.deepcopy(self.not_found), 'events': copy.deepcopy(self.events)}

    def search(self, query):
        """BM25 over title and content; ties by ID, so results are deterministic."""
        query_terms = [t for t in dict.fromkeys(words(query)) if t in self.idf]
        if not query_terms or len(query) > 512:
            raise ValueError('a search needs at least one known keyword')
        scored = []
        for key in self.documents:
            score = 0.0
            for term in query_terms:
                count = self.counts[key].get(term, 0)
                if count:
                    norm = count + 1.2 * (1 - 0.75 + 0.75 * self.lengths[key] / self.average)
                    score += self.idf[term] * count * 2.2 / norm
            if score > 0:
                scored.append((-score, key))
        rows = []
        for _, key in sorted(scored)[:TOP]:
            text = self.documents[key]['content']
            first = min((text.lower().find(t) for t in query_terms if t in text.lower()), default=0)
            start = max(0, first - 80)
            rows.append({'id': key, 'title': self.documents[key]['title'], 'snippet': text[start:start + 200]})
        return {'results': rows}

    def execute(self, call):
        before = identity(self.snapshot())
        try:
            if set(call) != {'name', 'arguments'} or call['name'] not in self.registry:
                raise ValueError('unknown tool')
            base._check_value(call['arguments'], self.registry[call['name']])
            result = self._apply(call['name'], call['arguments'])
        except (ValueError, KeyError, OverflowError, TypeError):
            result = {'error': 'invalid tool arguments or unavailable workspace object'}
        self.events.append({'call': copy.deepcopy(call), 'result': copy.deepcopy(result), 'before_root': before})
        return result

    def _apply(self, name, args):
        if name == 'search_documents':
            result = self.search(args['query'])
            self.found_ids.update(row['id'] for row in result['results'])
            return result
        if name == 'read_document':
            document = self.documents[args['document_id']]
            self.read_ids.add(document['id'])
            return copy.deepcopy(document)
        if name in ('calculate', 'shift_date'):
            helper = base.Workspace.__new__(base.Workspace)
            return base.Workspace._apply(helper, name, args)
        if name == 'submit_answer':
            ids = args['source_ids']
            if (not 1 <= len(args['answer']) <= 2000 or not 1 <= len(ids) <= 4 or len(set(ids)) != len(ids)
                    or not set(ids) <= (self.read_ids | self.found_ids if self.version == 2 else self.read_ids)):
                raise ValueError('an answer needs text and distinct sources found or read here')
            self.answer = {'answer': args['answer'], 'source_ids': sorted(ids)}
            self.not_found = None
            return {'submitted': True}
        if name == 'report_not_found':
            if not 1 <= len(args['reason']) <= 500:
                raise ValueError('a not-found report needs a short reason')
            self.not_found, self.answer = {'reason': args['reason']}, None
            return {'reported': True}
        raise ValueError('unsupported tool')


def normalize(text):
    return ' '.join(words(unicodedata.normalize('NFKC', text).translate(HYPHENS).replace(',', '')))


def citations(text):
    """Document IDs a prose answer cites, tolerant of typographic hyphens."""
    return set(CITATION.findall(unicodedata.normalize('NFKC', text).translate(HYPHENS).lower()))


def score_round_v1(snapshot, expected, terminated, final_text):
    """Version 1, as the baseline declared it: only a ``submit_answer`` counts, citing read documents."""
    if not terminated or not isinstance(final_text, str) or not final_text.strip():
        return False
    if expected.get('not_found'):
        return snapshot['not_found'] is not None and snapshot['answer'] is None
    answer = snapshot['answer']
    if answer is None or snapshot['not_found'] is not None:
        return False
    if not all(' '.join(words(f.replace(',', ''))) in ' '.join(words(answer['answer'].replace(',', '')))
               for f in expected['facts']):
        return False
    return set(expected['source_ids']) <= set(answer['source_ids']) <= set(expected['source_ids']) | set(
        expected.get('allowed_extra', ()))


def score_round(snapshot, expected, terminated, final_text, version=2):
    """Every expected fact stated, and exactly the needed sources cited, each found or read in this conversation.

    The answer may be structured (``submit_answer``) or the final prose reply with document IDs in it; a
    structured answer takes precedence. Not-found must be reported with ``report_not_found``, never guessed.
    ``version=1`` applies the baseline's declared rule instead.
    """
    if version == 1:
        return score_round_v1(snapshot, expected, terminated, final_text)
    if not terminated or not isinstance(final_text, str) or not final_text.strip():
        return False
    if expected.get('not_found'):
        return snapshot['not_found'] is not None and snapshot['answer'] is None
    if snapshot['not_found'] is not None:
        return False
    if snapshot['answer'] is not None:
        text, cited = snapshot['answer']['answer'], set(snapshot['answer']['source_ids'])
    else:
        text, cited = final_text, citations(final_text)
    if not all(normalize(fact) in normalize(text) for fact in expected['facts']):
        return False
    allowed = set(expected['source_ids']) | set(expected.get('allowed_extra', ()))
    return set(expected['source_ids']) <= cited <= allowed and cited <= set(snapshot['found_ids'])


def replay_transcript(public_world, calls, version=2):
    workspace = Workspace(public_world, version)
    for row in calls:
        if workspace.execute(row['call']) != row['result']:
            raise ValueError('document tool result differs on replay')
    return workspace.snapshot()
