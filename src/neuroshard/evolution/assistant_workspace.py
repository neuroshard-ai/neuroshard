"""Deterministic, in-memory workspace for complete assistant workflows.

Only declared tools execute. The sandbox has no network/filesystem effects and
receives no evaluation goals. Expected outcomes belong to the separate scorer.
"""

import copy
from datetime import date, timedelta
import json
import re

from neuroshard.evolution.granite_reference import strict_json
from neuroshard.evolution.modular_tools import _check_value
from neuroshard.evolution.modular_reference_execution import identity


def tool(name, description, properties):
    return {'type': 'function', 'function': {'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': properties,
                       'required': list(properties), 'additionalProperties': False}}}


STRING = {'type': 'string'}
INTEGER = {'type': 'integer'}
DRAFT_FIELDS = {'project': STRING, 'recipient': STRING, 'due_date': STRING,
                'total': INTEGER, 'source_ids': {'type': 'array', 'items': STRING}}
TOOLS = [
    tool('list_documents', 'List document metadata for the exact project name; content requires read_document.',
         {'project': STRING}),
    tool('read_document', 'Read a document by its ID from the workspace.', {'document_id': STRING}),
    tool('calculate', 'Perform exact integer addition, subtraction or multiplication.',
         {'operation': {'type': 'string', 'enum': ['add', 'subtract', 'multiply']},
          'left': INTEGER, 'right': INTEGER}),
    tool('shift_date', 'Shift an ISO YYYY-MM-DD date by a signed number of calendar days.',
         {'start_date': STRING, 'days': INTEGER}),
    tool('save_draft', 'Create or replace this project draft. This saves a local draft only; nothing is sent. '
         'Cite all documents used by ID; each must have been read in this conversation.', DRAFT_FIELDS),
]
REGISTRY = {item['function']['name']: item['function']['parameters'] for item in TOOLS}


def parse_calls(text, registry=REGISTRY):
    """Native Granite tool envelope; never evaluate generated code or repair JSON."""
    if not isinstance(text, str) or len(text.encode()) > 16384:
        raise ValueError('invalid response size')
    if '<tool_call' not in text and '</tool_call>' not in text:
        return []
    matches = list(re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL))
    if not 1 <= len(matches) <= 2 or re.sub(r'<tool_call>\s*.*?\s*</tool_call>', '', text, flags=re.DOTALL).strip():
        raise ValueError('require one or two complete tool calls without extra text')
    calls = []
    for match in matches:
        call = strict_json(match[1])
        if (not isinstance(call, dict) or set(call) != {'name', 'arguments'}
                or not isinstance(call['name'], str) or call['name'] not in registry):
            raise ValueError('unknown tool or call fields')
        _check_value(call['arguments'], registry[call['name']])
        calls.append(call)
    return calls


def iso_date(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError('date must be YYYY-MM-DD')
    return date.fromisoformat(value)


class Workspace:
    registry = REGISTRY

    def __init__(self, public_world):
        if set(public_world) != {'documents'}:
            raise ValueError('workspace accepts public documents only, not scoring metadata')
        self.documents = {}
        for document in copy.deepcopy(public_world['documents']):
            if set(document) != {'id', 'project', 'title', 'revision', 'status', 'content'}:
                raise ValueError('invalid document inventory')
            if (not isinstance(document['id'], str) or not document['id'] or document['id'] in self.documents
                    or document['status'] not in ('approved', 'draft') or type(document['revision']) is not int
                    or not isinstance(document['content'], str) or len(document['content'].encode()) > 8192):
                raise ValueError('invalid document identity or content')
            self.documents[document['id']] = document
        if not 1 <= len(self.documents) <= 32:
            raise ValueError('workspace document bound exceeded')
        self.read_ids = set()
        self.drafts = {}
        self.events = []
        self.world_root = identity(public_world)

    def snapshot(self):
        return {'world_root': self.world_root, 'read_ids': sorted(self.read_ids),
                'drafts': copy.deepcopy(self.drafts), 'events': copy.deepcopy(self.events)}

    def execute(self, call):
        before = identity(self.snapshot())
        try:
            if set(call) != {'name', 'arguments'} or call['name'] not in self.registry:
                raise ValueError('unknown tool')
            _check_value(call['arguments'], self.registry[call['name']])
            result = self._apply(call['name'], call['arguments'])
        except (ValueError, KeyError, OverflowError, TypeError):
            result = {'error': 'invalid tool arguments or unavailable workspace object'}
        self.events.append({'call': copy.deepcopy(call), 'result': copy.deepcopy(result), 'before_root': before})
        return result

    def _apply(self, name, args):
        if name == 'list_documents':
            return {'documents': [{k: d[k] for k in ('id', 'project', 'title', 'revision', 'status')}
                                 for d in sorted(self.documents.values(), key=lambda d: d['id'])
                                 if d['project'] == args['project']]}
        if name == 'read_document':
            document = self.documents[args['document_id']]
            self.read_ids.add(document['id'])
            return copy.deepcopy(document)
        if name == 'calculate':
            left, right = args['left'], args['right']
            if max(abs(left), abs(right)) > 1000000:
                raise ValueError('integer input bound exceeded')
            value = {'add': lambda: left + right, 'subtract': lambda: left - right,
                     'multiply': lambda: left * right}[args['operation']]()
            return {'value': value}
        if name == 'shift_date':
            if abs(args['days']) > 365:
                raise ValueError('date offset bound exceeded')
            return {'date': (iso_date(args['start_date']) + timedelta(days=args['days'])).isoformat()}
        if name == 'save_draft':
            if (not 1 <= len(args['project']) <= 120 or not 1 <= len(args['recipient']) <= 120
                    or not 0 <= args['total'] <= 1000000000000
                    or not 1 <= len(args['source_ids']) <= 4
                    or len(set(args['source_ids'])) != len(args['source_ids'])
                    or not set(args['source_ids']) <= self.read_ids):
                raise ValueError('invalid draft fields or unread evidence')
            iso_date(args['due_date'])
            if not any(d['project'] == args['project'] for d in self.documents.values()):
                raise ValueError('project is absent from this workspace')
            old = self.drafts.get(args['project'])
            revision = old['revision'] + 1 if old else 1
            draft = {**copy.deepcopy(args), 'source_ids': sorted(args['source_ids']), 'revision': revision}
            self.drafts[args['project']] = draft
            return {'saved': True, 'draft': copy.deepcopy(draft)}
        raise ValueError('unsupported tool')


def score_round(snapshot, expected, terminated, final_text):
    """Outcome scoring, not a prescribed tool trace or exact prose template."""
    drafts = snapshot['drafts']
    actual = drafts.get(expected['project'])
    if not terminated or not isinstance(final_text, str) or not final_text.strip() or actual is None:
        return False
    # Additional or wrong-project writes must not be hidden by one correct draft.
    if set(drafts) != {expected['project']}:
        return False
    projected = {key: actual[key] for key in DRAFT_FIELDS}
    return identity(projected) == identity({**expected, 'source_ids': sorted(expected['source_ids'])})


def replay_transcript(public_world, calls):
    workspace = Workspace(public_world)
    for row in calls:
        if workspace.execute(row['call']) != row['result']:
            raise ValueError('tool transcript result differs from deterministic execution')
    return workspace.snapshot()
