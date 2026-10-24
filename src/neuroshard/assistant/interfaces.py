"""Every workspace interface a served route may declare, behind one shape: tools, call parser, workspace.

The drafting and calendar interfaces come from ``assistant_workflow.interface`` unchanged. The general
route adds the user's documents (``workspace-documents/2`` and ``/3``) and chat tools
(``chat-calculator/1`` and ``chat-tools/1``). Every tool runs on the user's device; nothing here
reaches a network or a file.
"""

import copy

from neuroshard.evolution import assistant_workflow as workflow

DOCUMENTS = ('workspace-documents/2', 'workspace-documents/3', 'workspace-documents/4')
CHAT = ('chat-calculator/1', 'chat-tools/1')


class Interface:
    def __init__(self, name, group, tools, parse, factory, ascii_results=False):
        self.name, self.group, self.TOOLS = name, group, tools
        self._parse, self._factory, self.ascii_results = parse, factory, ascii_results

    def result_text(self, result):
        """A tool result as the model sees it, byte for byte as the route's accepted runner rendered it."""
        import json

        return json.dumps(result, ensure_ascii=self.ascii_results, sort_keys=True)

    def parse_calls(self, text):
        return self._parse(text)

    def Workspace(self, world):
        return self._factory(world)


class ChatWorkspace:
    """The chat route's tools as a workspace: a calculator and, under chat-tools/1, check_text."""

    def __init__(self, kit):
        self.tools = kit()

    def execute(self, call):
        return self.tools.execute(call)

    def snapshot(self):
        return {'events': copy.deepcopy(self.tools.events)}


def superseded(documents):
    """IDs of documents an earlier interface must not offer: older or unapproved revisions of a family that has
    one latest approved revision, judged from the user's own metadata (``project``, ``title``, ``revision``,
    ``status``).

    Interfaces before ``workspace-documents/4`` show the model only id, title and content, so revisions of one
    document look identical. The model then answers from whichever ranks first: in the pilot-2-rc2 soak, revision
    1 of the demo delivery plan instead of its approved revision 2, on every run. The selection reuses
    ``document_revisions.current`` (the /4 design's rule), applied on this device before the route sees the
    files. A family that rule does not decide is left as it was: no approved revision, a tie, incomplete or
    invalid metadata, or no metadata at all. So nothing is hidden unless a single current revision is known.
    """
    from neuroshard.evolution import document_revisions as revisions

    groups = {}
    for document in documents:
        if 'revision' in document and 'status' in document:
            key = (document.get('project'), document.get('family', document.get('title')))
            fields, problems = revisions.metadata(document)
            groups.setdefault(key, []).append({'id': document['id'], 'metadata': fields, 'problems': problems})
    hidden = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        outcome, chosen, _ = revisions.current(members)
        if outcome == 'known_current':
            hidden |= {m['id'] for m in members if m['id'] != chosen['id']}
    return hidden


def documents_world(world, version=3):
    """The documents workspace's public view of the user's files: id, title and content only.

    From ``workspace-documents/4``, the structured revision metadata fields pass through too (never any other
    field), and the model selects among revisions with the /4 tools. Earlier versions cannot see revision
    metadata, so they are offered only the latest approved revision of each document whose metadata decides one
    (``superseded``); documents without such metadata are offered exactly as before."""
    documents = world.get('documents', [])
    hidden = superseded(documents) if version < 4 else set()
    rows = []
    for document in documents:
        if document['id'] in hidden:
            continue
        content = document.get('content')
        if content is None:  # a drafting document: its fields, as text
            content = '; '.join(f'{key}: {value}' for key, value in sorted(document.items())
                                if key not in ('id', 'title'))
        row = {'id': document['id'], 'title': document.get('title', document['id']), 'content': content}
        if version >= 4:
            from neuroshard.evolution.document_revisions import FIELDS  # only /4 needs it

            row.update({key: document[key] for key in FIELDS if key in document})
        rows.append(row)
    return {'documents': rows}


def interface(policy):
    """The interface a policy declares, or the drafting sandbox when it declares none."""
    name = policy.get('interface')
    if name in DOCUMENTS:
        from neuroshard.evolution import assistant_documents as documents

        version = DOCUMENTS.index(name) + 2
        return Interface(name, 'documents', documents.tools(version),
                         lambda text: documents.parse_calls(text, version),
                         lambda world: documents.Workspace(documents_world(world, version), version))
    if name in CHAT:
        from neuroshard.evolution import assistant_general as general
        from neuroshard.evolution import assistant_workspace as base

        kit = general.ChatTools if name == 'chat-tools/1' else general.Calculator
        # The accepted chat runner (assistant_general.calculator_chat_episode) renders results ASCII-escaped.
        return Interface(name, 'chat', kit.TOOLS, lambda text: base.parse_calls(text, kit.REGISTRY),
                         lambda world: ChatWorkspace(kit), ascii_results=True)
    module = workflow.interface(policy)
    return Interface(name, 'calendar', module.TOOLS, module.parse_calls, module.Workspace)
