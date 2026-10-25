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


def revision_labels(documents):
    """Title annotations for interfaces before ``workspace-documents/4``: ``{document id: label}``.

    Those interfaces show the model only id, title and content, so revisions of one document look identical (the
    pilot-2-rc2 soak answered from revision 1 of the demo plan instead of approved revision 2). Nothing is hidden:
    every document stays readable for history, comparison and effective-date questions. Only documents in a family
    the /4 design's own rules decide get a label (a family of one included: its only revision is decided):
    - identity is an explicit ``(project, family)``, never a title (``document_revisions.families``/``members``,
      which also pull in malformed would-be members);
    - selection is ``document_revisions.current``; a family that is ``incomplete``, ``ambiguous`` or has no
      approved revision gets no label at all.
    The current revision is labelled ``revision N, approved; latest approved`` (the phrase users ask with, so search
    ranks it too), the others with their own revision and status.
    """
    from neuroshard.evolution import document_revisions as revisions

    parsed = []
    for document in documents:
        if not isinstance(document, dict) or not isinstance(document.get('id'), str):
            continue
        fields, problems = revisions.metadata(document)
        parsed.append({'id': document['id'], 'metadata': fields, 'problems': problems})
    labels = {}
    for (project, family) in revisions.families(parsed):
        group = revisions.members(parsed, project, family)
        outcome, chosen, _ = revisions.current(group)
        if outcome != 'known_current':
            continue
        for member in group:
            meta = member['metadata']
            labels[member['id']] = (f"revision {meta['revision']}, {meta['status']}"
                                    + ('; latest approved' if member['id'] == chosen['id'] else ''))
    return labels


def labelled_files(files, families):
    """An explicit, opt-in view for interfaces before /4: the user's files with revision labels in their titles.

    ``families`` is ``{document id: family}``, supplied by the user's device beside the files (never inferred). The
    returned files have the same documents and fields; only titles of documents in a decided family change
    (``revision_labels``). Every document stays readable."""
    enriched = copy.deepcopy(files)
    for document in enriched.get('documents', []):
        if document.get('id') in families:
            document['family'] = families[document['id']]
    labels = revision_labels(enriched.get('documents', []))
    out = copy.deepcopy(files)
    for document in out.get('documents', []):
        if document.get('id') in labels:
            document['title'] = f"{document.get('title', document['id'])} ({labels[document['id']]})"
    return out


def documents_world(world, version=3):
    """The documents workspace's public view of the user's files: id, title and content only.

    From ``workspace-documents/4``, the structured revision metadata fields pass through too (never any other
    field), and the model selects among revisions with the /4 tools. Earlier versions see exactly what they always
    saw (a session can opt into labelled titles: ``labelled_files``)."""
    rows = []
    for document in world.get('documents', []):
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
