"""Every workspace interface a served route may declare, behind one shape: tools, call parser, workspace.

The drafting and calendar interfaces come from ``assistant_workflow.interface`` unchanged. The general
route adds the user's documents (``workspace-documents/2`` and ``/3``) and chat tools
(``chat-calculator/1`` and ``chat-tools/1``). Every tool runs on the user's device; nothing here
reaches a network or a file.
"""

import copy

from neuroshard.evolution import assistant_workflow as workflow

DOCUMENTS = ('workspace-documents/2', 'workspace-documents/3')
CHAT = ('chat-calculator/1', 'chat-tools/1')


class Interface:
    def __init__(self, name, group, tools, parse, factory):
        self.name, self.group, self.TOOLS = name, group, tools
        self._parse, self._factory = parse, factory

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


def documents_world(world):
    """The documents workspace's public view of the user's files: id, title and content only."""
    rows = []
    for document in world.get('documents', []):
        content = document.get('content')
        if content is None:  # a drafting document: its fields, as text
            content = '; '.join(f'{key}: {value}' for key, value in sorted(document.items())
                                if key not in ('id', 'title'))
        rows.append({'id': document['id'], 'title': document.get('title', document['id']), 'content': content})
    return {'documents': rows}


def interface(policy):
    """The interface a policy declares, or the drafting sandbox when it declares none."""
    name = policy.get('interface')
    if name in DOCUMENTS:
        from neuroshard.evolution import assistant_documents as documents

        version = DOCUMENTS.index(name) + 2
        return Interface(name, 'documents', documents.tools(version),
                         lambda text: documents.parse_calls(text, version),
                         lambda world: documents.Workspace(documents_world(world), version))
    if name in CHAT:
        from neuroshard.evolution import assistant_general as general
        from neuroshard.evolution import assistant_workspace as base

        kit = general.ChatTools if name == 'chat-tools/1' else general.Calculator
        return Interface(name, 'chat', kit.TOOLS, lambda text: base.parse_calls(text, kit.REGISTRY),
                         lambda world: ChatWorkspace(kit))
    module = workflow.interface(policy)
    return Interface(name, 'calendar', module.TOOLS, module.parse_calls, module.Workspace)
