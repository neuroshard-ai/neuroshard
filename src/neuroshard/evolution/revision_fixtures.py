"""Versioned document fixtures with explicit revision-family metadata (DESIGN_LATEST_REVISION.md, "migration").

The original demo workspace (``assistant_workflow_data`` development sample 0) carries ``project``, ``revision``
and ``status`` but no ``family``, so under ``workspace-documents/4`` every document would be ``no_metadata``.
This module defines the enriched version, ``demo-v2``: the original world plus authoritative family IDs,
assigned by an explicit mapping from document ID. Nothing is inferred from titles or text. The original
fixture is never modified, so every earlier transcript replays against exactly the world it was recorded on.
"""

import copy

from neuroshard.evolution.modular_reference_execution import identity

ORIGINAL_ROOT = 'bc2bb684340e08cd268539644ceaf57b74ca4c0ad7327b7f8a1f856a046bca78'  # the demo world, as recorded
# Authoritative mapping for demo-v2: document ID -> family. Families are scoped by the documents' own project.
DEMO_V2_FAMILIES = {
    'doc-292283f7a9': 'delivery-plan',   # Cedar 6581, revision 1, approved
    'doc-88b36697f8': 'delivery-plan',   # Cedar 6581, revision 2, approved
    'doc-9e6d7ba44d': 'delivery-plan',   # Cedar 6581, revision 3, draft
    'doc-35d32ed303': 'delivery-plan',   # Cedar 6581 Annex, revision 2, approved (a separate project, so a separate family)
}


def original():
    from neuroshard.evolution import assistant_workflow_data as data

    world = copy.deepcopy(data.cases('development')[0]['world'])
    if identity(world) != ORIGINAL_ROOT:
        raise ValueError('the original demo fixture changed; earlier transcripts would no longer replay against it')
    return world


def demo_v2():
    """The enriched demo world: the original, with each document's ``family`` set from the explicit mapping."""
    world = original()
    ids = {document['id'] for document in world['documents']}
    if ids != set(DEMO_V2_FAMILIES):
        raise ValueError('the demo fixture changed: its documents no longer match the demo-v2 mapping')
    for document in world['documents']:
        document['family'] = DEMO_V2_FAMILIES[document['id']]
    return world


def roots():
    """Content identities of both fixtures, for pinning in tests and evaluation records."""
    return {'original': identity(original()), 'demo-v2': identity(demo_v2())}
