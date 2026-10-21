"""Print workspace-documents/4 results for a fixed set of inputs, as JSON, so CI can compare them across Python
versions: tool results must not depend on the interpreter, or recorded transcripts would not replay."""
import json
import sys

from neuroshard.evolution import assistant_documents as d

CASES = [('2027-01-01', '2027-01-15'), ('20270101', '2027-01-15'), ('2027-W01-1', '2027-01-15'),
         ('2027-01-01', '20270115'), ('2027-01-01', '2027-W02-5'), ('2027-02-30', '2027-03-01')]
out = []
for stored, asked in CASES:
    w = d.Workspace({'documents': [{'id': 'doc-00000001', 'title': 'Plan', 'content': 'x', 'project': 'P',
                                    'family': 'plan', 'revision': 1, 'status': 'approved',
                                    'effective_from': stored}]}, 4)
    out.append([w.execute({'name': 'current_document', 'arguments': {'project': 'P', 'family': 'plan'}}),
                w.execute({'name': 'effective_document',
                           'arguments': {'project': 'P', 'family': 'plan', 'date': asked}}),
                w.snapshot()])
json.dump(out, sys.stdout, sort_keys=True)
