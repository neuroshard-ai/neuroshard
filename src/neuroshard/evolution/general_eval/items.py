"""Item schema, loading, validation and the sealed-split commitment for the general-capability evaluation.

Every item is one JSON object (one per line in ``data/<split>/<area>.jsonl``)::

    {"id": "reasoning-dev-001", "area": "reasoning", "split": "development", "family": "rates",
     "turns": [{"user": "..."}],            # what the assistant sees; usually one turn
     "scoring": {"kind": "answer", ...},    # never shown to the assistant
     "fixture": {...},                      # tool use only: the tools' backing state
     "judged": {"rubric": "writing/1"},     # optional: also compared pairwise by the blind judge
     "provenance": {"source": "fresh", "author": "...", "written": "YYYY-MM-DD"}}

Only ``turns`` (and, for tool use, the tool schemas and tool results) ever reach a model. ``public_view``
is the only function that builds what a system receives.

A sealed split is held off-repo. Before it is used, its custodian publishes ``commitment(items, salt)``:
a salted digest over the canonical file plus per-area counts, and nothing that would let anyone test
whether a given question is in the set. At opening, the salt and the file are revealed to the auditor,
who checks ``verify_commitment`` before any score is accepted.
"""

import hashlib
import json
from pathlib import Path

AREAS = ('reasoning', 'coding', 'writing', 'factual', 'planning', 'tool_use')
SCORING_KINDS = {'reasoning': ('answer',), 'coding': ('code',), 'writing': ('constraints',),
                 'factual': ('short_answer',), 'planning': ('plan',), 'tool_use': ('tools',)}
SOURCES = ('fresh', 'reference')  # reference = a labelled public benchmark item, never part of a gate
DATA = Path(__file__).resolve().parent / 'data'
FORMAT = 'neuroshard-general-eval-items/1'
COMMITMENT_FORMAT = 'neuroshard-general-eval-commitment/1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def validate(item):
    """Raise ValueError on any malformed item; returns the item."""
    required = {'id', 'area', 'split', 'family', 'turns', 'scoring', 'provenance'}
    allowed = required | {'fixture', 'judged', 'notes'}
    if not isinstance(item, dict) or not required <= set(item) <= allowed:
        raise ValueError(f'item fields must be {sorted(required)} plus optional {sorted(allowed - required)}: '
                         f'{item.get("id") if isinstance(item, dict) else item!r}')
    if item['area'] not in AREAS:
        raise ValueError(f'{item["id"]}: unknown area {item["area"]!r}')
    if item['scoring'].get('kind') not in SCORING_KINDS[item['area']]:
        raise ValueError(f'{item["id"]}: scoring kind does not fit area {item["area"]}')
    turns = item['turns']
    if not (isinstance(turns, list) and turns and all(isinstance(t, dict) and set(t) == {'user'}
                                                       and isinstance(t['user'], str) and t['user'].strip()
                                                       for t in turns)):
        raise ValueError(f'{item["id"]}: turns must be non-empty user messages only')
    if item['provenance'].get('source') not in SOURCES:
        raise ValueError(f'{item["id"]}: provenance.source must be one of {SOURCES}')
    if item['area'] == 'tool_use' and 'fixture' not in item:
        raise ValueError(f'{item["id"]}: tool-use items need a fixture')
    from neuroshard.evolution.general_eval import scorers

    scorers.check_spec(item)
    return item


def load_file(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), start=1):
        if line.strip():
            try:
                rows.append(validate(json.loads(line)))
            except ValueError as error:
                raise ValueError(f'{path}:{number}: {error}') from None
    return rows


def load_split(split='development', root=None, areas=AREAS):
    """All items of a split, in area order then file order; IDs must be unique and match the split."""
    directory = Path(root or DATA) / split
    items = []
    for area in areas:
        path = directory / f'{area}.jsonl'
        if path.exists():
            items += load_file(path)
    ids = [item['id'] for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate item IDs')
    if any(item['split'] != split for item in items):
        raise ValueError('an item names another split')
    return items


def public_view(item, turn=0):
    """What a system under test may see for one turn: the user's words, nothing about scoring."""
    return {'id': item['id'], 'area': item['area'], 'user': item['turns'][turn]['user']}


def split_digest(items):
    """Unsalted digest of a split's canonical content (published for an open split such as development)."""
    return digest({'format': FORMAT, 'items': sorted(items, key=lambda item: item['id'])})


def counts(items):
    out = {area: 0 for area in AREAS}
    for item in items:
        out[item['area']] += 1
    return out


def commitment(items, salt, split):
    """Published before a sealed split is used. ``salt`` is at least 32 random bytes (hex), kept private
    until opening, so the digest cannot be used to test guesses about the contents."""
    if not isinstance(salt, str) or len(bytes.fromhex(salt)) < 32:
        raise ValueError('the salt must be at least 32 random bytes, hex encoded')
    if any(item['split'] != split for item in items):
        raise ValueError('an item names another split')
    body = hashlib.sha256(bytes.fromhex(salt) + canonical(sorted(items, key=lambda item: item['id']))).hexdigest()
    return {'format': COMMITMENT_FORMAT, 'split': split, 'counts': counts(items), 'items': len(items),
            'salted_sha256': body}


def verify_commitment(items, salt, published):
    actual = commitment(items, salt, published['split'])
    if actual != published:
        raise ValueError('the opened split does not match its published commitment')
    return True
