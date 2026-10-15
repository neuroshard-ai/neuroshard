"""Sealed evaluation from public randomness: a test set nobody, the project included, can know in advance.

A growth job's evaluation is derived from the drand round the ledger named when the job settled
(``seal_round``), which did not exist while the unit was trained. Its randomness seeds every choice:
which wording templates (from the full published pool), which workspaces and documents, which values,
cities, vendors, dates and amounts, which chat topics and constraints. The expected outcomes are computed
by the same deterministic builders as every earlier split, so anyone with the beacon recomputes the
identical set, and the case-ID digest pins it.

What this seals: the content of every case (workspaces, facts, values). What it does not: the wording
templates and the families, which are published. A unit can be trained toward the families and the
wording style; it cannot be trained on the cases.
"""

import hashlib
import json
import random

from neuroshard.evolution.modular_reference_execution import identity

FORMAT = 'neuroshard-beacon-sealed-evaluation/1'
POOLS = ('train', 'development', 'confirmation4')  # every published template set, merged per kind
TOPIC_WORDS = ('sailing', 'pottery', 'baking', 'cycling', 'gardening', 'astronomy', 'chess', 'jazz', 'hiking', 'origami',
               'beekeeping', 'birding', 'rowing', 'knitting', 'surfing', 'calligraphy', 'cheese', 'tea', 'maps', 'kites')


def seed_of(randomness, purpose):
    return hashlib.sha256(f'{FORMAT}:{purpose}:{randomness}'.encode()).hexdigest()


def templates(route):
    """Every published template of a route, by kind: the pool a sealed set draws its wording from."""
    from neuroshard.evolution import assistant_phrasings as phrasings
    from neuroshard.evolution.modular_reference_execution import ROOT, read

    pool = {}
    for path in (phrasings.FILE, phrasings.FILE2, phrasings.FILE3, phrasings.FILE4, phrasings.FILE5):
        for name, by_kind in read(ROOT / path)['sets'][route].items():
            for kind, rows in by_kind.items():
                pool.setdefault(kind, set()).update(rows)
    return {kind: sorted(rows) for kind, rows in sorted(pool.items())}


def documents(randomness, per_kind=12):
    from neuroshard.evolution import assistant_documents_data as docs

    pool, out = templates('documents'), []
    for kind in docs.KINDS:
        for index in range(per_kind):
            rng = random.Random(seed_of(randomness, f'documents:{kind}:{index}'))
            world, facts = docs._world(rng, 'sealed', f'{kind}-{index}')
            expected = docs._kinds(facts)[kind][2]
            template = pool[kind][rng.randrange(len(pool[kind]))]
            out.append({'id': f'docs-sealed-{kind}-{index:02d}', 'split': 'sealed', 'family': kind, 'world': world,
                        'turns': [{'user': docs._fill(template, facts), 'expected': expected}]})
    return out


def chat(randomness, per_kind=12):
    from neuroshard.evolution import assistant_chat_data as chat_data

    pool, out = templates('chat'), []
    for kind in chat_data.KINDS:
        for index in range(per_kind):
            rng = random.Random(seed_of(randomness, f'chat:{kind}:{index}'))
            topic = TOPIC_WORDS[rng.randrange(len(TOPIC_WORDS))]
            rows = {k: [v[rng.randrange(len(v))]] for k, v in pool.items()}
            text, constraints = chat_data._generated_task(kind, rng, topic, rows, index)
            out.append({'id': f'chat-sealed-{kind}-{index:02d}', 'split': 'sealed', 'family': kind,
                        'turns': [{'user': text, 'expected': {'constraints': constraints}}]})
    return out


def sealed(round_number, signature, randomness_hex, per_kind=12):
    """The verified beacon's evaluation set and its digest; refuses a beacon that does not verify."""
    from neuroshard.inference import beacon

    if not beacon.verify(round_number, signature, randomness_hex):
        raise ValueError('beacon does not verify')
    sets = {'documents': documents(randomness_hex, per_kind), 'chat': chat(randomness_hex, per_kind)}
    return {'format': FORMAT, 'round': round_number, 'randomness': randomness_hex, 'sets': sets,
            'sha256': identity(sets), 'case_ids_sha256': hashlib.sha256(
                json.dumps({k: [c['id'] for c in v] for k, v in sets.items()}).encode()).hexdigest()}
