"""The network decides what to learn next: sealed probes of the serving version, and recipes for its weak spots.

``probe`` runs the serving model on a beacon-sealed evaluation (``sealed_evaluation``) and reports its pass
rate per family. Any agent can run it, at any drand round, and anyone can recompute the same report.

``weakest`` picks the families below a target rate; ``recipe`` turns them into a training recipe:
verified demonstrations of fresh beacon-seeded cases of exactly those families (the reference solver
writes and checks every one), mixed with the serving model's own replay so earlier behaviour is kept,
on the route's unit arm. The recipe and its start state are what ``recipe_propose`` names; owners vote it
into the growth queue, and the growth pool opens it as a job when usage has paid for it.

The probe's cases and the recipe's cases come from different beacon rounds and different purposes, so a
recipe never trains on the cases that measured the weakness, nor on the cases that will seal its job.
"""

import hashlib
import json
import random

from neuroshard.evolution import sealed_evaluation as sealed


def probe(run, cases, route):
    """Pass rate per family of the serving model on one sealed set: ``run(case) -> passed``."""
    by = {}
    for case in cases['sets'][route]:
        hit, total = by.get(case['family'], (0, 0))
        by[case['family']] = (hit + bool(run(case)), total + 1)
    return {'round': cases['round'], 'cases_sha256': cases['sha256'], 'route': route,
            'families': {family: {'passed': h, 'cases': t, 'rate': h / t} for family, (h, t) in sorted(by.items())}}


def weakest(report, target=0.9, limit=3):
    """The families below ``target``, weakest first; ties by name, so every agent picks the same ones."""
    rows = sorted((v['rate'], name) for name, v in report['families'].items() if v['rate'] < target)
    return [name for _, name in rows[:limit]]


def demonstrations(families, randomness, per_family=60):
    """Fresh documents cases of the weak families, seeded for training (never the probe's or a seal's cases)."""
    from neuroshard.evolution import assistant_documents_data as docs

    pool, out = sealed.templates('documents'), []
    for family in families:
        for index in range(per_family):
            rng = random.Random(sealed.seed_of(randomness, f'recipe:{family}:{index}'))
            world, facts = docs._world(rng, 'recipe', f'{family}-{index}')
            expected = docs._kinds(facts)[family][2]
            template = pool[family][rng.randrange(len(pool[family]))]
            out.append({'id': f'docs-recipe-{family}-{index:02d}', 'split': 'generated-train', 'family': family,
                        'world': world, 'turns': [{'user': docs._fill(template, facts), 'expected': expected}]})
    return out


def recipes(report, randomness, policy, spec, model_root, encode, replay, per_family=60, target=0.9):
    """One recipe per weak family, so each trains as its own job on its own contributor nodes."""
    return [r for family in weakest(report, target)
            for r in [recipe({**report, 'families': {family: report['families'][family]}}, randomness, policy, spec,
                             model_root, encode, replay, per_family, target)] if r]


def recipe(report, randomness, policy, spec, model_root, encode, replay, per_family=60, target=0.9):
    """A training recipe for the documents route's weak families, every demonstration verified.

    ``encode(trajectory) -> sequence`` tokenizes under the route's tools; ``replay`` is the serving model's
    own verified sequences. Returns None if nothing is below target: a strong network does not grow for its own sake.
    """
    from neuroshard.evolution import assistant_documents_demonstration as demo

    families = weakest(report, target)
    if not families:
        return None
    experience = []
    for case in demonstrations(families, randomness, per_family):
        row = demo.demonstration(case, policy)          # raises unless the solver's demonstration passes
        experience.append(encode(demo.trajectory(case, row, policy)))
    return {'format': 'neuroshard-growth-recipe/1', 'route': 'documents', 'families': families,
            'probe': {'round': report['round'], 'cases_sha256': report['cases_sha256']},
            'arm': 'addition', 'spec': spec, 'model_root': model_root, 'base': 'granite-parent',
            'sources': {'experience': experience, 'replay': replay},
            'derived_from': hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()}
