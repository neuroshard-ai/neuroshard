"""Fit the general-route version's top router from canonical router features, and check it before release.

    PYTHONPATH=src python scripts/fit_general_router.py --features <evidence dir> --out <dir>

Routes: ``a3`` (any drafting or scheduling turn; A3's own router then decides between them, unchanged),
``documents`` and ``chat``. Features are the accepted router feature of each turn with the conversation's
opening turn (``assistant_turn_router.context_feature``), exactly as served.

Fit only on training turns: documents generated-train, generated-train-extra and generated-train-styles3;
chat generated-train and generated-train-styles3; drafting train; scheduling train and cross-train.
Report development turns (documents and chat generated-development5, drafting development, scheduling
development and cross-development). Record the route predicted for every turn of the release workload
(general confirmation7 and A3's confirmation sets). The release check itself is the test; this
prediction only avoids spending on a run that would stop at its first misroute.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from neuroshard.evolution import assistant_chat_data as chat
from neuroshard.evolution import assistant_documents_data as docs
from neuroshard.evolution import assistant_schedule_data as schedule
from neuroshard.evolution import assistant_turn_router as router
from neuroshard.evolution import assistant_workflow_data as drafting

FIT = [('documents', docs, 'documents', s) for s in ('generated-train', 'generated-train-extra', 'generated-train-styles3')]
FIT += [('chat', chat, 'chat', s) for s in ('generated-train', 'generated-train-styles3')]
FIT += [('a3', drafting, 'drafting', 'train'), ('a3', schedule, 'scheduling', 'train'),
        ('a3', schedule, 'scheduling', 'cross-train')]
DEVELOPMENT = [('documents', docs, 'documents', 'generated-development5'), ('chat', chat, 'chat', 'generated-development5'),
               ('a3', drafting, 'drafting', 'development'), ('a3', schedule, 'scheduling', 'development'),
               ('a3', schedule, 'scheduling', 'cross-development')]
WORKLOAD = [('documents', docs, 'documents', 'confirmation7'), ('chat', chat, 'chat', 'confirmation7'),
            ('a3', drafting, 'drafting', 'confirmation6'), ('a3', schedule, 'scheduling', 'confirmation3'),
            ('a3', schedule, 'scheduling', 'cross-confirmation3')]


def turns(sets, lookup):
    """``{key: (label, context feature)}`` for every turn of every case in ``sets``."""
    out = {}
    for label, module, name, split in sets:
        for case in module.cases(split):
            states = [lookup[turn['user']] for turn in case['turns']]
            for index, feature in enumerate(router.conversation_features(states)):
                out[f"{name}/{split}/{case['id']}#{index}"] = (label, feature)
    return out


def accuracy(fitted, rows):
    keys = sorted(rows)
    predicted = router.route_many(fitted, [rows[k][1] for k in keys])
    by = {}
    for key, choice in zip(keys, predicted):
        label = rows[key][0]
        hit, total = by.get(label, (0, 0))
        by[label] = (hit + (choice == label), total + 1)
    wrong = [(key, rows[key][0], choice) for key, choice in zip(keys, predicted) if choice != rows[key][0]]
    return {label: f'{h}/{t}' for label, (h, t) in sorted(by.items())}, wrong


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--features', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    directory = Path(args.features)
    texts = json.loads((directory / 'texts.json').read_text())
    vectors = np.load(directory / 'features.npz')['layer-1']
    lookup = {text: vector.tolist() for text, vector in zip(texts, vectors)}
    fit_rows = turns(FIT, lookup)
    fitted = router.fit({k: v[1] for k, v in fit_rows.items()}, {k: (v[0], 1.0) for k, v in fit_rows.items()},
                        steps=400, seed=27101401)
    report = {'format': 'neuroshard-assistant-general-router-fit/1', 'routes': fitted['routes'],
              'fit_turns': len(fit_rows), 'router_sha256': fitted['sha256']}
    for name, sets in (('fit', FIT), ('development', DEVELOPMENT), ('workload', WORKLOAD)):
        by, wrong = accuracy(fitted, turns(sets, lookup))
        report[name] = {'by_route': by, 'misrouted': wrong[:40], 'misrouted_count': len(wrong)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'general-router.json').write_text(json.dumps(fitted, sort_keys=True))
    (out / 'general-router-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('routes', 'fit_turns')} |
                     {k: {'by_route': report[k]['by_route'], 'misrouted': report[k]['misrouted_count']}
                      for k in ('fit', 'development', 'workload')}, indent=1))
    return 0 if report['development']['misrouted_count'] == 0 and report['workload']['misrouted_count'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
