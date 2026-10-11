"""The three-route router (general, drafting, scheduling) on the general baseline's Granite features.

    PYTHONPATH=src python scripts/analyse_general_router.py --general <baseline evidence dir> \
        --scaling <router-scaling Granite evidence dir> --out <dir>

General turns are the documents and chat train/development turns. Drafting and scheduling turns
are the router scaling study's real anchors, whose Granite features exist from that run. The
router is the packaged ``assistant_turn_router`` (logistic over message plus opening). Fit on
train turns; report on development turns, including reworded real requests, and calibrate the
fallback, where the fallback is the general route itself.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from neuroshard.evolution import assistant_chat_data as chat
from neuroshard.evolution import assistant_documents_data as docs
from neuroshard.evolution import assistant_turn_router as router
from neuroshard.evolution import router_scaling as scaling
from neuroshard.evolution import router_scaling_study as study

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_router_scaling as runner  # noqa: E402


def general_cases(split):
    rows = []
    for module, kind in ((docs, 'documents'), (chat, 'chat')):
        for case in module.cases(split):
            turns = [turn['user'] for turn in case['turns']]
            rows.append({'id': case['id'], 'capability': kind, 'user_turns': turns, 'labels': ['general'] * len(turns)})
    return rows


def vectors(general_dir, scaling_dir):
    texts = json.loads((Path(general_dir) / 'texts.json').read_text())
    general = dict(zip(texts, np.load(Path(general_dir) / 'features.npz')['layer-1']))
    real = runner.load_vectors(scaling_dir)['layer-1']
    return {**real, **general}


def features(cases, lookup):
    out = {}
    for case in cases:
        for turn, feature in enumerate(router.conversation_features([lookup[t].tolist() for t in case['user_turns']])):
            out[f'{case["id"]}#{turn}'] = feature
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--general', required=True)
    parser.add_argument('--scaling', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    lookup = vectors(args.general, args.scaling)
    _, fit, sets, para_fit, _ = study.everything()
    real_fit = [c for c in fit if c['id'].startswith('real-')]
    real_test = [c for c in sets['test'] if c['id'].startswith('real-')]
    reworded = [c for c in sets['unseen'] if c['id'].startswith('unseen-real-')]
    routes = ('general', 'drafting', 'scheduling')
    fit_cases = real_fit + general_cases('train')
    checks = {'familiar': real_test + general_cases('development'), 'reworded-real': reworded}
    every = fit_cases + [c for cases in checks.values() for c in cases]
    table = features(every, lookup)
    rows = scaling.turn_rows(fit_cases, routes)
    fitted = router.fit({k: table[k] for k in rows}, rows, fallback='general')
    report = {'fit_turns': len(rows), 'routes': fitted['routes'], 'checks': {}}
    for name, cases in checks.items():
        check = scaling.turn_rows(cases, routes)
        report['checks'][name] = {'turns': len(check), **{key: value for key, value in router.recall(
            fitted, table, check).items()}}
    calibrated, calibration = router.calibrate(fitted, {k: table[k] for k in rows}, rows, target=0.99)
    report['calibration'] = calibration
    for name, cases in checks.items():
        check = scaling.turn_rows(cases, routes)
        report['checks'][name + ' (calibrated)'] = router.recall(calibrated, table, check)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'general-router.json').write_text(json.dumps(calibrated, sort_keys=True) + '\n')
    (out / 'general-router-report.json').write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    print(json.dumps(report, indent=1)[:4000])
    return 0


if __name__ == '__main__':
    sys.exit(main())
