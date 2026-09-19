#!/usr/bin/env python3
"""Score frozen ordinary-serving traces without neural execution or a new final."""
import argparse
import json
from pathlib import Path

from neuroshard.evolution import serving_diagnosis as diagnosis
from neuroshard.evolution.reference_data import identity, save


def score(plan_path, traces, destination, source, facts):
    plan = json.loads(Path(plan_path).read_bytes())
    curation = json.loads(Path(facts).read_bytes())
    diagnosis.validate(plan, curation, source)
    result = diagnosis.score_traces(plan, json.loads(Path(traces).read_bytes()))
    destination = Path(destination)
    save(destination, result)
    print(json.dumps({'plan': identity(plan), 'summary': result['summary']}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--traces', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--facts', type=Path, required=True)
    args = parser.parse_args()
    score(args.plan, args.traces, args.destination, args.source, args.facts)
