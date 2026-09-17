#!/usr/bin/env python3
"""Freeze the answering-path repair with the previous router and experts."""
import argparse
import json
from pathlib import Path
import shutil

from neuroshard.evolution.reference_data import identity, save
from neuroshard.evolution.request_planning import FORMAT
from prepare_ordinary_access_trial import ROOT, finalize


def prepare(previous, output):
    read = lambda path: json.loads(path.read_bytes())
    if output.exists():
        raise ValueError('Preserve previous trial evidence')
    old = read(previous/'inputs/planned.json')
    if identity(old) != '6060f1ca53f63681bb85d6365f4d6c42092c18cf457244e0da58f2abb1b3d867':
        raise ValueError('Continue from the measured ordinary-access candidate')
    output.mkdir(parents=True)
    for folder in ('inputs', 'seed'):
        shutil.copytree(previous/folder, output/folder)
    for name in ('fitting.json', 'fit-result.json'):
        shutil.copyfile(previous/name, output/name)
    shutil.copyfile(previous/'result.json', output/'previous-result.json')
    old['request_policy'] = FORMAT
    save(output/'inputs/planned.json', old)
    finalize(output, ROOT/'config/experiments/request-preservation-trial.json')
    trial = read(output/'inputs/access-trial.json')
    trial['decision'] = {
        'hypothesis': 'Preserve clear single requests and ground unresolved plan references before expert calls.',
        'changed': ['request planning and its complete-call metering'],
        'preserved': ['router', 'expert checkpoints', 'expert input contracts', 'diagnostic cases', 'scoring rules'],
        'engineering_gate': 'Zero selection or decomposition failures, preserve every previously correct case, exact replay.',
        'quality_gate': 'The original all-automatic-answers gate remains unchanged and may still fail knowledge.',
        'learning': 'No training. Diagnose C separately after access is measured.'}
    save(output/'inputs/access-trial.json', trial)
    save(ROOT/'config/experiments/request-preservation-trial.json', trial)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.previous, args.output)
