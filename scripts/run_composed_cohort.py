#!/usr/bin/env python3
"""Evaluate the archived expert with fixed prompts and actual composed calls."""
import argparse
from pathlib import Path

from neuroshard.evolution import composed_cohort
from neuroshard.evolution.sharded.cohort_job import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['evaluate', 'final'])
    for name in ('inputs', 'parent', 'first', 'second', 'objects', 'first-objects',
                 'seed', 'home', 'interpreter'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args(), experiment=composed_cohort, network_factory=composed_cohort.network)


if __name__ == '__main__':
    main()
