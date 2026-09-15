#!/usr/bin/env python3
"""Execute the frozen second cohort with its preserved question interpreter."""
import argparse
from pathlib import Path

from neuroshard.evolution import interpreted_cohort
from neuroshard.evolution.sharded.cohort_job import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'final'])
    for name in ('inputs', 'parent', 'first', 'objects', 'first-objects', 'seed', 'home', 'interpreter'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--second', type=Path)
    run(parser.parse_args(), experiment=interpreted_cohort, network_factory=interpreted_cohort.network)


if __name__ == '__main__':
    main()
