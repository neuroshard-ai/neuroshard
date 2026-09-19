#!/usr/bin/env python3
"""Execute a committed, operated second-expert learning experiment."""
import argparse
from pathlib import Path

from neuroshard.evolution.sharded.cohort_job import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'final'])
    for name in ('inputs', 'parent', 'first', 'objects', 'first-objects', 'seed', 'home'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--second', type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
