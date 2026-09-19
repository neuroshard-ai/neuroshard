"""Evaluate a frozen learned transformer branch with independent parent fallback."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('development', 'final'))
    for name in ('home', 'inputs', 'parent', 'expert', 'objects', 'expert-objects', 'seed'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    from neuroshard.evolution.sharded.branch_job import run
    run(args)


if __name__ == '__main__':
    main()
