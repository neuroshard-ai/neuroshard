"""Evaluate preserved question interpretation through four separate GPU owners."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('development', 'final'))
    for name in ('home', 'inputs', 'parent', 'expert', 'objects', 'expert-objects', 'seed', 'interpreter'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    from neuroshard.evolution import preserved_interpreter as experiment
    from neuroshard.evolution.sharded.branch_job import run
    run(args, experiment=experiment, network_factory=experiment.network)


if __name__ == '__main__':
    main()
