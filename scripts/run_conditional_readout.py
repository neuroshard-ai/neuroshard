"""Run the frozen research decoder; no native activation or automatic issuance."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('train', 'final'))
    for name in ('home', 'inputs', 'parent', 'objects', 'seed'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--decoder', type=Path)
    args = parser.parse_args()
    if args.command == 'final' and args.decoder is None:
        parser.error('Final evaluation requires the committed decoder')
    from neuroshard.evolution.sharded.readout_job import run
    run(args)


if __name__ == '__main__':
    main()
