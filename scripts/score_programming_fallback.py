"""Score the opened-development complementary policy. Not an admission result."""
import argparse
import json
from pathlib import Path
import sys

from neuroshard.evolution import programming_expert as parent
from neuroshard.evolution import programming_fallback as experiment
from neuroshard.evolution.reference_data import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True,
                        help='Parent trial inputs directory containing prepared.json and dev.jsonl')
    parser.add_argument('--outputs', type=Path,
                        default=Path('config/experiments/programming-expert-development-outputs.json'))
    parser.add_argument('--report', type=Path,
                        default=Path('config/experiments/programming-fallback-development-diagnostic.json'))
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from programming_sandbox import check
    prepared = json.loads((args.inputs / 'prepared.json').read_text())
    rows = parent.read_role(args.inputs, prepared, 'dev')
    outputs = json.loads(args.outputs.read_text())
    report = experiment.score_opened_development_diagnostic(rows, outputs, check)
    save(args.report, report)
    print(json.dumps({k: report[k] for k in report if k != 'details'}))


if __name__ == '__main__':
    main()
