#!/usr/bin/env python3
"""Recompute learning, training and serving comparisons from retained evidence."""
import argparse
import json
from pathlib import Path

from neuroshard.evolution import cooperative_report as report
from neuroshard.evolution import reference_data as data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve previous report output")
    def read(relative):
        return json.loads((args.home / relative).read_bytes())
    prepared, selection = read("prepared.json"), read("selection.json")
    records = data.read_records(args.home / "inputs/test.jsonl", prepared["roles"]["test"]["sha256"])
    evaluations = {arm: read(f"evaluation/{arm}-test.json") for arm in ("seed", *report.ARMS)}
    training = {arm: [read(f"{arm}/rank-{rank}-result.json") for rank in range(2 if arm == "clean-pair" else 1)]
                for arm in report.ARMS}
    for arm, ranks in training.items():
        if {key: ranks[0][key] for key in ("candidate", "binding", "parameter_digest")} != selection["candidates"][arm]:
            raise ValueError("Training results differ from the selected candidate")
    result = {
        "learning": report.learning_report(prepared, selection, records, evaluations),
        "training": report.training_report(prepared, training),
        "serving": report.serving_report({phase: read(f"serving/{phase}.json") for phase in ("single", "pair", "failure")},
                                         selection["candidates"]["clean-pair"]["parameter_digest"]),
    }
    result["serving"]["arm"] = "clean-pair"
    data.save(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
