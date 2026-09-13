#!/usr/bin/env python3
"""Audit a prepared JSONL corpus for explicit instruction-template failures.

Reads messages only and never trains, repairs data, or writes to a ledger.
Unrecognized instructions and factual accuracy are outside this audit's scope.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from neuroshard.evolution.target_audit import FORMAT, audit_conversation


def audit_file(path):
    counts = Counter()
    kinds = Counter()
    findings = []
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for line in source:
            digest.update(line)
            record = json.loads(line)
            checks = audit_conversation(record["messages"])
            counts["records_total"] += 1
            counts["records_checked"] += bool(checks)
            counts["checks_total"] += len(checks)
            counts["checks_failed"] += sum(not check["passes"] for check in checks)
            counts["records_with_failed_checks"] += any(not check["passes"] for check in checks)
            kinds.update(check["kind"] for check in checks)
            if checks:
                findings.append({"id": record.get("id"), "source": record.get("source"),
                                 "row": record.get("row"), "checks": checks})
    return {"format": FORMAT, "source_sha256": digest.hexdigest(), "counts": dict(counts),
            "checks_by_kind": dict(kinds), "findings": findings,
            "scope": "Report-only recognition of specific English instruction templates. A passing match does not establish target quality; unrecognized, quoted or conflicting instructions require separate review. No records were altered or admitted by this audit."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="Prepared JSONL file to inspect")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report; existing files are preserved")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve the existing report; choose a new output path")
    report = audit_file(args.records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as target:
        json.dump(report, target, indent=2)
        target.write("\n")
    print(json.dumps(report["counts"]))


if __name__ == "__main__":
    main()
