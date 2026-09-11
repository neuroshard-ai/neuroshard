#!/usr/bin/env python3
"""Reject local/publishing material in distributions and verify required wheel assets."""
import argparse
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


EXCLUDED = {"archive", ".neuroshard", "website", "docs-site", "legacy",
            "node_modules", "venv_build", ".git", "__pycache__"}
REQUIRED = {
    "neuroshard/client/networks/llm-testnet.json",
    "neuroshard/client/consensus/go.mod", "neuroshard/client/consensus/go.sum",
    "neuroshard/publicnet/data/input.txt",
    "neuroshard/demo/abci.proto", "neuroshard/lab/abci.proto",
    "neuroshard/evolution/settlement.py",
    "neuroshard/evolution/text.py",
    "neuroshard/evolution/lifecycle.py", "neuroshard/evolution/cohorts.py",
    "neuroshard/evolution/forward.py",
}
REQUIRED_SOURCE = {
    "docs/eval/data/input.txt", "docs/llm-requirements.txt",
    "tests/evolution/conftest.py", "tests/evolution/test_settlement.py",
    "config/evolution-epoch.example.json", "config/experiments/response-from-seed-plan.json",
    "scripts/experiment_evolution_native.py", "scripts/experiment_evolution_response.py",
    "scripts/experiment_text_profile.py", "tests/evolution/test_text.py",
    "scripts/native_rpc.py", "tests/evolution/test_experiment_rpc.py",
    "docs/TEXT_PROTOCOL.md", "docs/evolution-requirements.txt",
    "docs/NATIVE_LIFECYCLE.md", "config/native-data.example.json",
    "scripts/prepare_native_cohort.py", "scripts/experiment_lifecycle_native.py",
    "scripts/experiment_forward_profile.py", "tests/evolution/test_lifecycle.py",
    "tests/evolution/test_cohort_preparation.py",
    "scripts/check_validator_topology.py", "tests/evolution/test_validator_topology.py",
    "scripts/review_native_cohort.py", "tests/evolution/test_cohort_review.py",
    "scripts/continue_lifecycle_native.py",
    "networks/neuroshard-llm-testnet-1/genesis.json",
}


def check_names(names):
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe package path: {name}")
        if (EXCLUDED.intersection(path.parts) or path.name.startswith(".env")
                or path.suffix.lower() in {".tex", ".pdf", ".pem", ".key", ".pyc", ".safetensors", ".ckpt"}):
            raise ValueError(f"Local, manuscript or publishing file packaged: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    directory = parser.parse_args().directory
    wheels, sources = list(directory.glob("*.whl")), list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("Use a fresh output directory containing one wheel and one source distribution")
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())
        check_names(names)
        if missing := REQUIRED - names:
            raise ValueError(f"Required wheel assets missing: {sorted(missing)}")
    with tarfile.open(sources[0]) as source:
        members = source.getmembers()
        check_names(m.name for m in members)
        if any(m.issym() or m.islnk() for m in members):
            raise ValueError("Source distribution contains links instead of standalone files")
        paths = {str(PurePosixPath(m.name).relative_to(PurePosixPath(m.name).parts[0]))
                 for m in members if m.isfile()}
        if missing := REQUIRED_SOURCE - paths:
            raise ValueError(f"Source distribution test/reproduction files missing: {sorted(missing)}")
        if any(p.startswith(("docs/archive/", "docs/whitepaper/", "docs/figures/",
                             "docs/experiments/", "docs/eval/results/")) for p in paths):
            raise ValueError("Historical documentation output entered the source distribution")
    print("Wheel and source distribution contain required runtime assets and exclude local/publishing material")


if __name__ == "__main__":
    main()
