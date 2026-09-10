#!/usr/bin/env python3
"""Build reproducible source assets from a public Git revision, never a working tree."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def build(ref, output, wheel=None):
    commit = git("rev-parse", "--verify", ref + "^{commit}").decode().strip()
    version = re.search(r'^version = "([0-9]+\.[0-9]+\.[0-9]+(?:a[0-9]+)?)"$', git("show", f"{commit}:pyproject.toml").decode(), re.M)[1]
    prefix = f"neuroshard-{version}"
    raw = git("archive", "--format=tar", f"--prefix={prefix}/", commit)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{prefix}-source.tar.gz"
    with archive.open("wb") as stream, gzip.GzipFile(fileobj=stream, filename="", mode="wb", mtime=0) as compressed:
        compressed.write(raw)
    files = []
    with tarfile.open(fileobj=io.BytesIO(raw)) as source:
        for member in source.getmembers():
            if member.isfile():
                files.append(f"{hashlib.sha256(source.extractfile(member).read()).hexdigest()}  {member.name}")
    (output / "SOURCE_FILES.sha256").write_text("\n".join(files) + "\n")
    if version.startswith('0.3.'):
        network_path = "networks/neuroshard-stage-8i5ghxq5"
    else:
        descriptor = json.loads(git('show', f'{commit}:src/neuroshard/client/networks/llm-testnet.json'))
        network_path = 'networks/' + descriptor['chain_id']
    names = ('genesis.json', 'network.json') if version.startswith('0.3.') else ('genesis.json', 'network.json', 'dataset.json', 'declarations.json')
    for name in names:
        (output / name).write_bytes(git("show", f"{commit}:{network_path}/{name}"))
    network = json.loads((output / "network.json").read_text())
    if hashlib.sha256((output / "genesis.json").read_bytes()).hexdigest() != network["genesis_sha256"]:
        raise ValueError("Release genesis digest differs from network declaration")
    metadata = {"version": version, "tag": f"v{version}", "source_commit": commit,
                "archive": archive.name, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "genesis_sha256": network["genesis_sha256"], "manifest_hash": network["manifest_hash"],
                "archive_url": f"https://github.com/neuroshard-ai/neuroshard/releases/download/v{version}/{archive.name}"}
    (output / "release.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if wheel:
        with zipfile.ZipFile(wheel) as package:
            for name in package.namelist():
                if name.startswith('neuroshard/') and name.endswith('.py'):
                    if package.read(name) != git('show', f'{commit}:src/{name}'):
                        raise ValueError(f'Wheel source differs from release commit: {name}')
        shutil.copy2(wheel, output / wheel.name)
    artifacts = sorted(p for p in output.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in artifacts))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    build(args.ref, args.output, args.wheel)
