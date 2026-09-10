#!/usr/bin/env python3
"""Compare three training steps across the host and an existing Docker image."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="neuroshard-trainer:latest")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", args.image], text=True))[0]
    (REPO / ".neuroshard").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="container-conformance-", dir=REPO / ".neuroshard") as tmp:
        root = Path(tmp)
        env = os.environ.copy()
        env.update(PYTHONPATH=str(REPO / "src"), ATEN_CPU_CAPABILITY="default", MKL_ENABLE_INSTRUCTIONS="SSE4_2")
        subprocess.run([sys.executable, "-m", "neuroshard.lab.experiments", "numeric-child", "--threads", "1",
                        "--output", str(root / "host.json")], env=env, check=True, capture_output=True)
        subprocess.run(["docker", "run", "--rm", "--network", "none", "--memory", "1g", "--cpus", "1",
            "--workdir", "/experiment", "--entrypoint", "python", "-e", "PYTHONPATH=/experiment/src",
            "-e", "ATEN_CPU_CAPABILITY=default", "-e", "MKL_ENABLE_INSTRUCTIONS=SSE4_2",
            "-v", f"{REPO / 'src'}:/experiment/src:ro", "-v", f"{REPO / 'docs/eval/data'}:/experiment/docs/eval/data:ro",
            "-v", f"{root}:/results", image["Id"], "-m", "neuroshard.lab.experiments", "numeric-child",
            "--threads", "1", "--output", "/results/container.json"], check=True, capture_output=True)
        host = json.loads((root / "host.json").read_text())
        container = json.loads((root / "container.json").read_text())
        match = host.pop("weights") == container.pop("weights") and host["results"] == container["results"]
    result = {"scope": "Same physical x86_64 host, separate container and Python/PyTorch build; CPU execution only",
              "image_id": image["Id"], "image_created": image["Created"], "host": host, "container": container,
              "pinned_cpu_dispatch": {"ATEN_CPU_CAPABILITY": "default", "MKL_ENABLE_INSTRUCTIONS": "SSE4_2"},
              "three_training_steps_bitwise_equal": match,
              "interpretation": "A finite conformance experiment, not a proof of equality for all hardware or all inputs"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"host_torch": host["torch"], "container_torch": container["torch"], "bitwise_equal": match}))


if __name__ == "__main__":
    main()
