#!/usr/bin/env python3
"""One disposable CPU reference host; collect evidence and retire it automatically."""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

import boto3
from botocore.exceptions import ClientError

from neuroshard.evolution.modular_reference_execution import (
    ROOT, PROFILES, committed_sources, read, save, sha256, wait_for_ci)


PROFILE = "fresh-reference-recovery"
RESOURCES = "config/experiments/modular-reference-fresh-recovery-resources.json"
RESOURCE_PROFILES = {PROFILE: RESOURCES,
                     "decoder-parity": "config/experiments/modular-decoder-parity-resources.json",
                     "granite-reference": "config/experiments/granite-reference-resources.json",
                     "granite-adapter-audit": "config/experiments/granite-adapter-audit-resources.json"}
GRANITE_PROFILES = {
    "granite-reference": ("granite_reference", "docs/granite-reference-requirements.txt"),
    "granite-adapter-audit": ("granite_adapter_audit", "docs/granite-adapter-audit-requirements.txt"),
}
REMOTE = "/home/ubuntu/neuroshard-reference"
PYTHON = REMOTE + "/.venv/bin/python"
STUDY = REMOTE + "/.study"


def source_freeze(profile):
    if profile == "granite-adapter-audit":
        from neuroshard.evolution.granite_adapter_audit import committed_sources as audit_sources
        return audit_sources()
    if profile == "granite-reference":
        from neuroshard.evolution.granite_reference import committed_sources as granite_sources
        return granite_sources()
    return committed_sources(profile=profile)


def resources(profile=PROFILE):
    value = read(ROOT / RESOURCE_PROFILES[profile])
    if (value["instances"] != 1 or value["instance_type"] != "r7i.4xlarge"
            or value["gpu"] or not 0 < value["hours"] <= 8 or value["disk_gib"] > 160
            or value["attempts"] != 1 or not value["shutdown_terminates"]
            or value["hours"] * value["price"]["usd_per_hour"] + 3 > value["planning_cap_usd"]
            or value["planning_cap_usd"] > 15):
        raise ValueError("resource contract exceeds the single-host allowance")
    prior = value["prior_resources"]
    if sha256(ROOT / prior["path"]) != prior["sha256"]:
        raise ValueError("prior resource receipt changed")
    receipt = read(ROOT / prior["path"])["resources_finished"]
    if receipt["remaining_instances"] or receipt["remaining_volumes"] or not receipt["security_group_retired"]:
        raise ValueError("prior allocation remains live")
    if profile == PROFILE and (
            receipt["conservative_instance_seconds"] + value["hours"] * 3600 > 8 * 3600
            or receipt["conservative_compute_usd"] + value["hours"] * value["price"]["usd_per_hour"] + 3 > 15):
        raise ValueError("recovery exceeds combined allowance or prior allocation remains live")
    if profile in ("decoder-parity", *GRANITE_PROFILES) and (value["hours"] > 2 or value["planning_cap_usd"] > 6):
        raise ValueError("reference profile exceeds its separate two-hour six-dollar allowance")
    return value


def instances(ec2, allocation):
    return [row for page in ec2.describe_instances(Filters=[
        {"Name": "tag:Name", "Values": [allocation["name"]]},
        {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped", "shutting-down"]}
    ])["Reservations"] for row in page["Instances"]]


def retire(home, ec2=None):
    """Only this allocation's exact purpose/name/source tags can authorize deletion."""
    home = Path(home)
    if (home / "resources-finished.json").exists():
        return read(home / "resources-finished.json")
    allocation = read(home / "allocation.json")
    ec2 = ec2 or boto3.client("ec2", region_name=allocation["resources"]["region"])
    found = instances(ec2, allocation)
    for row in found:
        tags = {tag["Key"]: tag["Value"] for tag in row.get("Tags", [])}
        if (row["InstanceId"] == allocation["resources"]["source_instance"]
                or tags.get("Name") != allocation["name"] or tags.get("Source") != allocation["commit"]
                or tags.get("Purpose") != allocation["resources"]["purpose"]):
            raise ValueError("allocation ownership mismatch; refuse retirement")
    if found:
        ec2.terminate_instances(InstanceIds=[row["InstanceId"] for row in found])
    stop = time.monotonic() + 600
    while instances(ec2, allocation):
        if time.monotonic() >= stop:
            raise TimeoutError("instance retirement did not complete")
        time.sleep(5)
    group = allocation.get("security_group")
    if group:
        for _ in range(60):
            try:
                ec2.delete_security_group(GroupId=group)
                break
            except ClientError as error:
                code = error.response["Error"]["Code"]
                if code == "InvalidGroup.NotFound":
                    break
                if code != "DependencyViolation":
                    raise
                time.sleep(3)
        else:
            raise TimeoutError("security group retirement did not complete")
    volumes = ec2.describe_volumes(Filters=[{"Name": "tag:Name", "Values": [allocation["name"]]}])["Volumes"]
    if volumes:
        raise RuntimeError("tagged volumes remain after instance retirement")
    elapsed = max(0, (datetime.now(timezone.utc) - datetime.fromisoformat(allocation["created"])).total_seconds())
    save(home / "resources-finished.json", {
        "finished_utc": datetime.now(timezone.utc).isoformat(), "remaining_instances": [],
        "remaining_volumes": [], "security_group_retired": True,
        "instance_ids": allocation.get("instance_ids", []),
        "conservative_instance_seconds": elapsed,
        "conservative_compute_usd": elapsed / 3600 * allocation["resources"]["price"]["usd_per_hour"],
        "storage_and_transfer_invoice_not_in_compute_total": True})


def allocate(home, commit, profile=PROFILE):
    limits = resources(profile)
    ec2 = boto3.client("ec2", region_name=limits["region"])
    source = ec2.describe_instances(InstanceIds=[limits["source_instance"]])["Reservations"][0]["Instances"][0]
    image = ec2.describe_images(ImageIds=[limits["image"]["ImageId"]])["Images"][0]
    if image["OwnerId"] != limits["image"]["OwnerId"] or image["State"] != "available":
        raise ValueError("frozen Ubuntu image is unavailable")
    now = datetime.now(timezone.utc)
    deadline = now + timedelta(hours=limits["hours"])
    name = "neuroshard-reference-" + uuid.uuid4().hex[:12]
    allocation = {"name": name, "commit": commit, "created": now.isoformat(),
                  "deadline": deadline.isoformat(), "resources": limits, "instance_ids": []}
    save(home / "allocation.json", allocation, exclusive=True)
    # A local guard covers setup failures; the instance also gets an absolute OS timer.
    subprocess.run(["systemd-run", "--user", "--unit=" + name + "-guard",
                    "--on-calendar=" + deadline.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "--timer-property=AccuracySec=1s", "--setenv=PYTHONPATH=" + str(ROOT / "src"),
                    sys.executable, str(ROOT / "scripts/modular_reference_cloud.py"),
                    "retire", "--home", str(home)], check=True, capture_output=True, timeout=30)
    group = ec2.create_security_group(GroupName=name, Description="Temporary CPU reference; SSH from controller only",
                                      VpcId=source["VpcId"])["GroupId"]
    allocation["security_group"] = group
    save(home / "allocation.json", allocation)
    ec2.authorize_security_group_ingress(GroupId=group, IpPermissions=[{
        "IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
        "IpRanges": [{"CidrIp": source["PrivateIpAddress"] + "/32"}]}])
    cloud = {"ssh_authorized_keys": [" ".join(Path("/home/ubuntu/.ssh/id_ed25519.pub").read_text().split()[:2])],
             "write_files": [
                 {"path": "/etc/systemd/system/neuroshard-reference-expiry.service", "content":
                  "[Service]\nType=oneshot\nExecStart=/usr/sbin/shutdown -h now\n"},
                 {"path": "/etc/systemd/system/neuroshard-reference-expiry.timer", "content":
                  "[Timer]\nOnCalendar=" + deadline.strftime("%Y-%m-%d %H:%M:%S UTC") +
                  "\nPersistent=true\nAccuracySec=1s\n[Install]\nWantedBy=timers.target\n"}],
             "runcmd": [["systemctl", "daemon-reload"],
                         ["systemctl", "enable", "--now", "neuroshard-reference-expiry.timer"],
                         ["loginctl", "enable-linger", "ubuntu"]]}
    tags = [{"Key": k, "Value": str(v)} for k, v in {
        "Name": name, "Project": "NeuroShard", "Purpose": limits["purpose"], "Source": commit,
        "ExpiresAt": deadline.isoformat(), "BudgetUSD": limits["planning_cap_usd"]}.items()]
    launched = ec2.run_instances(ImageId=image["ImageId"], InstanceType=limits["instance_type"],
        MinCount=1, MaxCount=1, KeyName=source["KeyName"], ClientToken=name,
        UserData="#cloud-config\n" + json.dumps(cloud), InstanceInitiatedShutdownBehavior="terminate",
        MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 1},
        NetworkInterfaces=[{"DeviceIndex": 0, "SubnetId": source["SubnetId"], "Groups": [group],
                            "AssociatePublicIpAddress": True, "DeleteOnTermination": True}],
        BlockDeviceMappings=[{"DeviceName": image["RootDeviceName"], "Ebs": {
            "VolumeSize": limits["disk_gib"], "VolumeType": "gp3", "Encrypted": True, "DeleteOnTermination": True}}],
        TagSpecifications=[{"ResourceType": kind, "Tags": tags} for kind in ("instance", "volume")])["Instances"]
    allocation["instance_ids"] = [r["InstanceId"] for r in launched]
    save(home / "allocation.json", allocation)
    stop = time.monotonic() + 600
    while time.monotonic() < stop:
        found = instances(ec2, allocation)
        if len(found) == 1 and found[0]["State"]["Name"] == "running" and found[0].get("PrivateIpAddress"):
            allocation["private_ip"] = found[0]["PrivateIpAddress"]
            allocation["public_ip"] = found[0].get("PublicIpAddress")
            save(home / "allocation.json", allocation)
            return allocation
        time.sleep(5)
    raise TimeoutError("temporary reference host did not start")


def ssh(home, allocation, args, *, data=None, timeout=60):
    return subprocess.run(["ssh", "-i", "/home/ubuntu/.ssh/id_ed25519", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=" + str(home / "known_hosts"),
        "ubuntu@" + allocation["private_ip"], shlex.join(args)],
        input=data, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True, timeout=timeout)


def bootstrap(home, allocation, source, profile=PROFILE):
    stop = time.monotonic() + 300
    while True:
        try:
            ssh(home, allocation, ["true"], timeout=15)
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if time.monotonic() >= stop:
                raise
            time.sleep(5)
    paths = ["/" + path for path in source["sources"]]
    setup_lines = [
        "set -eu",
        "sudo cloud-init status --wait",
        "sudo systemctl is-active neuroshard-reference-expiry.timer",
        "sudo apt-get -o DPkg::Lock::Timeout=300 -o Acquire::Retries=3 update -qq",
        "sudo apt-get -o DPkg::Lock::Timeout=300 -o Acquire::Retries=3 install -y -qq git python3-venv",
        shlex.join(["mkdir", "-p", REMOTE]), "cd " + shlex.quote(REMOTE),
        "git init -q", "git remote add origin https://github.com/neuroshard-ai/neuroshard.git",
        "git sparse-checkout init --no-cone", shlex.join(["git", "sparse-checkout", "set", "--no-cone", *paths]),
        shlex.join(["git", "fetch", "--depth=1", "--filter=blob:none", "origin", source["commit"]]),
        "git checkout --detach FETCH_HEAD"]
    if profile in GRANITE_PROFILES:
        from neuroshard.evolution.granite_reference import ARTIFACTS
        upstream = read(ROOT / ARTIFACTS)["upstream"]
        module, requirements = GRANITE_PROFILES[profile]
        setup_lines += [
            "python3 -m venv .bootstrap",
            ".bootstrap/bin/pip install uv==0.12.19",
            ".bootstrap/bin/uv venv --python 3.12.12 .venv",
            shlex.join([".bootstrap/bin/uv", "pip", "install", "--python", ".venv/bin/python", "-r", requirements]),
            "mkdir -p .upstream/granite-switch",
            "git -C .upstream/granite-switch init -q",
            shlex.join(["git", "-C", ".upstream/granite-switch", "remote", "add", "origin", upstream["repo"]]),
            shlex.join(["git", "-C", ".upstream/granite-switch", "fetch", "--depth=1", "origin", upstream["commit"]]),
            "git -C .upstream/granite-switch checkout --detach FETCH_HEAD",
            "PYTHONPATH=src .venv/bin/python -c " + shlex.quote(
                f"from neuroshard.evolution.{module} import configure,freeze; "
                "configure(); print(freeze()['commit'])")]
    else:
        setup_lines += ["python3 -m venv .venv",
            ".venv/bin/python -m pip install -r docs/evolution-requirements.txt",
            "PYTHONPATH=src .venv/bin/python -c " + shlex.quote(
            "from neuroshard.evolution.modular_reference_execution import configure_runtime,freeze; "
            f"configure_runtime({profile!r}); print(freeze(profile={profile!r})['commit'])")]
    setup = "\n".join(setup_lines)
    setup_seconds = allocation["resources"]["setup_seconds"]
    try:
        result = ssh(home, allocation, ["timeout", "--kill-after=10", str(setup_seconds), "bash", "-s"],
                     data=setup.encode(), timeout=setup_seconds + 15)
        (home / "setup.log").write_bytes(result.stdout)
    except subprocess.CalledProcessError as error:
        (home / "setup.log").write_bytes(error.stdout or b"")
        raise


def collect(home, allocation):
    size = int(ssh(home, allocation, ["du", "-sb", STUDY]).stdout.split()[0])
    if size > allocation["resources"]["outbound_upload_cap_bytes"]:
        raise ValueError("evidence directory exceeded its upload allowance")
    payload = ssh(home, allocation, ["tar", "-czf", "-", "-C", REMOTE, ".study"], timeout=600).stdout
    (home / "evidence.tar.gz").write_bytes(payload)
    # The generated evidence contains only regular directories/files; no model weights.
    import io
    import tarfile
    target = home / "evidence"
    target.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError("unsafe evidence archive")
        archive.extractall(target)
    if (target / ".study/result.json").exists():
        save(home / "result.json", read(target / ".study/result.json"))


def run(home, profile=PROFILE):
    home.mkdir(parents=True, exist_ok=True)
    source = source_freeze(profile)
    wait_for_ci(home, source["commit"])
    if source_freeze(profile) != source:
        raise ValueError("source changed while waiting for CI")
    failure = None
    try:
        save(home / "status.json", {"state": "allocating", "commit": source["commit"]})
        allocation = allocate(home, source["commit"], profile)
        save(home / "status.json", {"state": "setup", "instance_ids": allocation["instance_ids"]})
        bootstrap(home, allocation, source, profile)
        run_args = remote_command(profile)
        remaining = int((datetime.fromisoformat(allocation["deadline"]) - datetime.now(timezone.utc)).total_seconds()) - 600
        if remaining < 60:
            raise TimeoutError("setup exhausted the allocation")
        ssh(home, allocation, ["systemd-run", "--user", "--unit=neuroshard-reference-controller",
            "-p", "RuntimeMaxSec=" + str(remaining), "-p", "MemoryMax=512M", "-p", "MemorySwapMax=0",
            "-p", "TimeoutStopSec=0", "-p", "KillMode=control-group", "--working-directory=" + REMOTE,
            "--setenv=PYTHONPATH=" + REMOTE + "/src", *run_args])
        end = time.monotonic() + remaining
        while time.monotonic() < end:
            status = ssh(home, allocation, ["systemctl", "--user", "show", "--value", "-p", "ActiveState",
                                           "neuroshard-reference-controller"]).stdout.strip()
            try:
                progress = json.loads(ssh(home, allocation, ["cat", STUDY + "/status.json"]).stdout)
                save(home / "status.json", {"state": "running", "remote": progress})
            except subprocess.CalledProcessError:
                pass  # Source/runtime preflight may fail before status exists; service exit is checked below.
            if status not in (b"active", b"activating"):
                break
            time.sleep(30)
        else:
            raise TimeoutError("reference controller exhausted its allocation")
        collect(home, allocation)
        if not (home / "result.json").exists():
            log = ssh(home, allocation, ["journalctl", "--user", "-u", "neuroshard-reference-controller",
                                        "--no-pager", "-n", "80"]).stdout
            (home / "controller.log").write_bytes(log)
            raise RuntimeError("reference service exited without a result")
        if not read(home / "result.json")["execution_completed"]:
            raise RuntimeError("reference execution stopped; inspect collected result")
    except BaseException as error:
        failure = str(error) or type(error).__name__
        save(home / "failure.json", {"error": failure})
        if (home / "allocation.json").exists():
            allocation = read(home / "allocation.json")
            if allocation.get("private_ip") and not (home / "evidence.tar.gz").exists():
                try:
                    collect(home, allocation)
                except Exception as copy_error:
                    save(home / "copy-failure.json", {"error": str(copy_error)})
    finally:
        if (home / "allocation.json").exists():
            retire(home)
        save(home / "status.json", {"state": "failed" if failure else "finished", "error": failure})
    if failure:
        raise RuntimeError(failure)


def remote_command(profile):
    if profile == "granite-adapter-audit":
        return [PYTHON, REMOTE + "/scripts/run_granite_adapter_audit.py", "run",
                "--home", STUDY, "--models", REMOTE + "/.models"]
    if profile == "granite-reference":
        return [PYTHON, REMOTE + "/scripts/run_granite_reference.py", "run",
                "--home", STUDY, "--models", REMOTE + "/.models"]
    if profile == "decoder-parity":
        return [PYTHON, REMOTE + "/scripts/run_modular_decoder_parity.py", "run",
                "--home", STUDY, "--models", REMOTE + "/.models"]
    if profile != PROFILE:
        raise ValueError("unsupported cloud execution profile")
    return [PYTHON, REMOTE + "/scripts/run_modular_reference.py", "run", "--profile", profile,
            "--home", STUDY, "--models", REMOTE + "/.models", "--legacy",
            REMOTE + "/config/experiments/modular-reference-a1-legacy-baseline-result.json"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "retire"))
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(RESOURCE_PROFILES), default=PROFILE)
    args = parser.parse_args()
    if args.command == "retire":
        retire(args.home.resolve())
    else:
        try:
            run(args.home.resolve(), args.profile)
        except Exception as error:
            save(args.home / "status.json", {"state": "stopped", "error": str(error)})
            raise
