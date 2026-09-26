import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from neuroshard.evolution import modular_reference_execution as execution
from neuroshard.evolution.modular_reference import canonical_call, load_plan, score_reply
from neuroshard.evolution.modular_reference_comparison import compare
from neuroshard.evolution.modular_tools import validate_reply


ROOT = Path(__file__).resolve().parents[2]
PLAN, AMENDMENT = execution.PROFILES["fresh-reference"]


def plan():
    return load_plan(ROOT / PLAN)


def rows(successes, model):
    return [{"model": model, "id": task["id"], "text": "fixture", "passed": i in successes,
             "stopped": False, "seconds": 10, "first_token_seconds": 1, "max_rss_bytes": 1000}
            for i, task in enumerate(plan()["tasks"])]


def test_comparison_separates_completed_reference_gains_losses_and_admission():
    contract = plan()
    baseline = rows(set(range(18)), "baseline")  # six successes in every interleaved category
    candidate = rows(set(range(21)), "modular")
    report = compare(contract, baseline + candidate, True)
    assert report["baseline_gate"] and report["reference_ready"] and report["growth_screen_passed"]
    assert len(report["gained_ids"]) == 3 and report["lost_ids"] == []
    assert not report["quality_ready"] and not report["admission_evidence"] and not report["milestone_complete"]
    candidate[0]["passed"] = False
    report = compare(contract, baseline + candidate, True)
    assert report["net_gain"] == 2 and report["reference_ready"]
    assert report["lost_ids"] == [contract["tasks"][0]["id"]]
    assert not report["growth_screen_passed"]  # a higher total cannot hide forgetting
    assert not compare(contract, baseline + candidate, False)["reference_ready"]
    with pytest.raises(ValueError, match="duplicate"):
        compare(contract, baseline + [baseline[0]], True)


def test_empty_category_and_slow_answers_cannot_be_hidden_by_aggregate_scores():
    baseline = rows({i for i in range(24) if i % 3 != 2}, "baseline")
    assert not compare(plan(), baseline)["baseline_gate"]  # 16 total, zero tool calls
    baseline = rows(set(range(18)), "baseline")
    candidate = rows(set(range(21)), "modular")
    for item in candidate[-2:]:
        item["seconds"] = 121  # failed answers are still charged and included in p95
    report = compare(plan(), baseline + candidate, True)
    assert report["latency"]["modular"]["p95_seconds"] == 121
    assert not report["latency_gate"] and not report["growth_screen_passed"]


def test_fresh_contract_has_new_inputs_strict_tool_scoring_and_pinned_prior_cost():
    contract = plan()
    old = load_plan(ROOT / execution.PLAN)
    prior = {json.dumps(t["messages"], sort_keys=True) for t in old["tasks"]}
    assert len(contract["tasks"]) == 24
    assert all(json.dumps(t["messages"], sort_keys=True) not in prior for t in contract["tasks"])
    amendment = execution.read(ROOT / AMENDMENT)
    assert execution.sha256(ROOT / PLAN) == amendment["plan_sha256"]
    assert amendment["new_evaluation_seconds"] == 7200
    assert amendment["environment"]["ATEN_CPU_CAPABILITY"] is None
    for receipt in amendment["additional_evaluations"]:
        assert execution.sha256(ROOT / receipt["path"]) == receipt["sha256"]
    for task in contract["tasks"]:
        if task["kind"] == "tool":
            answer = "<function_calls>" + canonical_call(task["expect"][0]) + "</function_calls>"
            assert score_reply(task, answer, True)["passed"]
            assert not score_reply(task, "Unrequested prose " + answer, True)["passed"]


@pytest.mark.parametrize("acceptable", [True, False])
def test_controller_blocks_modular_until_baseline_passes_and_charges_all_prior_work(tmp_path, monkeypatch, acceptable):
    contract = plan()
    amendment = execution.read(ROOT / AMENDMENT)
    root = tmp_path / "source"
    execution.save(root / PLAN, contract)
    execution.save(root / AMENDMENT, amendment)
    for receipt in amendment["additional_evaluations"]:
        dest = root / receipt["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / receipt["path"]).read_bytes())
    legacy = ROOT / "config/experiments/modular-reference-a1-legacy-baseline-result.json"
    frozen = {"plan_sha256": amendment["plan_sha256"], "commit": "fixture"}
    monkeypatch.setattr(execution, "ROOT", root)
    monkeypatch.setattr(execution, "freeze", lambda **kw: frozen)
    calls = []

    def launch(home, models, binding, which, phase, seconds, memory_bytes, *, task=None, stats=None):
        calls.append((which, phase, seconds))
        if phase == "prepare":
            return {"file_state": {}}
        text = (task["accept"][0] if task["kind"] == "exact" else
                "<function_calls>" + canonical_call(task["expect"][0]) + "</function_calls>")
        if not acceptable and task["kind"] == "tool":
            text = "No function call."
        row = {"model": which, "id": task["id"], "category": task["category"], "phase": phase,
               "binding": execution.identity(binding), "task_sha256": execution.identity(task),
               "text": text, "terminated": True, "token_ids": [1, 2], "stopped": False,
               "seconds": 1, "first_token_seconds": .5, "max_rss_bytes": 1000,
               **score_reply(task, text, True)}
        if task["kind"] == "tool":
            row["wire_validation"] = validate_reply(text, json.loads(task["messages"][0]["functions"]))
        return row

    monkeypatch.setattr(execution, "launch", launch)
    result = execution.run(tmp_path / "study", tmp_path / "models", legacy, "fresh-reference")
    assert result["execution_completed"] is acceptable
    assert any(which == "modular" for which, _, _ in calls) is acceptable
    assert result["accounting"]["baseline"]["historical_evaluation_seconds"] == pytest.approx(11328.797205466837)
    assert result["accounting"]["baseline"]["new_evaluation_seconds"] == 0  # launches mocked, never reported as model work
    if acceptable:
        assert result["reference_ready"] and not result["growth_screen_passed"]  # equal perfect fixture scores
        assert len([1 for _, phase, _ in calls if phase == "replay"]) == 48
    else:
        assert "baseline quality gate failed" in result["error"]


@pytest.mark.parametrize("profile", execution.NATIVE_CPU_PROFILES)
def test_runtime_opt_in_occurs_before_torch_and_does_not_change_default_profile(tmp_path, monkeypatch, profile):
    # pytest's source path does not propagate to subprocesses. CI also installs
    # a wheel, which intentionally excludes the repository experiment contracts.
    # Make a foreign package fail loudly if the child inherits its import path.
    foreign = tmp_path / "neuroshard"
    foreign.mkdir()
    (foreign / "__init__.py").write_text("raise RuntimeError('foreign installed package selected')\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    code = (
        "import os; from neuroshard.evolution.modular_reference_execution import configure_runtime; "
        f"assert os.environ['ATEN_CPU_CAPABILITY']=='default'; configure_runtime({profile!r}); "
        "assert 'ATEN_CPU_CAPABILITY' not in os.environ; assert os.environ['OMP_NUM_THREADS']=='8'; "
        "import torch; "
        f"configure_runtime({profile!r})")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert result.returncode != 0 and "before importing torch" in result.stderr


def cloud_module():
    spec = importlib.util.spec_from_file_location("modular_reference_cloud", ROOT / "scripts/modular_reference_cloud.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cloud_budget_is_checked_before_any_allocation(monkeypatch):
    cloud = cloud_module()
    good = cloud.resources()
    for changes in ({"instances": 2}, {"hours": 9}, {"gpu": True}, {"planning_cap_usd": 2}):
        monkeypatch.setattr(cloud, "read", lambda p: {**good, **changes})
        with pytest.raises(ValueError, match="allowance"):
            cloud.resources()


def test_recovery_resource_cap_counts_prior_allocation_and_requires_retirement(monkeypatch):
    cloud = cloud_module()
    good = cloud.resources()
    original_read = cloud.read
    # Eight new hours would pass a standalone cap, but exceed the combined cap.
    monkeypatch.setattr(cloud, "read", lambda p: {**good, "hours": 8} if p == cloud.ROOT / cloud.RESOURCES
                        else original_read(p))
    with pytest.raises(ValueError, match="combined allowance"):
        cloud.resources()
    prior = original_read(cloud.ROOT / good["prior_resources"]["path"])
    prior["resources_finished"]["remaining_instances"] = ["old-worker"]
    monkeypatch.setattr(cloud, "read", lambda p: good if p == cloud.ROOT / cloud.RESOURCES else prior)
    with pytest.raises(ValueError, match="prior allocation remains live"):
        cloud.resources()


def test_retirement_refuses_a_different_operator_or_protected_instance(tmp_path):
    cloud = cloud_module()
    allocation = {"name": "ours", "commit": "our-source", "created": "2026-09-26T00:00:00+00:00",
                  "resources": cloud.resources()}
    execution.save(tmp_path / "allocation.json", allocation)

    class EC2:
        def describe_instances(self, **kw):
            return {"Reservations": [{"Instances": [{"InstanceId": "someone-else", "Tags": [
                {"Key": "Name", "Value": "ours"}, {"Key": "Source", "Value": "other-source"},
                {"Key": "Purpose", "Value": allocation["resources"]["purpose"]}]}]}]}

        def terminate_instances(self, **kw):
            raise AssertionError("must not terminate mismatched infrastructure")

    with pytest.raises(ValueError, match="ownership mismatch"):
        cloud.retire(tmp_path, EC2())


def test_allocation_has_one_cpu_host_two_deadlines_and_no_surviving_disk(tmp_path, monkeypatch):
    cloud = cloud_module()
    limit = cloud.resources()
    calls = []
    request = {}

    class EC2:
        def describe_instances(self, **kw):
            if "InstanceIds" in kw:
                return {"Reservations": [{"Instances": [{"VpcId": "vpc-fixture", "PrivateIpAddress": "10.0.0.2",
                    "SubnetId": "subnet-fixture", "KeyName": "fixture"}]}]}
            return {"Reservations": [{"Instances": [{"InstanceId": "temporary", "PrivateIpAddress": "10.0.0.3",
                                                       "State": {"Name": "running"}}]}]}

        def describe_images(self, **kw):
            return {"Images": [{**limit["image"], "State": "available"}]}

        def create_security_group(self, **kw):
            calls.append("group")
            return {"GroupId": "sg-fixture"}

        def authorize_security_group_ingress(self, **kw):
            assert kw["IpPermissions"] == [{"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
                                            "IpRanges": [{"CidrIp": "10.0.0.2/32"}]}]

        def run_instances(self, **kw):
            calls.append("instance")
            request.update(kw)
            return {"Instances": [{"InstanceId": "temporary"}]}

    monkeypatch.setattr(cloud.boto3, "client", lambda *a, **kw: EC2())
    monkeypatch.setattr(cloud.subprocess, "run", lambda *a, **kw: calls.append("guard"))
    old_read = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda p, *a, **kw: "ssh-ed25519 fixture-key" if p.name == "id_ed25519.pub"
                        else old_read(p, *a, **kw))
    result = cloud.allocate(tmp_path, "fixture-commit")
    assert calls == ["guard", "group", "instance"]
    assert request["MinCount"] == request["MaxCount"] == 1
    assert request["InstanceType"] == "r7i.4xlarge"
    assert request["InstanceInitiatedShutdownBehavior"] == "terminate"
    assert "IamInstanceProfile" not in request
    disk = request["BlockDeviceMappings"][0]["Ebs"]
    assert disk["DeleteOnTermination"] and disk["Encrypted"]
    assert disk["VolumeSize"] == 160
    cloud_config = json.loads(request["UserData"].split("\n", 1)[1])
    assert any("OnCalendar=" in f["content"] and "Persistent=true" in f["content"]
               for f in cloud_config["write_files"])
    assert result["instance_ids"] == ["temporary"]
