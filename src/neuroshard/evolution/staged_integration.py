"""A separate CPU mechanism candidate: expert training, then learned routing.

This does not change the failed learned-integration experiment or authorize
GPUs, model promotion, a larger backbone, or independent hosting.
"""
import copy
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path

import torch
from torch import nn

from neuroshard.evolution.reference_data import identity, save, sha256
from neuroshard.evolution.seed import FILES, MODEL_REPO, MODEL_REVISION


FORMAT = "neuroshard-staged-integration-v1"
CONTRACT_IDENTITY = "73183bed8c7c0d4f5055fc1ce8b2e0c38dc61923a03777edf708650f3a1cb3d6"
PLAN = "config/experiments/staged-integration.json"
DATA = "config/experiments/staged-integration-data.json"
FREEZE = "config/experiments/staged-integration-freeze.json"
ROLES = ("expert_new", "expert_replay", "gate_new", "gate_replay", "development", "retention")
TRAIN_ROLES = ROLES[:4]
EVAL_ROLES = ROLES[4:]
SOURCES = (
    PLAN, DATA, "docs/STAGED_INTEGRATION.md", "scripts/run_staged_integration.py",
    "src/neuroshard/evolution/staged_integration.py",
    "src/neuroshard/evolution/staged_integration_run.py",
    "src/neuroshard/evolution/seed.py",
    "src/neuroshard/evolution/reference_data.py",
    "src/neuroshard/evolution/data.py", "src/neuroshard/dataflow/store.py",
    "docs/evolution-requirements.txt", "docs/llm-requirements.txt",
)


def root():
    for directory in Path(__file__).resolve().parents:
        if (directory / PLAN).is_file():
            return directory
    raise FileNotFoundError("Staged integration requires its source checkout")


def load_spec():
    spec = json.loads((root() / PLAN).read_text())
    if identity(spec) != CONTRACT_IDENTITY:
        raise ValueError("Staged-integration contract changed; declare a separate candidate")
    if (spec["model"] != {"repo": MODEL_REPO, "revision": MODEL_REVISION, "files": FILES}
            or spec["host"] != "cpu" or spec["gpu_launch_authorized"] is not False
            or spec["admission_evidence"] is not False):
        raise ValueError("Only the pinned 135M CPU mechanism is allowed")
    if spec["confirmation_opened"] or spec["original_final_opened"] or spec["reuses_opened_64"]:
        raise ValueError("Earlier evaluation sets remain closed to this candidate")
    return spec


def make_data(spec):
    """New generated inputs; no MBPP files, IDs, labels or model outputs are read.

    Unordered operand pairs are unique across all roles, including across task
    families. Different values, not surface paraphrases, define the split.
    """
    pairs = [(a, b) for a in range(20) for b in range(a, 20)]
    pairs.sort(key=lambda pair: identity({"namespace": FORMAT, "seed": spec["data"]["seed"],
                                        "pair": pair}))
    rows, offset = {}, 0
    for role in ROLES:
        count = spec["data"]["counts"][role]
        selected = pairs[offset:offset + count]
        if len(selected) != count:
            raise ValueError("The generated operand pool is exhausted")
        offset += count
        modular = role in {"expert_new", "gate_new", "development"}
        rows[role] = []
        for a, b in selected:
            task = {"family": "modular-addition" if modular else "addition", "a": a, "b": b}
            if modular:
                prompt = f"What is the remainder when {a} + {b} is divided by 7? Reply with only the number."
                answer = str((a + b) % 7)
            else:
                prompt = f"What is {a} + {b}? Reply with only the number."
                answer = str(a + b)
            rows[role].append({"id": identity({"namespace": FORMAT, **task}), **task,
                               "messages": [{"role": "user", "content": prompt}], "answer": answer})
    return {"format": FORMAT + "/data", "source": "generated-arithmetic-mechanism-only",
            "license": "Apache-2.0", "roles": rows, "confirmation": None}


def load_data(spec):
    data = json.loads((root() / DATA).read_text())
    if data != make_data(spec):
        raise ValueError("Staged data differs from the frozen generator and splits")
    return data


def freeze_inventory():
    spec = load_spec()
    load_data(spec)
    return {"format": FORMAT + "/freeze", "contract": identity(spec),
            "host": "cpu", "gpu_launch_authorized": False, "admission_evidence": False,
            "files": {name: sha256(root() / name) for name in SOURCES}}


def bind_freeze(*, committed=False):
    saved = json.loads((root() / FREEZE).read_text())
    if saved != freeze_inventory():
        raise ValueError("Staged source/data differ from the candidate freeze")
    if committed:
        for name in (*SOURCES, FREEZE):
            try:
                content = subprocess.check_output(["git", "show", "HEAD:" + name], cwd=root(),
                                                  stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as error:
                raise ValueError("Commit the staged candidate before running: " + name) from error
            if content != (root() / name).read_bytes():
                raise ValueError("Commit the staged candidate before running: " + name)
    return identity(saved)


class StagedMixture(nn.Module):
    """Two copies of one MLP, trained in explicit, non-overlapping phases.

    Expert training guarantees access. Gate training uses both frozen outputs
    and a differentiable mixture. Serving uses one hard choice with incumbent
    tie-break; it cannot force the added expert. The soft/hard mismatch is
    intentional and evaluated, never counted as automatic success by itself.
    """

    def __init__(self, parent):
        super().__init__()
        self.incumbent = copy.deepcopy(parent)
        self.added = copy.deepcopy(parent)
        self.router = nn.Linear(parent.gate_proj.in_features, 2, bias=True)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        self.phase = "serve"
        self.trace = []
        self.training_routes = None
        self.record_routes = False
        self.set_phase("serve")

    def set_phase(self, phase):
        if phase not in {"expert", "gate", "serve"}:
            raise ValueError("Unknown staged phase")
        self.phase = phase
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        target = self.added if phase == "expert" else self.router if phase == "gate" else None
        if target is not None:
            for parameter in target.parameters():
                parameter.requires_grad_(True)

    def forward(self, hidden):
        if hidden.device.type != "cpu":
            raise ValueError("Staged integration is CPU-only")
        if self.phase != "serve" and not self.training:
            raise ValueError("Forced/soft routing is training-only")
        if self.phase == "expert":
            self.training_routes = {"mode": "forced-training-only", "tokens": hidden.numel() // hidden.shape[-1],
                                    "expert_evaluations_per_token": 1}
            return self.added(hidden)
        logits = self.router(hidden)
        if self.phase == "gate":
            probabilities = logits.softmax(dim=-1)
            self.training_routes = {
                "mode": "soft-training-only", "tokens": hidden.numel() // hidden.shape[-1],
                "expert_evaluations_per_token": 2,
                "hard_added_tokens": int((logits.argmax(-1) == 1).sum().detach()),
                "mean_added_probability": float(probabilities[..., 1].mean().detach()),
                "mean_added_margin": float((logits[..., 1] - logits[..., 0]).mean().detach()),
            }
            return (probabilities[..., :1] * self.incumbent(hidden)
                    + probabilities[..., 1:] * self.added(hidden))
        choices = logits.argmax(dim=-1)
        flat, selected = hidden.reshape(-1, hidden.shape[-1]), choices.reshape(-1)
        output = torch.zeros_like(flat)
        for index, module in enumerate((self.incumbent, self.added)):
            mask = selected == index
            if bool(mask.any()):
                output[mask] = module(flat[mask])
        if self.record_routes:
            margin = (logits[..., 1] - logits[..., 0]).detach()
            self.trace.append({"choices": choices.detach().tolist(), "added_margin": margin.tolist(),
                               # Batch size is one for generation. Only each call's
                               # last position directly supplies the next output token.
                               "answer_choices": choices[..., -1].detach().reshape(-1).tolist(),
                               "answer_margins": margin[..., -1].reshape(-1).tolist()})
        return output.reshape(hidden.shape)


def tensor_identity(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write_checkpoint(directory, module, optimizer, *, binding, phase, step):
    """Save numerical state and RNG before scoring; never overwrite a candidate."""
    from safetensors.torch import save_file
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    save_file({name: tensor.detach().cpu().contiguous().clone()
               for name, tensor in module.state_dict().items()}, str(directory / "weights.safetensors"))
    torch.save({"optimizer": optimizer.state_dict() if optimizer is not None else None,
                "torch_rng": torch.get_rng_state(), "python_rng": random.getstate()},
               directory / "training.pt")
    manifest = {"format": FORMAT + "/checkpoint", "binding": binding, "phase": phase,
                "step": step, "module": tensor_identity(module),
                "files": {name: sha256(directory / name) for name in ("weights.safetensors", "training.pt")}}
    save(directory / "manifest.json", manifest)
    return identity(manifest)


def read_checkpoint(directory, module, *, binding, phase):
    from safetensors.torch import load_file
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("binding") != binding or manifest.get("phase") != phase:
        raise ValueError("Checkpoint belongs to another phase or experiment")
    if set(manifest["files"]) != {"weights.safetensors", "training.pt"}:
        raise ValueError("Incomplete checkpoint inventory")
    for name, digest in manifest["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError("Checkpoint bytes changed: " + name)
    module.load_state_dict(load_file(str(directory / "weights.safetensors"), device="cpu"), strict=True)
    if tensor_identity(module) != manifest["module"]:
        raise ValueError("Restored checkpoint differs")
    return manifest


def _aligned(reference, candidate):
    ids = [row["id"] for row in reference]
    if len(ids) != len(set(ids)) or ids != [row["id"] for row in candidate]:
        raise ValueError("Evaluation identities differ or repeat")


def preservation(parent, candidate, minimum):
    _aligned(parent, candidate)
    protected = [row["id"] for row in parent if row["passed"]]
    lost = [a["id"] for a, b in zip(parent, candidate) if a["passed"] and not b["passed"]]
    gained = [a["id"] for a, b in zip(parent, candidate) if not a["passed"] and b["passed"]]
    return {"protected": protected, "lost": lost, "gained": gained,
            "nonempty_baseline": len(protected) >= minimum,
            "passed": len(protected) >= minimum and not lost}


def p95(values):
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Missing or invalid cost measurement")
    return sorted(values)[math.ceil(len(values) * .95) - 1]


def validate_routes(row, maximum_tokens):
    tokens = row.get("generated_tokens")
    trace = row.get("routes")
    if type(tokens) is not int or not 1 <= tokens <= maximum_tokens or not isinstance(trace, list):
        raise ValueError("Missing generation/routing trace")
    if len(trace) != tokens:
        raise ValueError("Each generated token needs its automatic routing decision")
    used = 0
    for call in trace:
        choices, margins = call.get("choices"), call.get("added_margin")
        if (not isinstance(choices, list) or len(choices) != 1 or not choices[0]
                or not isinstance(margins, list) or len(margins) != 1
                or len(choices[0]) != len(margins[0])):
            raise ValueError("Invalid batch-one routing trace")
        for choice, margin in zip(choices[0], margins[0]):
            if (type(choice) is not int or type(margin) not in (int, float)
                    or not math.isfinite(margin) or choice != int(margin > 0)):
                raise ValueError("Trace disagrees with automatic top-1 routing")
        if (call.get("answer_choices") != [choices[0][-1]]
                or call.get("answer_margins") != [margins[0][-1]]):
            raise ValueError("Answer decision must use the last position of each call")
        used += choices[0][-1]
    if row.get("added_answer_tokens") != used:
        raise ValueError("Added use disagrees with the recorded trace")


def score(spec, parent, expansion, control, training, control_training):
    """Only automatic generation can pass. Training loss is diagnostic evidence."""
    expected = make_data(spec)["roles"]
    for role in EVAL_ROLES:
        for arm in (parent, expansion, control):
            _aligned(expected[role], arm[role])
            for row, truth in zip(arm[role], expected[role]):
                if row.get("automatic") is not True:
                    raise ValueError("Only automatic serving may be scored")
                if (type(row.get("passed")) is not bool or row.get("answer") != truth["answer"]
                        or row["passed"] != (row["text"].strip() == truth["answer"])):
                    raise ValueError("Score does not match generated text and frozen answer")
        _aligned(parent[role], expansion[role])
        _aligned(parent[role], control[role])
        for row in expansion[role]:
            validate_routes(row, spec["training"]["generation_tokens"])
    kept = preservation(parent["retention"], expansion["retention"],
                        spec["gates"]["minimum_parent_retention_correct"])
    counts = {name: sum(row["passed"] for row in arm["development"])
              for name, arm in (("parent", parent), ("expansion", expansion), ("control", control))}
    gain = (counts["expansion"] >= max(counts["parent"], counts["control"])
            + spec["gates"]["minimum_new_gain"])
    added_on_gain = any(b["passed"] and not a["passed"] and b["added_answer_tokens"] > 0
                        for a, b in zip(parent["development"], expansion["development"]))
    latency_exp = p95([row["seconds"] for role in EVAL_ROLES for row in expansion[role]])
    latency_control = p95([row["seconds"] for role in EVAL_ROLES for row in control[role]])
    gates = {
        "automatic_gain": gain, "per_answer_preservation": kept["passed"],
        "added_module_used_on_gain": added_on_gain,
        "latency": latency_exp <= min(spec["gates"]["maximum_p95_seconds"],
                                      latency_control * spec["gates"]["maximum_latency_ratio"]),
        "isolated_peak_memory": (expansion["peak_rss_bytes"] <= spec["gates"]["maximum_peak_rss_bytes"]
                                 and expansion["peak_rss_bytes"] <= control["peak_rss_bytes"]
                                 * spec["gates"]["maximum_memory_ratio"]),
        "training_budget_control": (control_training["matched_budget"]
                                    and control_training["training_cpu_seconds"] >= training["comparison_cpu_seconds"]),
        "frozen_incumbent": training["incumbent_unchanged"],
        "gate_only_phase": training["added_unchanged_during_gate"],
    }
    passed = all(gates.values())
    if gain and added_on_gain:
        diagnosis = "complete_system_evaluated"
    elif not training["expert_training_signal"]:
        diagnosis = "expert_learning_not_observed"
    else:
        diagnosis = "automatic_integration_not_effective"
    return {"format": FORMAT + "/result", "passed": passed, "gates": gates,
            "counts": counts, "retention": kept, "diagnosis": diagnosis,
            "diagnosis_scope": "Training loss supports a mechanism diagnosis, not useful generalization.",
            "expansion_p95_seconds": latency_exp, "control_p95_seconds": latency_control,
            "admission_evidence": False, "gpu_launch_authorized": False,
            "confirmation_opened": False, "item4_complete": False,
            "next": "review-cpu-mechanism-evidence" if passed else "stop-this-candidate"}
