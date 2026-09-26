# Fresh BAR reference comparison

**Pre-execution status:** the first queue stopped at CI, before allocation.
The [test correction](MODULAR_REFERENCE_FRESH_CI.md) preserves the entire experiment
contract and requires green CI at its new committed revision before launch.

**Purpose:** measure what published neural composition adds, what it loses, and
what it costs, before training a NeuroShard expert. This is the next A1 reference
measurement for the [assistant checklist](../TODO_ASSISTANT.md). It does not
prove repeated growth, sharded execution or independent operation.

The [interface diagnostic](MODULAR_TOOL_INTERFACE_RESULTS.md) resolved the
observed formatting failures on three opened cases. Its 3/3 cannot be combined
with the original answers or reused as fresh evidence. This contract uses 24
new authored cases, frozen before generation. The former failed run stays closed.

## Inputs and gates

The [case plan](../config/experiments/modular-reference-fresh.json) contains eight
conversation/context tasks, eight instruction tasks and eight tool-selection
tasks, interleaved in that order. Context tasks include revisions to remembered
information, constraints and short reasoning. Tool tasks include competing
functions, argument types and account/event identifiers from earlier turns.
All outputs have deterministic scoring; the tool calls are parsed as data and
never executed against a live service.

These are 24 short reference cases, not a general chat benchmark. They are
public at the freeze; secrecy and freedom from pretrained contamination are
not claimed. They were authored after the interface diagnosis, without seeing
any new model replies. No earlier programming final or opened 64-case set is
used. Rendered inputs are at most 315 tokens; expected replies including EOS
need at most 38 tokens, within the declared 128-token cap.

Both checkpoints and all their artifact hashes remain those in the
[original reference decision](MODULAR_REFERENCE.md): published BAR-7B and
BAR-5x7B, without training, quantization, merging changes or top-k changes.
The modular checkpoint still activates all five experts; this comparison makes
no claim of constant computation as experts are added.

The execution order and gates are fixed:

1. Download/hash the baseline and generate all 24 answers once. Require at least
   **16/24 correct overall and 4/8 in each category**. Otherwise stop before
   downloading the modular checkpoint.
2. Independently replay every baseline answer, including wrong answers, with
   fresh processes. Require exact decoded text, termination and score agreement;
   also report token agreement. Only then prepare the modular checkpoint.
3. Generate and independently replay its same 24 cases. Any execution failure
   or replay disagreement invalidates the completed-reference claim.
4. Publish every baseline success, each gained and lost answer, category scores,
   and latency/memory for all primary replies. A higher total cannot hide a loss.

`reference_ready` means the baseline gate and complete paired comparison with
replay passed. It does not mean the composed model is better. A separate
`growth_screen_passed` requires **at least two newly correct answers, zero lost
baseline successes, modular p95 time at most 120 seconds and at most four times
baseline p95**. Even that is limited supporting evidence on 24 cases. Both
paths keep `quality_ready`, `admission_evidence` and `milestone_complete` false.
A1 requires review of the complete evidence and published recipe/placement
deviations; A2 still needs our own useful contributed capability.

## Execution amendment and resources

The [execution contract](../config/experiments/modular-reference-fresh-execution.json)
declares a new numerical profile. It does not resume either old study.
PyTorch remains 2.9.1+cpu, Transformers 4.57.3 and bfloat16 eager attention.
Greedy decoding and cache remain enabled. Native CPU instruction selection and
eight threads replace the legacy one-thread SSE profile. Runtime setup happens
before PyTorch import, requires the declared AMX/BF16 CPU features, and is
included in each source/runtime binding. Replays must match on the new host;
cross-profile equality with the old CPU answers is not claimed. NeuroShard's
native ledger numerical profile is unchanged.

The [resource contract](../config/experiments/modular-reference-fresh-resources.json)
allows exactly one temporary `r7i.4xlarge` in us-east-1: 16 vCPUs, 128 GiB RAM,
160 GiB gp3 storage, no GPU. [AWS lists its CPU and memory specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/mo.html).
The published on-demand price record is SKU `QN6NV26MDYTYVNSE`, effective
2026-09-01, **$1.0584/hour**. Eight hours is **$8.4672 compute**, with a **$15 total
planning cap** including reserved storage/network headroom. Other running
services are not part of this allowance and are not changed.

- CI must pass on the exact committed source before allocation. Maximum CI
  waiting time is one hour, without a paid worker.
- One allocation attempt; at most eight hours from allocation, including setup,
  download, generation, replay and evidence copy. No automatic relaunch.
- Setup is bounded at one hour. Each checkpoint gets at most one hour for
  artifact preparation and two hours of new primary/replay execution.
- Each reply has a 600-second limit, 128 generated tokens and a 112 GiB cgroup
  memory cap, with swap disabled. All wrong answers and failed/partial launches
  are charged. A timed-out reply is an execution failure, not an ordinary miss.
- Earlier baseline and interface evaluation remain recorded and charged at
  **11,328.797 seconds**. The historical preparation record includes 78.283
  known seconds plus the original unmeasured download. New allocation costs
  are reported separately; no earlier work becomes free or gets counted twice.
- A host timer terminates the instance at the absolute deadline, and a separate
  controller-side guard also retires the allocation. The successful path copies
  evidence first, then terminates the instance and confirms volume/security-group
  removal. Shutdown is configured to terminate rather than leave a stopped disk.

The single host is a reference oracle with sufficient memory. It is not evidence
of decentralized hosting, faster multi-peer inference or an independent operator.
Weights remain on that disposable host and are not committed or copied back.

## Run and handoff

Commit the contract and implementation first. From an isolated checkout at that
commit, with the existing AWS provisioning role and GitHub CLI:

```bash
PYTHONPATH=src python scripts/modular_reference_cloud.py run \
  --home /absolute/path/to/new-reference-allocation
```

Run the controller as a persistent user service. It waits for exact-commit CI,
allocates and configures the host, installs pinned dependencies, executes the
comparison and copies evidence. `status.json` reports the current phase.
`result.json`, `evidence/`, `setup.log` and `resources-finished.json` provide the
answer record, worker receipts, setup evidence and resource retirement. A failed
run records its error and does not restart.

After completion, publish the full gain/loss record before choosing the A2
training contract. A failed baseline changes the foundation decision. A valid
comparison with forgetting is evidence against adopting that composition as
our growth rule; it must not be described as preserved assistant capability.
