# Granite shard training results (A4, second execution)

## Second attempt: passed

**All seven checks passed.** With the declared warm-up on every owner and on
the reference host, the three owners trained the addition arm on the real model
to the same bits as the unchanged single-host trainer: six losses, twelve
preference margins and all 16 LoRA tensors (1,048,576 parameters).

The arm's owner was lost when told to take step 3. The group was relaunched from
its step-3 checkpoint and owner 0's saved references, and it again finished with
the reference's exact tensors and the remaining three losses.

Evidence: [result](../config/experiments/granite-shard-training2-result.json) and
[report](../config/experiments/granite-shard-training2-report.json), commit `01fe9b0`.

- **Memory.** Owner peaks were 5.30, 4.93 and 17.05 GB, against 22.43 GB for the
  reference host holding the complete model.
- **Time.** The owners took 599 s for six steps; the reference took 609 s.
- **Cost.** Four r7i.4xlarge hosts, $1.94, all retired.

Training across machines is now exact and recoverable. The learned module can be
trained by owners who never hold the backbone, can survive the loss of the owner
holding it, and ends where a single host would.

## First attempt

**Failed one of seven declared checks (recovery), so the execution fails.** The
main question still got a clear answer. Training the assistant's addition arm
across three owner hosts, with none holding the backbone, produced exactly what
the unchanged single-host trainer produced on the complete model. All six step
losses, all twelve preference margins and the SHA-256 of all sixteen LoRA tensors
are identical. The outage run did not reproduce those bits. Its very first
forward pass differed in the last place from the uninterrupted run on the same
hosts, so its checkpoint and resumed tensors differ too.

Declaration: [GRANITE_SHARD_TRAINING.md](GRANITE_SHARD_TRAINING.md).
Evidence: [result](../config/experiments/granite-shard-training-result.json) and
[report](../config/experiments/granite-shard-training-report.json), commit `b789206`.

## Checks

| Check | Outcome |
| --- | --- |
| Complete | Pass: the reference and all three owners finished 6 steps. |
| Fetched only owned | Pass: owners fetched 2.24, 2.05 and 2.05 GB in about 30 s each. The reference downloaded the full checkpoint. |
| Tensors | Pass: 16/16 LoRA tensors (1,048,576 parameters) have the reference's SHA-256. |
| Losses | Pass: all 6 losses and 12 preference margins equal the reference's exactly. |
| Outage injected | Pass: owner 2 exited when told to take step 3, and owners 0 and 1 saw the connection reset. |
| Recovery | **Fail**: the resumed tensors and the losses of steps 3–5 differ from the reference. |
| Memory | Pass: owner peaks 5.3, 4.9 and 17.1 GB against 22.4 GB for the reference host. |

## Why recovery failed

- **What differed.** Owner 0 computes eight preference references before step 0.
  In the outage run, the first of them (the first forward pass of each fresh
  owner process) came out as −0.155652. The uninterrupted run and the reference
  host both computed −0.155187. The other seven references are identical.
- **How it propagated.** Every outage-run step inherited that difference. Its
  step-0 gradient norm was 0.6610495 against 0.6610818, so its step-3 checkpoint
  was not the uninterrupted state.
- **What is not implicated.** The resume path itself was consistent: the
  checkpoint step, the saved references and the continued schedule all lined up.
  Resumes on small checkpoints reproduce the trainer bit for bit.

This is a finding about the runtime, not the protocol. A fresh process on the
same host can produce a different first forward pass. Earlier evidence could not
have caught this, because the canonical runtime had only been checked at the
token level. Before this is repeated, a diagnostic should measure first-pass
variation across fresh processes on this instance type, and the owners should
make their first pass reproducible, for example by a declared warm-up.

## Other measurements

Owners took 585 s for the six steps; the single reference host took 625 s. Owner
0 sent 0.95 GB: hidden states to owner 1 and boundary gradients to owner 2.
Owners 1 and 2 each sent 0.50 GB of hidden states. Only owner 2 held trainable
tensors, optimizer state and checkpoints.

## Resources

Four r7i.4xlarge hosts cost a conservative $1.89. Every host is retired.

## Meaning for A4

A4 now has three pieces of evidence:

- Generation across owners equals the complete model.
- Training across owners equals the complete model, on the real 3B assistant.
- A lost inference owner recovers to the canonical tokens.

Bit-exact recovery of training after a lost owner is not established. No
checklist credit or admission evidence.
