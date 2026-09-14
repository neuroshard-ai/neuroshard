# Adaptive shards and native GPU settlement, 2026-09-14

A 1,711,376,384-parameter model changed from two GPU owners to three while
preserving every learned weight and Adam moment. Replaying the next two updates
under either layout produced the same learned-state root. A fourth GPU replayed
all three partitions sequentially, and an isolated native chain rejected a forged
checkpoint and paid for the genuine updates. These are operated experiments on
matching A10G hardware, with one infrastructure owner.

The [frozen plan](../config/experiments/adaptive-shards.json),
[input/source commitments](../config/experiments/adaptive-shards-inputs.json),
[learning implementation](ADAPTIVE_SHARDS.md), and
[native verification rules](NATIVE_SHARD_REPLAY.md) define separate computation,
quality and settlement claims. Numerical sources were frozen at `ad7d2ca` before
training. The public 0.4.0 chain and its serving model were not changed.

## Ownership and exact continuation

| Layout | Parameters held by each worker | Layer boundaries |
| --- | --- | --- |
| Two workers | 838,907,904; 872,468,480 | 0, 11, 24 |
| Three workers | 503,343,104; 604,016,640; 604,016,640 | 0, 6, 15, 24 |

Each owner holds its student partition, local Adam state and a frozen reference
partition. The reference supplies a KL regularizer on previously trained replay
examples. No training worker constructs the full student or reference model.
The first worker owns the tied embedding/output head and final norm.

Redistribution at update 32 took 54.32 seconds, including transfer, verification,
durable staging and agreement on the new commit. Old learned tensors, optimizer
groups and parameter ages were unchanged. The learned-state root after updates
33–34 was exactly
`bfd1b7eeca535ce6d82e98c674a2e2e33fde2f7a3c55fc5989b09810b8caf217`
under both layouts. Full checkpoint roots differ because they also commit
ownership, parent and process RNG metadata. Excluding RNG from the learned root
is restricted to this dropout-free numerical profile; its updates consume no
random draws.

The operated controller restarts the process group at a committed cursor. This
demonstrates redistribution and continued training, rather than automatic public
discovery or uninterrupted membership changes. The separate
[host-replacement experiment](SHARDED_TRAINING_RESULTS.md) exercised physical
failure recovery; that failure was not reinjected in this redistribution trial.

## Complete replay and its cost

Three complete replay executions covered every partition of updates 33–34.
All final parameter and Adam tensors matched. A further execution on a fourth
24 GB GPU processed the three partitions sequentially: at most 604,016,640
student parameters were resident, with a peak CUDA allocation of 13,917,486,080
bytes including the local reference, optimizer and intermediates.

That one-GPU execution took 595.12 seconds end to end. The three reported
replay-and-write intervals total 138.16 seconds; they exclude initial checkpoint
loading. The complete three-worker training process for this captured two-update
window took 115.83 seconds. These differently provisioned paths are not an
equal-hardware performance comparison. Loading, hashing and writing many tensor
files materially affected the single auditor's wall time.

The closed transcript contains 3,995,785,128 unique tensor bytes, plus metadata,
for just two updates. Its commitment is
`49bfba54fcf2206de3223cc6c00cef7b99141b45c4ec96d14bf164361ec68274`.
Each send matches a receive, collective outputs follow their inputs, and actual
replay checks local computation and outgoing values. Metadata closure alone is
insufficient. A deliberately altered output tensor commitment, with recomputed
checkpoint and transcript bindings, failed numerical replay. CPU adversarial
tests also changed matching send/receive values together; neural replay rejected
that self-consistent forged witness.

Verification now starts from a recent complete checkpoint and fits one shard's
GPU memory. It still repeats all computation and requires substantial artifact
bandwidth. Multiple audit signers were controlled by one operator; these runs
do not demonstrate independent ownership or economical permissionless auditing.

## Native settlement and recovery

Four local CometBFT validators ran the optional native replay-quorum profile.
They used the real GPU checkpoint/transcript commitments and locally verified
results from the completed numerical replay. The trial reused these results;
it did not rerun the GPUs for every ledger transaction. Cached reports were
checked against their exact input, output and recorded computation.

A wrong-output claim received one conflicting positive report and three negative
reports. The native quorum rejected it without issuing training rewards. The
genuine window then settled while the fourth validator and its application were
offline. The ledger issued **2,000,000 atoms, or 2 experimental NEURO**, for
updates 33 and 34. Six paid audit services received **3,600,000 atoms from sponsor
escrow** across the rejected and accepted claims. Duplicate/stale work failed.

After the fourth validator restarted, all four agreed at height 68 on app hash
`2470F111FE35789C79DE5F196AF4101CC26BDAF1723DF533568F7FE738B3A498`.
All four stored states satisfied supply conservation. The chain ID was
`neuroshard-sharded-replay-1271401c9417`, with genesis SHA-256
`9197ff65006f7ac0a036eb695b350ec74d6e8212f3e59745ff43919d7da36517`.
All temporary native processes were stopped after the trial.

The trial exposed and corrected an audit-window liveness bug: a quorum of opaque
commitments does not imply a quorum for either eventual verdict. Early commitment
closure now requires every bonded participant; otherwise the declared commitment
deadline applies. Reveal closure still uses an actual consistent verdict quorum.
The corrected path passed the real four-validator trial and regression tests.

The security assumption is native bonded voting weight, not proof that signatures
represent different people. A threshold of colluding signers can approve a
forgery; an explicit test retains that counterexample. This bridge enables one
frozen job in a dedicated genesis. It does not yet activate future datasets,
settle growth, or connect these 1.7B checkpoints to public paid inference.

## Learning decision

The fixed 128-update endpoint is evaluated separately against the frozen seed.
Acceptance requires held-out task improvement, conversation retention within the
declared margin, and no reduction in correctly generated answers. Phase B and
its larger-model comparison depend on that result; successful computation or
native issuance cannot substitute for the quality gate.
