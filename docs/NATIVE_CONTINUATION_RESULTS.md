# Native settlement of sharded continual learning

The 15 September 2026 integration connects a passing learning recipe to native
training rewards, separate serving approval and token-paid inference. Three GPU
shard owners reproduce all 64 updates of the successful
[answer-balanced continuation](BALANCED_CONTINUATION_RESULTS.md). Workers start
with zero balances and earn **64 experimental NEURO**. Complete quality replay
then promotes the checkpoint, and a customer spends transferred worker earnings
on a complete distributed response.

The [machine-readable results](../config/experiments/native-continuation-results.json)
include every accepted window, financial amounts, quality decisions, recovery
checks and evidence identifiers. This is an isolated integration network under
one operator. The public 0.4.0 network and PyPI release retain their existing
execution profile.

## What the model reproduces

The model has 1,711,376,384 parameters and 24 layers, partitioned at
`[0, 6, 15, 24]`. Its learned weights and Adam state exactly reproduce R4 at both
the midpoint and terminal checkpoint. Every baseline and candidate answer on
the repeated quality sets also matches R4 exactly.

| Measurement | Baseline | Candidate | Decision |
| --- | ---: | ---: | --- |
| New-task correct answers | 378/512 | 462/512 | Pass |
| Arithmetic correct answers | 4/128 | 87/128 | Pass |
| Prior-task correct answers | 188/256 | 188/256 | No individual correct answer lost |
| Conversation response loss | 0.544084 | 0.546741 | Delta upper bound +0.004065 nats, below +0.02 |

These are already-exposed evaluation cases. Repeating them establishes numerical
and protocol integration; it supplies no new independent learning evidence.
The tasks cover four generated families, with bounded greedy generation. They
do not measure general assistant capability. The frozen final policy requires
non-decreasing prior family counts; zero individual losses are additionally
observed here and were required by the preceding development gate.

## Training acceptance and issuance

The activated job starts at step 384 and ends at 448. Each four-update
reservation names the current checkpoint and three workers. Each worker trains
its partition, signs its work and makes the corresponding evidence available.
Three audit services each replay all three partitions before acceptance.
The 16 accepted reservations contain 48 complete audit reports and 144
partition replays. Sixty-four distinct update identities receive payment.

Training issues exactly 64,000,000 atoms; this profile uses 1,000,000 atoms per
NEURO. Worker earnings are 18.823424, 22.588288 and 22.588288 NEURO.
Training acceptance advances the trainable checkpoint while leaving the
previous serving checkpoint in place. A separate quality statement binds the
completed job, baseline, candidate and evaluation policy. All three quality
auditors replay every partition before the ledger changes `serving_root`.
Quality promotion issues zero tokens.

Audit fees come from existing sponsor balances: 57.6 NEURO for training,
614.4 for quality and 42 for the inference demonstration, totaling 714 NEURO.
The sponsor also provides 20 NEURO of collateral capital to the first provider,
whose training earnings alone do not cover the required bond. These are
disclosed subsidies, separate from the 64 NEURO of work issuance.

## Spending earned tokens

A fresh customer receives 1 NEURO from a rewarded worker and locks a
256-token inference budget. The serving checkpoint generates 140 tokens,
including the termination token, in 61.25 seconds. Its actual response is:

```text
<work>
A664: 19 * 4 = 76; running total = 76
P214: 16 * 18 = 288; running total = 364
E577: 12 * 15 = 180; running total = 544
C763: 8 * 17 = 136; running total = 680
H597: 11 * 3 = 33; running total = 713
</work>
{"total":713}
```

No repair is applied. This is a fixed, exposed demonstration request, not an
additional quality test. Three auditors again replay all three partitions;
their complete replay durations are 111.91, 108.41 and 122.53 seconds.

The accepted receipt pays providers 0.014 NEURO, refunds 0.0116 NEURO of unused
budget and leaves the customer with 0.985 NEURO after the 0.001 transaction fee.
Inference issues no tokens. The 42 NEURO sponsor audit payment greatly exceeds
the service price; this experiment demonstrates accounting, not sustainable
pricing. Prompts, response tokens and financial records are public.

## Recovery and ledger verification

During transfer of the successfully trained step-444 checkpoint, rsync times
out before claim submission. The controller restarts the same chain with the
same keys and 56 settled updates. It resumes transfer of the preserved output
without repeating training or signing a replacement claim. Controller recovery
takes 384.36 seconds. A separate 897.57-second pause stages verified reference
objects on existing local SSDs. Both interruptions remain in the reported
training-window wall time of 17,216.70 seconds.

After paid inference completes, the verification driver checks:

- Restarting the same four validator homes preserves financial state and work
  identities.
- Resubmitting the exact original signed paid-claim bytes is rejected for its
  old nonce, with no additional payment or issuance.
- Three validators continue for ten blocks while the fourth process is down;
  the same fourth validator rejoins and catches up.
- All four saved states match their committing header hashes.
- Offline application replay verifies 30,408 consecutive exported headers and
  221 signed transactions, reconstructs the saved state and confirms supply
  conservation and 64 distinct paid updates.
- Missing headers, altered commitments, omitted transactions, forged
  signatures, different consensus sources and changed saved balances are
  rejected by the exported-history verifier.

The offline verifier checks application history against the exported headers;
it is not a consensus light-client proof. The outage is a process outage on
one CPU host, not a validator-host failure or independent-ownership trial.

An earlier isolated attempt correctly settled eight updates before a large
quality-manifest transport limit required a source change and fresh genesis.
Its history and evidence are preserved separately. Those eight payments are
not included in this chain's 64-update result.

## Reproduction, cost and remaining bottlenecks

| Commitment | SHA-256 or revision |
| --- | --- |
| Numerical source | `20fb2a44699ed349a633ffbb0828d298f8ecc698` |
| RPC-normalized genesis | `29db702f851093d4e4a3563c6b6c489e03a94e66161871c37d062e6daabde965` |
| Activated native job | `5e60749b8a3a9ea9bdbf73d75c291d0797b1108d92cb4e05cab5ea0d1ba5cc14` |
| Terminal checkpoint | `e2379f6f819b63508fe4648b21b9919a410f1924ce860485e0cfb65787378074` |
| Learned state | `f10c25c8795f40432819f3f67b7a273ba5374e374eb1645f7c356d65eac815ff` |
| Complete evidence archive | `db194c786638c93f6c46eaddb5b61138b68d4afeacad9ff259f5704dde1e6688` |

The stored genesis differs only in CometBFT's normalization of initial height
0 to 1. The result records and verifies both identities. Native checkpoint
metadata differs from the research checkpoint because it binds the activated
job; learned weights and optimizer state are identical.

The 43,478,948-byte evidence archive contains 864 members: exact sources,
public inputs, outcomes, ledger exports and complete model/witness backup
receipts. Large tensors are separate content-addressed objects. New backups
and the archive pass full S3 readback hashing. Storage remains operator
controlled; hashes and receipts do not establish a public artifact market.

Actual neural updates take approximately 1,906 seconds on each shard. The 48
training audit invocations total 21,151.65 seconds, with a median of 432.81
seconds. Quality production takes 3,857.19 seconds; its three audit invocations
each take about 3,257 seconds. Quality evidence alone is about 126.69 GB per
original shard, before replicated copies. These costs make inference caching,
smaller evidence and cheaper verification concrete next engineering targets.

One operator controls three A10G hosts, three audit services and four native
validator processes on one CPU host. Training and replay fit one partition in
GPU memory, but audit services stage all partition artifacts on disk. All
auditors perform full replay. This establishes neither independent providers
nor economical permissionless verification.

The three temporary GPU instances, their root disks and dedicated security
group were confirmed removed at 06:35:45 UTC. Protected CPU hosts remain
unchanged; both public testnet RPCs were caught up at the same height after
cleanup. Compute is estimated at $31.54 through confirmed termination. A
conservative allowance charging all 3.74 TB of lifetime transmitted traffic
at $0.02/GB adds $74.71; those counters include private and S3 traffic. These
are planning estimates, not an invoice. Disk and retained S3 storage costs
are separate. The final bounded allocation cap was $150.

The integration closes the path from a passing sharded learning recipe through
native work settlement to serving and spending earned tokens. Useful parameter
growth, further independently evaluated cohorts, broad assistant quality,
cheaper verification and independent providers remain the next measured work.
