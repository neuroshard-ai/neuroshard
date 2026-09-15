# Retained transformer branch results

The added expert learned new facts while the complete earlier model remained
available. Final accuracy was **719/1,024 (70.21%)**, below the frozen 75% gate.
The experiment therefore failed. Retention and continued parent serving passed.

| Measurement | Development | Final |
| --- | --- | --- |
| Newly correct factual answers | 221/256 (86.33%) | 719/1,024 (70.21%) |
| Earlier skill answers reproduced exactly | 192/192 | 768/768 |
| Earlier conversation losses reproduced exactly | 128/128 | 256/256 |
| Earlier correct skill answers lost | 0 | 0 |
| Parent answers after observed expert process exit | Passed | Passed |

The earlier skill score remained 151/192 on development and 650/768 on final.
Exact reproduction includes incorrect answers: retention preserves the prior
function, and is not evidence of universal correctness. The final factual
accuracy gain's entity-cluster interval was 67.68%–72.75%. Conversation loss
changes and both confidence bounds were exactly zero.

Three owners held disjoint parts of the original 1,711,376,384-parameter model.
A fourth owned the 134,225,920-parameter learned tail, making 1,845,602,304
stored transformer parameters. Ordinary full-vocabulary greedy generation used
the learned tail for an explicit domain named in user text and the unchanged
parent for other questions. The expert's weights came from the completed
1,024-update rehearsal; composition performed no additional training.

The first branch check reproduced every one of the rehearsal's 256 knowledge
outputs. The new branch wording was committed before use; the underlying facts
were training material. Final results concern synthetic factual recall, not
broad assistant usefulness. Static domain routing, four workers and one
operator do not establish permissionless admission or a general learned router.
The controller observed the expert process exit before the surviving parent
answered; arbitrary network partitions were not tested.

## Diagnosis after the frozen evaluation

All 1,024 factual outputs were valid JSON. The 305 failures were wrong facts,
with sensitivity to wording. A separate teacher-forced check recovered all
320 sampled training responses. That diagnostic measures training recall;
it is not a generated-answer or independent quality pass.

An exploratory question interpreter tried to rewrite 32 already-exposed
questions into a training-style question before ordinary expert generation.
No factual values were supplied to the interpreter. The unchanged trained
parent produced the requested name/field interpretation on **0/32** examples;
it returned `{"answer":null}` each time. End-to-end answers stayed **16/32**.
A separate untouched 135M interpreter recovered both fields on 8/32. These
failed probes do not change the original final result. Prompt placement and
instruction following require a controlled comparison before attributing cause.

The [result record](../config/experiments/branch-growth-results.json) binds
sources, graph, inputs, actual optimizer provenance, controller receipts and
separate diagnostic freezes. The evidence archive is 95,705,313 bytes with SHA-256
`f6df1970766462d888fca5962fea763c1f6b4cc04192f00a8a17dc04e948e6f2`;
full S3 readback passed. Parent and expert tensors have separately verified
inventories included in the archive.

This shares the rehearsal's allocation: $20.5751 compute, $0.4925 gp3 and
$18.3073 conservative transfer allowance, inclusive of both experiments and
diagnostics. These are estimates, not an invoice, and must not be counted
twice. All four temporary GPUs, their disks and the temporary security group
were removed. Public-network instances remained unchanged. No tokens were
issued and no native serving checkpoint was activated. The next cohort's GPU
trial remains deferred because this prerequisite did not pass.
