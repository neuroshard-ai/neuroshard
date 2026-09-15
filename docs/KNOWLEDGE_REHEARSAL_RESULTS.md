# Repeated-exposure learning results

Both candidates completed 1,024 updates across four GPU owners. Neither passed
the frozen learning contract. Repeating the fact training improved recall in
the existing tail, but appending two blocks did not establish useful growth.

| Candidate | Development facts | Final facts | Earlier correct skill answers lost | Outcome |
| --- | --- | --- | --- | --- |
| Two appended blocks | 47/256 (18.36%) | Not opened | 2 on development | Failed development |
| Existing last two blocks | 218/256 (85.16%) | 763/1,024 (74.51%) | 3 on final | Failed final |

The parent answered none of the new factual questions correctly. The control
missed the frozen 75% final accuracy threshold by five answers. Its earlier
skill score increased from 650 to 657 of 768, with ten gains and three losses;
the aggregate gain cannot satisfy the requirement to retain every earlier
correct answer. Conversation retention passed (mean loss change +0.006693 nats,
95% upper bound +0.010772). The failed appended candidate never opened finals.

Facts were synthetic training material; question wording was held out.
Earlier skill and conversation probes were exposed retention checks. These
results establish neither broad assistant quality nor a capacity advantage.
They issued no tokens and changed no native serving checkpoint.

Immutable prefix caching kept ordinary update semantics. Median update plus
feature loading was 0.825 seconds for appended blocks. Full feature banks,
actual weights and Adam states, sources, generation logs, scores and input
identities were preserved with content hashes and full S3 readback. See the
[machine-readable report](../config/experiments/knowledge-rehearsal-results.json).
No replacement-worker trial was run for the failed growth candidate.

The same allocation subsequently ran the separately frozen
[retained-branch experiment](https://github.com/neuroshard-ai/neuroshard/pull/45).
Total allocation estimates, inclusive of both trials and diagnostics, are
$20.5751 compute, $0.4925 gp3 and a conservative $18.3073 transfer allowance.
These are estimates, not an invoice; retained S3 storage is separate. Do not
add the branch report's shared totals a second time. All four temporary
instances and their disks were deleted, the temporary security group was
removed, and protected public-network instances remained unchanged.
