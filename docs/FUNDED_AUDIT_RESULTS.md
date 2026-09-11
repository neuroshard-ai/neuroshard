# Funded audit integration evidence — September 11, 2026

This candidate pays a separate process to replay complete neural execution and
tests recovery on a native ledger. It uses a new genesis and does not change the
public 0.4.0 network. All accounts and machines in these experiments belong to
one operator. The [audit rules](FUNDED_AUDITING.md) describe its assumptions;
the [operations guide](CANDIDATE_OPERATIONS.md) provides reproduction commands.

## Source and topology

The bounded trial uses the scripts at
[3710226](https://github.com/neuroshard-ai/neuroshard/commit/3710226).
Both its synthetic and real-model genesis commit package source
`2875b6e2de443c136cfffe3288b35d03a3f763ee521ce4f231d6704043e1a125`.
The source hash covers package Python files; it is distinct from a Git revision
or genesis hash. Later off-chain operator checks bound inference length and
require explicit subsidy approval. They were tested separately and were not
part of the frozen trial's second-cohort training path.

Four validators run on the primary Intel Xeon Platinum 8259CL host. The real
trial places two workers there and two workers, the funded auditor and a
non-voting full node on an Intel Xeon Platinum 8175M host. Each host has four
logical CPUs and approximately 16 GiB RAM. Remote workers and the auditor each
have a one-CPU quota. Computation uses the pinned float32 CPU profile, one
numerical thread, default ATen capability and SSE4.2 MKL instructions. Existing
public services also use these machines; timings are not isolated benchmarks.

Workers and the artifact mirror use HTTP through SSH. The second-host follower
first joins through a native peer tunnel, then restarts with the same keys and
genesis through the public peer at `54.210.225.74:26666`. The peer tunnel is
removed. All five full nodes agree on the application hash in block 3,588:
`AA3A1C5372395FB98A70C3857D8C41B58A57C8E0802499A426BB662737DF3FE3`.
This verifies public peer connectivity and replay across two CPUs. It does not
establish five independent operators or survival of the validator host.

The real chain is `neuroshard-lifecycle-check-2f4ad2ad7d32`, with genesis SHA-256
`d7695cbb6a8eccdaa76854c3e40ddbbe11894e2442a5fc7cce13a7cddf0ac654`.
The synthetic chain is `neuroshard-lifecycle-check-b84227cd51c0`, with genesis
SHA-256 `4e27a2feb9258b9bb3d2f09c10ddba4e94ee97fcb66fa0b86af2ac6eb5f30029`.
Each starts with 4,010 experimental NEURO, including validator collateral.
These allocations and subsequent rewards are unrelated to public balances.

## Completed synthetic rehearsal

The synthetic model has 10,384 parameters. Two cohorts complete eight training
steps and both quality evaluations. Killing the continuous operator after its
durable reservation and restarting it preserves that reservation and completes
the second cohort. Exactly 8,000,000 atoms are issued for eight distinct paid
tasks. The auditor receives 21,100,000 atoms from existing sponsor balances for
73 accepted services. Auditing creates no issuance.

The auditor records 214 stage replays across 74 reports: 211 covered stages in
paid services and three stages reaching the forged output head. Native replay
rejects that head using 4,520 uploaded bytes. Honest one-token inference pays
1,000 atoms and refunds 6,000. Both synthetic promotion decisions pass; this is
a protocol fixture, not evidence of LLM improvement.

Three of four validators continue producing blocks. Half the voting power
halts finality. Restarting the retained validators restores agreement in block
1,393, with application hash
`90EEEC60E21758CE0812CCB114FC5FFAF88EEB2995650A8136E0842A74B9D580`.
Final supply accounting is 3,997,602,000 liquid atoms, 10,000,000 validator
collateral, zero remaining escrow and 10,398,000 burned atoms. Their sum equals
the 4,010,000,000 initial atoms plus 8,000,000 issued atoms.

## Real model and data

The initial SmolLM2 model has 134,515,008 parameters. Four identity layers expand
the learning candidate to 148,675,392 parameters under four declared
48M-parameter partition capacities. Growth earns no training issuance. The
first four full-model updates issue 4,000,000 atoms.

The tokenizer and two reviewed Smol-SmolTalk cohorts are the same immutable
inputs documented in the [earlier lifecycle results](NATIVE_LIFECYCLE_RESULTS.md#fresh-data-evidence).
Each evaluation group contains 32 documents. The first cohort has 69 retention
and 75 fresh-response windows; the second has 89 and 76. Windows are aggregated
within each document before paired scoring. Previously unused rows are not
evidence of newly acquired world knowledge. Four training steps do not consume
every admitted training window, and these public cohorts can be overfit.

The first evaluation rejects promotion. Retention passes its tolerance; the
fresh-response result does not establish the required 0.001-nat improvement.
The original serving root remains active and becomes the next learning parent.
Verified computation is paid even when its candidate fails the quality gate.

## Recovery during a real fraud dispute

The remote auditor detects a deliberately forged output head. After three
native evidence chunks have finalized, the failure injector pauses the auditor,
checks its durable outbox and kills it while an upload has no locally recorded
receipt. The restarted process recovers the same signed bytes and nonce 244.
That operation then has a finalized receipt; recovery does not sign a new nonce
to replace it. A missing local receipt does not mean the transaction was absent
from the chain when the process died.

The auditor completes the 113,334,000-byte dispute. All validators use the native
referee to reject the forged response at block 5,265 with the reason
`objective replay mismatch: forward evaluation result`. This demonstrates
recovery of an interrupted large evidence upload and an objective rejection,
not a cheap neural proof. The auditor had detected the mismatch after four
stage checks in approximately 5.79 seconds, including its object handling;
native upload and settlement take several minutes on this deployment.

## What this candidate does not establish

Funded reports remain attestations under an honest-observer assumption. The
adversarial tests deliberately settle a forged update when the selected auditor
colludes and no observer challenges it. An objective challenge rejects the same
forgery and slashes a false revealed report. A cheap coverage digest, token
conservation and a paid-service counter do not prove honest computation.

The prototype's global pool can be filled by 16 unaccepted, refundable offers.
The admission-saturation test demonstrates that another sponsor is then denied
access until an offer is cancelled or expires. This is an unresolved admission
denial of service, not a fair permissionless market. Selected-auditor withholding,
challenge griefing and complete future-cohort funding remain separate limits.

The real inference fixture generates one token on the retained seed, placed
in three partitions. One complete auditor covers those partitions and the
output head: 400,000 atoms of audit fees plus two 1,000-atom publisher submission
fees, against a 1,000-atom payment. A four-partition serving model would instead
require 502,000 atoms for those costs. The later operator refuses this subsidy
unless explicitly configured. Its default one-token bound fits within the native
eight-token maximum; practical chat throughput has not been established.

The ledger still needs independent ownership, resilient hosting, fair resource
admission, artifact retention, sustainable prices, useful held-out improvement
and a rehearsed upgrade or launch procedure before a production cutover. These
experiments exercise a funded settlement candidate, not a permanent mainnet.
