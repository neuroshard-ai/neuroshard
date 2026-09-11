# Funded audit integration evidence — September 11, 2026

This candidate pays a separate process to replay complete neural execution and
tests recovery on a native ledger. It uses a new genesis and does not change the
public 0.4.0 network. All accounts and machines in these experiments belong to
one operator. The [audit rules](FUNDED_AUDITING.md) describe its assumptions;
the [operations guide](CANDIDATE_OPERATIONS.md) provides reproduction commands.

The [evidence bundle](https://neuroshard.com/experiments/funded-audit-20260911/README.md)
contains genesis files, final results, replay reports, recovery records and
four-validator supply checks, with a SHA-256 manifest. These are inspectable
operator measurements, not proof of independent ownership.

## Source and topology

The bounded trial uses the scripts at
[3710226](https://github.com/neuroshard-ai/neuroshard/commit/3710226).
Both its synthetic and real-model genesis commit package source
`2875b6e2de443c136cfffe3288b35d03a3f763ee521ce4f231d6704043e1a125`.
The source hash covers package Python files; it is distinct from a Git revision
or genesis hash. Later off-chain operator checks bound inference length and
require explicit subsidy approval. They were tested separately and were not
part of the frozen trial's second-cohort training path.
The current driver also extends its configurable second-cohort wait to two
hours after the frozen real-model trial exceeded its one-hour limit; it does
not change ledger deadlines or package source.

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

The second cohort completes four more updates and its full evaluation on a
134,515,008-parameter candidate. Retention again passes and fresh-response
improvement again fails. The two decisions are:

| Cohort | Candidate parameters | Mean retention loss change (nats) | Mean fresh loss change (nats) | Promotion |
| --- | ---: | ---: | ---: | --- |
| First | 148,675,392 | −0.00039184 | −0.00120472 | Rejected |
| Second | 134,515,008 | +0.00030163 | +0.00002372 | Rejected |

These means use the recorded quantized document differences; negative values
mean lower loss. The first fresh point estimate improves, but its uncertainty
bound does not establish the required improvement. Each decision uses the
predefined 32-document paired gate, including its quantization allowance.
Neither result establishes a generally more useful language model.

Across both cohorts, eight distinct training tasks issue exactly 8,000,000 atoms.
The remote auditor records 169 reports and 706 stage checks: 168 accepted
services cover 702 paid stages, and the forged response reaches four checks.
The sponsor pays 70,200,000 existing atoms for those services. No audit payment
creates issuance. The publisher separately replays 353 stages on the first
cohort and honest inference path; this count is reconstructed from the frozen
driver's completed path and excludes its separate growth replay.

## Driver timeout, completion and quorum recovery

The coordinator is killed after reserving the second cohort's first task. Its
automatic restart recovers that reservation and completes all four remaining
training steps. The frozen experiment driver then exceeds its one-hour wait
while evaluation is still in progress. At the retained height 8,968, only 68
candidate fresh-response windows remain. The driver stops its child services;
this is an experiment-controller timeout, not a completed quality evaluation.

A bounded manual continuation restarts the same validator homes, keys, signing
state, workers, auditor, artifact store and operator outbox. It finishes those
windows without new training or a genesis reset. The result records the resume
height and continuation-helper digest. The second-host follower also needs a
manual restart to reconnect promptly after the long outage. All five nodes
then agree in block 9,086. This demonstrates retained-state recovery with
intervention, not fully unattended operational resilience.

After evaluation, three of four validators continue producing blocks. Half the
voting power halts finality. Restarting the retained validators restores common
headers in block 9,811, with application hash
`00073FC58D6D2DB580AC832864C6313870EF0571FCC0CB8FCF4F16532EB740B2`.
All four final databases independently pass the supply invariant: 3,997,019,000
liquid atoms plus 10,000,000 validator collateral, zero remaining escrow and
10,981,000 burned atoms equal 4,010,000,000 initial plus 8,000,000 issued atoms.
Both bounded trial networks are stopped with their state preserved. The
existing public 0.4.0 network continues separately.

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
stage checks in approximately 5.79 seconds, including its object handling.
The dispute comprises 116 unique signed transaction envelopes totalling
151,171,234 bytes. The interval between their first and last consensus block
timestamps is 342.559577 seconds, including the injected crash. This is neither
a monotonic client-latency measurement nor total network bandwidth: retries,
RPC framing and peer gossip are excluded.

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

Acceptance also fails to reserve separate refutation capital. An adversarial
test gives an auditor enough for its audit bond but insufficient liquid funds
for the challenge bond. It detects fraud and refuses to attest; the claim cannot
mint, but the missed-report rule burns its bond. The real trial starts its
auditor with 100,000,000 atoms. Its measured head dispute needs another 2,000,000
atoms of challenge collateral and 116,000 atoms of transaction fees while the
5,000,000-atom audit bond is locked. Prefunding that obligation is a required
protocol change, described in the [admission RFC](AUDIT_ADMISSION_RFC.md).

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
