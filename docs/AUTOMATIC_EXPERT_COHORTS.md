# Automatic expert cohorts

The publisher controller now runs successive native training jobs and continues
after rejected quality. The operated integration also covers publisher restart,
rejected repeated source data and an accepted model that keeps answering while
new cohorts execute.

The [complete evidence release](https://github.com/neuroshard-ai/neuroshard/releases/tag/research-automatic-cohorts-20260917)
contains the source, numerical objects, original signed blocks, final state,
reproduction drivers and the earlier failed driver attempt.

| Measured operation | Result |
| --- | --- |
| Native consensus | Four CometBFT validators; one host and administrator |
| Automatically executed cohorts | Two, under one unchanged genesis |
| Numerical work | Real prefix production, two updates per cohort, three fresh audits per claim |
| Issuance | Exactly four trial NEURO for four verified updates |
| Quality | Both correctly measured failures rejected; serving root unchanged |
| Data rejection | Repeated original documents refused before a new proposal |
| Recovery | Publisher journal and signer reopened after the first rejection |
| Serving during work | Nine identical responses from persistent accepted shard processes |
| Validator agreement | Four matching app hashes at the final commitment height |
| Full ledger replay | All 74 transactions, 449 headers, exact final state |
| Operated duration | 406.98 seconds, plus export/replay |

These are small synthetic models and public test identities. The nine serving
probes check continued execution, not useful language answers or paid inference.
Correct training can earn the prescribed trial reward while a failed quality gate
keeps those weights out of serving. The separate larger-model lifecycle evidence
is recorded under task 3 in [TODO](../TODO.md).

## Execution and recovery

`neuroshard.evolution.expert_operator.Operator` follows committed state and
journals preparation, funding, reservations, numerical requests and signed
transactions. The preparation and execution commands are operator-installed
programs; network data reaches them as bounded JSON. Curators and auditors retain
their own signing authority. Mechanical source review does not itself approve a
cohort or establish that its content is trustworthy.

The operated trial used `expert_preparation` to read consecutive source windows,
preserve tokenization and provenance, check prior history, and seal the next job.
Actual prefix/training executors produced the claims. Separate numerical runs
replayed those claims and the complete quality evaluation before audit verdicts.
The rejected candidate closed normally, allowing preparation of the next cohort.

The first driver attempt waited for its original audit deadline after all bonded
auditors had committed and consensus had shortened the window. It missed reveal
and received no training reward. The corrected driver reads the current deadline.
The audit daemon also refreshes state after replay. An unknown signed audit may
be retired only when trusted committed history identifies its permanently closed
context; its original bytes and unknown outcome remain journaled. CheckTx errors
and a merely absent current claim cannot release its pending nonce.

## Reproduction and remaining integration

The release binds source commit `130a2e2598b4b9ecbdb9a4d03663f51275ce2e00`
and includes every numerical artifact for the recorded run. The integration test
is `tests/evolution/test_expert_operator_numerical.py`. The archived `operate.py`
driver runs that same numerical workflow against disposable CometBFT 0.38.26
validators on ports 39900–39932. It requires a fresh output directory, the pinned
native/numerical dependencies and sufficient RAM for concurrent shard processes.
Set `NEUROSHARD_COHORT_SOURCE` to the extracted source checkout and
`NEUROSHARD_COMETBFT` to the pinned binary; then run the driver from a fresh
directory. `export_replay.py` restarts those disposable validators, exports their
public blocks and checks the complete application replay. Private validator keys
are generated locally and are excluded from the published evidence.

Tasks 1 and 2 remain open. The source-selection and numerical commands in this
integration use local synthetic fixtures. Connecting the controller to a live
immutable feed and deployed GPU owners, and automatically promoting successive
candidates that pass cumulative LLM answer-quality gates, remain required.


## Structured immutable source feeds

`expert_source.collect` publishes bounded windows of original conversation
messages, preserving roles, source revision, license and row positions. Held-out
windows additionally preserve their scoring metadata. Each window and feed head
is content-addressed; published updates append to the previous inventory. A
crash after upload but before journal acknowledgement recreates the same objects
and does not skip rows. Publication and native admission have separate cursors.

`expert_source.Feed` reads directly from the existing local or S3 object stores
and is the `upstream` argument accepted by `expert_preparation.prepare` and
`seal`. It verifies hashes, byte bounds, row counts and contiguous windows. A
running reader accepts only an extension of its pinned head. Missing data raises
an availability error rather than becoming a rejected or silently skipped row.
The curator still checks allowed publishers, original source correspondence,
tokenization, historical duplicates and contamination before approving admission.
An immutable feed is not evidence that its publisher's content is true or safe.

Five checks cover cross-window reads, publisher and reader restart, an
upload-before-journal crash, rollback/overlap rejection, corrupted or missing
objects, bounded S3 reads and actual native proposal preparation and review.
The semantic trial's 192 training conversations were also published in three
64-row windows: nine immutable S3 objects, 42,090 bytes, all verified through full
public downloads. The reconstructed conversations exactly match the training
source; this publication opens no evaluation set and admits no native job.

A deployment's installed preparation command can consume this reader and return
the sealed job to the existing `Operator`. The complete integration with deployed
GPU owners and passing cumulative LLM quality remains outstanding; the collector
and feed reader do not claim that integration by themselves.
