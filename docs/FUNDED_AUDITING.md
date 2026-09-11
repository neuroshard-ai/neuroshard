# Funded complete-replay candidate

This opt-in evolution profile connects prepaid audit obligations to training,
growth, evaluation and paid inference. It requires a **new genesis**. It has not
upgraded the public 0.4.0 network or established a production mainnet.

The security assumption remains explicit: an honest observer must check the
execution and be able to get a refutation finalized. Auditors are selected by the
sponsor, and every selected auditor must attest to the entire execution graph.
Different keys do not establish different owners or independent computation.
Colluding producers and auditors can still agree on a false claim. This is a
purchased, accountable replay service; permissionless audit selection and its
collusion incentives remain release blockers.

## Funding and obligations

A genesis enables `manifest.auditing` with format
`neuroshard-funded-audit-v1`. Its fixed parameters are:

| Parameter | Reference value | Meaning |
| --- | ---: | --- |
| `price_per_stage` | 100,000 atoms | Existing tokens paid per covered execution stage per auditor |
| `auditor_bond` | 5,000,000 atoms | Collateral accepted before work is reserved |
| `commit_blocks` | 128 | Maximum initial reporting commitment window |
| `reveal_blocks` | 32 | Maximum reveal window after commitments |

The experiment can set longer commitment windows for measured hardware and
transfer costs. The absolute claim lifetime must fit both audit windows and the
ordinary challenge window. Values are experimental service prices, not measured
market clearing prices. Training issuance retains its existing period limit
and numerical work identity; audit fees create no additional issuance.

`fund_audit(publisher, auditors, stage_limit, expires_in)` escrows
`price_per_stage × stage_limit × number_of_auditors` from its sender. The selected
keys must be sorted and unique, with 1–8 auditors, a 1–4,096 stage cap and a bounded
offer lifetime. At most 16 offers can be outstanding. Each auditor submits
`accept_audit(budget_id)` and locks collateral. An auditor cannot use the
publisher's or reserved workers' key. There is no claim that this excludes their
other identities.

The publisher supplies `audit_budget` when reserving training or submitting a
growth, score or inference response. Training locks its offer **before** the
workers execute. The other claims must attach an already accepted offer. Native
metadata determines the exact stage count, including every generation step and
its output head. A cap smaller than that count rejects the claim. A contract is
bound to one reservation and one claim; it cannot fund overlapping work.

The sponsor can cancel an unreserved offer. Cancellation, offer expiry and an
abandoned training reservation return unspent audit funding and collateral.
The original training reservation penalty remains separate. Reserved funding
cannot be withdrawn while the associated claim is pending.

## Reports and settlement

The audit daemon retrieves the committed input and output objects, checks their
hashes, and runs the existing referee for **every** stage. A training audit covers
forward execution, backward execution and the optimizer, including cross-stage
dependencies. Growth replays the permitted identity extension. Score and
inference audits cover all committed forward records. A missing object cannot
produce a successful report. The daemon requests native availability or submits
a native stage dispute when appropriate.

`audit_commit(claim_id, commitment)` commits to a domain-separated digest of the
chain, claim, auditor, complete-coverage root and a 32-byte salt.
`audit_reveal(claim_id, salt, coverage_root)` opens it. The coverage root commits
to the record, claim kind and all stage indices. **This digest is public and
cheap to calculate: it specifies the obligation, and does not prove replay.**
The daemon's replay log is separate evidence of the measured implementation.

Reveals start after the commitment phase. If all auditors commit early, that
phase ends at the current block and reveals can start in the next block. Once
all reports are revealed, a full ordinary challenge window starts. No new
auditor or report replacement can be introduced after disclosure. Private
sharing, prearranged reports and collusion remain possible.

An unchallenged claim with missing reports **does not settle or mint**. On
acceptance, each auditor receives `actual_stages × price_per_stage`, its
collateral returns, and the sponsor receives the unused stage allowance.
Training rewards are issued once under the existing paid-work rule. Growth,
evaluation and audit payments do not issue training rewards; inference pays
from the customer's previously locked budget.

An objectively refuted execution pays no audit service fee. Revealed false
attestations lose collateral, split between the successful challenger and burn
under the same half-and-remainder convention as publisher collateral. A
successful fraud detector is not penalized for refusing to attest to a false
execution.

Missing reports lose auditor collateral only after an **uninterrupted** audit
window. An availability or full fraud challenge marks that window interrupted;
publisher withholding, unresolved disputes and absolute expiry do not establish
auditor misconduct. Such failures return unearned service fees and unslashed
audit collateral. This conservative rule means a bonded challenge can avoid a
timeliness penalty and deny progress. The existing challenge bond and absolute
deadline bound that episode; they do not solve service selection or griefing
economics. False revealed attestations remain slashable after an interruption.

Every transition checks:

`initial supply + training issuance = liquid balances + validator collateral + all escrow + burned tokens`

`issued atoms = training rounds × reward per round`

Audit escrow includes offer funds and every accepted auditor bond. A second
payment cannot arise from changing the publisher, partitioning the same paid
task again, retrying an RPC, or restarting an operator.

## Operational evidence and remaining scope

The [candidate operations guide](CANDIDATE_OPERATIONS.md) covers source and genesis
pins, the separate auditor process, the recovering full-model operator and
failure rehearsals. `/auditing` exposes outstanding obligations and a bounded
history of payments, refunds and penalties. It complements the existing
numerical-dispute counters; a paid service count is not a count of independent
operators.

Funded coverage is per execution claim. This version does not prefinance an
entire future cohort's availability, evaluation, serving and dispute costs in
one contract. Sponsors must retain enough funds to finish their admitted work;
the quality deadline still rejects an unfinished evaluation. Long-term artifact
retention, independent operators, committee selection, useful held-out model
improvement, an economic policy and external review are still necessary before
a permanent network can be called production ready.
