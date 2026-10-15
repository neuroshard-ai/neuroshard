# RFC: stewarded assistant testnet promotion

Status: proposed in PR #85 for the assistant testnet only. This adds a native
transition; it does not alter 0.4.0 genesis or grant token holders governance.

The assistant ledger previously bound one model forever, while accepted learning
results advanced separately. `model_promote(model_root, quality_root)` lets the
testnet's genesis funding authority adopt a complete, previously approved serving
version. Each validator must configure the same authority and allowlist of model
and quality digests. The quality record is an externally reviewed experiment
report; this transition does not prove quality mathematically.

Admission requires a valid signature, nonce and fee, an allowlisted quality
digest, a different model root, and **no open jobs**. Consensus records the
previous root, new root, authority, quality digest and height. It preserves all
accounts, bonds and initial supply, burning only the ordinary transaction fee.
Rollback is another authorized promotion to an approved previous version after
draining jobs. Changing the shard count is outside this amendment.

Validators judge fraud under the currently served version. Replay configuration
must retain both the old plain update and the routed module bank during migration.
Existing owner keys remain bonded; operators restart with the approved version.
Old clients fail the model-root check rather than buying incompatible work.

This is deliberately stewarded. An authority can approve a poor model, incompatible
validator allowlists can halt consensus, and continuously opening jobs can delay
a drain. Operators must agree on the upgrade and maintenance window. Permissionless
quality admission, independent voting power, automatically funded honest audits,
and censorship-resistant governance are separate unresolved protocol work.

The alternative is a fresh genesis for each model, losing continuous balances and
history. Automatically treating a training receipt or mining ticket as quality
approval would violate the separation of computation and usefulness. Neither is
used here.

Validation covers unauthorized and wrong-quality rejection, open-job refusal,
supply conservation, ABCI admission/proposal/commit, routed transcript replay and
native paid-chat integration. Activation additionally requires the
[release qualification](ASSISTANT_RELEASE_CHECK.md).

## Amendment 1: owner governance of the serving model

Status: implemented in code and tests on `research/general-route`. Not yet deployed to
`neuroshard-assistant-1`; deploying it is a validator software upgrade, and enabling it is
the steward's last promotion-related act.

The stewarded transition leaves one key deciding what every owner serves. This amendment
adds a one-way exit from it:

- **`governance_handover(approval_ppm, vote_blocks)`.** Only the configured promotion
  authority can submit it, and only once. The approval share must be a strict majority
  (at least 500,001 ppm) and the window at least 20 blocks. Afterwards `model_promote` is
  refused for every signer. No validator configuration can restore the steward, because
  the chain state, not `settlement.json`, now decides.
- **`model_propose(model_root, quality_root)`.** Anyone may submit it with a 2 NEURO deposit,
  held in escrow and counted by the supply invariant. It snapshots the bond of every
  active owner as its voting weights, so bonding after a proposal opens buys no vote on it.
  At most four proposals are open at once.
- **`model_vote(proposal_id, log_key, approve)`.** The account that bonded `log_key` casts one
  vote per proposal before the deadline.
- **`model_enact(proposal_id)`.** Anyone may submit it. If the approving weight reaches the
  share, the model changes (open jobs must be drained, as before), the history records
  `authority: owners` with the tally, and the deposit returns to the proposer. A proposal
  that closes without passing burns its deposit.

The people who must restart on the new version, the owners, are the ones who decide it.
The quality root stays an attestation, not a proof. Owners should check the published
release-check evidence it names, which anyone can rerun from the committed contract,
before voting.

This does not decentralise everything:

- One validator still orders the votes. Validator-set changes are separate work (A5 needs
  four independent operators).
- An owner majority can still approve a poor model.
- The quality evidence is still produced by whoever runs the release check.

Tests: `tests/evolution/test_owner_governance.py`. Operators use
`neuroshard assistant govern list|propose|vote|enact`, all signed by their own keys.
