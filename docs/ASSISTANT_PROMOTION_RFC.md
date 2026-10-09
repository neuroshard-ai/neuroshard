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
