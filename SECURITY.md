# Security policy

The supported native release is **0.3.0a1**. Earlier `0.2.x` packages, the archived registration/observer-ledger stack, and historical prototypes are unsupported. This is an experimental testnet; an independent production security audit has not been completed.

Use [GitHub private vulnerability reporting](https://github.com/neuroshard-ai/neuroshard/security/advisories/new), which is enabled for this repository. Include the affected revision, execution profile, impact, and minimal reproduction. Never include real signing keys, credentials, or user records.

Avoid publishing a working exploit before maintainers can assess it. Maintainers will coordinate a fix, release notes, and an advisory where appropriate. There is no guaranteed response time or funded bounty program.

Relevant boundaries include consensus acceptance, numerical conformance, signing-state recovery, monetary invariants, resource limits, worker authorization, retries, dependency integrity, and genesis/checkpoint trust. Use disposable chains for experiments. Never duplicate a live validator's signing identity.

The explorer is a view from a full node. Run your own node to verify independently. Account inclusion proofs and state-sync snapshots are not implemented.
