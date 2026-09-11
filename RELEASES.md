# Native releases

## Repository maintenance — protocol source scope

The current tree contains protocol/client source, tests, example configuration, network manifests, technical documentation and reproducibility tools. Website publishing projects, manuscripts, generated figures, old prototypes and raw measurement dumps have moved out of the tracked tree. Historical evidence remains linked to immutable revision `108b4ba3d6c6fb5760ff211b447ee95a67fa9112`; history and released artifacts are preserved.

No Python runtime source, genesis, balance or execution profile changed in this cleanup. CI now checks repository boundaries, Markdown links, package contents, Python behavior and the consensus build. This is not a new PyPI or network release.

## Unreleased — model-evolution research tools

The source checkout adds full-backbone pipeline training, immutable fresh/replay windows, response evaluation, identity depth growth, and a separate native application for optimistic training/growth settlement. Paid-task identities reject duplicate numerical work even when model ancestry changes. Reproduction scripts and compact input plans are included; the [working paper](https://neuroshard.com/papers/FINE2026_neuroshard_short.pdf) and linked historical measurements are published separately.

These tools have not been published as a new PyPI version or activated on the public chain. Native rolling-data activation, verifiable quality promotion and paid inference for evolving models remain incomplete. See [the execution guide and measured boundaries](docs/EVOLUTION_PROTOCOL.md); do not run a changed source tree against an existing genesis-bound validator home.

## 0.4.0 — LLM training and native paid inference (experimental)

The old PyPI registration client is replaced by a lightweight local-key client: `neuroshard doctor`, `join`, `wallet`, `chat` and request recovery. Joining installs a separate pinned CPU runtime and follows NeuroShard's own consensus; it needs no starting token balance. The website adds browser signing and paid inference using the same account backup.

The new execution profile freezes SmolLM2-135M-Instruct and trains a 4,608-parameter residual adapter. Four fixed public validation sequences gate serving promotion. Native inference jobs lock an explicit customer budget, pay only after full replay and refund the budget on expiry. Training and inference are separate job lifecycles; inference creates no tokens.

Immutable S3 ingestion replaces the old mutable-ID writer. A pinned, bounded collector journals progress, uses conditional content-addressed creation and does not autoactivate new data in an existing chain. Legacy conflicting objects are preserved separately and remain quarantined.

Compatible network: `neuroshard-llm-testnet-1`.

- Genesis SHA-256: `cf74dba2e15af1c66e893cb7a8b079273e157c5585bd8af15d7194588ba47cb7`.
- Manifest hash: `5118e92e1072a1351719becc46629b16bb5bb2c34c42393dce7d3764697101b4`.
- Four genesis validators, one operator, two hosts; 90 disclosed genesis NEURO.
- Maximum 10,000 training tasks; 1 NEURO issuance per accepted task. A 32-token inference request costs 0.033 NEURO including its fee.
- CometBFT 0.38.26 with a bundled dependency lock and Go 1.27.1; Linux x86_64; Python 3.10–3.12; pinned CPU profile. Remaining model-library advisory assessments are in [SECURITY.md](SECURITY.md).

[Full protocol](docs/LLM_PROTOCOL.md), [measured evidence](docs/LLM_EXPERIMENTS.md), [model card](docs/MODEL_CARD.md) and [migration/operator guide](docs/PUBLIC_TESTNET.md). Version 0.4.0 is a regular PyPI version so `pip install --upgrade neuroshard-ai` replaces the obsolete 0.2 stable client; its project maturity remains Alpha and the network remains experimental. Old-chain balances are not migrated. Full replay cost, public evaluation overfitting, operator concentration, provider discovery and long-term economics remain limitations.

## 0.3.0a1 — experimental public testnet

This release replaces the observer-ledger and account-registration deployment with native consensus, local keys, two-stage CPU training, and verified-work issuance. The application, explorer, model view, documentation, paper, experiments, and supported operator tools are developed in one public repository.

The Python package exposes `neuroshard-chain`, `neuroshard-work`, and `neuroshard` (native chain alias). The installer supplies the pinned CPU environment and CometBFT 0.38.26. Wheels include the corpus and protocol definitions. Install pinned CPU dependencies first; generic dependency resolution does not establish the genesis numerical profile. Earlier `0.2.x` PyPI packages and `--token` commands are unsupported for this chain.

Compatible network: `neuroshard-stage-8i5ghxq5`.

- Genesis SHA-256: `c3af664ca708318286c098d146dc962e781b55ff878a8a03a508f8f889898be6`.
- Manifest hash: `012fdc5f493a6bccf782629fc96dee48be15155e887323281aebb82471bd43d1`.
- CometBFT 0.38.26; Linux x86_64; Python 3.10–3.12; pinned CPU arithmetic.
- Consensus-bound source is unchanged by the public consolidation.

Release assets include source, wheel, genesis, network declaration, and SHA-256 checksums. Build a source release from its public revision with `python3 scripts/build_release.py --ref v0.3.0a1 --output dist/release`. Git snapshots exclude untracked keys and local databases.

The website needs no authentication backend. Old signup/login/download URLs lead to participation instructions; retired API endpoints return a retirement response. Existing user records are preserved operationally outside source control. No conversion of old balances or identities is defined.

Limits at this release: 34,976 parameters, full replay at every validator, concentrated launch ownership, bounded sponsorship, unresolved fair assignment, no inclusion proofs/state sync, and no established production monetary policy or sustained-load envelope. Accepted work does not prove model improvement or profitability. The five-page paper accompanied the earlier formulation and is now [archived](https://github.com/neuroshard-ai/neuroshard/blob/108b4ba3d6c6fb5760ff211b447ee95a67fa9112/docs/archive/FINE2026_neuroshard_short_pre_evolution.pdf); subsequent experiments and protocol revisions are documented separately.

## Release procedure

1. Run native tests, consensus checks, repository/link checks, wheel installation/conformance, package-content checks and a secret scan. Validate frontend changes in the separate publishing workspace when that deployment changes.
2. Build from the exact public revision. Compare manifest and genesis. Change chain/release when consensus compatibility changes.
3. Publish a named tag, checksummed assets, and concrete validation results. Mark network maturity explicitly. PyPI client version ordering must allow users to leave obsolete stable clients; it does not establish production network maturity.
4. Deploy from pinned directories. Upgrade validators sequentially, preserving signing state. Never auto-deploy from `main`.
5. Verify a fresh public checkout and node/worker trial. Retain web/service rollback records.
