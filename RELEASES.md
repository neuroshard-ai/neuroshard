# Native releases

## Repository maintenance — protocol source scope

The current tree contains protocol/client source, tests, example configuration, network manifests, technical documentation and reproducibility tools. Website publishing projects, manuscripts, generated figures, old prototypes and raw measurement dumps have moved out of the tracked tree. Historical evidence remains linked to immutable revision `108b4ba3d6c6fb5760ff211b447ee95a67fa9112`; history and released artifacts are preserved.

No Python runtime source, genesis, balance or execution profile changed in this cleanup. CI now checks repository boundaries, Markdown links, package contents, Python behavior and the consensus build. This is not a new PyPI or network release.

## Unreleased — model-evolution research tools

The source checkout adds full-backbone pipeline training, immutable fresh/replay windows, response evaluation, identity depth growth, and a separate native application for optimistic training/growth settlement. Paid-task identities reject duplicate numerical work even when model ancestry changes. Reproduction scripts and compact input plans are included; the [working paper](https://neuroshard.com/papers/FINE2026_neuroshard_short.pdf) and linked historical measurements are published separately.

An opt-in [native lifecycle](docs/NATIVE_LIFECYCLE.md) now integrates curated rolling-data admission, challengeable multi-window evaluation, serving decisions and bounded paid generation on isolated networks. Its [results](docs/NATIVE_LIFECYCLE_RESULTS.md) include a rejected real-model promotion and continued paid training after a second data admission without resetting the ledger.

The opt-in [compact optimizer extension](docs/COMPACT_UPDATE_DISPUTES.md) adds Merkle commitments around unchanged float32 SGD. A transaction can refute an inconsistent update using at most 32 KiB of witness data without model-artifact retrieval or full neural replay by validators. False accusations burn collateral without extending the deadline. Complete replay still checks gradients and tensor/file relationships; consistent invented gradients explicitly remain outside the compact path's coverage. No second issuance is authorized by adding commitments to an existing numerical task.

The [scaling design](docs/SCALING_DESIGN.md) prioritizes complete audit coverage, artifact replication, serving and sustained training capacity before model growth. It separates implemented mechanisms from the remaining audit economy, compute-group scheduling and public-independence requirements. The [learning milestone](docs/LEARNING_MILESTONE.md) now has a Git-sealed prepare/train/score driver with durable recovery and complete document evaluation. Its [128-step result](docs/LEARNING_MILESTONE_RESULTS.md) fails the sealed-test confidence bound, with all generation pairs and the rejected checkpoint published. Later phases remain blocked; this does not change 0.4.0.

The [1.7B GPU reference](docs/LEARNING_REFERENCE_RESULTS.md) completed 256 full-model AdamW updates and exactly reproduced the last 64 updates after checkpoint restoration on the same L40S. Its final-test improvement remains inconclusive and generated answers show regressions. The public evidence includes every selected answer, exact candidate weights, resource accounting and a report-only instruction-target audit. No serving promotion, native GPU transition or token issuance accompanies this development experiment.

The [two-GPU experiment](docs/COOPERATIVE_LEARNING_RESULTS.md) adds checked generated targets, a deliberately damaged control, weighted full-model DDP, committed final candidates and a recomputable quality/resource report. Clean task accuracy improves from 23/256 to 169/256; shared training reaches 168/256 with identical parameters on both ranks. Invoice totals remain 0/64 for every model. Synchronous communication makes updates 4.39× slower, while two inference replicas deliver 1.67× throughput and recover all requests after one provider stops. These are bounded operated research results; native consensus, paid inference and the public PyPI release retain their existing execution profile.

The [four-worker experiment](docs/LOCAL_TRAINING_WINDOWS_RESULTS.md) adds bounded local Adam windows, outer Nesterov aggregation, common group manifests, rank-bound optimizer/RNG restoration, an exclusive GPU process lock and report recomputation. A 1.7B run reduces transmitted data by 93.8% against DDP and reproduces all rank states after a killed process. Its quality contract fails: retention worsens and task accuracy trails DDP beyond the declared margin. A separate two-GPU PowerSGD feasibility probe completes after CPU offload of dense error feedback, with exact small-model conformance checks on CPU and CUDA. Four inference replicas deliver 3.00× throughput, and all requests complete after one stops. The full original/recovered states are backed up and all four temporary GPU instances, volumes and their dedicated security group are removed. These research tools authorize no native model promotion or GPU rewards.

The [funded candidate](docs/FUNDED_AUDITING.md) now escrows sponsor payments and auditor collateral before work, requires all selected auditors to report complete coverage, and pays honest replay without extra issuance. Missing coverage cannot mint; objective false reports can lose collateral. Sponsor selection, copying and collusion remain explicit assumptions. The separate auditor daemon retrieves and replays the entire graph, requests missing data and refutes incorrect stages through native transactions.

The [continuous operator and candidate joining guide](docs/CANDIDATE_OPERATIONS.md) cover admitted-data training, evaluation, durable worker recovery and non-voting full nodes. A signed-transaction outbox resolves uncertain outcomes by the hash of the original signed bytes before another nonce can be used. The operator refuses inference below its audit/submission price floor unless a subsidy is explicitly enabled, and bounds response length before escrowing audit fees. These mechanisms do not implement consensus upgrades or migrate existing balances.

The [completed funded-audit experiments](docs/FUNDED_AUDIT_RESULTS.md) settle eight full-model training tasks across two cohorts on two hosts, pay 168 audit services from existing balances, reject a forged response after an interrupted native upload, and pass quorum recovery and supply accounting. Both real-model promotion gates fail. The real driver exceeds its original one-hour wait and requires a retained-state continuation; future driver waits are configurable. Tests also expose colluding attestations, admission-pool saturation and missing refutation capital. The [admission RFC](docs/AUDIT_ADMISSION_RFC.md) proposes a next profile for the latter two problems; it is not implemented or activated.

The [continued-learning contract](docs/CONTINUED_LEARNING.md) freezes a 1.7B continuation from the passing adaptive phase-A checkpoint. It binds parent weights and Adam, tokenizer, runtime and unused evaluation data, requires generated-answer gain with per-family floors, and forbids growth. Native settlement of a pass is later job activation plus reserved-window receipts, not an imported checkpoint. Promotion remains a separate mint-zero serving decision. This does not change 0.4.0.

These tools have not been published as a new PyPI version or activated on the public chain. Public worker admission, independent audit selection, artifact retention and release integration remain incomplete. See [the execution guide and measured boundaries](docs/EVOLUTION_PROTOCOL.md); do not run a changed source tree against an existing genesis-bound validator home.

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

The [continued-learning result](docs/CONTINUED_LEARNING_RESULTS.md) is a reported failure: three workers completed 96 updates, but generated answers regressed on both fresh task sets. The prepare/train/score driver now enforces artifact freezes, content exclusions, actual checkpoint selection and development aborts; 412 tests pass. This remains research work with no serving promotion or public-network change.

The [calculation-step experiment](docs/REASONED_LEARNING_RESULTS.md) adds supervised intermediate calculations and a correct-reference-margin penalty to the existing sharded trainer. Across 256 new updates, fresh answers improve 192→242, including arithmetic 2→53 out of 64, but prior answers fall 93→90. The frozen retention floors reject it. All 384 final answer pairs are published, 425 tests pass, and all temporary GPU resources were removed after verified checkpoint backups. Native settlement and public serving remain unchanged.
