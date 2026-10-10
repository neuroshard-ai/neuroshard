# Security policy

The supported native release is **0.4.0**. Earlier `0.2.x` packages, the archived registration/observer-ledger stack, and historical prototypes are unsupported. This is an experimental testnet; an independent production security audit has not been completed.

Use [GitHub private vulnerability reporting](https://github.com/neuroshard-ai/neuroshard/security/advisories/new), which is enabled for this repository. Include the affected revision, execution profile, impact, and minimal reproduction. Never include real signing keys, credentials, or user records.

Avoid publishing a working exploit before maintainers can assess it. Maintainers will coordinate a fix, release notes, and an advisory where appropriate. There is no guaranteed response time or funded bounty program.

Relevant boundaries include consensus acceptance, numerical conformance, signing-state recovery, monetary invariants, resource limits, worker authorization, retries, dependency integrity, and genesis/checkpoint trust. Use disposable chains for experiments. Never duplicate a live validator's signing identity.

The explorer is a view from a full node. Run your own node to verify independently. Account inclusion proofs and state-sync snapshots are not implemented.

## Dependency review for 0.4.0

The release updates cryptography, HTTP, protobuf/gRPC and collector dependencies. Native consensus remains CometBFT 0.38.26, built with checksum-verified Go 1.27.1 and the bundled `client/consensus/go.mod` and `go.sum`. These pin patched transitive dependencies; a version string alone does not identify this build. The client checks a build-manifest digest and executable checksum before reusing its cached engine.

The numerical execution profile still requires PyTorch 2.9.1, Transformers 4.57.3, NumPy 2.2.6, tokenizers 0.22.1 and safetensors 0.7.0. Changing these versions requires a conformance assessment and potentially a new execution profile. The following outstanding advisories were assessed against the supported fixed-model path; they are not blanket suppressions or claims that the libraries are safe for arbitrary workloads.

| Advisory | Assessment of this release's path |
| --- | --- |
| [CVE-2025-3000](https://github.com/advisories/GHSA-rrmf-rvhw-rf47), [CVE-2025-3001](https://github.com/advisories/GHSA-qfhq-4f3w-5fph) | The eager Llama forward pass and adapter SGD do not use `torch.jit.script` or `torch.lstm_cell`. Participants submit bounded tensors/text, not executable Python or model checkpoints. |
| CVE-2025-14929 | The X-CLIP checkpoint conversion tool is not invoked. |
| [CVE-2026-1839](https://github.com/advisories/GHSA-69w3-r845-3855) | The protocol does not use Transformers Trainer or restore pickled RNG state. |
| [CVE-2026-4372](https://github.com/advisories/GHSA-29pf-2h5f-8g72) | Model configuration and weights are checked against the published genesis before loading; the inspected configuration uses Llama with explicit eager attention and local files. `trust_remote_code=False` alone is not a sufficient mitigation for this advisory. Never substitute an unreviewed model or genesis. |
| [CVE-2026-5241](https://github.com/advisories/GHSA-fgcw-684q-jj6r) | LightGlue is not used. |
| [CVE-2026-9856](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w) | Runtime inference and collection do not call tokenizer/processor `save_pretrained`; tokenizer files and chat templates are checksum-verified before use. |

The Go binary scan found no affected imported packages or symbols. Its module-level CometBFT finding GO-2025-3442 does not account for the [0.38.17 backport](https://github.com/cometbft/cometbft/security/advisories/GHSA-22qq-3xwm-r5x4), which is included in 0.38.26. The x/crypto OpenPGP module advisory concerns a package not linked into this binary. Retain raw scan findings and review new advisories; absence of a scanner finding is not a security proof.

## Dependency review for the assistant network runtime

October 10, 2026. The assistant network (`docs/granite-shard-chain-requirements.txt`) pins Python 3.12.12,
PyTorch 2.10.0, Transformers 5.5.4, setuptools 78.1.0 and no urllib3. Every owner, client, auditor and the
release qualification replay this runtime bit for bit, so a version change needs a new execution profile and
requalification. The other files under `docs/` are frozen runtimes of published experiments, whose records
pin their digests; they are not upgraded in place. The advisories open against these manifests were
assessed against the served path:

| Advisory | Assessment of the assistant network path |
| --- | --- |
| [CVE-2026-80047](https://github.com/advisories/GHSA-x9r9-c232-4q39) (no patched version) | Serving never calls `generate` or downloads generation code: owners and clients build partitions from checksummed safetensors byte ranges of a pinned revision and decode with their own loop. Configuration is read with `local_files_only=True`. |
| [CVE-2026-9856](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w) | The served path never calls `save_pretrained`; tokenizer files and the chat template are checksum-verified before use. |
| [CVE-2026-4372](https://github.com/advisories/GHSA-29pf-2h5f-8g72), [CVE-2026-5241](https://github.com/advisories/GHSA-fgcw-684q-jj6r), [CVE-2026-1839](https://github.com/advisories/GHSA-69w3-r845-3855) | Granite configuration and weights are pinned and checked before loading; no remote code, LightGlue or Trainer. Never substitute an unreviewed model, module or descriptor. |
| [CVE-2025-3000](https://github.com/advisories/GHSA-rrmf-rvhw-rf47), [CVE-2025-3001](https://github.com/advisories/GHSA-qfhq-4f3w-5fph) | No `torch.jit.script` or `torch.lstm_cell`; 2.10.0 already fixes CVE-2025-3001. Peers exchange bounded activation frames, not executable code or checkpoints. |
| [CVE-2025-47273](https://github.com/advisories/GHSA-5rjg-fvgr-3xxf), [CVE-2026-59890](https://github.com/advisories/GHSA-h35f-9h28-mq5c) | setuptools is only an install-time dependency; `PackageIndex.download` and sdist building are never invoked by any role. Install with the pinned requirements, from the pinned index. |
| urllib3 [CVE-2026-97687](https://github.com/advisories/GHSA-8988-9cw3-xx77), [CVE-2026-97688](https://github.com/advisories/GHSA-gh4c-6fx4-qh6g), [CVE-2026-97689](https://github.com/advisories/GHSA-vxq7-64xx-v4gw) | Not installed in the assistant runtime. The supported 0.4.0 package already requires urllib3 2.8.0. The frozen experiment runtimes that pin 2.7.0 are for reproducing published results on project hosts, not for serving. |

These are path assessments, not claims that the libraries are safe for other workloads. The assistant runtime
should move to patched versions in its next execution profile, after a conformance check of every served route.

Historical dependency manifests remain in Git history and the local archive, outside the current source tree and supported package. Archived applications must be reviewed and upgraded before anyone reactivates them. The retained v0.3 reference network is unsupported and exposes a read-only historical API; it is not the joining or inference endpoint. Never publish the ignored archive: it can contain local publishing configuration and operational logs.

## Model-evolution experiments

The `neuroshard.evolution` package is an experimental implementation available from the source checkout. It has not replaced the public 0.4.0 network. Its importer verifies the pinned seed files; workers accept bounded float32 safetensors components with checked shapes, and use no pickle checkpoint loading, remote model code, Transformers Trainer, JIT or LSTM execution in the implemented path. These restrictions are relevant to the advisories above and do not establish safety for other uses of the dependencies.

The opt-in [native lifecycle](docs/NATIVE_LIFECYCLE.md) adds data-admission votes and challengeable evaluation/generation graphs. Data votes attest to off-chain curation; they do not prove provenance, licensing or usefulness. Every accepted execution graph still requires complete honest observer coverage. The new cohort command serializes collection before reading source cursors, and native token batches remain immutable once admitted. Its topology checker can detect declared concentration of voting power but cannot prove independent operators. Neither passing unit tests nor a successful isolated lifecycle establishes public deployment security.

Native evolution claims rely on an honest observer obtaining the data and completing a challenge in time. Compact graph checks and signatures alone cannot reject an arithmetically forged update. The current terminal referee is expensive, audit funding is incomplete, and advertised worker capacity does not prove available independent hardware. Worker RPC binds to loopback and uses private operator-managed bearer tokens. Public discovery, model retention, native quality promotion and migration still need integration and independent review; this interface is not a new public mining endpoint.
