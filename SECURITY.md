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

Archived dependency manifests under `legacy/` may continue to trigger repository alerts. Those applications are not installed by the supported package or deployed by the current website. They must be reviewed and upgraded before anyone reactivates them. The retained v0.3 reference network is unsupported and exposes a read-only historical API; it is not the joining or inference endpoint.

## Model-evolution experiments

The `neuroshard.evolution` package is an experimental implementation available from the source checkout. It has not replaced the public 0.4.0 network. Its importer verifies the pinned seed files; workers accept bounded float32 safetensors components with checked shapes, and use no pickle checkpoint loading, remote model code, Transformers Trainer, JIT or LSTM execution in the implemented path. These restrictions are relevant to the advisories above and do not establish safety for other uses of the dependencies.

Native evolution claims rely on an honest observer obtaining the data and completing a challenge in time. Compact graph checks and signatures alone cannot reject an arithmetically forged update. The current terminal referee is expensive, audit funding is incomplete, and advertised worker capacity does not prove available independent hardware. Worker RPC binds to loopback and uses private operator-managed bearer tokens. Public discovery, model retention, native quality promotion and migration still need integration and independent review; this interface is not a new public mining endpoint.
