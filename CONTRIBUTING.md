# Contributing to NeuroShard

The native deployment candidate is the supported development path. It implements a small CPU training workload with native consensus and settlement. See the [operator guide](docs/PUBLIC_TESTNET.md), [protocol](docs/PROTOCOL_CANDIDATE_V2.md), and [experiments](docs/PROTOCOL_EXPERIMENTS.md) before changing its behavior.

The home for all development is [neuroshard-ai/neuroshard](https://github.com/neuroshard-ai/neuroshard). The [open-source transition](docs/OPEN_SOURCE_TRANSITION.md) records the consolidation decisions. A NeuroShard website account is not required to contribute or operate a native node.

## Install and check the native implementation

Use Linux x86_64 and Python 3.10–3.12 for the recorded CPU profile:

```bash
bash scripts/install_native.sh
ATEN_CPU_CAPABILITY=default MKL_ENABLE_INSTRUCTIONS=SSE4_2 PYTHONPATH=src \
  venv_build/bin/python -m pytest -q tests/test_verified_demo.py \
  tests/test_protocol_candidate.py tests/test_public_node.py tests/test_outbound_work.py
```

The installer creates a project-local environment and installs the pinned native engine. Old `neuroshard --token ...` instructions and the published legacy package do not start this chain. Historical tests are in `legacy/tests` with separate prototype dependencies. Run `venv_build/bin/python -m pytest -q` for the complete native suite.

To work on the website, use Node 22.12 or newer in the Node 22 line:

```bash
cd website
npm ci
npm run build
npm run dev
```

The development server proxies native API requests to `127.0.0.1:38659`. Configure that target for your own local gateway when necessary. The production build copies the current protocol documents and manuscript into its public assets. It needs no legacy authentication backend to render the native interface.

## Propose a change

For a bug report, include the source revision, chain ID when relevant, numerical profile, expected behavior, actual behavior, and reproduction steps. Share public transaction/block identifiers or reduced test cases. Exclude private keys, signing state, credentials, and user records.

For changes to consensus, validator admission, verification, reward rules, or execution semantics, open a design proposal explaining the invariant or behavior being changed, compatibility implications, assumptions, and how it can be tested. Small documentation and interface fixes can proceed directly as pull requests.

A pull request should explain the concrete problem, resulting behavior, and relevant validation. Add regression coverage when a change affects security, accounting, consensus, or failure recovery. Report which checks ran and any limitations; a passing small-model experiment does not establish LLM-scale performance or adversarial security.

The execution manifest binds numerical code and consensus source. Editing or moving those files can require a new compatible genesis/release. Test against disposable chain homes and preserve live validators' keys, databases, and signing state. Operators choose their supported release; merging a pull request must not automatically upgrade the running network.

## Useful contributions

- Reproduce the installation and training trial on additional compatible machines.
- Improve node synchronization, diagnostics, bounded APIs, and worker failure handling.
- Measure public-load behavior and verification cost with reproducible workloads.
- Investigate complete training verification and portable numerical execution.
- Improve the model/explorer interface and documentation using measured behavior.

Consensus safety, monetary accounting, and claims in the paper should remain explicit about their assumptions. Reproducible negative results are useful contributions.

## License

Contributions to project code are made under the existing [Apache License 2.0](LICENSE). Preserve applicable attribution and identify the provenance and license of any added third-party code, data, models, or assets.

Build documentation with `cd docs-site && npm ci && npm run build`. Its pages are generated from canonical repository documents. Browser checks: `cd website && npx playwright install chromium && npm test`.
