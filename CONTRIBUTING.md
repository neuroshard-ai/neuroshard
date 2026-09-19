# Contributing to NeuroShard

The native deployment candidate is the supported development path. It implements adapter training on a frozen pretrained language model, native consensus and paid inference. See the [operator guide](docs/PUBLIC_TESTNET.md), [protocol](docs/LLM_PROTOCOL.md), and [experiments](docs/LLM_EXPERIMENTS.md) before changing its behavior.

The home for protocol and client development is [neuroshard-ai/neuroshard](https://github.com/neuroshard-ai/neuroshard). A NeuroShard website account is not required to contribute or operate a native node. Keep the public tree focused on code, tests, configuration, network manifests and technical documentation.

## Install and check the native implementation

Use Linux x86_64 and Python 3.10–3.12 for the recorded CPU profile:

```bash
bash scripts/install_native.sh
ATEN_CPU_CAPABILITY=default MKL_ENABLE_INSTRUCTIONS=SSE4_2 PYTHONPATH=src \
  venv_build/bin/python -m pytest -q tests
```

The installer creates a project-local environment and installs the pinned native engine. The current `neuroshard-ai` client supports `neuroshard join`. Old `neuroshard --token ...` instructions and 0.2 packages do not start this chain. Run `venv_build/bin/python -m pytest -q` for the complete native suite.

Check repository links and build the distributable package:

```bash
python scripts/check_repository.py
venv_build/bin/python -m build --outdir .neuroshard/package-check
python scripts/check_distribution.py .neuroshard/package-check
```

Manuscripts, web publishing projects, raw experiment output, models and node state do not belong in the tracked repository or Python distributions. Keep local recovery material under ignored `archive/` and new experiment output under `.neuroshard/`. Preserve compact executable input fixtures when needed for reproduction. Link historical evidence to an immutable source revision instead of restoring its generated files to the active tree. Repository checks inspect the Git index, so stage intended file moves before running them locally.

## Propose a change

For a bug report, include the source revision, chain ID when relevant, numerical profile, expected behavior, actual behavior, and reproduction steps. Share public transaction/block identifiers or reduced test cases. Exclude private keys, signing state, credentials, and user records.

For changes to consensus, validator admission, verification, reward rules, or execution semantics, open a design proposal explaining the invariant or behavior being changed, compatibility implications, assumptions, and how it can be tested. Small documentation and interface fixes can proceed directly as pull requests.

A pull request should explain the concrete problem, resulting behavior, and relevant validation. Add regression coverage when a change affects security, accounting, consensus, or failure recovery. Report which checks ran and any limitations; a passing small-model experiment does not establish LLM-scale performance or adversarial security.

The execution manifest binds numerical code and consensus source. Editing or moving those files can require a new compatible genesis/release. Test against disposable chain homes and preserve live validators' keys, databases, and signing state. Operators choose their supported release; merging a pull request must not automatically upgrade the running network.

## Useful contributions

- Reproduce the installation and training trial on additional compatible machines.
- Improve node synchronization, diagnostics, bounded APIs, and worker failure handling.
- Measure public-load behavior and verification cost with reproducible workloads.
- Reproduce or refute the frozen [learning milestone](docs/LEARNING_MILESTONE.md) and its [rejected 128-step result](docs/LEARNING_MILESTONE_RESULTS.md) from public sources; do not relax its sealed-set or margin rules.
- Prepare or refute the frozen [continued-learning](docs/CONTINUED_LEARNING.md) contract from the passing phase-A checkpoint; do not reuse adaptive finals, grow the model, or treat a checkpoint file as payment.
- Investigate complete training verification and portable numerical execution.
- Improve the client, protocol APIs and technical documentation using measured behavior.

Consensus safety, monetary accounting, and claims in the paper should remain explicit about their assumptions. Reproducible negative results are useful contributions.

## License

Contributions to project code are made under the existing [Apache License 2.0](LICENSE). Preserve applicable attribution and identify the provenance and license of any added third-party code, data, models, or assets.

CI checks both supported Python versions, the pinned consensus build, repository links and package contents. Website publishing and browser checks belong to the separately maintained publishing workspace.
