# One public NeuroShard project

The public consolidation uses [neuroshard-ai/neuroshard](https://github.com/neuroshard-ai/neuroshard) as the single development home. The native node, workers, gateway, website, docs, paper, experiments, and deployment templates are all available there. No privately maintained implementation is required to operate the supported release.

## Source migration

The public history through `d3b9068` is preserved. Its six snapshot commits and the private development history through `d9d123b` had unrelated roots, so the consolidation imports reviewed source rather than rewriting public history or exposing private operational history. Original authorship is recorded in AUTHORS and source notices.

The [migration inventory](migration-inventory.json) accounts for all 433 previously tracked private paths and the transition document. Useful source and research remain public. Historical registration/observer code, docs, deployment files, and tests are marked unsupported under `legacy/`. Obsolete private synchronization/cloud-maintenance scripts and unused template/build artifacts are retained only in the private recovery archive. These are not dependencies of the native protocol or application.

Current consensus-bound files and paths are preserved: the release matches the existing genesis and numerical manifest. The public release builder uses a named Git snapshot, excluding untracked runtime state. Private node keys, signing state, credentials, old user records, and local databases remain outside source control.

## Supported experience

The root application provides the ledger, validator/account views, live model card, checkpoint download, accepted training history, and node/worker onboarding. The docs site builds from canonical markdown in the same repository. The old registration API is retired; participants create keys locally. No balance or identity conversion from the previous observer ledger is defined.

The training-history index is separate from consensus and reconstructs successful submissions from retained blocks. Checkpoint downloads are content-addressed observations, not implemented state-sync or account inclusion proofs. Supply values are displayed without JavaScript integer rounding. Worker availability and sponsor limits are shown explicitly.

The native Python package exposes chain/worker commands, includes the corpus, and is tested outside a checkout against the published genesis. The old `--token` onboarding and `0.2.x` PyPI releases do not start this chain. The website, documentation, package metadata, and contribution guide now identify one supported path.

## Operation and community

GitHub issues, discussions, private vulnerability reporting, dependency alerts, secret scanning, and push protection are enabled for the public project. CI checks native Python versions, browser behavior, website/docs builds, and dependency audits. Maintainer responsibilities and protocol-change review are described in [governance](../GOVERNANCE.md).

The cutover preserves the old account database and existing chain homes. Web routing can be rolled back separately from consensus. The private repository remains private and is archived after the checked public cutover; future changes belong in the public project. See [deployment](DEPLOYMENT.md) and [release notes](../RELEASES.md).

The release is an experimental testnet. Its small CPU model, full replay, concentrated launch ownership, bounded sponsorship, and unresolved production economics remain explicit. Public source enables independent operation; it does not by itself establish independent ownership or economical LLM-scale verification.
