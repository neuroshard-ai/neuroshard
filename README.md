# NeuroShard

NeuroShard is a research project toward a decentralized network that trains language models and rewards verified computation on its own blockchain.

The current native release combines **NeuroShard-native CometBFT consensus**, a public ledger, two-stage neural training, full verification replay, token rewards, earned-stake validator entry, delayed exits, and equivocation penalties. Full nodes generate their keys locally and join without a website account or registration token. Outbound workers can execute sponsored tasks without an initial balance or an inbound worker port.

The supported model has 34,976 parameters and runs on a pinned CPU profile. The deployment is experimental, with one operator currently controlling the test validators. Economical LLM-scale verification, independent ownership, fair worker assignment, sustained public load, and production monetary policy remain open work.

## Start here

Read the [native operator guide](docs/PUBLIC_TESTNET.md) for installation, genesis verification, joining, training, validation, and recovery. The public application is [neuroshard.com](https://neuroshard.com), with [documentation](https://docs.neuroshard.com) and [releases](https://github.com/neuroshard-ai/neuroshard/releases).

```bash
git clone --branch v0.3.0a1 --depth 1 https://github.com/neuroshard-ai/neuroshard.git
cd neuroshard
bash scripts/install_native.sh
venv_build/bin/python scripts/neuroshard_chain.py --help
venv_build/bin/python scripts/neuroshard_work.py --help
```

Use the reviewed source revision matching your network's genesis. The installer targets Linux x86_64 and Python 3.10–3.12. The old PyPI package, `neuroshard --token ...` command, and observer ledger belong to an earlier prototype and do not connect to this native chain. No balance migration is defined.

## Protocol and evidence

- [Protocol candidate v2](docs/PROTOCOL_CANDIDATE_V2.md): accepted transactions, execution rules, issuance, evidence, exit, and threat assumptions.
- [Experiments and reproduction](docs/PROTOCOL_EXPERIMENTS.md): native admission, cross-machine numerical conformance, real equivocation evidence, verification-cost experiments, and their limits.
- [Five-page manuscript](docs/FINE2026_neuroshard_short.pdf) and [source](docs/FINE2026_neuroshard_short.tex): earlier research formulation; new deployment experiments are documented separately before integration into the paper.
- [Fundamentals review](docs/FUNDAMENTALS_REVIEW.md) and [research roadmap](docs/RESEARCH_ROADMAP.md).

Signatures identify the worker that made a claim. Validators independently replay the prescribed computation to verify it. Rewarding a correct training step does not prove that it improved model quality, that a particular physical processor performed it, or that mining is economically sustainable.

## Check the implementation

```bash
ATEN_CPU_CAPABILITY=default MKL_ENABLE_INSTRUCTIONS=SSE4_2 PYTHONPATH=src \
  venv_build/bin/python -m pytest -q tests/test_verified_demo.py \
  tests/test_protocol_candidate.py tests/test_public_node.py tests/test_outbound_work.py
```

The reference code is in `src/neuroshard/demo`, the candidate state machine in `src/neuroshard/lab`, public node and worker tooling in `src/neuroshard/publicnet`, and the replacement site in `website`. Earlier implementation modules are retained for research history; the native entry points above define the supported deployment path.

[Apache 2.0 license](LICENSE).

## One open-source project

All supported code, website, documentation, paper, experiments, and deployment templates live here. Develop directly in this repository; there is no private-to-public sync workflow. The former private repository is retained as a recovery archive.

- [Contribute](CONTRIBUTING.md), [governance](GOVERNANCE.md), and [security](SECURITY.md).
- [Model card](docs/MODEL_CARD.md), [API](docs/API.md), and [deployment](docs/DEPLOYMENT.md).
- [Release notes](RELEASES.md) and [migration inventory](docs/migration-inventory.json).
- [Historical prototypes](legacy/README.md), outside the supported deployment and test path.
