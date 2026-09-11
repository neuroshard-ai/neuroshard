# NeuroShard

Contribute neural computation, earn native **NEURO**, and use it to pay for language-model responses. NeuroShard runs its own blockchain; keys stay on your machine and participation requires no website registration.

Release **0.4.0** is an experimental public testnet using **SmolLM2-135M-Instruct with a 4,608-parameter trainable adapter**. Native validators replay training and inference before accepting work or paying providers. The pretrained backbone is frozen. The initial validators share one operator across two hosts; this is a working protocol baseline, with economical large-model verification and independent ownership still to solve.

## Join

```bash
python3 -m venv ~/.venvs/neuroshard
source ~/.venvs/neuroshard/bin/activate
python -m pip install --upgrade pip setuptools neuroshard-ai
neuroshard doctor
neuroshard join
```

The lightweight client installs a separate pinned CPU runtime when you join, verifies the model/data/genesis, follows the native ledger and offers stage 1 work. Full nodes require Linux x86_64, Python 3.10–3.12, approximately 8 GiB RAM and 5 GiB initial free disk. Wallet and remote inference commands do not download the model. Leave `join` running; Ctrl+C preserves keys and history. Use a Python virtual environment if your OS manages the system Python.

After an accepted task earns NEURO, use another terminal:

```bash
neuroshard wallet balance
neuroshard chat "What is the capital of France?" --max-price 0.1
neuroshard wallet export neuroshard-key.json
```

Keep the backup private. It also works in the [browser inference interface](https://neuroshard.com/chat). **Prompts and responses are public.** A 32-token request costs 0.033 NEURO including its submission fee; expiry unlocks its budget but not the fee. Test balances have no promised monetary value or migration to a future chain.

[Website and ledger](https://neuroshard.com) · [PyPI](https://pypi.org/project/neuroshard-ai/) · [Operator guide](docs/PUBLIC_TESTNET.md) · [Documentation](https://docs.neuroshard.com) · [Releases](https://github.com/neuroshard-ai/neuroshard/releases)

## Protocol and evidence

- [Complete LLM protocol](docs/LLM_PROTOCOL.md): native consensus and bonds, training leases/rewards, serving promotion, paid inference, locks/refunds, limits and assumptions.
- [Experiments](docs/LLM_EXPERIMENTS.md): multi-host numerical and native settlement records, model probes, failures and reproduction.
- [Model card](docs/MODEL_CARD.md) and [immutable S3/data pipeline](docs/DATA_PIPELINE.md).
- [Network/genesis/allocations](networks/neuroshard-llm-testnet-1) and [research roadmap](docs/RESEARCH_ROADMAP.md).
- [Continual model evolution](docs/EVOLUTION_PROTOCOL.md): full-model training, response-focused evaluation, model growth and native disputes. These experiments have not replaced the public 0.4.0 network.
- [Inherited native ledger rules](docs/PROTOCOL_CANDIDATE_V2.md) and the [published research paper](https://neuroshard.com/papers/FINE2026_neuroshard_short.pdf).

Correct computation, improved model quality, decentralization and economic sustainability are separate claims. Full replay provides a precise acceptance rule but duplicates computation. Four fixed public validation sequences gate serving promotion and can be overfit. Signatures do not prove new physical energy expenditure. The initial allocation is 90 NEURO and the profile caps training issuance at 10,000 tasks. See the specification for these explicit limits.

## Develop and verify

```bash
python3 -m venv venv_build
venv_build/bin/python -m pip install -r docs/llm-requirements.txt '.[dev,data]'
ATEN_CPU_CAPABILITY=default MKL_ENABLE_INSTRUCTIONS=SSE4_2 \
  venv_build/bin/python -m pytest -q
```

The supported client is `src/neuroshard/client`, the LLM application is `src/neuroshard/inference`, immutable ingestion is `src/neuroshard/dataflow`, and the inherited native ledger is `src/neuroshard/lab`. Reference execution/transport remains in `demo` and `publicnet`. The `evolution` package contains the separately tested continual-model implementation. Some earlier modules remain for source compatibility; they do not define the current public entry points. See the [runtime map](src/neuroshard/README.md).

## Repository scope

This repository contains the open-source protocol and client, tests, reproducibility tools, network manifests, example configuration and technical documentation. Operating a node does not require the project's website source or a website account.

Website publishing projects, manuscripts, historical logs and measurement dumps are maintained outside the tracked tree. The paper remains available on neuroshard.com. Existing evidence links point to an immutable historical revision; removing generated material from the current tree does not erase those results. Local recovery material belongs in the ignored `archive/` directory, while keys and running node state belong in ignored homes such as `.neuroshard/`. Neither is included in releases.

Versions 0.2.x and 0.3's tiny reference chain are separate histories; registration tokens and old balances do not migrate. Never reuse an initialized node home for a different genesis. [Documentation](docs/README.md) explains the retained reference fixtures and repository checks.

[Contribute](CONTRIBUTING.md) · [Governance](GOVERNANCE.md) · [Security](SECURITY.md) · [Deployment](docs/DEPLOYMENT.md) · [API](docs/API.md) · [Release notes](RELEASES.md) · [Third-party provenance](THIRD_PARTY.md) · [Apache 2.0](LICENSE)
