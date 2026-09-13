# NeuroShard

NeuroShard is an experimental native blockchain for verifying and rewarding neural computation. Its public testnet demonstrates small-scale training and paid inference; permissionless full-model training remains under development. The goal is a collectively trained LLM whose usable capacity can expand as reliable compute joins. Keys stay on your machine and participation requires no website registration.

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
- [Continual model evolution](docs/EVOLUTION_PROTOCOL.md): full-model training, [versioned text tokenization](docs/TEXT_PROTOCOL.md), response-focused evaluation, model growth and native disputes. The [native lifecycle](docs/NATIVE_LIFECYCLE.md) connects curated fresh-data admission to training, serving decisions and paid generation on isolated integration networks. These experiments have not replaced the public 0.4.0 network.
- [Learning milestone](docs/LEARNING_MILESTONE.md): a Git-sealed prepare/train/score experiment for full-model learning. The [128-step result](docs/LEARNING_MILESTONE_RESULTS.md) lowers mean response loss but fails its sealed-test confidence bound; all measurements, generation pairs and the rejected checkpoint are public. The later continual-learning and two-host phases remain blocked.
- [GPU learning reference](docs/LEARNING_REFERENCE.md): the [completed 1.7B run](docs/LEARNING_REFERENCE_RESULTS.md) trained all parameters and reproduced checkpoint recovery exactly on one GPU. Final-test improvement remains inconclusive and answers show regressions; all measured pairs and the target audit are public.
- [Two-GPU learning and serving](docs/COOPERATIVE_LEARNING_RESULTS.md): exact accuracy rises from 23/256 to 169/256 on four generated task families; shared training reaches 168/256, with slower training and 1.67× inference throughput from two replicas. Arithmetic remains failed; this operated experiment changes no native model or rewards.
- [Scaling design](docs/SCALING_DESIGN.md): how added capacity should support complete auditing, reliable training and measured model growth. [Compact optimizer disputes](docs/COMPACT_UPDATE_DISPUTES.md) implement a first bounded refutation path; they do not replace complete training verification.
- [Funded audit candidate](docs/FUNDED_AUDITING.md): prepaid complete-graph replay services, collateral, refunds and a [recovering full-model operator](docs/CANDIDATE_OPERATIONS.md), with [integration evidence](docs/FUNDED_AUDIT_RESULTS.md). The sponsor selects auditors; independent ownership and collusion resistance remain open requirements.
- [Inherited native ledger rules](docs/PROTOCOL_CANDIDATE_V2.md) and the [published research paper](https://neuroshard.com/papers/FINE2026_neuroshard_short.pdf).

Correct computation, improved model quality, decentralization and economic sustainability are separate claims. Full replay provides a precise acceptance rule but duplicates computation. Four fixed public validation sequences gate serving promotion and can be overfit. Signatures do not prove new physical energy expenditure. The initial allocation is 90 NEURO and the profile caps training issuance at 10,000 tasks. See the specification for these explicit limits.

## Develop and verify

```bash
python3 -m venv venv_build
venv_build/bin/python -m pip install -r docs/evolution-requirements.txt '.[dev,data]'
ATEN_CPU_CAPABILITY=default MKL_ENABLE_INSTRUCTIONS=SSE4_2 \
  venv_build/bin/python -m pytest -q
```

The supported client is `src/neuroshard/client`, the LLM application is `src/neuroshard/inference`, immutable ingestion is `src/neuroshard/dataflow`, and the inherited native ledger is `src/neuroshard/lab`. Reference execution/transport remains in `demo` and `publicnet`. The `evolution` package contains the separately tested continual-model implementation. Some earlier modules remain for source compatibility; they do not define the current public entry points. See the [runtime map](src/neuroshard/README.md).

[Governance](GOVERNANCE.md) · [Security](SECURITY.md) · [Deployment](docs/DEPLOYMENT.md) · [API](docs/API.md) · [Release notes](RELEASES.md)

---

[Contributions](CONTRIBUTING.md), protocol reviews, and reproducible experiments are welcome.

If you use NeuroShard in research, please [cite the software](CITATION.cff) and include the release or commit you used.

Licensed under the [Apache License 2.0](LICENSE). Attribution and third-party acknowledgements are in [NOTICE](NOTICE) and [THIRD_PARTY.md](THIRD_PARTY.md).
