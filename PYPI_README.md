# NeuroShard

Contribute neural computation and use native **NEURO** to pay for inference.
NeuroShard runs its own blockchain. Your keys stay on your machine; joining
requires no website account or registration token.

**Experimental protocol testnet.** This package serves the 0.4.0 ledger and a
small pretrained instruction model with a trainable adapter. It is not the
1.7B bounded-activation research assistant. Validators replay the computation
before accepting work or paying providers. Test NEURO has no promised monetary
value or mainnet conversion.

## Install and join

```bash
python3 -m venv ~/.venvs/neuroshard
source ~/.venvs/neuroshard/bin/activate
python -m pip install --upgrade pip setuptools neuroshard-ai
neuroshard doctor
neuroshard join
```

The Python package is a lightweight client. `join` installs a separate, pinned CPU
runtime, downloads and checks the model and network data, creates a local key,
starts a full node, and contributes training stage 1 when sponsored work is available.
It displays its progress. Press **Ctrl+C** to stop; your key and history remain on disk.

Full nodes require **Linux x86_64 and Python 3.10–3.12**. Allow at least 8 GiB RAM
and 5 GiB free disk for the runtime, model, and initial history. Supported workers
use the pinned CPU profile; arbitrary GPU execution is not accepted by this release.
Wallet and remote inference commands do not download the neural runtime.

```bash
neuroshard status
neuroshard wallet export ./neuroshard-key.json
```

Keep the exported key private. It controls the account. Rewards are paid only when
the complete task is accepted by native validators; returning work alone is not payment.

## Use the model

After earning NEURO or receiving a native transfer:

```bash
neuroshard chat "Explain what a blockchain is in one sentence." --max-price 0.1
```

The client shows the total budget and request ID. The chain locks the inference
budget, checks the provider's output, and settles payment. A timed-out request
unlocks its budget; the submission fee is spent. Use `neuroshard request REQUEST_ID`
to inspect an uncertain or pending result before submitting another request.

**Prompts and responses are public.** This small model can make mistakes and is not
suitable for sensitive information. The current deployment has one operator across
two hosts; independent operators and economical large-model verification remain open work.

## Choose how to participate

```bash
neuroshard join --role observer    # Verify the ledger without contributing training
neuroshard join --role provider    # Run a node and serve requests addressed to your key
neuroshard --help                 # Wallet, transfer, request, and connection options
```

An existing participant can use `neuroshard start` to resume. Use `--home` for a
separate identity and `--network-file` for another reviewed native network. A remote
RPC supplies a view of the ledger; running a full node independently replays it.

Upgrading from **0.2.x** replaces the old `neuroshard --token ...` client. Website
registration tokens and the earlier observer balances do not migrate to the native
chain. Version 0.3's tiny reference chain remains a separate research network.

[Website](https://neuroshard.com) · [Operator guide](https://docs.neuroshard.com/generated/PUBLIC_TESTNET)
· [Source and issues](https://github.com/neuroshard-ai/neuroshard)

Apache License 2.0. Pretrained model and dataset provenance are documented in the repository.
