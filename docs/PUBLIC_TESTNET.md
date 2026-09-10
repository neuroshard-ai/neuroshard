# Join NeuroShard LLM testnet 1

Release **0.4.0** uses a small pretrained instruction model, native training rewards and NEURO-paid inference. It runs NeuroShard's own CometBFT chain. The [network descriptor](../networks/neuroshard-llm-testnet-1/network.json), [genesis](../networks/neuroshard-llm-testnet-1/genesis.json), [protocol](LLM_PROTOCOL.md) and [model card](MODEL_CARD.md) define this experimental network.

## Install, inspect, join

```bash
python3 -m venv ~/.venvs/neuroshard
source ~/.venvs/neuroshard/bin/activate
python -m pip install --upgrade pip setuptools neuroshard-ai
neuroshard doctor
neuroshard join
```

If your operating system manages its Python installation, use a virtual environment or `pipx install neuroshard-ai`; do not override the OS package manager. Full nodes require Linux x86_64 and Python 3.10–3.12. Allow 8 GiB RAM and 5 GiB free disk initially; the two tested hosts have 16 GiB RAM. History grows, so monitor disk space. Wallet and remote chat do not require the neural runtime.

The small client installs a separate pinned CPU environment on explicit `join`/`setup`, prepares CometBFT with a checksum-verified Go toolchain, downloads hash-checked model/data files, generates local keys, checks numerical conformance and replays the native ledger. It then offers stage 1 work to the project sponsor. The first setup can take several minutes. Status reports show synchronization, round and balance; rewards require a finalized complete task.

No registration, starting balance or inbound worker port is required. Outbound HTTPS connects to the sponsor; native TCP connects to peers. Listen on TCP 26656 if you want to accept inbound peers. RPC and application gRPC stay on loopback. The default node/key home is `~/.neuroshard/llm-testnet`. Runtime and model caches are under `~/.neuroshard`. Set `NEUROSHARD_STATE_DIR` before installing/starting to use another root.

Leave `join` running to contribute. Ctrl+C stops its node and worker while retaining keys and history. Resume with `neuroshard start`. Default training cadence is roughly five minutes plus execution time; assignments depend on sponsor availability and a finite attempt budget. Operator fallback workers keep both stages available and yield to available public workers. This selection policy is not a Sybil-resistant allocation mechanism.

## Keys, balances and inference

In another terminal:

```bash
neuroshard status
neuroshard wallet balance
neuroshard wallet export ./neuroshard-key.json
neuroshard chat "What is the capital of France?" --max-tokens 32 --max-price 0.1
```

The backup contains the private native seed; anyone with it can spend that account's funds. The CLI never prints it. Keep an offline private copy. `neuroshard wallet import FILE --home NEW_HOME` restores an account into a new home. This does not restore validator signing state; validators require their separate consistent node backup. Never run a second validator with copied live consensus keys.

Import the same account backup at [the inference page](https://neuroshard.com/chat) to spend worker rewards in the browser. The page retains keys only in memory and stores only pending request IDs in session storage. Reloading requires reimport. The website code is part of your signing trust boundary; the CLI with your own RPC avoids relying on the website to sign.

A 32-token request currently costs 0.033 NEURO including a 0.001 submission fee. This is a fixed price for the requested output limit, even if EOS finishes early. Inference locks its budget, pays only after validators reproduce the output, and unlocks the budget if the provider misses its block deadline. The fee remains spent. All prompts and responses are public.

The client saves its signed request before broadcast and displays its immutable request ID. If the connection fails or settlement is pending, inspect the same request before paying again:

```bash
neuroshard request REQUEST_ID
neuroshard transfer --to RECIPIENT_PUBLIC_KEY --amount 0.1
```

The API retains the latest 128 completed/expired requests; older results require retained native history. Transfer submission is not a claim of final acceptance; inspect its block or account nonce.

## Other roles and independent verification

```bash
neuroshard join --role observer
neuroshard join --role provider
neuroshard work --stage 0 --coordinator https://YOUR_SPONSOR
neuroshard serve
neuroshard chat "Hello" --provider PROVIDER_PUBLIC_KEY
```

`work` and `serve` use an already running local node. An inference provider receives jobs only when customers select its key; starting a provider does not automatically list it on the project website. Use distinct homes/identities when running multiple instances and distinct base ports. Do not mix two processes that spend from the same account without coordinating nonces.

The bundled descriptor pins chain `neuroshard-llm-testnet-1` and genesis SHA-256 `cf74dba2e15af1c66e893cb7a8b079273e157c5585bd8af15d7194588ba47cb7`. Compare these with the public release. Startup uses the project's current height/hash as a checkpoint by default. An old stake history needs a recent trusted checkpoint; for independent operation obtain it from independently trusted operators and supply:

```bash
neuroshard join --trusted-height HEIGHT --trusted-hash BLOCK_HASH
```

A custom `--network-file` must contain the full reviewed descriptor; `--rpc` selects another HTTP RPC for wallet/chat reads and submissions. Full-node execution still uses its own local native RPC. A public RPC is a server's view, without account proofs; a local full node independently replays history. A genesis download alone is not a solution to long-range stake-history attacks.

## Bonding and native operator tools

The lightweight client handles onboarding and payments. Advanced native tools run in the installed CPU environment:

```bash
~/.neuroshard/runtimes/0.4.0/bin/neuroshard-chain --help
~/.neuroshard/runtimes/0.4.0/bin/neuroshard-work --help
```

The `bond`, `unbond`, `withdraw`, `account` and `status` commands operate on native accounts; amounts in these advanced commands are integer atoms. Use the same node home and inspect their help for arguments. Use `neuroshard start` for this LLM profile: the older `neuroshard-chain init/run` commands are for the v0.3 reference profile. Bonding uses a minimum unit of 0.25 NEURO, a consensus-key possession proof and delayed activation. Read [the protocol](LLM_PROTOCOL.md) before becoming a validator, especially evidence windows and withdrawal. The launch validators are controlled by one operator; permissionless entry does not itself establish independent ownership.

## Failures and recovery

- Runtime installation: inspect `~/.neuroshard/runtimes/0.4.0/install.log`, fix disk/network errors and rerun `neuroshard setup`. Installation is locked against concurrent setup.
- Consensus build: inspect `~/.neuroshard/tools/consensus-build.log`. Pinned Go and module versions are required.
- Startup/conformance: inspect the node home's `logs/`. Never bypass a failed profile or checkpoint check.
- No work: inspect [sponsor status](https://neuroshard.com/work/status), the node's sync state and `logs/worker.log`. A returned receipt can still be pending payment. A failed sponsor lease pays no worker reward.
- Inference unavailable: inspect [provider status](https://neuroshard.com/api/inference). Do not submit repeated payments to recover an uncertain request.
- Restart: preserve `account.key`, `config/`, `data/`, `candidate.sqlite` and associated WAL consistently. Replay history from peers if rebuilding; preserve validator signing state. Do not edit balances or reset nonces.

## Earlier clients and chains

Version 0.2.x used registration tokens and an observer ledger. Upgrading replaces that client; old credentials and balances do not migrate. Version 0.3's 34,976-parameter native reference chain is also separate. Its [historical instructions](REFERENCE_NODE_V03.md) describe that release, and its [network bundle](../networks/neuroshard-stage-8i5ghxq5) remains in Git. TCP 26656 on the public seed now serves the new LLM chain; old peers must not assume the same address identifies the old network. A retained reference explorer is at `/reference/v03/api/network` on neuroshard.com. Never initialize a different chain in an existing home.
