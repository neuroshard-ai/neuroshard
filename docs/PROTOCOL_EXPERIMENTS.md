# Protocol experiments and decisions

The latest deployment checks are in [Public deployment candidate](#public-deployment-candidate-direct-p2p-and-outbound-workers) below and the [operator guide](PUBLIC_TESTNET.md). The earlier tables describe their original runs.

Date: 2026-09-10. The paper is unchanged in this iteration. The executable [v2 protocol candidate](PROTOCOL_CANDIDATE_V2.md) records what the experiments currently justify, including its limits.

## Results

| Question | Result | Decision |
|---|---|---|
| Can a participant earn its first tokens and enter native consensus without an admission key? | A worker earned 800,000 atoms, bonded 250,000, and joined the actual CometBFT set: 4 → 5 validators | Implement public bond transactions, proof of consensus-key possession, epoch scheduling, and H+2 effective updates |
| Can an exited validator escape a known offense immediately? | Genuine duplicate-vote evidence was accepted after removal and burned 62,500 of its still-reserved atoms | Require both height and time horizons before returning collateral |
| Does an offline validator keep receiving verifier pay? | It received its legitimate worker pay but no verifier share while its consensus process was offline | Pay verifier shares using the actual task-block commit, one block later |
| Can a miner collect twice or claim the wrong result? | Incorrect, expired, and repeated submissions were rejected without advancing the model or issuance | Keep exact replay, atomic settlement, and per-account nonces |
| Is a data provider a mandatory trusted authority? | The newcomer rejected all 69 corrupt chunk responses and fetched the same 1,115,394-byte corpus from another peer | Authenticate content, permit peer fallback, and fail closed when no source works |
| Is deterministic PyTorch enough for bitwise agreement? | Repeated identical profiles agreed; changed CPU dispatch profiles differed by up to about 3.35×10⁻⁸ in parameters | Pin dispatch before import, bind numerical conformance vectors in genesis, and keep actual-task replay |
| Can different packaged runtimes agree under that profile? | Host `2.9.1+cpu` and a container's `2.9.1+cu128`, both executing on CPU with pinned dispatch, matched three complete training updates | A useful conformance result; the container build is not automatically admitted by the stricter versioned chain manifest |
| Does the profile agree on another machine? | Xeon 8259CL/Python 3.10.12 and Xeon 8175M/Python 3.12.3 matched 20 full updates, including gradients, stage receipts, inference, and genesis manifest | Admit this tested pair to the candidate profile; keep broader hardware support conditional on conformance |
| Does the native lifecycle work across those machines? | Three local nodes and two remote nodes, with one training stage on each host, completed the entire admission/exit/slashing/expiry/inference scenario and agreed on a common block | The candidate now has an inter-host execution result; independent ownership and public peer discovery remain untested |
| Can exact matrix verification be cheaper than computing the product? | At 512 and 1,024 square dimensions, measured 3.40× and 4.46× speedups against the same integer-product baseline; small products were slower | Retain as an experimental operator verifier, not a general training proof |
| Does useful-work issuance preserve the honest-stake assumption automatically? | In a stated scenario, an actor with 10% initial stake and 80% of work crosses one-third voting weight at epoch 71 | Treat resource ownership/concentration as an explicit security assumption and economic experiment |

The native run completed five valid training steps, reduced the fixed held-out batch loss from approximately 5.7411 to 4.8006, and issued exactly 5,000,000 atoms. Paid inference credited its provider 80,000 atoms from a funded request and minted nothing. An expired reservation burned its 2,000,000-atom collateral. After joining, exit, evidence, expiry, and payment, all five full nodes agreed on model state, accounts, bond histories, and a common native block. The experiment finished with four voting validators.

The newcomer used an ordinary sponsored training task to acquire its balance; there was no faucet allocation to it and no privileged join endpoint. A sponsor still selected the worker. This does not demonstrate fair access to work or independence of the local operators.

The same scenario subsequently passed across both machines. Training still ended at the same five-step model root, `bd96cb5975655b6fd0025dcfbcf7213b1e8efb86c3d3ec4918c6cf1bf5a80b74`, with 5,000,000 atoms issued and 80,000 paid to the remote inference provider. The final common block at height 91 is recorded in the [two-host native result](eval/results/protocol_two_host_native.json). The joining/exiting validator ran on the remote machine, and its post-exit evidence and withdrawal were settled by the combined native chain.

## What failed and what it taught us

CPU dispatch changed even the initial parameter root. Floating results that are numerically close are not interchangeable commitments. Genesis now records actual CPU dispatch and three full gradient/update vectors; incompatible validators fail initialization. Those vectors are finite tests, so wide hardware support still requires a stronger numerical contract.

A fixed all-ones projection accepted a deliberately incorrect matrix: adding to one output column and subtracting from another preserved the tested sum. Deriving checks from a complete output-bound statement rejected the same forgery. The probability claim for that construction is conditional on its random-oracle/grinding assumptions, not established by the 64 injected corruption tests.

Exact reassociation also failed for honest float32 arithmetic, while loose tolerances accepted a modified output. Integer bounds avoid that particular ambiguity; they do not establish correct quantized attention, backpropagation, SGD, or model quality. Verification timing includes statement hashing and bounds checks, but excludes network transfer and any new cost of converting a training system to the integer profile.

The offline-payment test initially observed a previously earned reward arriving during the measurement interval. The final test waits for prior deferred rewards to settle before measuring the new offline task. Native rewards now have an explicit earning height, signer snapshot, and settlement height.

The unit checks also cover a dangerous exit corner case: a merely pending replacement must not let the last validator leave before replacement activation. The rule requires another activation to have been emitted before the last active bond can schedule exit. Genesis validation additionally rejects mismatches between the native engine's evidence windows and the application's unbond rules, as well as unsupported block limits, key types, and vote extensions.

## Reproduce in this workspace

The existing Python environment and native engine are prepared. The native probe creates a disposable localhost chain and stops its own processes when finished:

```bash
venv_build/bin/python scripts/protocol_lab.py --output .neuroshard/native-v2-result.json
PYTHONPATH=src venv_build/bin/python -m pytest tests/test_protocol_candidate.py tests/test_verified_demo.py -q
PYTHONPATH=src venv_build/bin/python -m neuroshard.lab.experiments run --output .neuroshard/operator-results.json
venv_build/bin/python scripts/check_numerics_container.py --image neuroshard-trainer:latest --output .neuroshard/container-result.json
python3 docs/eval/analyze_membership.py --output .neuroshard/economic-scenarios.json
```

The combined unit suite contains **59 passing tests on each machine**. It includes exact pipeline gradients from the v1 reference plus v2 account conservation, adversarial claims, delayed power, split identities, copied possession proofs, both unbond clocks, evidence replay, numerical/native-parameter genesis mismatch, and interrupted validator-update commit recovery.

The container command requires Docker and an existing image with compatible dependencies. It uses the image's recorded immutable local ID and disables networking for the numerical execution. It does not run a GPU or provide an independent physical machine. `--base-port` on the native probe selects another localhost port range when needed.

For a fresh checkout, install the CPU dependencies in [demo-requirements.txt](demo-requirements.txt), install Go, and build both pinned native tools:

```bash
bash scripts/install_demo_consensus.sh
bash scripts/build_protocol_probe.sh
```

The second tool deliberately creates conflicting signatures only for a key generated inside the disposable test chain. The normal validator path uses CometBFT's signing-state protection. The build is pinned by [go.mod](../tools/protocolprobe/go.mod) and [go.sum](../tools/protocolprobe/go.sum). The lab's additional ABCI schema can be regenerated with `grpcio-tools==1.76.0`:

```bash
python -m grpc_tools.protoc -I src --python_out=src src/neuroshard/lab/abci.proto
```

## Second-machine environment

The supplied Ubuntu 24.04 machine is prepared at `/home/ubuntu/neuroshard-lab` with Python 3.12.3, its own `venv_build`, the pinned CPU dependencies, matching source/data, and the CometBFT 0.38.26 engine and evidence helper. The native binaries were copied from the already-built local tools and their SHA-256 digests checked on both machines. `system-setup.log`, `python-setup.log`, and `python-environment.txt` record installation details there. No Docker or GPU is required.

On another owner-authorized Ubuntu host, the setup script installs `python3-venv`, copies the experiment sources and the two prebuilt native binaries, and installs the CPU requirements in that isolated directory. It does not copy existing validator/account keys or chain databases. The combined network experiment later generates its own disposable test keys and transfers only the homes/keys needed for its remote roles.

```bash
bash scripts/setup_remote_lab.sh ubuntu@YOUR_HOST
venv_build/bin/python scripts/check_numerics_remote.py --host ubuntu@YOUR_HOST --steps 20 --output .neuroshard/hardware-conformance.json
venv_build/bin/python scripts/protocol_lab_remote.py --host ubuntu@YOUR_HOST --output .neuroshard/two-host-native.json
```

The numerical command compares the full manifest and each step's parameter, gradient, and stage-receipt commitments, plus inference after the twentieth update. The network command places full nodes 0–2 and training stage 0 locally; nodes 3–4, stage 1, and the inference provider run remotely. SSH forwards carry real inter-host P2P and HTTP/RPC traffic while services bind to loopback. The runner stops its local/remote processes and tunnel after the scenario; the prepared environment and logs remain available. Both machines are controlled by one operator.

## Evidence and interpretation

The [native protocol result](eval/results/protocol_native_v2.json) binds the source and numerical manifest, records actual membership heights and evidence hash, and identifies a common block. The [operator/profiles report](eval/results/protocol_candidates.json) contains timing and corruption observations. The [container report](eval/results/protocol_container_conformance.json) identifies the image, packaged runtime, and compared vectors. The [two-machine conformance report](eval/results/protocol_remote_conformance.json) contains all 20 compared training vectors and hardware/runtime identities. The [economic scenarios](eval/results/protocol_economics.json) are calculations under stated assumptions, not measured adversarial participation or market forecasts.

The initial experiments ran on one four-vCPU x86_64 host. The additional machine has four vCPUs and a different reported Xeon model; both have approximately 16 GiB RAM. The pinned native execution is a new profile, so its parameter roots differ from the earlier native-dispatch reference even when displayed losses are almost identical. Old v1 results and the paper are preserved.

The economic concentration scenario assumes all rewards are restaked after four epochs, with 80% of each reward allocated to workers and 20% to consensus. An actor may perform valid work while acquiring enough ownership to threaten future consensus. Slashing for proven faults does not prevent that acquisition. A separate reservation-budget calculation shows that 10,000,000 atoms can finance four completely forfeited reservations at the current fee/bond, occupying up to 68 block intervals without replenishment. This bounds one attack's resource cost, not the network's total denial-of-service risk.

The next decisive experiment is a complete training graph under cheaper, independently reproducible verification, compared at equal total compute/bandwidth against full replay. It must include nonlinearities, quantization/range constraints, boundary dependencies, backward operators, the optimizer, and retained state. Extend the current two-machine work to independently controlled operators, broader CPU/GPU profiles, genuine network partitions, and adversarial task discovery. Enlarging the manuscript should follow those measured claims.

## Public deployment candidate: direct P2P and outbound workers

The subsequent [direct-network record](eval/results/public_network_candidate.json) uses a new genesis and the `testnet` timing profile. Native P2P travels directly over the remote server's public TCP 26656; SSH is used for setup and process administration. Three initial validators run on one host and the fourth on the other. A fifth full node joins from genesis using only the public peer, starts with zero balance, earns 800,000 atoms from two accepted training steps, bonds 250,000 atoms plus its fee, and becomes a voting validator after the longer activation/epoch delay. Its bond transaction was committed at height 16 and active membership was observed at height 122.

Stopping the remote seed preserves progress with sufficient remaining voting weight. Stopping another validator leaves 21 of 41 voting units, and the chain halts at height 126. Restarting the validator restores progress; the remote seed then rejoins and catches up. Both machines agree on the three-step model and block 136, hash `112FD6B7ACCCA10D30861FCC647726EB7F435C6D73117763F3A87BF4AD8337A7`. The test issued exactly 3,000,000 atoms. All keys remain controlled by one operator; this is not evidence of independent ownership or resistance to a sustained Internet attack.

The [outbound-worker record](eval/results/outbound_work_candidate.json) starts from a separate fresh installation on the remote machine, including a new Python environment and a locally built pinned consensus engine. Its new full node checks the HTTPS genesis against the recorded digest and verifies the chain locally. A zero-balance worker supplies stage 1 over outbound HTTPS to the preview site, with no inbound worker port. The fourth accepted training step credits it 400,000 atoms, and the local/remote model roots match. The optional sponsor process cannot accept an invalid update or issue its own reward; it submits the signed receipts to native validators for full replay.

Public gateway tests cover administrative-RPC rejection, exact genesis bytes, profile isolation, paginated/bounded state reads, nanosecond timestamps on Python 3.10, and signed transaction validation. Worker tests reject foreign sponsors/chains, changed models/batches, wrong workers, expired leases, and stale messages; durable retries reuse completed computation. Actual browser checks cover desktop/mobile views, native block details, public account lookup, public peer instructions, and an unavailable-node state. The replacement frontend builds with Vite 7.3.6 and the recorded dependency audit reports no known vulnerabilities.

The first attempts exposed three harness/interface problems that were corrected before the passing run: observing a reward before peer propagation, parsing native nanoseconds on Python 3.10, and treating a transient SSH-wrapped connection refusal during restart as a terminal test failure. Polls now wait for the required committed state and retry bounded startup errors. None of these corrections weakens the ledger acceptance predicate.

Earlier JSON artifacts retain the manifests and source hashes from their original runs. This deployment changes the candidate source manifest and adds an explicit chain parameter profile; earlier manifest hashes must not be interpreted as matching the current source. The new direct-network record includes the current full manifest. The original numerical experiments remain finite observations, not a proof of arbitrary hardware compatibility. The [operator guide](PUBLIC_TESTNET.md) defines this release's scope and reproduction commands.

A longer-lived public sponsor exposed one further availability issue: immediately reusing a worker's recent registration after settlement could reserve another task just as that worker disconnected. The bounded sponsor stopped on timeout, and the native chain burned the abandoned reservation's collateral without advancing the model. The coordinator now clears selected workers' registrations after settlement and requires fresh polls for the next lease. Workers support `--max-tasks 1` for a bounded trial. A subsequent [public sponsor recovery check](eval/results/public_sponsor_recovery.json) finalized the sixth training step, confirmed the trial worker exited, and observed no extra reservation after exit. This improves normal disconnect behavior; deliberate worker withholding remains a sponsor risk.
