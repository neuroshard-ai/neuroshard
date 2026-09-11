# LLM experiments — September 2026

These measurements concern the 0.4 LLM profile. Earlier tiny-model, stake-admission, equivocation and verifier-cost experiments remain in [the v2 report](https://github.com/neuroshard-ai/neuroshard/blob/108b4ba3d6c6fb5760ff211b447ee95a67fa9112/docs/PROTOCOL_EXPERIMENTS.md); they are not substituted for LLM measurements.

## Model and cross-host conformance

The pretrained backbone has 134,515,008 frozen parameters and the residual adapter has 4,608 trainable parameters. Both tested hosts use CPU float32, one thread, eager attention and the pinned SSE4_2/default ATen profile, with Python 3.10 locally and 3.12 remotely. Three feature roots, gradient roots, updated adapter roots, losses and an inference result root matched exactly. Startup repeats these checks against genesis.

[Remote conformance record](https://github.com/neuroshard-ai/neuroshard/blob/108b4ba3d6c6fb5760ff211b447ee95a67fa9112/docs/experiments/llm-2026-09-10/remote-conformance.json) binds the current public manifest. This is evidence for these machines and versions, not universal floating-point portability.

[Base-model probes](https://github.com/neuroshard-ai/neuroshard/blob/108b4ba3d6c6fb5760ff211b447ee95a67fa9112/docs/experiments/llm-2026-09-10/base-model-probes.json) preserve all three manual prompts: France/Paris was correct (~1.30 seconds), a blockchain definition was coherent (~3.04 seconds), and a polite-rewrite instruction failed (~4.36 seconds). Those times exclude native consensus and are not end-to-end API latency or throughput measurements.

## Complete two-host native cycle

[The recorded disposable-network run](https://github.com/neuroshard-ai/neuroshard/blob/108b4ba3d6c6fb5760ff211b447ee95a67fa9112/docs/experiments/llm-2026-09-10/two-host-cycle.json) used two local validators, two remote validators and a fresh remote full-node worker. Native traffic crossed an SSH transport while the older public reference chain remained online. Both stages produced receipts, validators replayed work, and the worker then paid for inference from the promoted adapter.

| Observation | Result |
|---|---|
| Fresh worker balance | 0 atoms |
| Accepted updates | 3 |
| Earned stage rewards | 1,200,000 atoms = 1.2 NEURO |
| Inference budget / submission fee | 32,000 / 1,000 atoms |
| Provider payment | 25,600 atoms |
| Worker's final balance | 1,167,000 atoms = 1.167 NEURO |
| Served adapter | `56c1e62d59f3aac596d8141f9bb4dc25104b69aff2961c0b26d73b4c01e164a0` |
| Answer | “The capital of France is Paris.” |
| Common finalized height | 35 |
| Common block hash | `7229E2B79086D75C79D3BC9A6269DDED9B2F7E64C0A715C857BF216063E3230F` |

The public launch uses a different, explicitly published chain/genesis. The numerical model profile is unchanged. A malformed-envelope guard was tightened before the public genesis was frozen. Do not present this disposable chain's block hash as a block on the public chain.

An earlier run failed after a valid training update because the HTTP `broadcast_tx_commit` response exceeded the RPC timeout. The sponsor could not know whether it had committed. The fix submits the exact signed bytes once, then polls the native transaction hash, including after a lost connection. Comet's JSON-RPC hash argument is base64 bytes. The failed run was not counted as a pass. Fault-injection tests now require recovery without a second broadcast or nonce.

## Data integrity and service failures

The [initial immutable S3 snapshot verification](https://github.com/neuroshard-ai/neuroshard/blob/108b4ba3d6c6fb5760ff211b447ee95a67fa9112/docs/experiments/llm-2026-09-10/initial-s3-snapshot.json) confirms all eight uploaded shards and 512 unique documents. Separate bounded legacy recovery preserved eight current objects and reported four unavailable historical hashes; it did not repair or activate the full old corpus. See [data pipeline](DATA_PIPELINE.md).

The first scheduled collector successfully committed cursor 128 and its snapshot, then hit an Arrow streaming shutdown crash. The cursor did not reset. Disabling pre-buffering and explicitly closing the stream appeared to recover once, but another run reproduced the shutdown crash after cursor 384. The final reader downloads the pinned Parquet source with an upstream SHA-256 check and reads local batches synchronously with threads disabled. Cursor order and rendered contents were compared with the previous reader before resuming. Service exit status and snapshot publication are checked separately. The [final collector retest](https://github.com/neuroshard-ai/neuroshard/blob/108b4ba3d6c6fb5760ff211b447ee95a67fa9112/docs/experiments/llm-2026-09-10/collector-service-retest.json) records three consecutive successful exits, advancing the cursor from 384 to 768, and a complete S3 readback. This failure is retained because upload success alone does not establish healthy process operation.

## Public installer, browser settlement and restart evidence

The [public browser cycle](https://github.com/neuroshard-ai/neuroshard/blob/108b4ba3d6c6fb5760ff211b447ee95a67fa9112/docs/experiments/llm-2026-09-10/public-browser-cycle.json) used a fresh wheel installation outside the checkout. `setup` installed a separate runtime and built the pinned consensus engine; `join` downloaded verified artifacts through the public website, connected directly to TCP 26656, replayed history and earned 0.4 NEURO as a stage-1 worker. Browser import of that worker's local key then signed exactly one 32-token request. The balance changed from 400,000 to 367,000 atoms; the provider received 25,600 atoms and the response finalized in public block 480. The browser scenario completed in approximately 8.7 seconds, including settlement checks and capture. Desktop and 390-pixel mobile layout checks passed with no browser runtime errors.

The [validator restart trial](https://github.com/neuroshard-ai/neuroshard/blob/108b4ba3d6c6fb5760ff211b447ee95a67fa9112/docs/experiments/llm-2026-09-10/validator-restart.json) stopped one of the four public validators. The remaining three advanced five blocks, from 279 to 284, in 70.12 seconds, including the configured proposal timeout. The stopped validator was restarted with its existing signing state. This tests one unavailable validator and the secondary transport; it is not a multi-operator or network-partition security test.

The [dependency and conformance review](https://github.com/neuroshard-ai/neuroshard/blob/108b4ba3d6c6fb5760ff211b447ee95a67fa9112/docs/experiments/llm-2026-09-10/release-dependency-review.json) records the patched runtime and the remaining advisory assessments. All four public validators were then [restarted sequentially](https://github.com/neuroshard-ai/neuroshard/blob/108b4ba3d6c6fb5760ff211b447ee95a67fa9112/docs/experiments/llm-2026-09-10/security-rollout.json) using the locked consensus build; they agreed on block 1293 after the rollout. Another [paid response after the upgrade](https://github.com/neuroshard-ai/neuroshard/blob/108b4ba3d6c6fb5760ff211b447ee95a67fa9112/docs/experiments/llm-2026-09-10/post-upgrade-chat.json) finalized in block 1370 using the checkpoint promoted at round 9. The [CLI payment](https://github.com/neuroshard-ai/neuroshard/blob/108b4ba3d6c6fb5760ff211b447ee95a67fa9112/docs/experiments/llm-2026-09-10/public-cli-cycle.json) also returned the expected answer in native block 732.

## Automated checks

The suite exercises native accounting, stake/evidence behavior inherited from v2, malformed envelopes, locked funds, provider authorization, wrong results, queue bounds, replay and expiry; light-client signature compatibility and key-file permissions; worker assignment authorization and durable receipt reuse; uncertain transaction submission; and immutable ingestion/recovery.

The separately maintained publishing workspace's browser tests exercise the minimal join flow, model history, ledger precision, mobile navigation, unavailable services, native verification of browser signatures including Unicode, zero-balance payment prevention, and request-ID recovery after a lost connection. Fixture tests and a live signed payment are separate checks. Published release/deployment records state their actual outcomes rather than treating an active process as a successful settlement. Browser tooling and raw output are outside the current protocol repository.

## Reproduce

Install the matching CPU runtime and package on both machines, retrieve the genesis-pinned model/data using the client, and check conformance:

```bash
ATEN_CPU_CAPABILITY=default MKL_ENABLE_INSTRUCTIONS=SSE4_2 \
  venv_build/bin/python scripts/check_llm_profile.py \
  --genesis networks/neuroshard-llm-testnet-1/genesis.json \
  --model-dir /PATH/TO/VERIFIED/MODEL \
  --data networks/neuroshard-llm-testnet-1/dataset.json
```

For a disposable independent chain, supply an authorized second machine with the matching wheel/runtime/model installed:

```bash
PYTHONPATH=src venv_build/bin/python scripts/experiment_llm_two_hosts.py \
  --ssh USER@HOST --model-dir /LOCAL/MODEL --remote-model-dir /REMOTE/MODEL \
  --engine /LOCAL/cometbft --remote-engine /REMOTE/cometbft \
  --remote-python /REMOTE/venv/bin/python --remote-work /home/USER/neuroshard-experiments
```

The script creates unique local/remote homes and declarations, runs four validators plus a fresh worker, asserts earned balance and a paid response from a promoted checkpoint, compares a common block, stops its processes and retains a JSON report. It needs `ssh`, `rsync`, free local ports 43656/43666/43660/45656/45679 and remote ports 44656/44666/44676/45666/43660, and SSH forwarding. It does not delete existing chain homes or change production services. Inspect the source before running it on another operator's machine.

## Practical limits

Four replaying validators duplicate substantial computation. A frozen backbone permits cached features, so accepted work does not prove fresh energy expenditure. Four short public validation sequences do not establish broad improvement. The deployment starts with one operator, one advertised bootstrap peer and one advertised inference provider, finite issuance and sponsor/data budgets. Sustained adversarial load, independent ownership, more capable evaluation and economical verification remain open gates.
