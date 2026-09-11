# Compact optimizer disputes — experimental extension

NeuroShard's full-model profile can opt into `update_witnesses: "neuroshard-sgd-chunks-v1"` in its initial model metadata. This changes the model commitment and consensus source; it belongs on a fresh experimental network, not an edited live genesis. The public 0.4.0 network is unchanged.

The goal is to reduce the consensus cost of refuting an inconsistent SGD update. This does **not** replace complete training audits, establish an audit market, or prove that gradients are useful or correct. Model parameter bytes, loss arithmetic, clipping and the identity of payable numerical work remain unchanged by the added commitments.

## Statements and trust boundary

Each signed training stage now asserts, for every owned parameter tensor:

1. The `before` tensor matches that parameter in the parent component's committed safetensors file.
2. The `gradient` tensor is the result of the prescribed forward/backward computation.
3. The `after` tensor matches the candidate component's committed safetensors file.
4. The three tensors satisfy the declared float32 SGD operation, with the globally committed clipping scale and learning rate.

The full referee checks all four by reconstructing the stage and recomputing the tensor commitments. The new witness can disprove statement 4 from three authenticated tensor chunks. A contradiction refutes the conjunction of the publisher's statements even without first proving the other three. Conversely, agreement on one chunk establishes none of statements 1–3 and says nothing about other chunks.

These Merkle roots are **additional optimistic assertions**. They are not cryptographic proofs that a slice belongs to the existing flat SHA-256 safetensors commitment. A producer can invent a consistent gradient and update: the compact check then passes while the full-stage referee rejects the claim. That adversarial case is an explicit regression test. Ordinary observers must still check the complete graph and all declared index/file/gradient relationships.

## Tensor commitments

Parameters use row-major, little-endian float32 bytes and chunks of 1,024 elements. Every leaf hash binds its domain, chunk position, byte length and bytes. Internal nodes have a separate domain. The tree pads to a power of two with a fixed empty-leaf hash. The signed descriptor binds the root and exact integer shape; tensor names and ownership are derived from the model and must cover the stage exactly once in a fixed order.

Workers compute roots around the existing in-place SGD update. They do not write every chunk as a separate artifact. A witness publisher reconstructs the gradient, reads the claimed candidate component, and produces before/gradient/after Merkle openings for one chunk. Witness preparation can therefore cost a full stage replay. The optimization targets adjudication and evidence transport, not the cost of discovering every possible error.

The numerical operation is the existing pinned CPU `torch.add_` with `alpha = -learning_rate * scale`. No epsilon comparison or randomized spot-check changes acceptance. Chunked execution must agree byte-for-byte with the whole-tensor operation on the supported numerical profile; cross-hardware measurements are required before broadening that profile. Nonfinite committed values constitute an invalid update assertion.

## Native transaction

`refute_update` carries `claim_id`, `stage`, `tensor_index` and a witness containing `chunk`, `before`, `gradient`, and `after` openings. The witness is at most 32 KiB. The validator verifies the paths and at most 1,024 elementwise updates. It reads no off-chain objects and does not invoke the neural stage referee.

The transaction requires a live training claim with no active dispute and an unexpired challenge window. It charges the normal fee and challenger bond. Invalid paths or malformed inputs reject the transaction without changing state. Valid openings of a consistent update burn the challenger bond and leave the original deadline unchanged. An inconsistent update rejects the claim, returns the challenger bond, pays half the publisher's bond to the challenger and burns the remainder, using the existing fraud accounting. No training reward is issued for that rejected claim.

`update_check_count` counts these bounded checks separately from full referee calls. Neither count establishes independent audit coverage. Unavailable evidence, false tensor/file relationships, incorrect gradients, clipping/norm fraud, attention errors and other numerical failures retain the complete stage dispute path. The ordinary challenge and absolute expiry limits remain in force.

## Reproduction and measurement

Use the pinned numerical dependencies. The benchmark accepts a real imported model and a response-labeled batch. Each invocation needs a new private home:

```bash
python scripts/experiment_update_witness.py \
  --home ./update-check --objects ./objects --model-root MODEL_ROOT \
  --batch ./batch.json --workers-config ./workers.json
```

It compares baseline and instrumented training in alternating order, reports the first cold pair separately, retrieves all resulting components, and checks unchanged parameter bytes, loss and paid-work identity. It fully replays every stage of both final records, corrupts one embedding element, compares the full oracle with the compact witness, and records witness-construction cost separately from adjudication. Repeated identical numerical tasks are timing samples, not additional payable training. Object traffic excludes HTTP, RPC-envelope and SSH overhead and is labeled accordingly.

The portable cases can be checked on another supported CPU without transferring model weights:

```bash
python scripts/experiment_update_witness.py \
  --verify ./update-check/result.json --expected-sha256 RESULT_SHA256
```

Then run a separate native test. `--compare-full` first uploads the same forged stage to the original referee, providing an actual native transport/finality baseline:

```bash
python scripts/experiment_update_native.py \
  --home ./update-native --record ./update-check/result.json \
  --objects ./objects --engine /path/to/cometbft --compare-full
```

The native test rejects the forged claim through both paths, exercises a false compact accusation, pays one correct training task, and checks common application hashes. Its four validators share one host and operator. It stops all its node processes on exit; it does not upgrade or reset the public network.

## Research relationship and remaining work

[Verde](https://arxiv.org/abs/2502.19405) motivates narrowing disagreements within an ML computation graph and using reproducible arithmetic. This extension is much narrower: it localizes only an asserted SGD relation and preserves the existing full-stage fallback. It does not implement Verde's complete arbitration protocol or inherit a theorem covering all neural operators.

Reducing adjudication cost does not fund honest observation. The [verifier's dilemma](https://arxiv.org/abs/2312.01549) and [Proof of Diligence](https://arxiv.org/abs/2402.07241) are relevant to that separate problem. Paying for a signature is insufficient evidence that a participant performed an independent check. Any proposed audit subsidy must account for copying, collusion, ownership concentration and the actual cost of complete coverage.

The experiment must report commitment overhead, full observer work, cold/warm transport, witness construction, native evidence bytes and settlement latency together. Until those measurements and broader operator coverage are established, this extension is a bounded refutation primitive, not a demonstrated economical verification system.
