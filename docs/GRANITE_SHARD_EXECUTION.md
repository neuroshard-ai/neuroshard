# Granite shard execution (A4, first execution)

Declared on September 28, 2026, before any sharded Granite run. The
[contract](../config/experiments/granite-shard-execution.json) pins every rule below.

## Question

Can the pinned Granite 4.1 3B assistant run across machines where no machine
downloads, stores or loads the complete backbone, and still produce exactly what
the canonical single-host runtime produces?

## Method

Three owner hosts split the 40 layers at `[0, 12, 26, 40]`. Owner 0 also holds the
tied embedding and final norm, so it embeds each prompt and scores the last
hidden state; owners 1 and 2 hold only their layers. Each owner keeps its own
key/value cache for its layers.

- **Fetch.** A committed [tensor inventory](../config/experiments/granite-tensor-inventory.json)
  records the byte range and SHA-256 of all 362 tensors, derived from the two
  checkpoint files after they matched their pinned digests. Each owner requests
  only its own byte ranges from the pinned Hugging Face revision and checks every
  tensor before writing its shard.
- **Runtime.** Every owner uses the canonical parent runtime: identical packages,
  Python, CPU instructions and numerical environment, eight threads, bf16 weights
  and eager attention. The partition uses the same decoder modules, masks, rotary
  tables, multipliers and last-position head as the complete model.
- **Transport.** gloo over TCP between the owners' private addresses. Hidden states
  travel as exact bf16 bytes; the last owner returns only the last position.
- **Workload.** Every recorded generation of the
  [canonical re-baseline](ASSISTANT_WORKFLOW_CANONICAL_RESULTS.md): 206 workflow
  generations and 24 anchors. Owner 0 prefills each recorded input and decodes
  greedily with the canonical cap (192) and end token. No new evaluation data is
  opened.
- **Outage.** Owner 2 exits at its 40th forward step of the longest recorded
  generation. The group is relaunched; owner 0 replays the committed tokens step
  by step, which rebuilds every cache exactly, and continues.

Before this run, tests on a small random Granite checkpoint showed:

- Partitioned logits are bit-identical to the complete model, and cached greedy
  generation is token-identical to `generate`. This holds on transformers 4.57
  and on the pinned 5.5.4.
- Boundary gradients match the complete model.
- Three owner processes reproduce `generate`, and a killed owner resumes to the
  same tokens.

## Checks

The execution passes only if all six hold:

1. Every owner completes the agreement phase, and owner 0 returns all 230 generations.
2. Every generation is token-identical to its canonical recording.
3. Each owner fetched exactly its owned tensors and their bytes, and less than the
   whole checkpoint.
4. Every owner's serving peak RSS stays under its 8 GiB kernel limit and below the
   canonical single-host peak (11.3 GB).
5. Owner 2 is lost mid-generation, and some but not all tokens are committed.
6. The relaunched group finishes that generation with the canonical tokens.

Fetch time and requests, traffic per owner, and generation time against the
single host are measured and not gated.

## Limits

This run does not claim:

- **Faster single requests.** The layer ranges run in sequence.
- **Other A4 evidence.** Throughput with concurrent requests, backward and
  training across owners, and a learned module served on shards are later
  executions.
- **Independent operation.** Owners run by independent operators belong to A5.

The run gives no checklist credit and is not admission evidence.

Resources: three r7i.4xlarge owner allocations, each with its own expiry and a
$8 allowance (at most $24), under the
[owner resource contract](../config/experiments/granite-shard-owner-resources.json).
One attempt.
