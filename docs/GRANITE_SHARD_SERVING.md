# Granite shard serving (A4, third execution)

Declared on September 29, 2026, before any sharded serving of the learned
assistant. The [contract](../config/experiments/granite-shard-serving.json) pins
every rule below.

## Question

Can the complete learned assistant be served across machines, with no machine
holding the backbone, and answer exactly as it does on one host? "Complete"
means the pinned parent plus the round-4 addition arm, the refitted gate and
prefix-cache serving. This is the deliverable named at the start of this work:
one useful capability, improved through training, served across machines, with
earlier successful behavior preserved.

## Method

Three owner hosts split the layers at `[0, 12, 26, 40]`. Each fetches only its
own verified byte ranges and the pinned small files.

- **Owner 0** holds the embedding, final norm, layers 0–11, the checked
  tokenizer, the gate and the workspace tools. It renders every request, runs
  the tools and decodes greedily.
- **Owner 1** holds layers 12–25.
- **Owner 2** holds layers 26–39 and the addition arm (LoRA on layers 32–39).
  The arm is uploaded with the digest pinned by development and is switched on
  or off per episode; switching it off restores the parent bit for bit.

For each development episode:

1. Owner 0 computes the parent's final-layer feature at the first request, with
   the arm off, in a cache-free pass.
2. The gate selects the parent or the arm for the whole episode.
3. Every owner keeps one episode cache and crops it to the longest token prefix
   shared with each new request, as the single-host responder does.

The target is the 24 served episodes of the pinned
[round-4 development result](ASSISTANT_EXPERIENCE_DEVELOPMENT_ROUND4_RESULTS.md).
For each episode the comparison covers the selection, every generation's
prompt, input and output tokens, text, termination, reused prefix length, and
the score. No new evaluation data is opened.

**Diagnostic (not gated).** The [shard training execution](GRANITE_SHARD_TRAINING_RESULTS.md)
showed that a fresh process can compute its first forward pass differently. On
each owner, six fresh processes push one fixed input through its layers three
times and record output digests.

Before this run, tests on a small Granite-shaped assistant showed that serving
on three owner processes reproduces the development evaluator's served system,
in both directions:

- Parent-selected and arm-selected episodes both match.
- Multi-turn conversations whose caches crop to the shared prefix match, with
  and without the arm.
- These hold on transformers 4.57 and 5.5.4.

## Checks

The execution passes only if all six hold:

1. All three owners finish, and owner 0 serves all 24 episodes.
2. Owner 0's tokenizer conforms to the pinned pipeline and fixture digests.
3. Every owner fetched only its verified byte ranges and the pinned small files.
4. Only owner 2 holds the arm, with the pinned digest.
5. Every episode's selection, generations and score equal the single-host result.
6. Every owner's serving peak RSS stays under its 8 GiB kernel limit.

The run also measures, without gating:

- p95 latency, including selection, against the single host's 92.7 s.
- Traffic per owner.
- The determinism diagnostic.

## Limits

This makes no new quality claim, because development data is already opened.
The quality claim for A2 comes from the fresh confirmation. Owners run by
independent operators belong to A5. The run gives no checklist credit and is not
admission evidence.

Resources: three r7i.4xlarge allocations, each with its own expiry and a $8
allowance (at most $24), under the
[resource contract](../config/experiments/granite-shard-serving-resources.json).
One attempt.
