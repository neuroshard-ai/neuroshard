# Granite shard training (A4, second execution)

Declared on September 29, 2026, after the [first shard execution](GRANITE_SHARD_EXECUTION_RESULTS.md)
passed and before any sharded training on the real model. The
[contract](../config/experiments/granite-shard-training.json) pins every rule below.

## Question

Can the assistant's learned module be trained across machines, with no machine
holding the complete backbone, and come out exactly as the single-host trainer
would make it?

## Method

The owners and their fetch are the same as in the first execution. Three owner
hosts split the layers at `[0, 12, 26, 40]`, and each fetches only its own
verified byte ranges. Owner 2 holds layers 26–39, so it holds every tensor of the
addition arm: rank-16 LoRA on the q/v projections of layers 32–39, 1,048,576
FP32 parameters. It also holds their AdamW state and checkpoints. Owners
0 and 1 never compute a backward pass.

- **Protocol.** Owner 0 runs the declared schedule. It embeds each sequence and
  receives the full final hidden state. It computes the trainer's loss, or its
  preference margin, through the norm and tied head, then returns each boundary
  gradient as exact bf16 bytes. Owner 2 runs one backward pass over the
  microbatch's graphs, so shared tensors accumulate exactly as they would on one
  host, then clips and steps on command.
- **Control.** A fourth reference host downloads the complete pinned checkpoint.
  It runs the unchanged `assistant_experience_train.train` on the same sequences
  and settings.
- **Settings.** The round-1 architecture and optimizer with the round-4 objective
  and learning rates, shortened to 6 steps with 2 preference pairs per step and a
  fresh seed.
- **Data.** 12 round-1 verified trajectories, 4 parent replay items and 4 round-4
  verified repair pairs of at most 2,048 tokens, encoded with the canonical
  checked tokenizer and committed with their source digests.
- **Outage.** Owner 2 exits when told to take step 3. The group is relaunched from
  its step-3 checkpoint and owner 0's saved preference references, and continues
  the same schedule.

Before this run, tests on a small random Granite checkpoint showed:

- Both arms trained across three owner processes reproduce the single-host
  trainer bit for bit. That covers losses, preference margins and final tensors.
- A resumed outage also reproduces it.
- Both results hold on transformers 4.57 and 5.5.4.

## Checks

The execution passes only if all seven hold:

1. The reference and all three owners complete the uninterrupted schedule.
2. Every owner fetched only its verified byte ranges.
3. Every final LoRA tensor has the same SHA-256 as the reference's.
4. Every step loss and preference margin equals the reference's exactly.
5. The arm's owner is lost mid-schedule, and owner 0 reports the loss.
6. After relaunch, the final tensors equal the reference's and the remaining step
   losses match.
7. Every owner's peak RSS stays below the reference host's.

## Limits

This run does not claim:

- **Quality.** The six-step arm is neither evaluated nor served.
- **Other A4 evidence.** Throughput and serving the learned module on shards are
  later executions.
- **Independent operation.** Owners run by independent operators belong to A5.

The run gives no checklist credit and is not admission evidence.

## Second attempt

Declared on September 29, 2026. The [first attempt](GRANITE_SHARD_TRAINING_RESULTS.md)
passed six of seven checks and failed recovery, and it stays failed. The
[ring determinism diagnostic](GRANITE_SHARD_DETERMINISM.md) traced that failure
to one cause: a fresh process's first forward pass can round differently. With
the declared warm-up, twelve fresh launches were identical.

In this attempt, each owner runs the warm-up after joining the ring and before
any work:

- discarded 1,024-token and 1-token passes;
- one discarded backward pass through the arm on its owner, with its gradients
  cleared.

The reference host warms up its complete model with the same discarded passes.
Owners, arm, sequences, settings, the outage at step 3 and all seven checks are
unchanged. Tests on small checkpoints show the warm-up leaves every trained
tensor bit-identical.

Resources: four r7i.4xlarge allocations, each with its own expiry and a $8
allowance (at most $32), under the
[resource contract](../config/experiments/granite-shard-training-resources.json).
One attempt.
