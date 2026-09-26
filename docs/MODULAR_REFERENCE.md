# Modular reference decision

**A1 contract, September 26, 2026.** No scored generation has been read.
The [assistant checklist](../TODO_ASSISTANT.md) stays open until the committed
run meets every A1 criterion. SmolLM2-135M-Instruct remains the protocol
fixture. It is not the assistant foundation. Historical experiment freezes are
unchanged.

## Choice

The reference recipe is **BAR** (Branch-Adapt-Route), executed from the
published checkpoints rather than retrained:

| Role | Checkpoint | Revision | Parameters | Storage |
| --- | --- | --- | --- | --- |
| Dense baseline | [allenai/BAR-7B](https://huggingface.co/allenai/BAR-7B) | `bee49db02284e88308e85c352e8d9a01b2b48ff5` | 7,298,011,136 | 14,596,063,712 bytes |
| Modular model | [allenai/BAR-5x7B](https://huggingface.co/allenai/BAR-5x7B) | `2e1b57d2e22fbf9c1a08a71298b53abcfdf827ad` | 24,612,753,408 | 49,225,600,720 bytes |

Both are Apache-2.0. BAR-7B is `Olmo2ForCausalLM`. BAR-5x7B is
`FlexOlmoForCausalLM` with 5 experts and `num_experts_per_tok` 5. The shared
chat template hash is
`1fd2eac6d6f73da1f9e30d9b0511f91a70cc8455ffffe48d20f48e77357f5029`.
Tokenizer, config, and generation-config hashes are in
[modular-reference-a1.json](../config/experiments/modular-reference-a1.json).
Upstream training code is the FlexOlmo branch `jacobm-flex-post-train` at
`24fa718332a4f14bce56c40ab77041028e00f8aa`. This repository does not vendor
that training stack.

FlexOlmo is not the reference. Its published composition,
[FlexOlmo-7x7B-1T](https://huggingface.co/allenai/FlexOlmo-7x7B-1T), is a 33B
pretraining mixture of about 132 GB. The current host has no GPU, 15 GiB of
RAM, and about 73 GiB free. BAR also reports that freezing every shared weight
during post-training, which is FlexOlmo's expert rule, leaves tool use near
the baseline (20.3 versus 46.4 when embeddings and the output head are
trained). FlexOlmo remains the architecture family of the released BAR router,
not the learning recipe we will follow.

## Recipe

BAR starts from a fully post-trained dense model. For each new domain it builds
a two-expert model:

- The anchor expert's feed-forward weights are copied from the post-trained
  model and frozen.
- The domain expert's feed-forward weights are copied from the pretrained,
  not post-trained, checkpoint.
- Mid-training updates the domain feed-forward weights and leaves attention,
  embeddings, and the output head frozen.
- Supervised fine-tuning unfreezes embeddings and the output head so new
  function-call tokens can be learned. Math and code mix domain data with
  general instruction data.
- Reinforcement learning with verifiable rewards, used for math and code,
  unfreezes attention as well. Tool use and safety stop after supervised
  fine-tuning.

Merging copies every domain feed-forward network plus the anchor into one
model. Shared tensors that diverged are averaged. There is no sign election
or magnitude trim. After the merge, only the per-layer router is trained, on
a stratified 5% sample of the supervised data, for two epochs at learning rate
1e-4. Expert and shared weights stay frozen during routing. The released
5-expert checkpoint then uses every expert on every token: the router is a
bias-free linear map, softmax is computed in float32, and the top 5 of 5
probabilities are mixed without renormalization (`norm_topk_prob` is false).
Sparse activation does not reduce this checkpoint's feed-forward work.

Reported category scores are context, not our pass thresholds. BAR's own chat
average falls from 48.9 on the dense model to 38.7 on the 5-expert model,
while tool use rises from 25.3 to 45.6. A1 has to measure that tradeoff on
one harness. It does not treat the paper's number as a result we have
reproduced.

Upstream training used one to eight 8×H100 nodes. Mid-training is a 50B-token
cosine decay from learning rate 9e-4. Supervised fine-tuning is two epochs at
1e-4 and sequence length 4,096. Reinforcement learning uses GRPO at learning
rate 6e-7. None of that training is part of this milestone.

## Differences from the released recipe

This run does not train. It loads the two published checkpoints and scores
them with the same prompts, stop tokens, and decoder. The differences below
were fixed before any task output:

- The harness is nine short tasks, three each for conversation, instruction
  following, and tool use. It is not AlpacaEval, IFEval, or BFCL.
- Execution is one CPU thread, eager attention, and bfloat16. Weights stream
  one decoder layer at a time from the safetensors files. No GPU, no new
  instance, and no quantized copy.
- KV cache is on for both models. BAR-7B's config says `use_cache: false`;
  that flag is not followed, so the two checkpoints share one decode policy.
- Greedy decoding stops on generation-config ids 100265 (`<|im_end|>`) and
  100257 (`<|endoftext|>`). A reply that hits 32 new tokens without either id
  is unterminated and fails.
- The installed transformers release is 4.57.3. The checkpoint metadata names
  4.52.4 and 4.57.6.
- BAR-7B uses RoPE theta 8,000,000. BAR-5x7B uses 500,000. Both values stay
  as released. Our prompts are short, so this does not remove the difference.
- CPU eager bfloat16 is not required to match AllenAI's GPU generations.
  Matching their benchmark tables is not a pass condition. A second run with
  the same code, plan hash, and weight hashes must reproduce the decoded text;
  a mismatch invalidates the record.

## Scoring contract

Exact tasks pass only when the finished reply, after stripping the ends and
normalizing newlines, is one of the declared strings. A sentence that contains
the right number fails. Tool tasks pass only when the single
`<function_calls>` block contains exactly the expected keyword call. Prose
outside that block is ignored. Positional arguments, a second call, and a
missing block fail. The prompts do not contain the expected call text.

The baseline is usable only when every category has at least one successful
task. An empty category cannot become a retention claim. The modular
checkpoint must finish the same nine tasks. A1 does not require it to beat
the baseline; the comparison is the measurement. Individual misses stay
visible. No task is regenerated after its reply file exists.

Resource limits on this host: 40 minutes per reply, 8 hours per checkpoint
evaluation, 3 hours per download, and 12 GiB resident memory. Crossing a
limit stops that checkpoint. Weights stay in `.neuroshard/modular-reference/`
and are not committed.

## Route toward sharded execution

These figures come from the released shapes: hidden size 4,096, intermediate
size 11,008, 32 layers, vocabulary 100,278. They match the published parameter
counts exactly. The existing 48M-parameter CPU shard cap cannot hold one
5-expert layer (743,477,248 parameters). The route adapts the layer pipeline
already used for the 1.7B shards, with every expert of a layer kept on the
same worker.

| Placement | Bytes | Hosts |
| --- | --- | --- |
| One 5-expert layer, bfloat16 weights | 1,486,954,496 | inference slice |
| Eight layers plus embeddings and the output head | 13,538,598,912 | largest of 4 inference workers |
| KV cache at 2,048 tokens, all layers | 1,073,741,824 | split across those workers |
| One layer, FP32 weights, gradients, and both Adam moments | 11,895,635,968 | one training worker per layer |
| Embedding and output head, same Adam state | 13,143,703,552 | a separate training worker |

Four 16 GiB hosts can hold inference without any host storing the full
backbone. The largest inference worker still fits under 16 GiB after its
share of the 2,048-token cache. Training state needs 32 layer workers plus
one embedding worker at the same 16 GiB budget, or fewer larger GPUs. A
generated token crossing a four-stage pipeline sends 24,576 bytes of hidden
state. That is small next to reading a layer. Active compute still touches
all five feed-forward networks, so adding experts does not by itself keep
latency constant. Reducing the top-k is a later serving decision; this
reproduction keeps the released value of 5.

## Run

The driver is [run_modular_reference.py](../scripts/run_modular_reference.py).
It refuses a config or template hash that differs from the plan before
decoding. Downloads and replies stay outside git until a result file is
copied into `config/experiments/`.

```bash
python scripts/run_modular_reference.py fetch --which baseline
python scripts/run_modular_reference.py evaluate --which baseline
python scripts/run_modular_reference.py fetch --which modular
python scripts/run_modular_reference.py evaluate --which modular
```

A1 closes only when both checkpoints finish, the baseline gate passes, the
placement estimate still fits, and the deviations above still describe the
run. A failed gate, a missing artifact, or an exhausted budget remains open.
