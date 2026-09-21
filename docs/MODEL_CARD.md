# NeuroShard LLM testnet model card

The release serves **SmolLM2-135M-Instruct plus a NeuroShard residual adapter**. It can produce short instruction responses, but its capability is limited. This card describes the 0.4.0 protocol testnet, not the 1.7B bounded-activation research assistant. This is a working CPU training/payment experiment, not a production assistant.

| Component | Current profile |
|---|---|
| Base model | HuggingFaceTB/SmolLM2-135M-Instruct, revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, Apache-2.0 |
| Frozen parameters | 134,515,008 |
| Trainable parameters | 4,608; rank-4 residual adapter after pretrained final normalization |
| Total parameters | 134,519,616 |
| Update | Float32 SGD, learning rate 0.02, gradient norm clipped to 1, one 64-token sequence |
| Execution | One CPU thread, eager attention, fixed numerical conformance checks |
| Source data | Smol-SmolTalk, revision `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`, Apache-2.0 |
| Initial snapshot | 512 distinct documents, 487 train / 25 validation; eight immutable shards |
| Execution sample | 128 training and four validation sequences of 64 tokens |
| Generation | Greedy, maximum 256 input tokens and 64 new tokens; short responses can end early |
| Serving gate | Promote the adapter only when mean loss on the four fixed validation sequences improves |

All asset hashes, source commitments and vectors are in the [genesis manifest](../networks/neuroshard-llm-testnet-1/genesis.json). Downloads use safetensors and verified files; pretrained weights are distributed separately from the client. See [third-party provenance](../THIRD_PARTY.md) and [the protocol](LLM_PROTOCOL.md) for exact execution semantics.

The initial adapter validation loss was 1.7775393724. In the recorded three-update two-host experiment it decreased to approximately 1.77636 (the hexadecimal exact value is in the report). This is a very small, fixed public evaluation set: it can be overfit and does not establish generalization, safety or meaningful broad model improvement. Training rewards pay correct prescribed computation even when a checkpoint is not promoted.

The standalone base-model probe answered the France/Paris question and produced a coherent blockchain definition, but failed a polite-rewrite instruction. Those examples are retained in [experiment records](LLM_EXPERIMENTS.md). Do not report only the successful examples. A correct native receipt establishes the specified computation, not the truth or usefulness of the generated text.

Every validator replays the neural work. Throughput, latency and total computation grow with this duplication. Larger backbones, full-backbone training, economical verification, stronger evaluation, diverse independent operators and extended adversarial load testing remain work. The network has no private inference: prompts, responses, addresses and transfers are public.

The v0.3 reference profile is a separately initialized 34,976-parameter byte-level model over Tiny Shakespeare. Its results must not be presented as measurements of this LLM profile.
