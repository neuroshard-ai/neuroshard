# Native reference model

The first public release trains a **34,976-parameter byte-level language model** to test prescribed computation, verification, and native settlement. It is not a general-purpose assistant or useful production LLM.

| Property | Current profile |
|---|---|
| Architecture | NeuroLLM, 2 decoder layers, hidden size 32, 4 attention heads, 2 KV heads, intermediate size 64 |
| Vocabulary | 256 byte values |
| Training | SGD without momentum, learning rate 0.1; 4 sequences of 32 bytes per update |
| Arithmetic | Linux x86_64, PyTorch 2.9.1+cpu, float32, single thread, fixed ATen/MKL dispatch, MKLDNN off |
| Pipeline | Two CPU stages; validators replay the full update |
| Data | Tiny Shakespeare; [provenance](../THIRD_PARTY.md) |
| Corpus SHA-256 | `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed` |

Genesis binds the corpus, configuration, conformance vectors, and consensus source. Initialization uses the fixed seed in `demo/work.py`. Training batches use the first 95% of bytes and a round-specific seed; the final 5% is reserved for the reference evaluator. This is a small reproducibility exercise, not a broad benchmark.

## Reading progress

The model page shows the latest accepted round and root. History is reconstructed from successful training submissions in native blocks and execution results. Plotted losses are training minibatch cross-entropies **before** each update. Batches change between steps. Decreasing losses do not establish held-out quality, generalization, economic value, or poisoning resistance.

Historical experiment files may describe different disposable chains. Their validation losses are not live measurements of this chain. Held-out evaluation is available through local research tooling; the public API avoids triggering neural evaluation for every visitor.

## Checkpoints

`GET /api/model/checkpoint.json` exports accepted model state as bounded JSON. Each tensor contains its shape and base64 little-endian float32 data, without pickle. The response includes genesis identity, state height, round, and root. `work.digest(checkpoint["weights"])` must equal `checkpoint["model_root"]`.

Compare the root and chain identity with your own node. A matching content hash detects different bytes; a website statement is not an inclusion proof or independent consensus verification. A download may be newer than a previously rendered page. Joining still replays from genesis; checkpoint download is not an implemented state-sync feature.

Broader evaluation, robust data selection, portable arithmetic, useful scale, economical verification of backward/optimizer operations, independent operators, and sustained reliability remain open. See the [roadmap](RESEARCH_ROADMAP.md) and [experiments](PROTOCOL_EXPERIMENTS.md).
