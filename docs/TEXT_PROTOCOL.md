# Versioned text protocol for model evolution

This is the implemented text profile for the experimental full-model pipeline. It binds the meaning of token IDs to a model and its data, and makes response coverage explicit. The public 0.4.0 adapter network continues to use its released execution profile.

## Model and tokenizer are one version

A tokenizer maps text to embedding-row indices. Matching vocabulary sizes alone does not establish that two tokenizers give those indices the same meaning. Chat formatting also affects the token stream; adding special tokens a second time after rendering can change the intended input. [Hugging Face chat-template documentation](https://huggingface.co/docs/transformers/chat_templating) explains the formatting and special-token distinction.

`TextCodec` stores two immutable SHA-256-addressed objects: the serialized fast-tokenizer backend, and a `neuroshard-text-codec-v1` manifest. The backend includes vocabulary/merges, normalization, pre-tokenization, decoding and added-token definitions. The manifest binds:

| Field | Meaning |
| --- | --- |
| `backend` | Digest of the canonical tokenizer backend JSON |
| `chat_template` | Exact supported template text |
| `vocabulary` | Number of contiguous IDs and corresponding model embedding rows |
| `special_tokens`, `special_ids` | BOS/EOS/padding/unknown definitions and all reserved IDs |
| `runtime` | Exact Transformers, Tokenizers and Jinja2 versions |
| `renderer` | Bounded Jinja rendering with trimmed blocks and stripped block indentation |
| `decode_cleanup` | `false`; no cleanup of decoded whitespace |
| `input_normalization` | `none`; no extra normalization outside the committed backend |

The current seed retains its 49,152-token vocabulary. The [experimental requirements](evolution-requirements.txt) pin Transformers 4.57.3, Tokenizers 0.22.1 and Jinja2 3.1.6. Loading a codec checks its runtime, reconstructs the backend, and verifies canonical reserialization. Changing vocabulary, special tokens or template produces a different root. Transient backend padding and truncation are disabled; bounded windowing is explicit.

A model manifest includes `tokenizer_root`. Binding a verified legacy seed adds that metadata and a parent link without changing tensor bytes. Subsequent training and identity-initialized depth growth must retain the root. Prepared windows carry it too. Batch construction rejects mixed roots, and the text generation helper requires the model and codec to agree. Corpus settings refuse an in-place tokenizer/objective change.

These commitments identify the chosen text semantics. They do not certify that a newly supplied tokenizer is useful, that its corpus is original, or that a coordinator selected good data. The importer verifies the supported seed files; arbitrary embedding/tokenizer migrations are outside this path.

## Bounded chat formatting

Inputs contain an optional initial system message followed by alternating user and assistant messages. Each message has only `role` and UTF-8 `content`. Generation requires a final user message. Literal reserved control tokens in content are rejected. This protects message framing; it is not a semantic prompt-injection defense.

The supported Jinja subset permits bounded message loops, conditionals, explicit role/content fields, simple comparisons and bounded concatenation. Clock access, calls, filters, recursive/nested loops, object introspection and rendering whole implementation objects are rejected. Rendering streams through a size limit, using the [Jinja streaming API](https://jinja.palletsprojects.com/en/stable/api/#jinja2.Template.generate). This deliberately does not support every Hugging Face chat template.

| Bound | Value |
| --- | --- |
| Tokenizer backend | 16 MiB |
| Template | 64 KiB, at most 32 concatenations |
| Conversation | 128 messages, 256 KiB content |
| Rendered conversation | 512 KiB, 32,768 encoded tokens |
| Numerical training window | At most 256 token positions |
| Default window | 64 preceding-context positions plus 64 target positions |
| Default document budget | Four windows; configurable from 1 to 64 |
| Generation input | 192 tokens; overlong prompts are rejected explicitly |

Chat formatting occurs once. Encoding then uses `add_special_tokens=False`; there is no automatic padding or truncation in the tokenizer. The seed's original and reconstructed chat streams are compared in the real-model conformance script.

## Response targets and coverage

For every nonempty assistant turn, the formatter must yield a stable generation prefix of at least two tokens and terminate the response with the committed EOS. Targets include the real answer and its EOS; the structural newline after EOS is excluded.

Responses are split into bounded chunks. Each chunk uses preceding conversation/answer tokens as context. Prompt and right-padding labels are `-100`, so only real response targets contribute to causal loss. No EOS is inserted at a chunk boundary. A one-word answer remains a valid example.

When the document budget is limited, the first chunk of each assistant turn is considered before later chunks. Each retained target appears once within that document's prepared windows. The corpus records total, scored and omitted target counts, target offsets, actual response endings and context truncation. Sampling training windows in later epochs can intentionally replay a target.

Training may accept a document with an explicit omitted-target count. Held-out ingestion rejects it as `incomplete_evaluation` if any response targets would be omitted. This trades evaluation coverage for a bounded workload and can bias the admitted population toward shorter conversations; report the rejection counts. Complete target coverage still uses truncated preceding context and teacher forcing. It is not whole-conversation generation quality.

The maintained `TextCorpus` uses this policy. Historical `Corpus` defaults and older experiment scripts preserve their old roots/objectives for reproduction. Moving to this profile requires new corpus and epoch homes.

## Document-level evaluation

Evaluation reserves entire documents, consumes all their windows, and never treats another chunk of a reserved document as a fresh evaluation example. Within each document, window mean losses are weighted by their actual unmasked causal targets. The quality gate receives one paired mean loss per document, rather than counting chunks as independent observations. Training/evaluation deduplication remains the corpus's existing normalized-text and SimHash heuristic; semantic contamination is still possible.

Candidate and parent must use the same tokenizer root for this token-loss comparison. Per-token perplexity depends on tokenization, so it is not a valid direct quality comparison across vocabulary changes. [Hugging Face's perplexity documentation](https://huggingface.co/docs/transformers/perplexity) discusses that dependency and context-window limitations.

This remains a local research gate. Public held-out examples can be learned or leaked; document reservation is not a proof of benchmark secrecy. Native settlement verifies committed numerical work and preserves the model's tokenizer identity. It does not independently prove that a batch was derived from good raw text or authorize public serving promotion.

## Vocabulary evolution

Continuous learning does not require continuously replacing the tokenizer. New concepts and words can be learned as sequences of existing tokens. Retaining the seed vocabulary lets existing weights, replay data and inference agree while the network trains more parameters or adds blocks.

The current implementation rejects an implicit vocabulary replacement. A future explicit migration needs a versioned embedding/output-head conversion, retokenization from retained raw data, compatible inference clients, and a separate quality decision on the same raw text/tasks. Token efficiency should be measured by language and domain, alongside throughput and task quality; fewer tokens alone is not a quality result. This migration is not implemented or activated by `grow`.

## Measured conformance, September 11, 2026

The final text-profile run used three worker processes across two CPU machines controlled by one operator. Both machines independently produced identical full codec manifests and token IDs for eight probes: English, Hebrew, Arabic, Chinese, emoji/accents, code, whitespace and mixed-direction text. Every probe round-tripped exactly. This tests encoding consistency, not multilingual answer quality.

| Check | Observed result |
| --- | --- |
| Seed / grown parameter count | 134,515,008 / 148,675,392 |
| Two-answer fixture coverage | 121 targets in three windows; the legacy first-response path retained 64 |
| Actual training batch | First two windows, 101 response targets, one full-model step |
| Training step wall time | 18.05 seconds |
| Independent stage replay | All three passed; 7.32, 6.41 and 5.76 seconds |
| Post-training generation | Eight output tokens decoded to “The capital of France is Paris.” |
| Codec through training/growth | Preserved; mismatched-codec transitions rejected in regression tests |

The codec root is `b797b60c9203884ec7d519b50dfd16e4375b239d249ac7377aa3f6c646b56e85`. The short generated answer is a functionality probe of the pretrained seed after one update; it does not establish improvement. The manually written training conversation is a conformance fixture, not a quality benchmark. No new candidate has qualified for public serving through this experiment.

The first native run stopped during dispute resolution when a long replay outlasted the synchronous RPC response. This was an unknown transaction outcome, not evidence of a rejected computation. The experiment driver now submits one signed envelope and checks its exact hash for a committed success or rejection, with a bounded confirmation deadline and a longer isolated RPC budget. Regression tests cover lost acknowledgments, final rejection and an unresolved deadline without resigning or resubmitting a payment.

## Reproduce

From a checkout, install the experimental profile and obtain the hash-verified seed:

```bash
python -m pip install -r docs/evolution-requirements.txt '.[collector,dev]'
python -m neuroshard.evolution download-seed --model-dir ./seed
python scripts/experiment_text_profile.py --home ./text-only --model-dir ./seed --codec-only
python scripts/experiment_text_profile.py --home ./text-model --model-dir ./seed
```

Every run requires a fresh home. `--codec-only` checks text without loading weights. The full run imports the actual seed, prepares all response windows, trains the first two windows in one batch, replays each stage, generates a short answer and verifies codec preservation during growth. The remaining window is checked for coverage but is not claimed as trained. The default workers share one process. Supply `--workers-config ./workers.json` with the `workers` array from the [epoch example](../config/evolution-epoch.example.json) to use separately running workers; [transport setup](EVOLUTION_PROTOCOL.md#run-a-complete-research-epoch) explains private worker tokens and forwarding.

Inspect the model-bound profile using the `model_root` from the result:

```bash
python -m neuroshard.evolution inspect-tokenizer --objects ./text-model/objects --model-root MODEL_ROOT
```

The resulting training record can also drive the existing native fraud/growth lifecycle on an isolated four-validator chain:

```bash
python scripts/experiment_evolution_native.py --home ./text-native \
  --engine /path/to/cometbft --record ./text-model/result.json --objects ./text-model/objects
```

The fixture conversations and language probes test conformance, not learning quality. Every text experiment records `quality_improvement_claimed: false`. Private keys, object stores and generated result files remain in the chosen experiment homes. These commands do not update the public testnet, promote an evolved serving model or change public balances.
