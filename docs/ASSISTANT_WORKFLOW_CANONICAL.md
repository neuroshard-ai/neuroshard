# Canonical-tokenizer workspace re-baseline

September 27, 2026. New execution; the [failed baseline](ASSISTANT_WORKFLOW_BASELINE_RESULTS.md)
remains failed and unchanged. This repeats the same parent measurement after
finding that every Granite execution encoded its prompts with the wrong tokenizer.

## The runtime defect

All Granite executions pin `transformers` 5.5.4. Its `AutoTokenizer` resolves the
checkpoint's `tokenizer_class` (`GPT2Tokenizer`) to a constructor that replaces
the serialized `Split(\p{N}{1,3}) + ByteLevel` pre-tokenizer with GPT-2's regex
([upstream issue](https://github.com/huggingface/transformers/issues/45812)).
Identifiers and digit runs then split differently from the tokenization Granite
was trained with: `save_draft` becomes `save`, `_`, `draft` instead of `save`,
`_draft`; `2027` becomes `20`, `27` instead of `202`, `7`.

All 193 recorded workflow prompts reproduce exactly under the pinned 5.5.4
`AutoTokenizer`; none reproduce under `tokenizer.json`. Eleven of the 24 original
anchor prompts are also affected. The model still generates canonical pieces, but
must copy strings from a context encoded the other way. That matches the observed
errors: 39 of 51 workflow tool errors call a misspelled tool (`save_raft`,
`read_annotated`), and most others corrupt dates, document IDs or project numbers
(`20-27-04-26`, `Cedar 64` for `Cedar 6460`). The reference, adapter audit,
context, evidence and answerability executions used the same runtime. Their
published outcomes are unchanged; they were measured under this defect.

## Frozen method

The [plan](../config/experiments/assistant-workflow-canonical.json) keeps the
model, policy, tools, limits, 24 development episodes, qualification thresholds
and replay cases of the first baseline. Two things change:

- The [tokenizer loader](../src/neuroshard/evolution/granite_tokenizer.py) reads
  the pinned `tokenizer.json` directly. Before any model call, its pipeline and a
  fixture of identifiers, dates and IDs must equal `tokenizers.Tokenizer.from_file`.
  Every runtime encode is compared again; one difference stops the worker. The
  encode count must match the executed requests, generations and anchors.
- The 18 prior anchor successes were measured under the defect. All 24 anchors
  run first under the original anchor gate; a lost prior success is published but
  does not stop the workflows. Canonical successes become the protected set.

The worker also records whether the old `AutoTokenizer` path conforms in the same
runtime; under 5.5.4 it should not. If the parent exceeds 20/24, the declared +4
development gain has no room; harder workflows are then required before training.

## Execution

One disposable r7i.4xlarge CPU host, no GPU, two-hour expiry and a $6 planning
allowance under the [resource contract](../config/experiments/assistant-workflow-canonical-resources.json).
Exact-commit CI must pass first. Primary worker 70 minutes; conditional two-case
fresh-process replay 10 minutes. No automatic retry. Evidence is copied and the
host, volume and security group retired on completion or failure.

```bash
PYTHONPATH=src venv_build/bin/python scripts/modular_reference_cloud.py run \
  --profile assistant-workflow-canonical \
  --home .neuroshard/assistant-workflow-canonical-20260927
```

The [execution inventory](../config/experiments/assistant-workflow-canonical-execution.json)
pins contracts, sources and runtime. This measures the parent for the
[verified-experience comparison](ASSISTANT_EXPERIENCE_LEARNING.md); it trains
nothing, opens no confirmation data and earns no checklist credit.
