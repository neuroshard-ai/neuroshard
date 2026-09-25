# Append-only execution

This is the CPU run authorized after [the serving rule](APPEND_ONLY_GROWTH.md).
The failed block-expert measurement stays closed. Questions are fresh. The
parent weights do not train. The added two blocks train on the new arithmetic
skill. At serving time the parent answer is kept whenever the parent was
correct, including every protected retention answer. The added shard is used
only where the parent missed and the shard is correct.

Training starts only when the parent has at least one correct retention answer.
That answer, and every other correct retention answer, is protected. Zero
protected answers stops the run before training.

The served system passes when it keeps every protected answer and gains at
least four correct new answers the parent missed. The in-place control is
trained and recorded. It is not the pass gate. A pass authorizes review of a
settlement freeze. It does not issue NEURO, upgrade public 0.4.0, or complete
item 4.

```bash
PYTHONPATH=src venv_build/bin/python scripts/run_append_only_execution.py \
  --run --seed .neuroshard/seed-smollm2-135m --home .neuroshard/append-only-NEW
```
