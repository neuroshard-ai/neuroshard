# Block expert measurement

This is a separate CPU contract after the [block-expert baseline stop](BLOCK_EXPERT_RESULTS.md).
That study required eight correct parent retention answers before training and
stopped at six. No expert was trained. This contract does not change that result
and does not reuse its opened cases.

The question is whether two added identity-initialized Transformer blocks learn
additional unseen arithmetic answers. Parent retention is recorded. It does not
block training, and it is not a pass gate. A reply that does not end in EOS, or
that is not one integer or one equation matching the question, is incorrect.
An unfinished reply does not stop the study.

The architecture, optimizer, step count, and cost rules match the stopped
contract. The expert must answer at least 24/64 new questions and beat both the
parent and the matched-cost control by at least four, with a positive lower
95% paired-bootstrap bound. Latency, memory, frozen parent weights, and matched
training spend still apply. Retention losses are listed and are not repaired.
A pass only permits review of a later selector contract. Selector training,
automatic serving, a GPU, a 1.7B run, and checklist credit stay unauthorized.
Item 4 still requires four independent operators.

Operands run from 0 through 95. Unordered pairs are unique and exclude every
staged-integration, staged-answering, calibration, and block-expert pair.

```bash
PYTHONPATH=src venv_build/bin/python scripts/run_block_expert_measure.py
PYTHONPATH=src venv_build/bin/python scripts/run_block_expert_measure.py \
  --run --seed .neuroshard/seed-smollm2-135m --home .neuroshard/block-expert-measure-NEW
```

The runner accepts `--run` only after these sources are committed. No run is
part of preparing this contract.
