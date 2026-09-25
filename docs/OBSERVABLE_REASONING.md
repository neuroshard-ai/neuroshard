# Observable reasoning: one bounded CPU experiment

Status: execution contract; no learning result yet. This is a new experiment,
not a repair or rerun of the append-only oracle result. The original programming
final and opened programming campaign are not read. No public model, token
issuance, checklist completion, GPU, or 1.7B run is authorized.

## Question and intervention

Can two added transformer blocks learn a compositional arithmetic task better
than spending the same training CPU on the existing final two blocks, while a
deployable question-only selector preserves the parent's correct answers?

The previous direct-digit task tied a constant. This trial changes supervision:
the model learns to generate both operand remainders, their sum, and the final
remainder. No arithmetic program supplies intermediate steps at inference. All
steps are neural generations. The in-place control gets identical supervision.
This is a worked-example hypothesis, not an established improvement.

The seed is the pinned SmolLM2-135M-Instruct. Both arms use the existing tested
identity-block initialization and frozen-prefix training cache. Only the two
blocks train; the head and original expert-path weights remain frozen. There
are 864 expert updates, with 8 documents per step and learning rate 0.0003.
The control trains until its optimization CPU covers the entire expert training
child plus the expert-specific prefix computation. Imports, checkpointing,
common preparation, selector fitting, evaluation and any failed work remain in
the total bill. No expensive run begins before the execution freeze is committed.

## Serving and data boundary

A multinomial naive Bayes router fits **only training question words and task
labels**, with equal priors and smoothing 1. Numbers normalize to one token;
unseen words are ignored. Candidate log odds must exceed log(9); otherwise use
parent. Both comparison systems use that same router. This tests task selection,
not a general correctness predictor or learned multi-skill composition.

The selector accepts a string, not an evaluation row. It receives no gold
answer, task family, case ID, correctness flag, protected list or generated
reply. Selection happens before exactly one greedy decode, capped at 64 new
tokens. The full timer includes selection, model-path switching, tokenization,
prefill, decode and text decoding. Model startup is separately billed. Both
parent and candidate partitions remain resident in the isolated serving worker;
peak RSS includes both. Forced-expert responses diagnose knowledge versus
selection and are separately billed; they never choose the served answer.

All unordered operand pairs exclude the six earlier arithmetic datasets.
Training contains 768 modular examples and 384 ordinary additions. The new
evaluation contains 96 modular questions and 128 ordinary additions; its
wording templates are absent from training. Test operands can occur in training
with other partners: the claim is recombination of familiar operands, not
extrapolation to unseen numbers. A deterministic generator and all inputs are
committed. Public commitment is not secrecy.

Baseline answers define protection **before training**. Every baseline-correct
answer in both evaluation roles is protected. The router never sees this list.
A small baseline does not prevent training, but fewer than four correct parent
retention answers fails the final protection gate. This avoids another costly
setup that stops without measuring whether the expert can learn.

Raw decisions and replies are recorded before joining evaluation labels. A
strict parser accepts one complete integer or the declared four-line trace,
with real EOS. It never searches arbitrary prose or repairs an answer. Generated
intermediate steps are recorded, but the primary score is the final answer.

## All gates must pass

- Automatic serving answers at least 32/96 new questions correctly.
- Net gain is at least 6/96 over each of parent, the same-route trained control,
  and the most common training answer. Each paired bootstrap 95% lower bound
  is positive (10,000 draws, fixed seed).
- Forced expert also beats forced control by at least 6/96 with positive lower
  bound. A routing improvement cannot disguise a weak added module.
- Every protected answer is correct under automatic serving; at least four
  protected retention answers exist. Per-answer losses are published.
- Frozen parameters match, the control pays the required CPU budget, and the
  full served-response p95 is at most 20 seconds and 1.5 times control.
- Isolated serving peak RSS is at most 4 GiB and 1.5 times control.

One CPU thread; six-hour total cap; no retries or automatic follow-up. Failure
stays failure, including failure to beat the control. An execution failure is
reported separately from a completed quality failure. Pass means one local
synthetic mechanism survived these gates. It would not establish an assistant,
continuous growth, distributed performance, or economical verification.
Item 4 still requires four independent operators.

## Execute and track

Use the existing pinned CPU environment and seed. Run from a detached worktree
at the committed freeze so other development cannot change its sources:

```bash
PYTHONPATH=src /home/ubuntu/neuroshard/venv_build/bin/python scripts/run_observable_reasoning.py \
  --run --seed /home/ubuntu/neuroshard/.neuroshard/seed-smollm2-135m \
  --home /home/ubuntu/neuroshard/.neuroshard/observable-reasoning-20260925
```

The home must not exist. `study.json` pins commit, contract, source and freeze;
`status.json` identifies the active arm, `*-history.json` reports updates,
and `result.json` records the final gates, receipts, answers and complete CPU
accounting. The status command has no neural imports and starts no work:

```bash
python3 scripts/run_observable_reasoning.py --status \
  --home /home/ubuntu/neuroshard/.neuroshard/observable-reasoning-20260925
```

Check the service exit state as well if the last status remains `running` after
the deadline: a host shutdown or cgroup kill cannot write a final result.
Do not automatically resume an interrupted study under this contract.
