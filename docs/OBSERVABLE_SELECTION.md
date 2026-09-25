# Observable selection

**Status: specified, not trained. No GPU. Not admission.**

The [append-only execution](APPEND_ONLY_EXECUTION_RESULTS.md) kept protected
answers only by reading whether each reply matched the hidden answer. The
added blocks also tied a constant training-label baseline, 9/64. This contract
is the next learning rule. It does not rerun that study and does not reuse its
opened questions as a success set.

A later execution may train only after this rule is committed. The selector's
inputs are the question text, the parent generation, the added generation, and
scores the model produced without a gold label. The selector must not receive
the gold answer, a `passed` flag, or any equality between a reply and the gold
answer. Evaluation may still compare the served text with the gold answer
after the choice is made.

Success for that later run requires both of these, on fresh questions:

- Served new answers beat the most common training label, not merely a parent
  that scored zero.
- Every predeclared protected answer is still correct in the served output.
  The protected list may be known as question identities from before the run.
  Correctness is checked after selection, not supplied to the selector.

No GPU, 1.7B run, NEURO issuance, 0.4.0 upgrade, or checklist credit is
authorized. Item 4 still requires four independent operators.
