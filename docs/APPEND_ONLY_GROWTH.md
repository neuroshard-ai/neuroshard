# Append-only growth

**Status: rule frozen, not trained. No GPU. Not admission. Item 4 remains open.**

The [block-expert measurement](BLOCK_EXPERT_MEASURE_RESULTS.md) showed that
training the path which answers every question can add a few new answers while
erasing the parent's protected answers. In-place training of the existing
blocks did the same. A growing assistant keeps the accepted answers and appends
a shard for questions the accepted model missed.

This contract does not train. It fixes the serving rule before any new run.
The opened measurement cases stay closed and cannot be rescored as a pass.

## Serving rule

1. The parent weights stay frozen. The parent is the default answer.
2. If the parent already answered a protected question correctly, the served
   answer is the parent's answer.
3. The added shard is consulted only when the parent did not answer correctly.
4. If the added shard is also wrong, the parent's answer remains the served
   answer.
5. An unfinished reply is incorrect.

The CPU execution is [append-only execution](APPEND_ONLY_EXECUTION.md). It uses
fresh questions. Success for that later run is: every protected parent answer
is still served, and the served system gains correct answers the parent missed.
Beating an in-place control that is allowed to forget is not the product test.
Selector training, a GPU, a 1.7B run, promotion, and checklist credit are not
authorized. Four independent operators remain item 4.
