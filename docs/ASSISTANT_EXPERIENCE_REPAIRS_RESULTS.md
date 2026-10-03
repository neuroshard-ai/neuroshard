# Date-base repairs results

The [repairs study](ASSISTANT_EXPERIENCE_REPAIRS.md) ran once on one A10G from
commit `2c741aa`. **It failed as declared:** no repaired rollout passed the
scorer, so no repair pair was found and no arm was continued. There is no third
attempt. The third confirmation split stays sealed. Evidence:
[report](../config/experiments/assistant-experience-repairs-report.json).

## What happened

- The pinned compositional small module sampled 2112 episodes on 528 training
  cases; 1922 passed.
- Of the 190 failures, nearly all were compositional practice cases: wrong
  drafts after the latest and scope corrections, and wrong first rounds in the
  date family.
- Only 9 failures started a date shift from a base the round cannot justify.
  Each got two repair attempts, and none of the 18 continuations passed. In all
  9, the module had already read a different revision than the goal source and
  computed the total from it, so repairing the shift alone could not succeed.

## Finding

The development error, applying the review interval twice, does not occur on
training cases. It arises only with the held-out correction that adds a day,
which no training split contains, so repairs on training cases carry no signal
for it.

## Resources

One GPU allocation, retired with nothing remaining ($4.90).
