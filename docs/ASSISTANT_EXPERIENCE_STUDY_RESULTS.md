# Methodology study results

The [methodology study](ASSISTANT_EXPERIENCE_LEARNING.md#methodology-study-before-a-third-attempt)
ran once on one GPU from commit `304f97c`. Under its declared rule, **neither
the larger module nor the committee is carried forward.** The study used
already-opened data and earns no A2 credit. The fresh confirmation split stays
sealed. Evidence: [report](../config/experiments/assistant-experience-study-report.json).

## Outcomes

Greedy successes on the 64 integration and 24 development cases, parent
successes each system lost on those episodes, and the declared score
(integration + development − 2 × lost):

| System | Integration | Development | Lost | Score |
|---|---|---|---|---|
| Update control (63M) | 59 | 19 | 2 | 74 |
| Small module (1M) | 61 | 16 | 3 | 71 |
| Large module (50M) | 56 | 19 | 3 | 69 |
| Committee (3 members + parent) | 49 | 16 | 0 | 65 |
| Member 0 | 49 | 17 | 1 | 64 |
| Member 1 | 56 | 19 | 3 | 69 |
| Member 2 | 47 | 16 | 1 | 61 |
| Parent | 34 | 9 | — | — |

Sampled integration success rates: update 92.2%, small 93.8%, large 85.9%,
committee 70.3%, parent 53.9%.

The large module outscored the committee (69 against 65) but not the small
module (71), so the rule carries neither forward.

## Findings

- **Splitting the experience cost more than voting gained.** Each member learned
  from a fixed third of the training cases (234–270 sequences) and solved 47–56
  integration cases alone. The small module, trained on all 740, solved 61.
- **The vote preserved the parent.** The committee was the only system that lost
  no parent success on either split. Both earlier confirmations failed partly on
  that check.
- **The vote was conservative.** The parent breaks ties, so members override it
  only with three of four votes, or with two when the other two proposals also
  disagree with each other. In 8 of the 53 integration cases where at least two
  members succeeded alone, the committee failed.
- **More parameters did not help.** The 50M module lost 5 integration cases to the
  1M module, sampled worse, and lost as many parent successes.

## Resources

One GPU allocation, retired with nothing remaining: 15,374 instance-seconds,
$5.18. An earlier launch stopped at startup ($0.22).
