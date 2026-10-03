# Experience growth results

The [growth study](ASSISTANT_EXPERIENCE_GROWTH.md) ran once on one L40S from
commit `94f378b`. Under its declared rule, **there is no third attempt.** The
candidate was the small module, because the committee scored below it, and the
small module lost 2 parent successes on the development split. The study used
already-opened data and earns no A2 credit. The third confirmation split stays
sealed. Evidence: [report](../config/experiments/assistant-experience-growth-report.json).

## Outcomes

Greedy successes on the 64 integration and 24 development cases, parent
successes lost on those episodes, and the declared score:

| System | Sequences | Integration | Development | Lost | Score |
|---|---|---|---|---|---|
| Update control | 2114 | 60 | 20 | 0 | 80 |
| Small module | 2114 | 62 | 16 | 2 | 74 |
| Member 1 | 690 | 57 | 19 | 1 | 74 |
| Member 0 | 701 | 51 | 16 | 0 | 67 |
| Committee (3 members + parent) | — | 46 | 13 | 0 | 59 |
| Member 2 | 723 | 44 | 14 | 3 | 52 |
| Parent | — | 34 | 8 | — | — |

Sampled integration success rates: small 95.3%, update 93.8%, committee 71.1%,
parent 50.8%.

## Findings

- **Growth helped the update most.** With 2.9 times the experience, the update
  control rose from 74 to 80 and lost no parent success. The small module rose
  from 71 to 74.
- **The small module generalizes less.** It solved 2 integration cases the update
  missed, but it missed 4 development cases the update solved, including both
  lost parent successes. The development split holds out combinations of
  corrections that training never shows.
- **Members vary widely.** Trained on 690–723 sequences each, about what the
  methodology study's small module had, they solved 44, 51 and 57 integration
  cases. That module solved 61.
- **The committee stays close to the parent.** Its outcome matched the parent's
  on 52 of 64 integration cases. In 8 of the 53 cases where at least two members
  succeeded alone, it still failed, because split member votes leave the
  parent's tie-break in charge.

## Resources

One GPU allocation, retired with nothing remaining ($4.08), after a launch
that found no capacity ($0.07). The growth effort cost $21.04 in total: $7.01
for the failed first collection, $9.88 for the second, and $4.15 for the study.
