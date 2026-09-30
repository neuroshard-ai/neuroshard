# Compositional practice results

The [compositional study](ASSISTANT_EXPERIENCE_COMPOSE.md) ran once on one A10G
from commit `1b8a7f4`. Under its declared rule, **there is no third attempt**:
the small module lost 3 parent successes on the development split. The study
used already-opened data and earns no A2 credit. The third confirmation split
stays sealed. Evidence: [report](../config/experiments/assistant-experience-compose-report.json).

## Outcomes

Both arms trained on the 3302-sequence pool, 36% of it compositional practice.

| System | Integration | Development | Lost | Score |
|---|---|---|---|---|
| Update control | 60 | 19 | 1 | 77 |
| Small module | 62 | 17 | 3 | 73 |
| Parent | 34 | 9 | — | — |

Sampled integration success rates: small 95.3%, update 93.8%, parent 52.3%.

## Findings

- **Practice did not transfer to the held-out addition.** All three lost cases
  are latest or scope development cases whose correction ends "Also move the
  resulting due date one calendar day later." The small module has lost two of
  them in every study.
- **Small effect overall.** Against the growth study, the small module's
  development successes moved from 16 to 17, and the update's from 20 to 19.
  Integration was unchanged at 62 and 60.
- **The parent varies between runs.** It solved `workflow-caf22532f0f4` here but
  not in either earlier study, because batched GPU decoding is not
  bit-reproducible. The update control lost that case too.

## Resources

One GPU allocation, retired with nothing remaining ($2.25). With the two
collections ($9.04), compositional practice cost $11.28.
