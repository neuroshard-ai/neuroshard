# Verified-experience confirmation result

**Finished September 28, 2026 at 13:18 UTC; confirmation failed on one check.**
Execution freeze `cd7fb626a0d8568b84123a173b89b359b5dcf44e` passed exact-commit CI.
After the [round-3 development pass](ASSISTANT_EXPERIENCE_DEVELOPMENT_ROUND3_RESULTS.md),
the 96 sealed confirmation episodes were opened once: the parent control under
recompute serving and the round-3 update and addition systems under prefix-cache
serving, each alone on its own CPU host.

| Measurement | Parent | Update system | Addition system | Required of addition |
| --- | ---: | ---: | ---: | ---: |
| Complete workflows | 49/96 | 86/96 | **88/96** | At least 77/96 ✓ |
| Fewest in any family | — | 8 | **8** (difference) | At least 8 ✓ |
| Net versus parent | — | — | **+39** | At least +10 ✓ |
| Lost parent successes | — | — | **0** | 0 ✓ |
| Lower 95% gain versus parent | — | — | **+31.3 points** | Above 0 ✓ |
| Lower 95% gain versus update | — | — | **−5.2 points** | At least −5 points ✗ |
| Episode p95, routing included | 177.9 s | 95.2 s | **98.9 s** | At most 180 s ✓ |

The addition gains 39 of the parent's failures and loses none of its 49 successes.
It completes two more workflows than the update control (six gained, four lost),
but the family-bootstrapped lower bound of that difference is −5.2 percentage
points, 0.2 points short of the declared non-inferiority margin. With eight
operation families, one family's swing moves this bound substantially; the
difference family, where the addition completes exactly the required eight, is the
weakest.

## Decision

Keep this result failed. A2 is not established for this capability, and
confirmation is not rerun. The failure concerns parity with the equal-data
update, the amended growth comparison, not learning itself. The same
comparison is lopsided against the unchanged parent: 49/96 to 88/96 with every
parent success preserved, from verified self-generated experience plus verified
decision preferences, a 1,048,576-parameter module in 4.2 MB, and p95 latency
halved by prefix-cache serving. Any new attempt needs fresh confirmation data
under a newly declared contract.

## Evidence and cost

Raw results for the [parent](../config/experiments/assistant-experience-confirmation-parent-result.json),
[update](../config/experiments/assistant-experience-confirmation-update-result.json) and
[addition](../config/experiments/assistant-experience-confirmation-addition-result.json)
and the [report](../config/experiments/assistant-experience-confirmation-report.json)
record every episode and receipt; the frozen scorer rescored all 288 episodes
before the gate. The parent host's controller reported a tagged volume still
present at its first retirement check; a repeated check found none and wrote the
receipt. The three hosts cost **$6.76**. The verified-experience comparison has
used **$26.38** of compute in total, within the $100 ceiling.
