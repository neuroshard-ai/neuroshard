# Leftover programming fallback result

**Status: leftover comparison passed its frozen gates. Not promoted. Original
128-task final remains closed.**

The [frozen comparison](PROGRAMMING_FALLBACK_COMPARISON.md) asked whether the
rejected programming tail helps when it is used only after the parent's public
example fails, versus giving the unchanged parent one extra decode on a repair
prompt. That is a test of complementary shard skill, not of an ever-growing
public assistant.

## Result

| Arm | Full-test correct / 32 |
| --- | --- |
| First parent attempt | 8 |
| Expert fallback | 12 |
| Parent repair | 8 |

Net +4 versus both controls, 4 unique gains and 0 losses, paired-bootstrap
lower bound versus repair +0.03125. Public example passed for 11/32 first
attempts; those answers were kept. Sixteen repair decodes repeated the first
answer tokens and were scored as failures, not discarded. Repair p95 input
tokens were 429 versus fallback 246: extra-attempt count and output-token cap
were shared, computation was not.

p95 serving time (generation plus example check) was 12.19s fallback versus
14.01s repair. An independent CPU rescore matched the GPU host on every leftover
task. No optimizer step ran. The original programming-expert final was not
read.

This does **not** promote the tail into the accepted graph, does not open the
128-task final, does not credit learned routing, and does not change 0.4.0 or
checklist item 4. It is one bounded measurement that an added shard can earn
work under an executable gate.

## What the four gains do and do not prove

The four unique wins are complementary capability: the trained tail produced
full-test-correct programs where the first parent attempt and the parent
repair did not. That is the mechanism worth keeping.

Zero unique losses are largely guaranteed by the policy. A fully correct
parent program passes the public example and is retained. That is protection
through selection, not learned resistance to forgetting. Eleven first parent
answers passed the example; eight of those also passed the full tests, so
three kept parent answers were still wrong on withheld tests. The example is
an imperfect gate.

## Research baseline

This exact configuration is the research baseline for further growth:

- rejected-trial terminal checkpoint `46bd2e76…`
- public-example checking rule
- one extra decode and a 256 output-token cap

The next growth experiment must compare an expanded system against this
complete fallback system, not restart against the unmodified parent. The
[second-capability growth](PROGRAMMING_GROWTH.md) campaign that asked that
question is closed: isolation passed, then independent-tail merges and
heuristic selectors failed their declared gates. This leftover extra remains
the research baseline because those challengers failed acceptance. The next
experiment is [learned integration](LEARNED_INTEGRATION.md). Public promotion
and opt-in research serving remain separate decisions. The 0.4.0 genesis is
unchanged.

## Resources

Four temporary g5.xlarge hosts in us-east-1. Conservative GPU instance-time
through observed retirement: **$0.913**, excluding preparation-host, storage
and transfer. All four instances are terminated, their volumes and network
interfaces are gone, the temporary security group is deleted, and the protected
hosts retain their prior states.

## Reproduction

Committed freeze: `config/experiments/programming-fallback-freeze.json`.
Outputs: `config/experiments/programming-fallback-outputs.json`.
Score: `config/experiments/programming-fallback-results.json`.
