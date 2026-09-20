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
