# Preserved interpretation: four-owner final passes

The distributed composition answered **949/1,024 newly worded knowledge
questions correctly (92.68%)**, retained every token of all 768 earlier skill
answers and reproduced all 256 earlier conversation losses exactly. The parent
continued serving after the controller observed the expert process exit with
code zero. All eight frozen final checks passed.

| Measurement | Development | Independent final |
| --- | ---: | ---: |
| Knowledge answers correct | 982/1,024 | 949/1,024 |
| Knowledge accuracy | 95.90% | 92.68% |
| Entity-cluster gain lower bound | 0.95020 | 0.91309 |
| Earlier skill outputs reproduced exactly | 192/192 | 768/768 |
| Earlier correct answers lost | 0/151 | 0/650 |
| Earlier conversation losses reproduced exactly | 128/128 | 256/256 |
| Parent serves after observed expert exit | Pass | Pass |

The [complete result](../config/experiments/preserved-interpreter-results.json)
contains the decisions, all-owner agreement, process-exit observations,
archival readback and resource accounting. The parent answered none of these
synthetic knowledge questions correctly before using the expert. The final
entity-cluster gain interval was [0.91309, 0.94043], over 128 entities. Facts were
training material; newly committed question wording was held out.

All four GPU instances, their disks and the study security group were removed;
the protected hosts remained unchanged. Compute was bounded at $9.80, with
$0.16 estimated disk cost and an $11.23 conservative transfer allowance.
These are planning estimates rather than an AWS invoice. The complete archive
was read back successfully with SHA-256
`59679091f8a3c15afbb697d7e78afabeb785fda0579ea6eea7301345f43abe36`.

## What made the difference

The learned expert already stored much of the knowledge, but did not answer
varied wording reliably. The preserved original 1.7B model now interprets a
question into a name and requested field. Strictly validated arguments produce
a canonical question that the learned expert answers through ordinary greedy
generation. Invalid interpretation falls back to the original question.
The interpreter receives no factual values, and serving uses no factual lookup
table. Questions outside the explicit directory domain use the prior parent.

Three machines each own disjoint portions of the trained parent and preserved
interpreter. A fourth owns the 134,225,920-parameter learned expert. No owner
holds either complete 1.7B model. Total stored parameters are 3,556,978,688;
this is additional capacity, with additional inference work and storage.

All weights stayed unchanged during composition and evaluation. The parent
retains its actual Adam state at update 448; the expert retains its separate
1,024 tail updates. The [preceding training result](https://github.com/neuroshard-ai/neuroshard/blob/research/knowledge-rehearsal/docs/KNOWLEDGE_REHEARSAL_RESULTS.md)
records how those expert weights were learned. The [method and frozen plan](PRESERVED_INTERPRETER.md)
describe the interpretation rule, runtime, ownership limits and final gate.

## Selection and reproduction

All 32 outputs of the earlier diagnostic reproduced exactly across the four
owners. Development then evaluated the already exposed branch-final questions.
The full development archive was read back and selection was committed as
`0f6334462733d6fd11e6471cfeeb2bdf818a391e` before the new final started.
Numerical source was frozen at `7029fb4e455d9babc86405b50668d83066f9981d`.
All four owners agreed on the final output identity. No tuning followed exposure
of the new final.

The [research model release](https://github.com/neuroshard-ai/neuroshard/releases/tag/research-shards-20260915)
provides the exact model and optimizer parts, tokenizer, numerical inputs,
all four owners' output records, notices and reproduction instructions. Its
manifest lets each owner download only its assigned portions and verify their
full SHA-256 hashes. Reproducing these published questions checks execution;
it is not another independent quality final.

All 459 published assets, including 454 model/optimizer objects, passed complete
anonymous download and SHA-256 verification: 25,577,867,850 bytes in total.
The [publication record](../config/experiments/preserved-interpreter-publication.json)
binds the release, numerical bundle and verification result. The download
procedure needs no operator AWS access or GitHub login.

Use the [current reproduction guide](REPRODUCE_EXPERT_GRAPH.md), which corrects
control-receipt publication to use atomic file replacement. The initial bundled
guide wrote the receipt directly, allowing a concurrent reader to see partial
JSON. Model objects, frozen neural execution and recorded results are unchanged.

## What this establishes

This is a passed, bounded example of accessing learned neural knowledge through
a larger distributed model while preserving earlier outputs. It supports the
next experiment: another owner learns a separate expert while this established
graph keeps its state and serves earlier requests.

The experiment used one operator and a fixed explicit domain rule. Earlier
skills are exposed retention probes. General assistant quality, automatic
routing, independent ownership, arbitrary membership and economical
permissionless verification remain unmeasured here. The serving graph was not
activated by native consensus; this composition issued no tokens. Earlier
failed experiments remain failed.
