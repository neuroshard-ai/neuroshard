# Preserve interpretation while peers learn experts

A controlled probe improved factual answers from **16/32 to 30/32** without
changing the learned expert. The original SmolLM2-1.7B-Instruct model interpreted
the question, then the expert answered a canonical version through ordinary
language-model generation. This is an exploratory result on exposed questions.
The subsequent [four-owner final passed](PRESERVED_INTERPRETER_RESULTS.md):
949/1,024 newly worded knowledge answers, with all 768 earlier skill outputs
and 256 conversation losses reproduced exactly.

| Interpreter | Instruction placement | Correct interpretations | End-to-end factual answers |
| --- | --- | --- | --- |
| Original model | Original prompt | 0/32 | 16/32 |
| Original model | Instruction repeated after the quoted question | 28/32 | 30/32 |
| Previously trained parent | Original prompt | 0/32 | 16/32 |
| Previously trained parent | Instruction repeated after the quoted question | 0/32 | 16/32 |

Both raw-answer controls reproduced every token from the earlier four-peer
run. All inference weights remained unchanged. The evidence supports a prompt
interaction and an instruction-following difference between checkpoints on
this task; it does not establish that every general ability was lost. Some
invalid interpretations fell back to an original question the expert already
answered correctly, explaining why answer accuracy exceeds interpretation
accuracy. No factual values entered the interpretation prompt.

The [diagnostic result](../config/experiments/interpretation-control-results.json)
binds the complete outputs, runtime, source and verified archive. Its temporary
GPU and disk were removed. The first allocation was retired before setup after
an EC2 discovery race; its bounded replacement used the original deadline.
Combined compute was less than $0.44; storage and transfer are separate.
No tokens were issued and no native serving checkpoint was activated.

## Distributed method

Three peers each hold disjoint portions of the preserved interpreter and the
previously trained parent. A fourth owns the learned two-layer expert. No
participant receives either whole model. Each of the two 1.7B models contains
1,711,376,384 parameters each; the expert adds 134,225,920, for 3,556,978,688
stored transformer parameters. This larger graph is not an equal-budget
scaling advantage.

Questions in the explicit fictional-directory domain first pass through the
preserved interpreter. It generates a person's name and one requested field.
Arguments must be valid JSON with exactly the declared keys, a supported field
and a name literally present in user text. Valid arguments produce a canonical
question; invalid arguments preserve the original question. The learned expert
then generates the answer from its weights with the full tokenizer vocabulary.
There is no factual lookup table or forced answer class.

Other questions execute the prior parent path directly. Interpretation uses
the three-parent process group, and expert serving uses all four owners. The
parent remains able to answer after the controller observes the expert process
exit. Every model stays read-only throughout this composition trial; the expert
retains the weights and actual optimizer provenance from its 1,024 training
updates. The [training result](https://github.com/neuroshard-ai/neuroshard/blob/research/knowledge-rehearsal/docs/KNOWLEDGE_REHEARSAL_RESULTS.md) is available in
the preceding research branch.

## Frozen evaluation

The [plan](../config/experiments/preserved-interpreter.json) and
[new final wording](../config/experiments/preserved-interpreter-questions.json)
are committed before evaluation. Preparation binds all numerical source, every
input, the partitioned original-model assets, and the retained-output cache.
Execution refuses an absent or changed prepared record.

First, all 32 composed outputs from the diagnostic must reproduce exactly
across four peers, including its two wrong answers. Development then uses the
1,024 already-exposed branch-final questions. They provide a selection check,
not an independent quality result. Only a committed eligible graph opens a
separate final with 1,024 newly worded questions. The facts were trained;
question wording is held out. The final cannot repeat training or earlier
questions byte for byte.

The gate still requires at least 75% factual accuracy, an entity-cluster gain
lower bound above 0.10, no earlier correct skill losses, exact previous skill
outputs and conversation losses, and parent serving after observed expert exit.
The final rechecks all 768 earlier skill answers and 256 conversation losses.
The four-GPU allocation is bounded to three hours and $50, including a
conservative transfer allowance. A failed gate stays failed. The previous
branch's failed final and the deferred next-cohort prerequisite remain unchanged.

This targets reliable access to newly learned neural knowledge. General
assistant usefulness, arbitrary routing, multiple independent operators,
permissionless verification and native graph settlement remain distinct work.
The controlled knowledge literature also distinguishes memorized content from
reliable extraction across contexts ([Knowledge Storage and Extraction](https://arxiv.org/abs/2309.14316)).
That motivates this experiment; its outcome must come from the measurements.
