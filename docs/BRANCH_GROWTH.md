# A learned transformer branch

The longer existing-tail learner reached 218/256 correct new-knowledge answers
and preserved all 151 previously correct skill answers on development. The
added-depth candidate failed. This composition experiment uses the successful
learned tail as a new expert while retaining the complete original model.

Three peers hold the original 24-layer model in disjoint partitions. Its first
22 layers are shared with a fourth peer's two learned transformer layers.
Questions mentioning the fictional Luma directory use that new tail. Other
questions use the original tail. Both paths use the original tokenizer and
ordinary greedy language-model generation. Routing receives only user text;
there is no entity or attribute oracle, factual lookup table, output-class list,
forced JSON prefix or answer repair.

This adds actual transformer parameters and preserves the original function
outside the explicit domain. Parent tensor transfers and token decisions use
their own three-peer process group. The experiment checks continued generation
after the controller observes the new expert process exit. It does not claim
arbitrary peer admission, Internet-scale availability or a general learned
router.

No new training is performed in this composition trial. The expert retains the
exact weights and Adam provenance from the frozen 1,024-update rehearsal. The
first GPU check must reproduce all 256 earlier knowledge outputs token for
token. New development and final question wording was committed in an earlier,
unexecuted readout proposal and remains unused. Its facts were training material.
Old skill and conversation probes measure retention only.

The recorded parent retention outputs are reused as a derived cache. Every
cached token and loss must match the new graph's actual parent-path execution.
This also checks reuse across the three-owner layout. New knowledge questions
receive fresh baseline generation. Cache identity and input hashes are recorded;
reusing it changes neither the questions nor the gates.

The graph must reach 75% new-question accuracy, pass the original gain and
retention gates, and reproduce parent skill outputs and conversation losses
exactly. Only a committed eligible graph may open finals. Growth here means a
useful added expert alongside the retained model. It does not establish that
this graph beats every fixed-size model at equal cost or prove a scaling law.
No token issuance or native serving promotion occurs in this experiment.

The plan and prepared identities are in `config/experiments/branch-growth.json`
and `config/experiments/branch-growth-prepared.json`. Numerical results are
pending until complete reports are published.
