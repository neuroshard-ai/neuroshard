# A conditional learned decoder

The added transformer blocks did not reliably learn the new fictional directory
facts. A training-only diagnostic found that the frozen parent's hidden vector
could support a much better joint value readout. This experiment tests that
specific lead through actual question-only inference on four separate owners.

Three owners retain the frozen 24-layer parent. A fourth owns a small learned
decoder. It receives only one causal hidden vector; no entity, attribute,
expected answer or evaluation identifier is supplied to prediction. The vector
is computed from the user's question followed by three globally common JSON
tokens, before any value-bearing token. Training computes fresh vectors with
exactly this inference shape, rather than reusing padded teacher-forced vectors.

One float64 ridge solve learns 128 value classes from all 5,120 previously
admitted question-answer examples. The decoder serializes its chosen class as
JSON. This is a structured neural decoder, not an unconstrained language-model
head. Its name roster contains no entity-to-value mapping. A simple explicit
domain selector sends questions mentioning the fictional Luma directory and
exactly one admitted full name to this expert; other questions use the frozen
parent. General routing, ambiguous requests and cross-domain conversation are
not established by this bounded profile.

New question forms are committed before training. Development and final forms
are distinct from the earlier growth and rehearsal experiments. Facts are
training material; held-out wording measures recall through new questions.
The decoder must reach 75% accuracy, preserve every previously correct skill
answer, pass the existing conversation-retention bound, and reproduce all
ordinary skill fallback outputs exactly. Only an eligible, committed decoder
may open the final questions. Previously exposed skill and conversation probes
measure retention; they are not independent new-quality evidence.

The parent control receives the same common JSON prefix for knowledge questions.
The experiment measures the learned decoder's gain over that control, records
feature-production and fitting costs separately, and verifies the causal wire
and actual parent fallback. It does not claim an advantage over a matched
transformer tail, broad assistant quality, independent operators or native
settlement. A passing result would establish one bounded mechanism for adding
learned capacity while preserving the parent.

The frozen plan is `config/experiments/conditional-readout.json`; source and
input identities are in `conditional-readout-prepared.json`. No result exists
until the complete reports and selection have been published.
