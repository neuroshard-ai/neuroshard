# Preserve explicit ordinary questions

The first full ordinary cohort completed 128 updates with three fresh replays
per four-update window. Its terminal evaluation improved single answers from
3 to 13/16 and combined answers from 2 to 9/16, losing no previously correct
retained answers. The required combined score was 12/16. That frozen result is
failed and cannot be converted into a passing prospective result by this repair.

Two combined cases failed reference repair. Other cases lost domain scope when
the planner rewrote complete user questions. Two single questions also exposed
real fact confusions, and another selected the wrong expert. One combined
answer failed exact scoring despite a scientifically correct paraphrase about
Rayleigh scattering. These are distinct failure mechanisms.

The new optional request policy preserves two explicitly stated question spans,
including their local context and capitalization. It strips only a sentence
connector between the questions. It does not generate question wording, provide
routing labels or change neural weights. Quoted text, shared trailing output
instructions, additional turns, extra questions and obvious external references
keep the existing planning path. A local noun phrase before an object or
possessive reference is a syntactic heuristic, not a proof of unambiguous meaning.
The complete answering system still needs prospective evaluation.

The diagnostic compares the old and new policies on the same terminal weights
and already opened cases. It requires reproduction of every old neural
transcript, records all new complete replies, and replays a combined request.
Billing charges only the calls actually executed and rejects altered literal
spans. This diagnostic cannot issue tokens, promote a model, reopen an independent
final or satisfy the three-cohort milestone.

The inference-only diagnostic reproduced every original neural response and
replayed a combined response exactly. Singles stayed 13/16; combined answers
improved 9→11/16. However, two previously correct diagnostic answers regressed.
Both were bootstrap questions gained by the unaccepted admission candidate;
none of the accepted seed's correct answers were lost. Literal wording confused
the proposal lifetime with another number in one case and selected the parent
instead of admission in another. Skills and conversation scores stayed unchanged.

This method failed the unchanged development gates and remains optional. It
repairs some planner rewrites but exposes the expert's sensitivity to wording.
The [result](../config/experiments/literal-question-diagnostic-20260918/results/result.json)
and [complete visible replies](../config/experiments/literal-question-diagnostic-20260918/results/answers.json)
are recorded without training, issuance or promotion.
