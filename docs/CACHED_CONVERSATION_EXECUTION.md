# Cached conversation execution

The experimental planned graph accepts alternating user/assistant turns, asks
the preserved neural model to split the latest request into at most two
standalone questions, and applies the existing learned classifier to each
question. Directory arguments are interpreted by the preserved model; learned
expert tails generate the answers. General answers use the preserved instruct
model. Two answers are displayed with their questions in order.

Each owner retains only its layer caches. The prompt is processed once; later
decode steps transmit one hidden vector per boundary. Selecting a learned tail
also excludes the original tail from the third parent's execution and cache.
Caches are rebuilt for every call and invalidated by changed weights or an
incomplete update. They never become trusted input from another peer.

The response includes every planning, argument and answer call, tokenized
prompt, owner set and generated token. Invalid plans return a clarification
status and empty final text, with their actual computation retained for replay.
The operator queue supports `generate_planned` and `replay_planned` under an
explicit service commitment. It is not a public chat endpoint or an admitted
native inference format. Prompt tokens and responses are visible in its
evidence; it provides no private inference guarantee.

Caching changes floating-point matrix shapes. Its own committed source and
quality decision are required before settlement activation. Seven small-model
checks cover full-model logit agreement, separate owner execution, changed
weights, interrupted caches, forged activation replay and reduced tensor bytes.
An additional five-process check verifies actual failed neural planning, no
expert execution and complete replay. The larger-model result is pending in
[the frozen trial](../config/experiments/cached-composition-trial.json).
