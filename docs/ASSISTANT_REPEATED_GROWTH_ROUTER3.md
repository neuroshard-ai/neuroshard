# A3 stage 1: the turn-need router on what the user wrote

Declared on October 5, 2026, after the [turn-need router's results](ASSISTANT_REPEATED_GROWTH_ROUTER2_RESULTS.md)
and before any development with it. Contract:
[assistant-growth-router3.json](../config/experiments/assistant-growth-router3.json).
The project owner chose this change.

## Why

Trained on 2,112 turns labelled by what their verified correct solutions need, the
router was only 74% accurate held out and 73% on the integration turns. That is
below the 79% of routing every turn to scheduling. The labels were clean, so the
feature was the limit. The parent's state at the first token of its reply encodes
what it is about to do, and little of what the user wrote.

## The change: the feature

A turn's feature is now the parent's mean final-layer state over the user's own
message, its end marker and the reply header. Everything before the message is the
same for every turn: the instruction, the tools and the user header. That prefix is
computed once into a key/value cache. Each turn extends the cache and is cropped
back, so a feature costs a fraction of a second instead of about three. On a small
model the cached computation matches a full forward pass to within 1e-5.

The router file names this feature, and development and confirmation compute it in
the same way on the same CPU runtime. Labels, data, the centroid rule and the
reported accuracies are unchanged. The router is fitted once and pinned by digest
before development.

## What stays

Round 4's units, routing, the three versions, and every development and
confirmation gate. Nothing is trained.

## Budget

A3 has spent $38.98. The refit allows at most $5.50 on one CPU host, development $18
on three and confirmation $47 on five. That is $109.48 in total, within the $110
ceiling.
