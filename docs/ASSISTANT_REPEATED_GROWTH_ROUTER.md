# A3 stage 1: a router that separates the two kinds of turn

Declared on October 5, 2026, after the [round-4 development results](ASSISTANT_REPEATED_GROWTH_DEVELOPMENT4_RESULTS.md)
and before any development rerun. Contract:
[assistant-growth-router.json](../config/experiments/assistant-growth-router.json).
The project owner chose this router from the options after round 4's development.

## Why

Round 4's candidate met the development gate's scheduling and cross levels, 18/24
and 6/8. It lost one drafting success because its selector sent every turn,
drafting included, to the scheduling route. The A2 logistic recipe has now failed
in both directions. Fitted on GPU features with every tie counted for drafting, it
sent every scheduling turn of the separate update to the drafting route. Refitted
on CPU features without the turns both routes failed, it sent every turn to the
scheduling route. Every turn feature shares a large component, the instruction and
tools rendered before the user message. The two kinds of turn differ only in a
small part of it.

## The router

Each development host computes the parent's features at the integration user turns
on its own CPU runtime, as in round 4, and subtracts their mean. It then normalises
them and forms each class's weighted mean from the round-4 targets. Turns that both
routes always failed are left out, and other ties count for drafting. A turn goes to
the scheduling route only if its centred feature is closer to the scheduling mean;
an exact tie keeps the drafting route. Held-out accuracy over four folds of
integration cases is reported, not gated.

## What stays

Round 4's units and integration outcomes, routing, the three versions, and every
development and confirmation gate. Nothing is trained. If the candidate passes, the
sealed confirmation opens once. Each confirmation host refits the same router and
must reproduce the selector development used.

## Budget

A3 has spent $33.12 since stage 0. The rerun allows at most $18 for development on
three CPU hosts and $47 for confirmation on five, $98.12 in total within the
unchanged $100 ceiling.
