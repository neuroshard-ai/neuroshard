# A3 cohort 3: an upgrade of drafting

Declared on October 7, 2026, after [cohort 2 was accepted](ASSISTANT_REPEATED_GROWTH_CONFIRMATION2_RESULTS.md)
and before any cohort-3 execution. Contract: [assistant-growth-cohort3.json](../config/experiments/assistant-growth-cohort3.json).
The project owner chose the upgrade.

## Why

[A3](ASSISTANT_REPEATED_GROWTH.md) asks for three accepted cohorts, at least one of them an
upgrade, and for an update that proves better than keeping the previous system under a
declared resource budget. Drafting, cohort 1, is the capability with room to improve. On the
192 sealed drafting episodes of cohort 2's confirmation, the accepted drafting route failed
15, all in the first turn:

- 9 cited approved revision 1 instead of the latest approved revision;
- 5 cited the unapproved draft when the user asked for the change from revision 1;
- 1 saved the right draft, then ran out of model turns before its confirmation.

On the opened development cases, 4 of its 5 drafting failures also saved the right draft
and then ran out of turns: it made one tool call per reply, and the policy allows six replies
per turn. Of the 4 sealed cross failures, one was a handoff's drafting turn that cited
revision 1. The other three were review requests, which go to the scheduling route; this
upgrade does not change it.

## The upgrade

L3 is a new rank-16 low-rank module on U1's projections, on top of the frozen accepted update
U1, the same shape as cohort 2's L2. It serves only the drafting route, behind A2's
per-episode gate. The scheduling route, the pinned router and every policy, tool and limit
stay as cohort 2 accepted them.

L3 learns from one verified demonstration per drafting training case, 1,248 in all, written by
a [drafting solver](../src/neuroshard/evolution/assistant_drafting_demonstration.py) as the
accepted version's own replies. A turn that needs a plan lists the project's documents and
reads the latest approved revision; it also reads approved revision 1 when the user asks for
the change from it, and reads revision 1 instead when the user switches to it. Calls that do
not depend on each other share a reply: two reads, or a date shift and a calculation. Every
demonstration must pass the scorer within the drafting policy's limits and read no unapproved
revision, or the job stops before training. The solver sees training cases only.

## What runs

- **GPU (one A10G host).** The demonstrations; L3's training for 512 steps on six
  demonstrations and two items of A2's parent replay per update, with stage 1's settings for
  the module; then U1 and L3, each alone, on the 64 drafting integration cases, reported.
- **Development (one CPU host).** The upgraded system routed turn by turn on the opened
  development cases: 24 drafting, 24 scheduling and 8 cross. The previous system's development
  episodes are pinned from [development 7](ASSISTANT_REPEATED_GROWTH_DEVELOPMENT7_RESULTS.md).
  The upgraded system must solve at least 21 of the 24 drafting cases (the previous system
  solved 19), lose none of the previous system's successes on any set, and keep p95 within
  180 s.
- **Confirmation (four CPU hosts), only after development passes.** Fresh sealed sets frozen
  in this declaration: 192 drafting episodes (`confirmation6`), 192 scheduling
  (`confirmation3`) and 48 cross (`cross-confirmation3`). The upgraded and the previous system
  each run all three.

## The confirmation gate and the resource budget

The upgraded system must:

- solve at least 5 more sealed drafting episodes than the previous system, net, with the lower
  95% bound on the gain per episode above zero (10,000 bootstrap samples over the eight drafting
  families);
- lose none of the previous system's successes on drafting, scheduling or cross;
- keep p95 within 180 s and at most 10% above the previous system's on the same episodes.

The upgrade adds exactly one rank-16 module, 1,048,576 parameters, and trains on one GPU host
within its allowance. Passing within that budget is A3's comparison with keeping the previous
system. Passing accepts cohort 3; with cohorts 1 and 2 that is three accepted cohorts, a new
capability and an upgrade. Acceptance is not native promotion.

## Budget

A3 has spent $91.18. Cohort 3 allows at most $10 on one A10G host, $6 for development on one
CPU host and $37.60 for the confirmation on four. That is $144.78 in total, within the $150
ceiling.
