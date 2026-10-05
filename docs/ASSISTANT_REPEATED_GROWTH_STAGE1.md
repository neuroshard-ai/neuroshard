# A3 stage 1: learning to schedule without relearning drafting

Declared on October 3, 2026, after the [stage-0 report](ASSISTANT_REPEATED_GROWTH_STAGE0_RESULTS.md)
and before any scheduling rollout or training. Authorized by the project owner,
who chose turn-level routing with each unit's own tool set and a $100 ceiling. The
[contract](../config/experiments/assistant-growth-stage1.json) pins every rule below.

## Why routing per turn, with tool sets

Stage 0 showed that three more tool definitions in every prompt cost the accepted
version four drafting successes. An assistant that keeps gaining capabilities
cannot show every tool in every prompt, so each learning unit keeps the interface
it is served with:

- **Drafting turns** go to the version's drafting unit with the drafting
  instruction and tools, exactly as A2 accepted them.
- **Scheduling turns** go to its scheduling unit with the calendar instruction
  and tools.
- **A conversation can switch at a follow-up.** One calendar workspace runs
  every call and keeps the history. The "save a draft, then schedule a handoff"
  tasks need this, since their first message is pure drafting.

A selector reads the frozen parent's state at each user message, rendered alone,
and picks the route. It is fitted as A2's gate was, from success rates on
integration episodes and never from evaluation goals. Each route runs every
integration case alone. A turn selects the scheduling route only where that
route succeeded more often on that round, weighted by the difference. Otherwise
it keeps the drafting route, the incumbent, as A2 kept the parent; a tie counts
for drafting with the weight of one episode. Anchor prompts stay with the parent.

## Three versions under one budget

| Version | Drafting turns | Scheduling turns |
| --- | --- | --- |
| Separate update | U1, the accepted update | U2, the update trained further from U1 |
| Separate module | U1 | L2, a new low-rank module on top of U1 |
| Shared | U2, replacing U1 | U2 |

U2 comes from one training run, so the separate-update and shared versions
differ only in whether U1 is kept for drafting. That isolates A3's question:
does keeping the earlier unit retain drafting better than updating shared
weights? L2 trains on the same mixture, schedule and steps.

## Collection

The accepted version samples the 256 scheduling and 32 cross training cases
under the calendar policy, eight times each. Stage 0 found almost no natural
successes, so cases without one get eight coached retries. The card is a fixed
scheduling procedure with no values: list each attendee's busy times, try
09:00, the user's earliest start and every busy end in order, and book the first
start whose whole meeting is free. Accepted conversations are stored without
the card.

A coached trajectory is kept only if the sampler, without the card, finds it no
less likely per token than the least likely natural success of this collection.
To keep that reference populated, 64 drafting training cases are also sampled
twice under the drafting policy, for this check only. At most four trajectories
are kept per case.

## Training

Each update draws four scheduling or cross trajectories, two of A2's pinned
verified drafting trajectories, and two of A2's parent replay items. Every
sequence is encoded with the tools of the interface it was produced under. The
schedule is 128 updates: the update at learning rate 1e-5 from U1's tensors with
a fresh optimizer, and the rank-16 module at 3e-4 on top of U1.

## Development (opened data)

Each version is routed per turn on the canonical CPU runtime over the 24
scheduling, 8 cross and 24 drafting development cases. The candidate is the
separate version with more scheduling and cross successes; a tie goes to the
low-rank module, which is cheaper to distribute. The candidate needs:

- at least 18 of 24 scheduling cases and 6 of 8 cross cases;
- every one of the accepted version's 19 drafting successes;
- p95 at most 180 s, including every selection pass.

All three versions are reported.

## Confirmation (sealed, once)

Only after the candidate passes, three sealed sets open once: 192 scheduling,
48 cross and 192 drafting cases (`confirmation4`). Four systems run on them:

- the candidate, on all three sets;
- the shared version, on drafting;
- the accepted version as accepted, on drafting;
- the accepted version under the calendar interface, on scheduling and cross, as
  the previous system.

The candidate needs at least 154 of 192 scheduling successes with 16 in each
family, and at least 36 of 48 cross successes. It must also be at least 20 above
the previous system, with a family-bootstrap lower bound above zero. It must lose
none of the accepted version's drafting successes, with p95 at most 180 s.

A3's comparison is decided on the same data. The separate candidate must lose
fewer of the accepted version's drafting successes than the shared version.

## Outcomes and budget

- **Pass:** scheduling becomes the assistant's second capability.
- **Fail:** a second round may be declared within the remaining budget.

| Step | Allowance |
| --- | --- |
| GPU host | $21 |
| Three development hosts | $18 |
| Five confirmation hosts | $47 |

The stage-1 ceiling is $100.
