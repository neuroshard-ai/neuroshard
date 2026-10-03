# Development diagnostic of the small module

Declared on September 30, 2026, after the [compositional study](ASSISTANT_EXPERIENCE_COMPOSE_RESULTS.md),
before any diagnostic run. Authorized by the project owner.

## Question

In three studies, the small module lost parent successes only on latest and
scope development cases whose correction ends "Also move the resulting due date
one calendar day later". The update control solved most of them. The studies
recorded only pass or fail. How exactly does the small module fail these
cases, and how fast is the unrouted serving path a third attempt would use?

## Method

- **Systems.** The update control and the small module from the compositional
  study, attached from their pinned checkpoints and served without routing under
  the prefix cache.
- **Cases.** The 24 opened development cases. The canonical parent result is
  the reference; no parent host runs.
- **Runtime.** One canonical CPU host per system (r7i.4xlarge, eight threads,
  pinned packages), through the
  [confirmation execution](../config/experiments/assistant-experience-confirmation-execution.json)
  in its development phase.
- **Output.** Every episode transcript and its wall time, and the second
  confirmation gate's latency checks computed on these episodes.

## Limits

This is a diagnostic. It has no gate and earns no credit, and the third
confirmation split stays sealed. One run per system, about $3 in total.
