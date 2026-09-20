# Operated alpha deployment result

**GPU serving closed on September 20, 2026 at the owner's request.** The seven
GPU hosts were retired after closing admission, draining work and preserving
recovery records. The four ledger hosts remain scheduled for expiry and refunds.
See the [retirement record](../config/experiments/operated-alpha-retirement.json)
and [historical joining guide](JOIN_ALPHA.md).

The accepted growing graph was served through a source-installed CLI alpha.
This was a finite service operated by one
administrator on seven GPU hosts and four separate ledger hosts. AWS participants
have separate keys and machines; they do not establish independent ownership.
No physical host contains the complete backbone.

The original ten-request gate passed: two warm-ups, six ordinary requests with
two concurrent customers, and two automatic provider-process recovery cases.
All eight measured ordinary/recovery responses and neural outputs exactly match
the preceding accepted-model service trial. This deployment runs no new training
and makes no new model-quality claim.

## Measured behavior

| Request | First visible text | Generation complete | Native settlement |
| --- | ---: | ---: | ---: |
| client-package | 34.02 s | 37.04 s | 99.68 s |
| retained-directory | 36.51 s | 40.14 s | 161.51 s |
| ordinary-general | 34.30 s | 49.34 s | 168.09 s |
| constrained-general | 33.15 s | 34.75 s | 99.62 s |
| multiturn-context | 36.21 s | 38.28 s | 111.47 s |
| combined-question | 42.17 s | 46.99 s | 180.27 s |

First-visible and generation timing includes atomic native admission. Settlement
includes the batch's end-to-end time. The frozen limits were 45 seconds for first
visible text, 90 seconds for generation and 1,200 seconds for ordinary settlement.
Each of the six ordinary cases passes; this small sample is not a population
latency guarantee. Full verification is substantially slower than draft delivery.

Automatic process-loss recovery settled once within the unchanged 1,500-second
limit: coordinator-loss: 446.36 seconds and backbone-loss: 504.27 seconds. The driver stopped a provider service,
preventing systemd from immediately restarting it. The recovery controller and
native assignment selected a previously advertised replica on another host.
The request, price ceiling and graph stayed pinned. These are process-loss
checks, not a physical-host or independent-operator availability soak.

A fresh observer synchronized from the public genesis and peer endpoints. A new
wallet obtained 500 sponsored trial NEURO; retrying the grant returned the same
transfer. Its first chat deliberately timed out, then resumed the same request
without signing another payment. The driver first requested an unsupported
1,200-second CLI wait; that rejected invocation is preserved, followed by a
600-second resume of the same admission. The second turn retained the requested phrase:
`violet kettle`. Both paid turns settled, provisional text was
observed, and the graph/tokenizer-pinned conversation contains four messages.
The first reply invented an irrelevant definition of “violet kettle.” That
poor answer remains in the published transcript. This check establishes context
retention and payment behavior, not broad answer quality.

## Evidence and reproduction

The snapshot reproduces **3,369 headers and
697 accepted transactions**, with zero rejected transactions, against all four
saved validator states. There are **36 complete numerical
replays across 12 claims**, covering the deployment gate
and onboarding. Serving issued **zero** new tokens. The final snapshot
state is `e8c9ceaac60be926c10a713bc1075e1532daaa6ec9f54207604dcb318dc77bf7`.

- [Deployment-gate ledger, numerical reports and replies](https://dwquwt9gkkeil.cloudfront.net/research/native-expert-live-20260916/objects/6579c3fef23c36f109711e3f8e757bf82509d5fda5a62ec2b15a6ecced33f679)
  — SHA-256 `6579c3fef23c36f109711e3f8e757bf82509d5fda5a62ec2b15a6ecced33f679`.
- [Exact source](https://dwquwt9gkkeil.cloudfront.net/research/native-expert-live-20260916/objects/7484da5237e1295cb6ccfb470136cb4ee3d349ee73247cb62ab5e035605c2362) — SHA-256
  `7484da5237e1295cb6ccfb470136cb4ee3d349ee73247cb62ab5e035605c2362`; commit `52ec72f49b5e77b682bc5e48174fdb9f9544ca25`.
- [Pinned public access descriptor](https://dwquwt9gkkeil.cloudfront.net/research/native-expert-live-20260916/objects/7a0a49c8d9f8731145fc88078cd3c2fcbd526d4483d63c29e3b1b8e490c40bfd) — SHA-256 `7a0a49c8d9f8731145fc88078cd3c2fcbd526d4483d63c29e3b1b8e490c40bfd`.
- [Machine-readable result](../config/experiments/operated-alpha-results.json).

After downloading and checking the evidence checksum, extract it into an empty
folder. From the pinned source with the documented CPU dependencies, replay it:

```bash
ATEN_CPU_CAPABILITY=default MKL_ENABLE_INSTRUCTIONS=SSE4_2 \
  PYTHONPATH=src:scripts python scripts/replay_provider_preflight.py \
  --home /absolute/path/to/extracted-evidence
```

Application replay verifies transitions against saved states, header linkage and
application hashes; it does not independently verify CometBFT commit signatures.
The separate joining observer follows native consensus from genesis. Numerical
reports represent three actual full replays under one administrator's audit keys;
they are not proof of independent auditing.

## Availability, funding and limits

Admission and GPU service **closed early on September 20, 2026**. The original
maximum deadlines were September 22 at 11:09 UTC for admission and 12:39 UTC for
GPUs. This serving-only window was retired to preserve resources for further
assistant-learning research; no new learning result is claimed. The ledger remains
scheduled through **September 26, 2026 at 23:49 UTC** for expiry/refunds, then private
validator signing state and block stores are preserved before retirement.
Final private provider-wallet backups were preserved before GPU retirement.
The scheduled ledger backup and absolute ledger retirement timers remain enabled.
Private recovery records are excluded from public evidence.

The aggregate ceiling is **$800**. The complete-window planning estimate at
publication is **$731.82**,
including all 60 GPU hours, seven ledger days, a worst-case CPU bursting allowance,
disks, networking, controller use and failed-deployment reserves. This is a
conservative forecast, not an invoice. New capacity closes at the $750 watch
threshold, leaving $50 for draining admitted work. Starter credits are finite
sponsor transfers with no demonstrated market price or Sybil-proof distribution.

The service has two simultaneous request slots and a 64-token response cap.
Prompts, prior context and settled responses are public ledger data. Draft text
is provisional until settlement. Full replay remains expensive. The website and
PyPI 0.4.0 client still use the older chain; use the pinned source joining guide.

[Earlier failures and their declared repairs](OPERATED_ALPHA.md) remain recorded:
preparation/heartbeat acknowledgement, the GPU library path, exhausted CPU credits,
a strict 45.016-second latency miss and a check against a lagging validator. No
failed result was rounded into a pass. The accepted weights, tokenizer and neural
behavior were preserved throughout these deployment repairs.

This completes item 6's bounded operated-alpha criterion. Item 4 remains open
for independently administered hosting, its declared availability soak and the
remaining permissionless admission/resilience evidence. Prior learning criteria
remain bounded demonstrations, not proof of ChatGPT-level general capability.
