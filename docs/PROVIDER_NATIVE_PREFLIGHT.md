# Native provider join and recovery preflight

The frozen CPU trial passed all three cases: initial service, loss of the
coordinator, and loss of a required backbone owner. Fresh provider keys joined
through native registration, restored their own committed partitions over HTTP,
and executed through authenticated HTTPS peers. Recovery preserved the original
request and graph, changed the assignment epoch, and paid each request once.

| Case | Assignment epoch at settlement | Complete numerical replays | Case duration |
| --- | ---: | ---: | ---: |
| Ordinary request | 0 | 3 | 104.81 s |
| Coordinator killed and replaced | 1 | 3 | 247.55 s |
| Backbone owner killed and replaced | 1 | 3 | 245.58 s |

The failed owners were killed after native acceptance and creation of their
execution-start marker. Each replacement used a fresh key and endpoint with an
empty model cache. The protocol restarted the pinned request; it did not recover
an in-memory activation stream. Case durations include fresh process startup,
funding, three full numerical replays and native settlement windows. They are
not measurements of LLM generation latency.

Every case paid 7 execution atoms and refunded 987 unused execution atoms. Its
separate provider reservation paid 510 atoms and refunded 490. The three
requests issued **zero tokens**. These fixture prices are not production tariffs.
All four validators agreed on the completed result. Fresh application replay
reproduced **660 headers, 84 accepted transactions and all four stored validator
states**, including collateral, audit payments and refunds.

This is a **33,760-parameter synthetic graph on one physical host under one
administrator**. Preparation uses the existing deterministic fixture's two tail
revisions. Its ordinary request emitted a terminal EOS token; the experiment
checks execution and settlement, not useful answering or accepted-LLM quality.
There were no new instances or GPUs. The 628.52-second trial used a 7 GiB memory
limit, 1.5-CPU quota and 20-minute cutoff. Every trial process stopped; the host's
two existing public services remained running.

The method was committed before execution as `71a6d7fde8fdc5c67a43424cdd7fed251e81e87e`.
The source commit that introduced the provider implementation, `7dc8dd6`, passed
all five CI checks, including **989 tests on each of Python 3.10 and 3.12**.
Later resident-model reuse and customer-quote changes have their own focused
checks and were not part of this frozen native run.

Public artifacts, each verified by complete public readback:

- [Evidence, model fixture, native blocks, four saved states and replay driver](https://dwquwt9gkkeil.cloudfront.net/research/native-expert-live-20260916/objects/8c30280e59ecabbf53bde07d8bd932a2bbabd8a95c0c2b5956783c48d3faf7a5),
  SHA-256 `8c30280e59ecabbf53bde07d8bd932a2bbabd8a95c0c2b5956783c48d3faf7a5`.
- [Matching source archive](https://dwquwt9gkkeil.cloudfront.net/research/native-expert-live-20260916/objects/abf642c08f95b2d70cb150132f31e866467c9ffe1718102c9db6ae7ea3c9ba62),
  SHA-256 `abf642c08f95b2d70cb150132f31e866467c9ffe1718102c9db6ae7ea3c9ba62`.

Verify both archive hashes and the evidence file inventory before use. With the
declared research CPU dependencies installed, application replay requires no
signing key, GPU or running native node:

```bash
PYTHONPATH=/absolute/path/to/matching-source/src \
  python /absolute/path/to/evidence/replay_provider_preflight.py \
  --home /absolute/path/to/evidence
```

The replay checks application transitions and saved states; it does not
independently authenticate CometBFT commit signatures or redo neural execution.
The separate complete numerical reports cover the three inference claims.

TODO 4–6 remain open. The accepted LLM still needs the frozen operated
recovery/load trial and complete cost accounting; chat delivery needs its
public integration. Independent administration and voting-power distribution
still require operators outside this administrator's infrastructure.
