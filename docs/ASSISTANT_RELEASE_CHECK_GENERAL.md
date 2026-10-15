# General-route release qualification

October 14, 2026. **Declared before execution.**

The candidate served version is `a4-general` (model root `71e2835b…`). It keeps everything A3 serves:
U1, scheduling unit L2, drafting upgrade L3, A3's router, policies and plan. It adds the general route
that passed its sealed confirmation7 ([results](ASSISTANT_GENERAL_ROUTE.md)):

| route | index | unit | base | policy |
| --- | --- | --- | --- | --- |
| drafting | 1 | L3 | U1 | A3's |
| scheduling | 2 | L2 | U1 | A3's |
| documents | 3 | L9 | the unchanged parent | `documents5` |
| chat | 4 | C1 | the unchanged parent | `chat-tools/1` |

**Routing.** A new top router
([router](../config/experiments/assistant-general-router-a4.json),
[fit report](../config/experiments/assistant-general-router-a4-report.json)) decides per turn between
A3 (`a3`), `documents` and `chat`. It uses the accepted router feature with the conversation's opening
turn, exactly as the user's device computes it. A3 turns then go to A3's own router, unchanged, so A3's
routing cannot move.

The router was fitted on 2,566 training turns only. The features came from one canonical CPU host
($0.28). Results:

- 816/816 A3, 560/560 chat and 1,190/1,190 documents training turns;
- 202/202 development turns;
- every one of the 876 turns in this check's workload, as a prediction.

**Workload and pass rule** ([contract](../config/experiments/assistant-network-release-check-general.json)).
The check covers 324 paid conversations through a fresh shadow chain:

- 132 documents and 96 chat conversations from confirmation7;
- an A3 regression set of all 48 cross-capability conversations, the first 24 drafting and the first 24
  scheduling.

Each must match its accepted run in every generated input and output token, tool call and result, final
reply and (for documents) workspace snapshot, under the route it was accepted under. The first mismatch
stops the attempt. Nothing is trained or tuned, and no evaluation set opens.

**Hosts.** The two protected owner hosts each run one shadow owner. One temporary m7i.2xlarge client
hosts stage 0, and no worker holds the whole backbone. The shadow services run from their own checkout
and virtual environment (`neuroshard-release-general`), never from the live services' checkout. Every
role checks out the same committed source and refuses to run on any other bytes.

**Budget.** At most 16 hours of serving (the worker budget) and an 18-hour hard stop on the client;
$40 including the client instance. The supervisor copies evidence before stopping anything, then
retires the temporary client and the shadow units only.

**After a pass:**

1. publish the report and immutable assets;
2. upgrade validators to software with owner governance and multi-route serving;
3. hand over from the steward key;
4. put `a4-general` to the owners as the first `model_propose`.

None of that is part of this check.

## Result: passed

October 15, 2026, from commit `7f0a97d`
([report](../config/experiments/assistant-network-release-check-general-report.json)).
**All 324 paid conversations reproduced their accepted runs token for token**, through a fresh shadow
chain and two separately hosted owners:

| set | matched |
| --- | --- |
| documents (confirmation7) | 132/132 |
| chat (confirmation7) | 96/96 |
| A3 cross-capability | 48/48 |
| A3 drafting | 24/24 |
| A3 scheduling | 24/24 |

- Serving took 9.8 hours, about 109 seconds per conversation. Each owner committed 324 signed job logs.
- The temporary client was retired after evidence copy, and its instance time cost $4.32.
- The live seed, owners and auditor kept running on A3's commit throughout. The shadow units were
  stopped only after evidence copy.
- The result digest, the quality root a proposal names, is `c36e4569…`.

`a4-general` (model root `71e2835b…`) is qualified. The steps that change the live testnet (publish the
assets, upgrade validators, hand over from the steward key, and put the version to the owners' vote)
need the project owner's go-ahead. This check gives no independent-operation credit.
