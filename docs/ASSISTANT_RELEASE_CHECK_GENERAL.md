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
