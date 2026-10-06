# Start contributing to the assistant

NeuroShard needs people who run nodes, contribute useful demonstrations and check
whether learned changes actually improve the assistant. You can start before the
six [assistant milestones](../TODO_ASSISTANT.md) are complete. This is a contributor
preview, not a release of the Granite assistant or a promise of continuous improvement.

## Run a network worker

Follow [the public testnet guide](PUBLIC_TESTNET.md). No registration is required;
you create keys locally. Linux x86_64 and Python 3.10–3.12 are the supported node
profile. `neuroshard join` follows the native ledger and offers training work.
Assignments depend on a finite sponsor budget; a returned result earns testnet
NEURO only after settlement. The public chain still uses SmolLM2-135M-Instruct
with a small trainable adapter. Joining it does not currently train Granite.

The CPU ledger and training were recovered on September 27 after a full disk
stopped consensus. The [recovery record](NETWORK_RECOVERY_20260927.md) distinguishes
process availability, fresh blocks and actual settled work. Before joining, check
[network health](https://neuroshard.com/healthz) and
[sponsor status](https://neuroshard.com/work/status).

The source client now reports block freshness in `neuroshard doctor`, rejects new
payments against a stale ledger, and checks availability before a fresh node setup.
These client changes are not yet a new PyPI release. The deployed public gateway
also checks ledger progress before advertising inference or relaying transactions.

Running your own machine with your own keys is useful participation. Independent
consensus operation additionally requires the [hosting trial](INDEPENDENT_HOSTING.md);
another machine managed by the project does not count as another administrator.

## Contribute an assistant correction

The source client provides an offline path for complete tool-use demonstrations.
It runs on the supported Linux development environment without downloading model
weights. Install the main branch in a separate virtual environment:

```bash
git clone --branch main https://github.com/neuroshard-ai/neuroshard.git
cd neuroshard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
git rev-parse HEAD
neuroshard contribute --help
```

Start with the [fictional, two-turn example](../examples/assistant-contribution.json).
Replace it with an original example using the same document/read/calculate/date/draft
tools. Include the assistant's actual tool calls, tool-free final acknowledgement,
and expected draft for every turn. Choose a different project and fresh source
content. Do not copy evaluation questions or use private/customer records.

```bash
neuroshard contribute package \
  --input examples/assistant-contribution.json \
  --output ./orchid-contribution.json \
  --origin 'NeuroShard Apache-2.0 fictional example; demonstration only' \
  --revision v1 --public-training
neuroshard contribute verify ./orchid-contribution.json
```

The flag explicitly attests that you have the rights to publish the entire example
and allow training reuse under Apache-2.0. The command creates or uses your local
account key and writes a signed packet. **It uploads nothing and pays no tokens.**
Check the packet before sharing it in a contribution pull request. Keep account
keys out of the submission. The content root and signature bind its author,
provenance, policy and bytes; they do not prove truth or ownership of the content.

Every turn must complete its declared draft using the deterministic in-memory
tools. Rejected calls, incorrect outcomes and responses assigned to the wrong turn
are refused. No supplied code runs, and the tools cannot contact external services.
This establishes replay consistency, not whether the example is good instruction:
a reviewer must check that the question, source, actions and expected result agree.

## Review and prepare training data

Maintainers review source rights, relevance, prompt injection, correctness and
overlap with evaluations. Review applies to exact packet bytes. Create a manifest
using the receipt's example root and `sha256sum orchid-contribution.json`:

```json
{
  "format": "neuroshard-assistant-review/v1",
  "purpose": "training",
  "entries": [{
    "file": "orchid-contribution.json",
    "sha256": "REPLACE_WITH_PACKET_SHA256",
    "example_root": "REPLACE_WITH_RECEIPT_EXAMPLE_ROOT",
    "accepted_for_training": true
  }]
}
```

```bash
neuroshard contribute export --review review.json --directory . \
  --output reviewed-training.jsonl
```

The export checks signatures and replay again, rejects changed/duplicate packets,
and creates one conversation window per assistant decision. Inputs contain the
actual preceding messages and tool responses; the last assistant message is the
training target. Expected scoring fields are excluded. Train on that final target
only, not the preceding user/tool/context tokens. No training is started by export.

Exact normalized workflow development/confirmation prompts are blocked. This is
only an overlap screen: it does not detect paraphrases, contamination of other
benchmarks, false instructions, or malicious but internally consistent examples.
No unreviewed folder is ingested, and a review manifest is a maintainer's explicit
local selection, not a permissionless correctness certificate.

The next learning comparison needs committed data, prepared target-token windows,
both trained controls and its execution amendment. Contributions can supply future
training cohorts, but cannot silently enter the already frozen comparison. A
candidate must demonstrate useful behavior and preservation before changing the
served model. The [failed workspace baseline](ASSISTANT_WORKFLOW_BASELINE_RESULTS.md)
remains failed; contributor onboarding does not claim any milestone is complete.
