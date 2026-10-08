# Public assistant interface

**Status: declared, not a public service. A6 remains open.**
Requires the accepted A1–A4 serving version. Independent operators (A5) and a
public soak are still missing. This is not an announcement that the assistant is
available.

Machine-readable session: `neuroshard.assistant.public`
(`neuroshard-assistant-public/1`). Current serving name `a3-cohort3`; rollback
target `a2-u1`.

## One version

A version binds policy, tool schemas, limits and the named modules (U1, L2, L3).
Promotion evaluates that whole bind. The previous name stays addressable so a
bad promotion can roll back without rewriting transcripts.

Chat, tool calls and workspace drafts share one session. Personal notes sit in a
separate memory map with a 32-key / 4 KiB-per-value bound. Tools still execute
only on the in-memory workspace; `external_effects` stays false.

## Consent

Default consent is deny:

- conversations are not training data
- nothing is shared outside the session
- no external action is authorized

`export()` returns the version and consent only, unless the caller opts into
sharing. `training_export()` raises unless training is opted in. The ledger field
is always empty: a session is not a transaction.

## What this does not do

It does not open a public endpoint, take payment or complete A6. Quality
monitoring, funding and the operating soak wait on A5. Join and recovery for
operators are in [ASSISTANT_JOIN_RECOVERY.md](ASSISTANT_JOIN_RECOVERY.md).
