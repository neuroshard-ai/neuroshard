# Current release gates

These gates apply to the native LLM profile in 0.4.0. Historical Docker/tracker/PoNW checks belong to retired prototypes; they are not evidence for this chain.

## Evidence required for the experimental release

- Exact model, tokenizer, source, dataset and three-step numerical commitments agree on the two tested hosts.
- Four native validators and a fresh worker agree on blocks; training earns NEURO and the worker spends earned NEURO on a replay-verified response from the promoted adapter.
- Accounting rejects locked-fund spending, wrong provider/output, replay, malformed envelopes and queue overflow; expiry refunds the budget without minting money.
- Ingestion survives interrupted upload/publication and refuses corrupted content or unavailable S3 access; old mutable objects remain preserved.
- A wheel installed outside the source checkout supports the minimal CLI; onboarding initializes a separate node without registration. The public PyPI version must be independently checked after upload.
- Browser signatures verify with native Python; zero balance cannot pay; uncertain submission retains the request ID. Site/docs build, navigation and failure behavior pass.
- Pinned services, public TCP peer, native training, provider settlement and the bounded data timer are checked after deployment. Restart tests preserve progress and budgets.

Measured results and their dates belong in [LLM experiments](LLM_EXPERIMENTS.md), including failures and limitations. Passing a fixture test is not a live-network result; an uploaded package is not a verified fresh install.

## Still required before a production network claim

Independent validator ownership and bootstrap operators; extended public load and Byzantine/network-partition testing; sustainable sponsor/work allocation and provider discovery; robust evaluation against overfitting and poisoning; economical verification covering forward, backward and optimizer; reliable data/history availability; independent security review; transparent protocol/release/checkpoint governance and a justified long-term monetary policy.

The initial four validators share one operator across two hosts. Full replay duplicates neural work. The four public validation sequences are too small to establish broad quality. These remain limitations even when every experimental release check passes.
