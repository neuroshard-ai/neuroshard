# Decentralization status of the assistant network

October 13, 2026. This page tracks every point at which `neuroshard-assistant-1`, or the way new
units reach it, still depends on one party. It records what has changed in code and what is still
true of the live network. "In code" means implemented and tested on `research/general-route`. Nothing
here is live until validators run the new software and the steward hands over (see the order below).

## What changed in code

| Dependency | Before | Now (in code) | Tests |
| --- | --- | --- | --- |
| Who changes the served model | One steward key (the genesis faucet) signs `model_promote`, checked against an allowlist each validator keeps locally | A one-way `governance_handover`, after which only a bond-weighted owner vote (`model_propose` / `model_vote` / `model_enact`) changes the model. Weights are snapshotted when a proposal opens, there is a deposit against spam, and enactment is permissionless. The steward key is then refused by consensus, not by configuration ([RFC amendment](ASSISTANT_PROMOTION_RFC.md#amendment-1-owner-governance-of-the-serving-model)) | `test_owner_governance.py` |
| Which ledger RPC a user trusts | One RPC address. Its state was taken as given | Any number of RPCs. Each reply must match its own state root, and every RPC reporting the same height must agree, so one honest RPC exposes a lying or forked one. Submissions fail over | `test_owner_governance.py::test_chain_reads_verify_roots_and_cross_check_every_rpc` |
| Where units and checkpoints are downloaded | One GitHub release URL | Any number of mirrors. Every file is pinned by size and SHA-256, so a source serving other bytes is refused and the next one is tried. Owners can serve their own verified copies to peers (`assistant host --mirror-port`), and a holder never offers bytes that differ from the pin | `test_owner_governance.py::test_any_mirror_serves_pinned_files_and_a_wrong_one_is_refused` |
| Which routes owners can serve | Two hard-coded routes (drafting and scheduling), each an addition on U1; owners, the relay budget and fraud replay all refused any other route index | Routes come from the descriptor, up to 8. A route may be an addition on U1 or on the unchanged parent (the general route's L7 and C1). Owners, relay and replay accept any declared route, and the bank switches bases bit-exactly | `test_assistant_route_shards.py::test_a_parent_based_route_serves_and_replays_beside_update_based_routes` |
| Which tools the user's device can run | Drafting and calendar workspaces only | The documents and chat tools as well (`assistant/interfaces.py`). Each route group keeps its own workspace on the device, and the K-route turn router (`logistic-context`, with a fallback) is served | `test_general_route_serving.py` |

A3's descriptor is unaffected. Its model root (`e3aaf34d…`), route table and file list are
unchanged, and its release-check pins are now verified against the commit it was qualified at.

## What is still centralized

These are true of the live network today, in the order they should be removed.

1. **One validator.** The seed is the only CometBFT validator, so it orders every transaction,
   including governance votes. Owner governance cannot be censorship-resistant until A5: four
   operators run validators, and none holds a third of the voting power. The ledger has no
   validator-set update path yet. That is the next protocol change.
2. **The steward key still governs.** The handover is in code, not on chain. It takes effect when
   validators run software with the new transitions and the steward submits `governance_handover`.
3. **One operator runs every host.** That covers the seed, both owners, the auditor and the proof
   store (`ASSISTANT_NETWORK.md`). Peers can bond and host a shard today. None has yet.
4. **The faucet holds the whole supply.** Test NEURO has no value. A valued token would need a
   different genesis.
5. **Units are trained and evaluated on project-run cloud hosts.** Every general-route unit (L4–L8,
   C1) was trained on one A10G and judged on one CPU host. The network-native path, where
   contributions train the next module, is milestone 8 and is not built. Training on the network
   would need the same bit-exact replay and receipts that serving has.
6. **Synthetic phrasings come from one hosted model.** Claude on Amazon Bedrock wrote only the
   user-side wording of the general route's training and evaluation sets. Every assistant message
   learned from is the parent's own or our verified solver's. The prompts, every response and
   every judgment are committed, so anyone can audit what entered training. Anyone can also replace
   the generator, because the sets are inputs to deterministic case builders, not to the model.
7. **CI and git hosting gate releases.** A release check requires committed sources and a passing
   GitHub Actions run. Anyone can rerun both from the committed contract, but the check itself
   names one repository.
8. **Stage 0, routing and tools are unaudited.** They run on the user's device, which protects the
   user's documents. No owner computes them, so there is nothing to slash. Owners' shard
   computation is what replay covers.
9. **One party decides releases.** A3's release-check amendment records that the project owner made
   one agent solely responsible for the release, and the promotion key signed it. Owner governance
   replaces the signature once handed over. The decision to propose a version, and the review of its
   evidence, still rest with whoever runs the release check, until owners review it themselves.
10. **The live validator allowlist is not in the repository.** The seed's `settlement.json`
    allowlist of approved versions was written on the host. The bundled descriptors carry no
    `approved_versions`, so the repository alone cannot show what the validator accepts. Owner
    governance moves that decision into chain state.

## Order of activation

1. Release-qualify a version with the general route (L7 and C1), exactly as A3 was qualified.
   Every confirmation conversation must reproduce token for token through paid, separately hosted
   shards.
2. Validators upgrade to the software with owner governance and multi-route serving.
3. The steward proposes nothing new. It submits `governance_handover` and then the qualified
   version as the first `model_propose`. Bonded owners vote, and anyone enacts.
4. Add mirrors and RPCs to the descriptor as independent operators offer them.
5. Then A5: a validator-set update path and four independent validators.
