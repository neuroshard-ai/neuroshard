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
   including governance votes. The validator-set update path now exists in code (`validator_propose`:
   owners vote validators in and out, and CometBFT applies the updates; tested on real CometBFT).
   A5 still needs four independent operators to run validators, with none holding a third of the
   voting power.
2. **Done: the steward key no longer governs.** The live chain runs owner governance (validator upgraded
   October 15, 2026). The steward handed over at block 491756, and consensus refuses its `model_promote`.
   `a4-general` was the first owner-voted version, enacted at block 491799. Both owners are still the
   project's, so the vote is the project's until independent owners bond.
3. **One operator runs every host.** That covers the seed, both owners, the auditor and the proof
   store (`ASSISTANT_NETWORK.md`). Peers can bond and host a shard today. None has yet.
4. **The faucet holds the whole supply.** Test NEURO has no value. A valued token would need a
   different genesis.
5. **Units are trained and evaluated on project-run cloud hosts.** Every general-route unit (L4–L9,
   C1) was trained on one A10G and judged on one CPU host. The network-native path is now in code
   ([growth on the network](NETWORK_GROWTH.md)): sponsors escrow a reward against a recipe,
   contributors train bonded segments, any step can be proven wrong on chain, and contributors are
   paid per step. It is not deployed, and no unit has been grown this way yet.
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

## October 15, 2026: who holds the sealed test

- **Before:** the project wrote every sealed split and decided when to open it.
- **Now (in code):** a growth job's sealed set is derived from a drand round that did not exist when the
  unit's training state was fixed. Anyone recomputes it, and owners vote only on evaluations they reproduce
  themselves ([growth on the network](NETWORK_GROWTH.md#sealed-tests-from-public-randomness-and-the-autonomous-loop)).
- **Still open:** the recipe's families and minimum are chosen by its sponsor, and owners still run the
  evaluation on their own hardware.

## Four validators, and a node joined from public instructions alone

October 18, 2026.

**Four validators.** Three new nodes ran `neuroshard assistant validator`. Each:
- took the genesis and the seed's address from the public RPC;
- generated its consensus key on its own machine;
- replayed the 769,000-block chain from block 1, at about 57 blocks per second. Committed blocks'
  proof verdicts are taken as committed, so syncing needs no model.

They then held the audited shards of the served version. Owners proposed the three consensus keys
and voted. The clean-room owner's vote was needed: the project's two owners hold 10 of the 15 million
bonded, just short of two-thirds. All three proposals were enacted by height 769,716. CometBFT now runs
4 validators at power 1 each, in us-east-1a, 1b, 1c and 1f.

- **Losing a validator.** With one stopped for 60 seconds, the chain made 31 blocks (normally ~60) and
  the stopped node rejoined by itself.
- **Losing the seed's RPC.** Every validator now serves a public RPC, and the bundled network
  descriptor lists all four. Clients cross-check them and keep working with the seed unreachable.

**A node joined from public instructions alone.** A bare Ubuntu 24.04 machine started with none of
the project's images, environments or keys. It followed `docs/ASSISTANT_NETWORK.md` and then:
- installed the pinned runtime;
- rebuilt the served version, G2, from committed state;
- bonded as a second owner of shard 2 and served live jobs, audited clean and paid 88,769 atoms
  each;
- voted on validators from its own account.

From that machine a G2 documents turn took 34 s.

**What the clean-room run fixed.**
- The instructions assumed a system Python 3.12 with `venv`. Ubuntu 24.04 ships 3.12.3 without it,
  and the runtime needs 3.12.12, so the instructions now use `uv`.
- `validator` and `seed` stop their node when the command exits, so they are run as services.
- A node that falls out of fast sync (because its only peer reloaded) restarts to resume it.

**What went wrong.** The three enactments were meant to land in one block, so the chain would go from
1 validator straight to 4. The node checks each account's nonce against committed state, so only the
first was admitted. For about 40 blocks the chain ran on 2 validators, both required, before the other
two enactments were sent from separate accounts. `govern enact` now enacts one proposal per block,
and the docs say to send enactments from separate owner accounts.

**Seen in passing.** Asked for the baseline quantity in the latest approved Cedar 6581 plan, G2
answered 58 (revision 1) rather than 61 (revision 2), citing the older document. Picking the current
version is still a failure mode on these drafting workspaces.

**What this does not change.**
- The project administers all four validators, both original owners, the auditor and the faucet. The
  clean-room machine shares the project's AWS account. Decentralization in fact needs operators who
  are not the project.
- The faucet and proof store are still single services on the seed, and model files are hosted on
  GitHub.
- Owners, auditors, contributors and judging validators need the same CPU class (Intel with AMX) so
  that work replays bit for bit.
- NEURO is a test token with no value, so nobody has a reason to run a node except to help test.
