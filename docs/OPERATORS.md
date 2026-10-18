# Run a NeuroShard node

NeuroShard is a learned assistant served across independent machines. Its ledger runs on CometBFT and
settles payment for the work, and its owners vote in new versions of the model. The network already
grows itself: usage funds training, contributors train, owners verify and vote, and every node switches
to the new version from what the chain committed. What it lacks is **operators who are not the project**.
Today the project administers every validator and every original owner. This page is for joining.

**Status, October 18, 2026:**
- Chain `neuroshard-assistant-1`, 4 validators, more than 780,000 blocks.
- Model G2: Granite 4.1 3B with a documents unit grown by the network itself.
- Two shards, three owners. One owner was set up from these instructions alone on a bare machine.
- Test tokens only: NEURO has **no monetary value**.

## Join in one command

On Linux x86_64 with an Intel CPU that has AMX (for example AWS `m7i` or `r7i`):

```bash
curl -LsSf https://raw.githubusercontent.com/neuroshard-ai/neuroshard/main/scripts/join.sh | bash -s -- ROLE
```

The script:
- installs the pinned runtime (Python 3.12.12 through `uv`);
- clones the code and checks the live network;
- rebuilds the model version the chain serves;
- installs the role as a systemd service.

Keys are generated on your machine and never leave it. Back up `~/.neuroshard/assistant`.

| Role | Command | Needs | Open ports |
| --- | --- | --- | --- |
| Validator | `validator` | 16 GB RAM, 60 GB disk, AMX, a few hours to replay the chain | 26656, 26657, 28780, 28781 |
| Owner of a shard | `owner --shard 1` or `--shard 2` | 8 GB RAM, 8 threads, AMX | 28700, 28781 |
| Auditor | `auditor` | 16 GB RAM, AMX | none |
| Contributor | `contributor` | 32 GB RAM, AMX (it trains the 3B model) | none |
| Just try it | `chat` | 6 GB RAM, any x86_64 | none |

The CPU requirement is real. Audits and training checks replay work bit for bit, which holds only on
one execution class. A node on another CPU would compute slightly different numbers. It could then be
accused of fraud it did not commit, or accuse others wrongly.

## What each role does, earns and risks

**Validator.** Orders transactions and judges fraud proofs. It also stores proofs, recipes and
training states, and mirrors the model's pinned files.
- Joining: after the node has caught up, owners vote it in
  (`govern propose-validator --consensus-key KEY --power 1`; owners vote, anyone enacts). The chain
  keeps working while any one of four validators is down.
- Earns: nothing directly today. Validators are not paid on this test network.
- Risks: none on chain. A validator that stays offline should be voted out (power 0).

**Owner.** Bonds 5 test NEURO (from the faucet), hosts one shard and serves the paid jobs that pick it.
Every message is signed, and the owner commits a log of each job.
- Earns: an equal share of each job's price for the positions it served, minus a 10% growth levy
  (owner-voted) that funds training. A typical documents conversation pays each owner about 0.08 NEURO.
- Votes: on new model versions, validators and the levy, in proportion to the bond.
- Risks: the whole bond, if an auditor proves the owner sent wrong work. On October 16, an honest owner
  was slashed because the auditor and validator held an older version's weights. That is fixed: they
  now judge only with the version a job ran. It shows the risk is not theoretical.

**Auditor.** Downloads owners' signed logs, replays them and proves faults on chain.
- Earns: half of a proven cheater's bond, nothing for clean audits.
- Risks: a challenge deposit (1 NEURO) that is forfeited if the proof fails.
- Run it on its own machine: replaying on an owner's CPUs slows that owner by up to 40%.

**Contributor.** Trains segments of growth jobs the network funds from its pool, and spot-checks other
contributors' steps.
- Earns: the job's reward for the steps it commits.
- Risks: a 2 NEURO bond per segment, returned when the job settles and slashed if a step is proven
  wrong.
- The agent refuses to build on a state it cannot reproduce.

**Owners deciding what the network learns.** A recipe (training data plus settings) needs an owner vote
before the pool funds it. Owners' agents verify each recipe themselves before voting:
- template recipes are rebuilt from the published builders;
- corpus recipes are checked by replaying every conversation and confirming each fact appears in a
  cited document.

A new version goes live only if it beats the served version on a beacon-sealed evaluation that includes
real requests, and owners reproduce that evaluation bit for bit.

## What still depends on the project

- **Administration:** every validator, the two original owners, the auditor and the faucet run on
  machines the project administers. Independent operators are the point of this page.
- **Single services:**
  - The faucet: one, on the seed.
  - Model files: the pinned files are mirrored by every validator, so GitHub is no longer required.
    The base model's tensors still come from Hugging Face at a pinned revision.
  - The ledger RPC is served by all four validators, and clients cross-check them.
  - Proof, recipe and state stores run on every validator and fill each other's gaps.
- **Economics:** NEURO is a test token. Running a node costs real money and earns nothing of value, so
  for now the reason to join is to help test.

## Talk to us

Open a GitHub issue titled `Operator: <name>` with your role, region and machine type. Do not send keys.
Ask in the same thread if something in `scripts/join.sh` fails on your machine: that is the most useful
report we can get.

## The website

[neuroshard.com/network](https://neuroshard.com/network) shows this network live: blocks, validators and
whether they signed, owners and their bonds, every model version and the vote that enacted it, the growth
pool, and settled conversations and growth jobs. It reads several validators' RPCs through a read-only
explorer (`python -m neuroshard.assistant.explorer`) that holds no keys. Anyone can run the same explorer
against their own node. `neuroshard.com/join.sh` serves `scripts/join.sh` from this repository.
