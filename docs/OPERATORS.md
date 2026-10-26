# Run a NeuroShard node

NeuroShard is a learned assistant served across independent machines. Its ledger runs on CometBFT and
settles payment for the work, and its owners vote in new versions of the model. The network already
grows itself: usage funds training, contributors train, owners verify and vote, and every node switches
to the new version from what the chain committed. What it lacks is **operators who are not the project**.
Today the project administers every validator and every original owner. This page is for joining.

**Status, October 26, 2026:**
- Chain `neuroshard-assistant-1`, 5 validators, more than 1,300,000 blocks.
- Model: Granite 4.1 3B with a documents unit grown by the network itself (version `a4-general` and its
  owner-enacted growth).
- Two shards, three owners, all on **`pilot-2-rc2`**, which passed a 72-hour soak under injected faults: 233 of
  233 paid conversations settled on chain. Every node is still administered by the project.
- The current release is **`pilot-2`**: `pilot-2-rc2`'s serving code plus three reviewed client fixes (stage 0
  always uses 8 threads; a warning before payment on CPUs the replies weren't evaluated on; an optional fast
  start, off by default). Owners on `pilot-2-rc2` and `pilot-2` serve together.
- An experimental, project-operated pilot. Test tokens only: NEURO has **no monetary value**.

**Who we're inviting now:** a first, small group of **owners**, **auditors** and **users**. **Validators and
training contributors are on hold**: see *Roles on hold* below.

## Before you start

- **Use the release, not `main`, for both the installer and the code.** The commands below fetch `join.sh`
  from the `pilot-2` tag and pass `--branch pilot-2`, so what runs is exactly the reviewed release. `--branch` alone
  pins only the code the installer checks out, not the installer itself. The maintainers will name later
  releases in your issue.
- **Your conversations aren't confidential from the network's operators.** The owners serving a conversation
  compute on it, and the activations they keep in their signed logs can practically be turned back into text.
  Auditors can download those logs. Use test documents, not personal or confidential ones.
- **Joining as an owner bonds 5 test NEURO and starts serving paid jobs at once.** You can leave at any
  time with `neuroshard assistant leave` (see *Leave the network* below).
- **Check your machine first.** It installs nothing:

  ```bash
  curl -LsSf https://raw.githubusercontent.com/neuroshard-ai/neuroshard/pilot-2/scripts/join.sh | bash -s -- check
  ```
- **Try it as a user first, if you like.** `chat` installs the same runtime and code, starts no service and
  bonds nothing. On a fresh m7i.2xlarge (October 22) it installed in 17 s, and the first answer arrived
  112 s after starting `chat`, including the stage download, model load, faucet grant and admission.
- **Users on CPUs without AMX** (other Intel, AMD): replies were evaluated on Intel CPUs with AMX. Elsewhere,
  `chat` says so and asks before it downloads anything or pays; scripts pass `--experimental-cpu`. Replies may be
  worded differently there, and their quality and speed aren't validated.

## Join in one command

Owners and auditors need Linux x86_64 with an Intel CPU that has AMX (for example AWS `m7i` or `r7i`):

```bash
curl -LsSf https://raw.githubusercontent.com/neuroshard-ai/neuroshard/pilot-2/scripts/join.sh | bash -s -- ROLE --branch pilot-2
```

`ROLE` is `owner --shard 1`, `owner --shard 2`, `auditor` or `chat`.

The script:
- installs the pinned runtime (Python 3.12.12 through `uv`);
- clones the code and checks the live network;
- rebuilds the model version the chain serves;
- installs the role as a systemd service.

Keys are generated on your machine and never leave it. Back up `~/.neuroshard/assistant`.

| Role | Command | Needs | Open ports |
| --- | --- | --- | --- |
| Owner of a shard | `owner --shard 1` or `--shard 2` | 8 GB RAM, exactly 8 threads, AMX, 20 GB disk | 28700, 28781 |
| Auditor | `auditor` | 16 GB RAM, AMX | none |
| Just try it | `chat` | 8 GB RAM (the installer checks this), Linux x86_64; AMX recommended | none |

The installer still accepts `validator` and `contributor`. Don't use them yet (see *Roles on hold*).

The CPU requirement is real for owners and auditors. Audits replay work bit for bit, which holds only on
one execution class. A node on another CPU would compute slightly different numbers. It could then be
accused of fraud it did not commit, or accuse others wrongly.

## What each role does, earns and risks

**Owner.** Bonds 5 test NEURO (from the faucet), hosts one shard and serves the paid jobs that pick it.
Every message is signed, and the owner commits a log of each job.
- Earns: an equal share of each job's price for the positions it served, minus a 10% growth levy
  (owner-voted) that funds training. A typical documents conversation pays each owner about 0.08 NEURO.
- Votes: only once admitted. Bonding and serving need no permission, but voting on new model versions,
  validators and the levy requires the existing members to admit you (`govern propose-member`; see
  *What still depends on the project*). Admitted members vote in proportion to their bond.
- Risks: the whole bond, if an auditor proves the owner sent wrong work. On October 16, an honest owner
  was slashed because the auditor and validator held an older version's weights. That is fixed: they
  now judge only with the version a job ran. It shows the risk is not theoretical.

**Auditor.** Downloads owners' signed logs, replays them and proves faults on chain.
- Earns: half of a proven cheater's bond, nothing for clean audits.
- Risks: a challenge deposit (1 NEURO) that is forfeited if the proof fails.
- Run it on its own machine: replaying on an owner's CPUs slows that owner by up to 40%.

**User.** Runs `chat`: stage 0 of the model on your machine, the rest on the owners, paid in test NEURO from
the faucet. Unused escrow is refunded when the job settles.

## Roles on hold

**Validators and training contributors are not part of this pilot yet.** Please don't run them, even though
`join.sh` still accepts `validator` and `contributor`.
- **Validators:** the network is moving to a validator release whose fraud-proof handling changed. That needs its
  own mixed-version qualification (judging growth proofs with old and new validators together) before any new
  or upgraded validator joins. Also, don't provision the parent model onto any additional `pilot-1` validator.
- **Contributors:** outside training contribution depends on that same growth-fraud enforcement, which the
  serving soak didn't test.

We'll open these roles in a later release, with their own instructions.

## After you join (owners)

**Confirm it is serving.**
- `journalctl -u neuroshard-owner -f` shows `Shard N of … is serving on 0.0.0.0:28700`.
- `neuroshard assistant status` (with the same `--home`) lists your endpoint among the shard's active
  owners.
- From another machine, port 28700 must be reachable. If your machine is behind NAT, pass
  `--public-host HOST`: the script otherwise uses the address `checkip.amazonaws.com` reports.

**Upgrade to a newer release.**
- Rerun the command with the new release in **both** places: the tag in the `join.sh` URL and `--branch`. The
  script fetches that release, checks it out in
  `~/.neuroshard/release`, reinstalls it into `~/.venvs/neuroshard`, and restarts the service.
- Check that the service runs the new code: `git -C ~/.neuroshard/release describe --tags` names the
  release, and the journal shows a fresh `is serving` line.
- It refuses if you changed files in `~/.neuroshard/release`.
- Upgrade between jobs: `neuroshard assistant status` shows `open jobs`, and a job cut off mid-conversation
  expires and is refunded to its user.

**Roll back.** Rerun the command with the previous release in both the `join.sh` URL and `--branch`
(`pilot-2-rc2` is the previous serving release).

**Recover from a crash or reboot.**
- The service restarts on failure and at boot.
- Your keys and job logs are in `~/.neuroshard/assistant`. A job your owner started before a crash is
  never served twice: it is committed from its log, or it expires and is refunded.
- Restore that directory from backup onto a new machine to keep the same owner identity.
- Never run two machines with the same owner home at once.

**Leave the network.** Two steps, with the same `--home` and `--shard` the service uses:

```bash
sudo systemctl disable --now neuroshard-owner        # stop serving first
neuroshard assistant leave status --shard N --home ~/.neuroshard/assistant
neuroshard assistant leave unbond --shard N --home ~/.neuroshard/assistant
# after the challenge window (600 blocks, about 10 minutes), while the bond can still be challenged:
neuroshard assistant leave withdraw --shard N --home ~/.neuroshard/assistant
```

- `unbond` stops new jobs from naming the owner. The ledger refuses it while a job naming this owner is
  unsettled: wait for those jobs to settle (`leave status` lists them).
- `withdraw` returns the bond to your account once the window has passed. A proven fault in that window
  would still slash it.
- A withdrawn log key can never bond again. To host again, start a new `--home`.

**Known failures and what they mean.**
- `this log key is no longer bonded; start a new owner home`: this owner was slashed or unbonded. Its bond
  is gone. Start a fresh `--home` to bond again.
- `the ledger serves another version than this owner holds`: the owners enacted a new model version.
  The host follows enacted versions and rebuilds them from committed state. If this persists, restart the
  service (`sudo systemctl restart neuroshard-owner`).
- A client reporting `the network stayed busy until the deadline; nothing was funded`: every owner of a
  shard was serving. That is the designed overload behaviour, not a fault of your node.
- Model-file downloads failing: the pinned files are mirrored by every validator. Retrying usually
  succeeds, since another mirror is tried.
## Run a faucet (optional)

New users and operators get test NEURO from a faucet. Any operator can run one from its own account, so
joining does not depend on the project's seed:

```bash
neuroshard assistant faucet --home ~/.neuroshard/faucet --port 28782
```

- It grants 20 NEURO per request to accounts holding less than 10, once per account per hour and five times
  per source address per hour (`--amount`, `--floor`, `--per-address`). It refuses to start without enough
  NEURO to grant.
- It only transfers its own account's balance, so nothing about it has to be trusted. Fund that account
  first, from your own balance or another faucet.
- Open the port, and name it in your operator issue. Clients try the descriptor's `faucet`, then its
  `faucets` list, then any URLs in `NEUROSHARD_FAUCETS` (comma-separated), until one grants.

## What still depends on the project

- **Administration:** every validator, every owner, the auditor and the faucet run on machines the
  project administers. Independent operators are the point of this page.
- **Governance:** voting is *admitted*. The founding members are the owners active at block 822,000, all
  project-run. Bonding and serving need no permission; voting needs members to admit you
  (`govern propose-member`).
- **Single services:**
  - The faucet: the descriptor names one, on the seed. Clients fall through to any other faucet they are
    given (see *Run a faucet*), but no independent faucet is listed yet.
  - Model files: the pinned files are mirrored by every validator, so GitHub is no longer required.
    The base model's tensors still come from Hugging Face at a pinned revision.
  - The ledger RPC is served by all five validators. Clients verify each state read against the pinned
    validators (light client).
  - Proof, recipe and state stores run on every validator and fill each other's gaps.
- **Economics:** NEURO is a test token. Running a node costs real money and earns nothing of value, so
  for now the reason to join is to help test.

## Talk to us

Open a GitHub issue titled `Operator: <name>` with your role (owner, auditor or user), region and machine type.
Do not send keys. Include the output of `join.sh check`. We're starting with two or three operators, so we may ask
you to wait for a later group.
Ask in the same thread if something in `scripts/join.sh` fails on your machine: that is the most useful
report we can get.

## The website

[neuroshard.com/network](https://neuroshard.com/network) shows this network live: blocks, validators and
whether they signed, owners and their bonds, every model version and the vote that enacted it, the growth
pool, and settled conversations and growth jobs. It reads several validators' RPCs through a read-only
explorer (`python -m neuroshard.assistant.explorer`) that holds no keys. Anyone can run the same explorer
against their own node. `neuroshard.com/join.sh` serves `scripts/join.sh` from `main`, which moves. For a
reproducible install, use the tagged URL above.
