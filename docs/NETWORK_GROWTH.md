# Growth on the network

October 13, 2026. Status: implemented and tested in code on `network/validators-growth`. Not deployed.

The high-level design has the assistant grow through contributions. Contributors train new units,
the network checks the work, owners decide what is served, and the network switches version while
keeping the previous one for rollback. Until now every unit was trained by the project on its own
cloud hosts. This is the protocol that lets anyone do that work and be paid for it, with fraud
provable on chain.

## The loop

1. **A sponsor opens a growth job** (`growth_open`) and escrows a reward against a recipe. The recipe
   fixes the arm, spec, data, seed and schedule, and the model it grows. The job names
   `recipe_root`, the digest of the start state, the number of optimizer steps, and the segment
   length. The recipe file is published, so anyone can train it or check it.
2. **Contributors train segments in order** (`growth_commit`). Each commit posts a bond and the digest
   of the full training state after every step in it: trainable tensors and optimizer moments, as
   raw bytes. The digests chain from the previous segment's last root. Contributors need only the
   committed state bytes and the recipe. `neuroshard assistant grow train` does it from the
   contributor's own keys.
3. **Anyone checks any step** (`growth_challenge`), with a deposit and the content address of a proof
   bundle. The bundle holds the committed state before the disputed step. Validators fetch it
   before `growth_prove`.
4. **Validators judge one step, never a whole run** (`growth_step`). They load the state, check that
   it hashes to the committed root, re-execute that single step under the recipe, and compare the
   result with the committed root. A validator without the recipe or the bundle gives no verdict;
   it never rejects. A proven fault slashes the segment's bond, half to the challenger and half
   burned. That segment and every later one reopen, and the later bonds are returned. A challenge
   whose proof never verifies forfeits its deposit.
5. **The job settles** once every segment has passed its window. Contributors are paid per step they
   committed and get their bonds back. The result records the job's `final_root`, the trained unit.
6. **Owners decide whether it is served.** A grown unit is a candidate, not an upgrade. Its quality is
   judged as before, by a sealed confirmation that anyone can rerun from the committed contract.
   The version is then put to an owner vote (`model_propose` / `model_vote` / `model_enact`). Nothing
   in the growth protocol judges quality. Settling a training job proves the work was done
   correctly, not that it was useful.

## Why one step is enough

Training is deterministic within a pinned execution class. A4 showed that training across three
owners reproduces the single-host trainer to the bit
([results](GRANITE_SHARD_TRAINING_RESULTS.md)). Each step is a pure function of the state before it,
the recipe and the step index. If a contributor's last root is wrong, some step's root is the first
wrong one, and re-executing that one step from the committed (correct) root before it proves it.
This is the same bisection argument as serving fraud, applied to training.

The tests check this on the protocol trainer (`growth_replay`):

- Segments trained by different contributors, resuming from committed bytes on another machine,
  reproduce the unchanged trainer's tensors exactly.
- An honest step re-executes to its committed root.
- A forged root is proven wrong.
- A bundle that does not start from the committed state proves nothing.

## Who decides what, after these changes

| Decision | Before | Now (in code) |
| --- | --- | --- |
| Which model is served | One steward key | Bond-weighted owner vote, after a one-way handover |
| Who orders transactions | One project validator | Owners vote validators in and out (`validator_propose`). The chain keeps the set in its state, and CometBFT applies the updates; this is tested end to end on real CometBFT validators |
| Who trains new units | The project, on its cloud hosts | Any contributor, segment by segment, paid per step, bonded, with every step checkable |
| Whether training was done correctly | Trusting the project's report | Any challenger can prove one wrong step on chain |
| Whether a unit is good enough | A sealed confirmation the project ran | The same evidence, which anyone can rerun, followed by an owner vote |

## Limits

- **Execution class.** Bit-exact replay holds within one pinned CPU instruction class and runtime.
  Contributors and judging validators must share it.
- **Recipes are inputs.** The data a recipe names is what the unit learns from. A sponsor can name poor
  data. Owners vote on the result after its sealed evaluation, and that vote is the defence.
- **Validators must hold the base model** to judge a disputed step of a Granite-scale recipe, as they
  hold shards to judge serving fraud. Sharded step replay, where each shard holder re-executes only
  its layers, is the next step for large units.
- **Proof availability.** As for serving, bundles reach validators' stores out of band. A proof
  commits only if validators with more than two thirds of the voting power hold its bundle.
- **Not deployed.** The live chain runs neither the growth transitions nor validator governance yet.

## Sealed tests from public randomness, and the autonomous loop

October 15, 2026. In code and tested; not yet run live.

**The problem it removes.** Every sealed test so far was a split the project wrote and kept unopened.
Whoever holds a sealed set could leak it, train toward it, or choose which one to open.

**Sealed evaluation from drand** (`inference/beacon.py`, `evolution/sealed_evaluation.py`).

- drand is a threshold randomness network run by many independent organisations. Every 3 seconds it
  publishes a BLS signature; its SHA-256 is that round's randomness.
- `beacon.verify` checks a beacon against drand quicknet's fixed group key in-process, so no API is
  trusted.
- When a growth job settles, the ledger records the block's BFT time, agreed by the validators, and
  `seal_round`: the first round after that time, plus 20 rounds (about one minute). The trained state is
  fixed before that randomness exists.
- The round's randomness seeds every case choice: workspaces, documents, facts, values, dates, topics,
  constraints, and which published wording template each case uses. The expected outcomes come from the
  same deterministic builders as every earlier split. Anyone with the beacon derives the identical 132
  documents and 96 chat cases, and the reference solver solves all of them.
- What it seals: the content of every case. What it does not seal: the families and the published wording
  pool. A unit can be trained toward the kinds of questions, never on the cases.

**The loop** (`evolution/growth_agent.py`, `evolution/growth_runtime.py`, `neuroshard assistant grow agent`).
Anyone runs the agent with their own keys, and it acts only through transactions it may sign:

1. **Contribute:** train the next segment from the committed state, but only if that state reproduces
   locally, never on top of work it cannot reproduce.
2. **Check:** re-execute a random step of every other contributor's segment. If it is wrong, find the
   first wrong step, challenge it with a proof bundle, and prove it.
3. **Evaluate:** once a job settles and its seal round exists, verify the beacon, derive the sealed set,
   and run the grown unit beside the serving model on the same cases.
4. **Propose:** if the unit meets the recipe's minimum and beats the serving model, propose the
   candidate version with the report's digest.
5. **Vote** (as an owner): derive the same sealed set, run the evaluation itself, and vote yes only if its
   own result reproduces the proposal's digest and passes. It never reads the proposer's report and never
   votes on judgment.

Committed training states are content-addressed files that any store may serve. A contributor accepts
bytes only if they hash to the root committed on chain.

**Tested** (`tests/evolution/test_growth_loop.py`), on the real ledger state machine, the real protocol
trainer and recorded drand rounds that the test verifies:

- one agent cheats on a segment, another catches it, proves the first wrong step and is paid half the
  bond, and the segment reopens;
- honest agents retrain and check each other;
- the job settles and its seal round is named;
- the evaluation is derived from the beacon and the version is proposed;
- owners recompute the evaluation and vote yes, and the proposal is enacted;
- an owner whose own evaluation differs from a proposal's evidence votes no.

**Limits.**

- The evaluator runs the model, so owners need the base model and the pinned execution class, as they do
  for serving.
- The minimum and the families are part of the recipe, and the recipe is chosen by its sponsor. Owners
  should judge recipes as well as results.
- Two consensus-rule changes came with this. Block time feeds the growth seal, and a model already in an
  open proposal cannot be proposed again until it closes. Validators must run this software before either
  rule matters on chain.
