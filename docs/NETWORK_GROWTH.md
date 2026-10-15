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
