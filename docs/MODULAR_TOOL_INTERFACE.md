# BAR tool-interface diagnostic

**Completed:** [3/3 correct calls and three exact replays](MODULAR_TOOL_INTERFACE_RESULTS.md).
This contract remains the pre-execution record; A1 is still open.

The completed BAR-7B baseline scored **0/3 tool calls**. Its prompt requested
function use but never specified the tagged, keyword-argument syntax required
by its scorer. This is an interface mismatch worth correcting before selecting
a different foundation. It does not yet explain every failed answer or show
that the checkpoint can use tools reliably. The [original failed result](MODULAR_REFERENCE_BASELINE_RESULTS.md)
remains closed and unchanged.

## Evidence for the correction

Ai2's [OLMES code-call handler at revision 5a51f502](https://github.com/allenai/olmes/blob/5a51f502d463b8cdc4a2dcad7d7096c41ff1197e/oe_eval/dependencies/BFCL/bfcl/model_handler/local_inference/allenai.py)
instructs models to emit a `function_calls` block containing calls with named
arguments. The published [Dolci tool-use data](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT-Tool-Use)
illustrates that format too. Function definitions use the OpenAI envelope that
our original prompts already used. No schema conversion is needed.

The adapter now supplies a generic syntax instruction, using the conversation
and available function definitions only. It receives no expected answer.
Literal arguments are parsed without executing Python, checked against the
declared function registry and argument types, and returned as data. This
diagnostic invokes no external tool. The upstream handler's use of `eval` is
not imported.

The [interface audit](../config/experiments/modular-tool-interface-audit.json)
pins the upstream source and records rendered prompt lengths. Corrected prompts
are 239–256 tokens. Expected calls including EOS need 15–22 tokens, within the
unchanged 32-token generation cap. This is a documented interface correction,
not a claim to reproduce the paper's complete benchmark configuration.

## Frozen execution

The [new diagnostic plan](../config/experiments/modular-tool-interface-diagnostic.json)
and [execution contract](../config/experiments/modular-tool-interface-execution.json)
define a separate experiment. The old task plan and scores are not revised.

- Use the already cached BAR-7B weights and tokenizer, at their original pinned
  revisions. CPU only; no downloads, new instances, training or modular model.
- Generate once for each opened tool case, in order: add, weather, balance.
  Keep the questions, function definitions, expected calls, greedy decoding,
  EOS handling, numerical profile and 32-token cap unchanged. Only the tool
  system instructions change. The other six cases in the copied plan are
  retained for comparison and are not generated in this diagnostic.
- Report the unchanged original score and additional native-format validation
  separately. Native validation also rejects prose outside the call block,
  unknown functions, duplicate arguments, executable expressions and wrong
  types. Do not turn a parser failure into a correct answer.
- Independently regenerate only correct, valid primary calls. Require exact
  decoded text, termination and score agreement; report token agreement too.
  Zero correct calls finishes as a negative diagnostic, without replay.
- At least one correct replayed call supports the narrow claim that the
  corrected interface can elicit a valid call on an opened case. **No result
  grants admission, A1 completion, or permission to launch the larger model.**
- Stop on execution failure, disagreement or the declared budget. No prompt
  search, score relaxation, automatic restart or alternative model follows.

Workers retain complete source/artifact binding and independent receipts from
the [bounded execution runner](MODULAR_REFERENCE_EXECUTION.md). Preparation
verifies cached files within 600 seconds. Each generation is limited to
2,400 seconds and 12 GiB, with swap disabled and process-group termination.
Three primary calls plus at most three replays allow **at most four hours of
new generation**, plus verification and small charged supervision overhead.
The original 5,203.386 seconds remain charged in the eight-hour per-checkpoint
evaluation envelope. Interrupted work remains charged; failed work never
silently restarts. Historical download duration remains unknown.

The background controller can wait up to one hour for successful push CI on
the exact committed source before starting. A failed CI, timeout or API error
stops before inference. CI waiting is reported separately from model work.

## Run and handoff

Commit first and run from a detached worktree at that commit. Use a new home;
the existing baseline directory supplies cached weights only. The Python
environment must match the contract's pinned packages.

```bash
PYTHONPATH=src python scripts/run_modular_reference.py run \
  --profile tool-interface --wait-for-ci \
  --home /absolute/path/to/new-study \
  --models /absolute/path/to/cached-modular-reference \
  --legacy config/experiments/modular-reference-a1-legacy-baseline-result.json
```

`status.json` identifies the current phase. `progress.json` contains completed
calls and replays; `result.json` records the final diagnostic and accounting.
The controller can run as a persistent user service without an active agent.

After completion, publish the result before choosing another experiment. If
tool use remains unusable, this corrected prompt stops too. If it works, A1
still needs a separately declared reference comparison with usable protected
categories. These opened questions cannot become fresh evidence.
