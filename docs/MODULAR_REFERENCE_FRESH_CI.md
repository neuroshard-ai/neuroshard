# Fresh reference pre-execution CI correction

The queued `aca7a59` comparison stopped at **07:09:54 UTC on September 26, 2026**,
because [its Python 3.10 CI job](https://github.com/neuroshard-ai/neuroshard/actions/runs/36224896821)
had one failing test (1,394 passed, one skipped). Python 3.12 was cancelled by
the matrix failure. Packaging, repository and consensus jobs passed.

No EC2 allocation, model preparation or generation started. This attempt has
no learning result and no new EC2 cost. Its local stopped record is preserved.

`test_runtime_opt_in_occurs_before_torch_and_does_not_change_default_profile`
started a Python child without explicitly selecting the repository source.
Pytest's configured `pythonpath` applies to its own process; CI's child therefore
imported the installed wheel. Experiment contracts are intentionally excluded
from that wheel, so the child raised a missing-file error before reaching the
startup-order guard that the test meant to exercise.

The corrected test explicitly selects the repository source, matching the
experiment launcher. It also supplies a deliberately failing foreign package
through the inherited path to verify that the child does not select it.

This is a test correction before execution. The [case plan](MODULAR_REFERENCE_FRESH.md),
model revisions, runtime implementation, scoring, thresholds, budget and stop
rule are unchanged. The comparison must pass CI at the new committed revision
before allocation; use a new controller home and retain the stopped one. The
single permitted allocation attempt has not yet been used. A1 stays open.
