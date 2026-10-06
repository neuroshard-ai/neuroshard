# Serving the accepted assistant across owners

Declared on October 6, 2026, after the sharded loader learned to serve an update and
before any sharded serving of one. Contract:
[granite-shard-serving-update.json](../config/experiments/granite-shard-serving-update.json).

## Why

[A4's serving execution](GRANITE_SHARD_SERVING_RESULTS.md) reproduced the single-host
assistant across three owners token for token, but with the round-4 *addition*. A2
accepted the round-4 *update* instead, so a network release would have served a unit the
assistant never accepted. The owner runtime now loads either arm.

## What runs

The same three owner hosts, layer ranges, runtime and gates as A4's serving execution.
Owner 2 holds layers 26-39 and the accepted update of layers 32-39, served as plain
projections in the backbone dtype as single-host serving stores them, and switched on or
off per episode by A2's gate. The 24 served development episodes must reproduce the
single-host development result of [A2's third attempt](ASSISTANT_EXPERIENCE_THIRD_RESULTS.md)
token for token: selection, every generation and score. No new evaluation data is opened.

## Budget

Three r7i.4xlarge hosts, each with a four-hour expiry and an eight-dollar allowance, at
most $24; A4's run cost $1.75. No GPU or training.
