# Public CPU testnet recovery — September 27, 2026

The public `neuroshard-llm-testnet-1` endpoint was responsive but its ledger had
stopped at height **554521**, timestamp **September 19, 20:13:29 UTC**, training
round **2420**. The inference provider heartbeat was fresh, so its old API
incorrectly advertised availability. A running process was not a working service.

## Cause and recovery

The second validator host's 100 GiB root volume was full. Its consensus logs
reported failed WAL/address-book writes with `no space left on device`. Four
validators across the two project-managed hosts could no longer finalize blocks.

The existing volume was expanded from 100 to 128 GiB, then its partition and ext4
filesystem were grown online. This restored approximately 28 GiB of free space.
The two affected validators were restarted individually. Ledger databases, keys,
signing state, balances, genesis and model artifacts were preserved. There was no
chain reset, new validator, new compute instance or GPU allocation. The extra
28 GiB is ongoing storage, not a temporary experiment volume.

After block production resumed, the sponsor waited for the previous lease to
expire. Both workers returned a new training result, settled at height **554772**:

```
A1276714EF8AD4F445B84158B8BBD60C05260BF4E97B45129700048A9B4A1AA6
```

The native transaction returned code zero and advanced training to round **2421**.
Both validators on the recovered host and the controller's checked validator
agreed on that block hash. The existing finite sponsor budget was retained;
failed attempts were not erased or replenished.

A subsequent request through the public HTTPS gateway completed at height
**554895**, request ID
`3f5f8cc0f85b1c7ea7f8dc833ce5226f9685fa7ad44ffac0b02a19a699d2fe57`.
It used a 12-token limit and spent 0.013 testnet NEURO including the fee. This
checks inference and native settlement after recovery, not answer quality.

## Availability correction

The deployed gateway now requires a fresh, ready ledger as well as a current
provider heartbeat before reporting `provider_online: true`. It exposes
`network_ready`, `provider_connected` and `seconds_since_block` separately. New
transaction relays return HTTP 503 while the ledger is not ready; reads remain
available so a caller can investigate an earlier signed request.

Only the gateway file was replaced in the installed runtime, followed by a
restart of the controller's public node. The inventory of all consensus-committed
source files was checked unchanged before and after deployment. Current blocks
and the new readiness fields were checked through the public HTTPS endpoint.

The source client separately checks ledger freshness before chat/payment and
fresh-node setup. An existing node may still start to synchronize. These client
changes require the updated source install until a later package release.

## What this establishes

This recovery restores the existing small-model CPU testnet, including native
training settlement. It does not make Granite public, improve assistant answers,
prove independent operation or renew the retired GPU alpha. Four consensus keys
on two project-controlled hosts still represent one administrator.

Operators should monitor `/healthz`, advancing block heights, disk space and
settled training separately. Keep growing experiment artifacts away from validator
storage and provision headroom before disk exhaustion. A storage expansion fixes
this incident; it does not establish an unattended operating soak.

Local operational receipts, source inventories and backups are retained under
`.neuroshard/contributor-alpha-20260927/`. No private keys or signing states are
included in this public report.
