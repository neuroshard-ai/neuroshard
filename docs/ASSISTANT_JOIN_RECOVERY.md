# Join and recover an independent assistant node

**Status: instructions for the A5 soak. Not authorized until four operators
exist and the CPU protocol preflight has passed.** The [operator kit](OPERATOR_KIT.md)
is the page to send volunteers. The contract is
[INDEPENDENT_HOSTING.md](INDEPENDENT_HOSTING.md).

This is a dedicated assistant test network. It is not the 0.4.0 protocol
testnet and not a public assistant.

## Join (operators)

1. Open a GitHub issue titled `Independent operator: your name or organization`,
   with region and machine type. Do not send keys.
2. Wait until four independently administered operators are confirmed. Extra
   machines under one administrator do not count.
3. On a Linux x86_64 machine you administer, generate validator keys locally.
   They never leave the machine.
4. Install the pinned release the maintainers name in the issue thread. Open one
   inbound TCP port for peers. About 16 GB RAM and 50 GB free disk.
5. Bond the test tokens the project sends. Optionally host one CPU shard of the
   accepted graph. No GPU is authorized.

## Soak

Once all four validators are live, the scripted soak is: join, leave, lose a
required backbone shard, lose the coordinator, recover state, and keep serving.
No administrator may hold a third or more of voting power. This operator may
hold at most one of the four validators.

## Recover

- Restart the process from the last committed height. Keys stay on the machine.
- A replaced shard or coordinator is a new process under the same operator, not
  a new administrator.
- If the current serving version fails its gate, operators keep the previous
  version (`a2-u1`) addressable; see [ASSISTANT_PUBLIC.md](ASSISTANT_PUBLIC.md).
- A lost session is gone unless the user opted into an export. Default consent
  stores nothing on the ledger.

## What does not count

Another AWS instance we can SSH to, a second key on your machine, or joining the
public 0.4.0 chain.
