# Run one of the first four independent NeuroShard nodes

NeuroShard serves a learned assistant across machines and settles payment for that
work on its own chain. The next test shows that no single administrator controls it.
It needs four operators who each administer their own machine; the project may hold
at most one of the four. The rules are in the [hosting contract](INDEPENDENT_HOSTING.md).

## What you would do

- Run one validator of a dedicated test network on a machine you control.
- Optionally host one shard of the assistant, on CPU; no GPU is needed.
- Take part in a short scripted soak: join, go offline and return, and keep serving
  while another operator's shard or the coordinator drops out and recovers.

## What you need

- A Linux x86_64 machine you administer: your own hardware or your own cloud account,
  not one the project gives you access to.
- About 16 GB of RAM, 50 GB of free disk, and a stable connection with one inbound TCP
  port open for peers.
- Python 3.10 to 3.12, a few hours to set up, and some attention during the soak.

## What stays yours

- Your keys are generated on your machine and never leave it.
- Only you can sign for your validator.
- Test tokens have no monetary value, and the test network is not a production service.
  Conversations on it are public.

## What the project provides

- A pinned release, setup instructions and testnet tokens to bond your validator.
- Help through GitHub while you set up. Hosting costs are your own unless you arrange
  otherwise with the maintainers.

## What it demonstrates

Milestone A5 of the [assistant plan](../TODO_ASSISTANT.md): the assistant keeps serving
and settling while no administrator holds a third of the voting power, and nodes can
join, leave, lose a shard and recover.

## To volunteer

Open a GitHub issue titled "Independent operator: your name or organization", with your
region and machine type. Setup, soak and recovery steps are in
[ASSISTANT_JOIN_RECOVERY.md](ASSISTANT_JOIN_RECOVERY.md). They start once four operators
are confirmed and the protocol preflight has passed.
