# Join the operated alpha

This deployment is being prepared. Use the published access descriptor only
after its availability field records a passing deployment gate. The earlier
0.4.0 chain and PyPI package are different releases.

The alpha runs the accepted shard graph, with paid provider discovery and three
complete replays before settlement. Its GPU window is funded for at most 60 hours;
the descriptor gives the exact service and ledger deadlines. All bootstrap
hosts belong to one administrator. Additional AWS hosts do not establish
independent ownership. Starter credits have no demonstrated market value.

## Run your own ledger observer

Use Linux x86_64 with Python 3.10–3.12, at least 4 GB RAM and several GB of free
disk. This observer needs the pinned CPU dependencies, but no GPU or model
weights. Review the descriptor and its exact source commit before installation.
From that checkout:

```bash
python3 -m venv .venv-alpha
.venv-alpha/bin/python -m pip install -r docs/evolution-requirements.txt
.venv-alpha/bin/python -m pip install .
source .venv-alpha/bin/activate
python scripts/join_operated_alpha.py --descriptor ./access.json --home ./alpha-node
```

The observer verifies native blocks from genesis. Keep it running and wait until
it reports `ready`; its RPC listens only on your machine. The descriptor pins
the genesis, manifest, executable checksum, complete answering graph and native
peer endpoints. Merely starting an observer neither spends nor bonds tokens.

## Get bounded starter credits and chat

In another terminal using the same environment:

```bash
neuroshard wallet create --home ./customer
python scripts/operated_alpha_credits.py --descriptor ./access.json --wallet-home ./customer
neuroshard chat "What is the name of the NeuroShard client package?" \
  --home ./customer --hosted-config ./alpha-node/hosting.json \
  --max-tokens 64 --quote-only
```

The credit service receives only your public key. It sends at most 500 trial
NEURO per key, with a global 10,000 NEURO sponsorship ceiling. This is not proof
of unique people: multiple keys can exhaust the finite pool. A pending request
must be retried with the same wallet; it never authorizes another grant.

Prompts, conversation context and settled replies are public ledger data. Read
the complete quote, including verification and occupied capacity, before paying:

```bash
neuroshard chat "What is the name of the NeuroShard client package?" \
  --home ./customer --hosted-config ./alpha-node/hosting.json \
  --max-tokens 64 --max-price 500 --session ./conversation.json --wait-seconds 1200
```

Provisional text appears before native settlement and remains unverified until
then. If the command times out, use its printed request ID with `--resume` and
the same wallet. Never sign another payment to replace an uncertain one. Reuse
`--session` for another turn only after the preceding request settles. Capacity
is limited to two simultaneous requests and the graph has a bounded context.
If both slots are occupied, a quote can be refused; retry later. A quote refusal
does not submit a payment.
Full replay is expensive: the maximum verification budget is locked before
execution, then unused funds return after settlement. A 500-credit grant can
start a bounded request; how many later turns fit depends on actual charges
and the next complete quote. It is not an unlimited chat subscription.

See [chat and recovery behavior](HOSTED_CHAT.md) and the [funding and trust
assumptions](OPERATED_ALPHA.md). Running your own observer establishes your own
ledger verification. Independent provider and validator operation also requires
control of those keys, resources and administration; the fixed TODO does not
count centrally controlled AWS hosts as outside operators.
