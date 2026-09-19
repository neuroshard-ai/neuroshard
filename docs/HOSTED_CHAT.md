# Chat with the accepted shard graph

The provider research profile connects the existing lightweight client to the
complete answering graph. It requires that profile's freshly pinned native
genesis, available provider offers and running full-replay auditors. It is not
enabled by the public 0.4.0 genesis. The accepted-LLM latency/load trial and
independent public deployment remain open in the fixed TODO.

These commands require the matching research source release. The existing
`neuroshard-ai` 0.4.0 package on PyPI does not include `--hosted-config`. From
the reviewed checkout, install the lightweight client in a separate environment:

```bash
python3 -m venv .venv-client
.venv-client/bin/python -m pip install .
source .venv-client/bin/activate
```

Create a local wallet with `neuroshard wallet create --home ./customer`. Run a
synchronized local full node for the reviewed hosting genesis. The client uses
that node as its authority; these ABCI queries do not have light-client proofs.
Put the node's loopback RPC, chain ID and manifest digest in `hosting.json`:

```json
{
  "node_rpc": "http://127.0.0.1:26657",
  "chain_id": "CHAIN_ID_FROM_THE_REVIEWED_GENESIS",
  "manifest_root": "SHA256_OF_THE_CANONICAL_GENESIS_MANIFEST"
}
```

Read the complete price before submitting:

```bash
neuroshard chat "What can this network help me with?" \
  --home ./customer --hosted-config ./hosting.json --max-tokens 64 --quote-only
```

The quote includes bounded neural execution, all selected providers' fixed fees,
full verification funding and a transaction-fee allowance. It expires after 64
blocks. Discovery does not reserve capacity. No model or GPU dependencies are
loaded by the client. The local full node is a separate service.

The signed request asks native settlement to select available offers atomically
inside that price ceiling. Two customers may read the same indicative quote;
their committed reservations acquire different available capacity. A quote is
not a guarantee of a particular provider key.

`fund_hosted_audit` binds the verification reservation to this customer, graph
and complete conversation hash. Another customer cannot front-run that budget,
and its selected coordinator cannot attach it to unrelated neural work. The
generic sponsor-funded `fund_audit` transaction is not accepted for hosted chat.

After funding the wallet, replace `YOUR_LIMIT` with the maximum total NEURO you
authorize. The client refuses a quote exceeding that limit before payment:

```bash
neuroshard chat "What can this network help me with?" \
  --home ./customer --hosted-config ./hosting.json --max-tokens 64 \
  --max-price YOUR_LIMIT --session ./conversation.json --wait-seconds 600
```

The terminal displays provisional text during generation. `--json` emits signed
provider update contents and native result records as JSON lines. Each update
replaces the previous draft; incremental token decoding is not always stable at
the character boundary. Planning and intermediate worked text are excluded from
the chat display. For a deterministic combined reply, each completed answer
part becomes visible while the next part runs. Neural composition and structured
rendering still require their complete inputs before visible output. Empty
intermediate answers do not produce question-only previews. A malformed later
answer retracts the provisional reply.

Drafts are **unverified**. Agreement among assigned provider signatures alone
cannot authorize payment. The final reply and refunds come from native
settlement after complete paid replay. A lost stream or customer disconnect does
not cancel the native job, establish a failed audit or trigger another payment.
Full replay and native audit windows also mean settlement can arrive much later
than visible text. No public latency guarantee has yet been established.

Every signed payment is journaled before broadcast. If the wait ends or the
client is interrupted, use its printed request ID to recover the same signed
operations:

```bash
neuroshard chat --resume REQUEST_ID --home ./customer \
  --hosted-config ./hosting.json --wait-seconds 600
```

Do not replace an uncertain request with a newly signed payment. An audit offer
that has not acquired a job is cancelled when its quote expires. Unused job,
provider and audit budgets return through native expiry; transaction fees remain
spent. A rejected audited claim does not authorize automatic new verification
funding. Provider replacement can restart the pinned request within its native
attempt, price and expiry bounds. The client follows a committed replacement and
clears the obsolete draft; it does not select replacement providers itself yet.

Only settled, nonempty replies extend the conversation file. Reusing `--session`
adds the next user turn. The client pins the graph and tokenizer; a changed
version requires a new conversation file. It never silently truncates context
or treats an unverified draft as assistant history. Recovering an old request
cannot overwrite later recorded turns. Run one process and outbox per wallet;
different customer wallets may reserve different available provider capacity.

**Visibility and storage:** prompts, prior conversation, neural-call token IDs
and final replies are public ledger data. This includes intermediate model calls
even though the chat display hides them. Assigned providers, auxiliary model
owners and auditors receive the conversation needed for execution. TLS protects
transport, not those recipients. Do not submit private information. The ledger
does not offer erasure. Local request journals, settled conversation files and
provider transcripts persist until their owners remove them. Draft delivery
keeps only the latest text per assignment, in bounded provider memory; native
blocks and final execution records have separate retention obligations.

Checks cover token-by-token observation against full-model generation, delivery
failure without numerical changes, wrong-customer and obsolete-epoch rejection,
complete spending caps, quote cancellation, and lost-acknowledgement recovery
after native expiry with all unused budgets refunded. These checks do not stand
in for the accepted model's prospective operated load/cost trial.
