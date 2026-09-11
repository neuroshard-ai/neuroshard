# Deploy the LLM testnet application

The public repository contains the client, native protocol, data writer, tests, network manifests and operator documentation. Deploy from a pinned release and preserve all node keys, signing state and databases outside the checkout. Website publishing is maintained separately. There is no registration database in the supported path.

## Build and run

```bash
git clone --branch v0.4.0 https://github.com/neuroshard-ai/neuroshard.git
cd neuroshard
python3 -m venv venv_build
venv_build/bin/python -m pip install -r docs/llm-requirements.txt
venv_build/bin/python -m pip install --no-deps .
```

Use Linux x86_64 and Python 3.10–3.12. Node.js and website build tools are not required to operate the protocol. CI checks native Python behavior, the pinned consensus build, repository links and package contents. A lightweight wallet installation does not install the numerical operator dependencies.

For an ordinary participant, use [the public client](PUBLIC_TESTNET.md). To establish another native chain, generate at least four locally controlled genesis declarations using the reference bootstrap tools, inspect all allocations and ownership, and run `python -m neuroshard.inference.node genesis --help`. `genesis` freezes model/data/source/runtime; never edit an initialized chain's genesis to apply a software update. The supplied network bundle starts from 90 disclosed genesis NEURO, with a 10,000-task issuance cap.

## Services and retained state

[Systemd templates](../config) show the node, sponsor, worker/provider and bounded collector. Adapt paths, accounts, network and budgets before installing. The launch services run wheel-installed code in isolated release directories, not an editable working tree. CPU arithmetic environment is set before Python imports. Node processes supervise native Comet, ABCI and the gateway. RPC/application listeners are loopback-only; native peers use TCP 26656. Keep validator private keys and signing state private, consistent and unique to one running process.

The launch has two local and two remote genesis validators under one operator. Public participants connect directly to `100.53.139.52:26656`. A secondary private SSH transport connects operator peers across the same two hosts; this is not a separate consensus or a public joining requirement. One published bootstrap peer is an availability limitation. Add independently operated peers and document ownership as participation grows.

Local application gateway/sponsor ports are 39659/39660. The old v0.3 reference gateway remains at 38659. Its old remote public seed is stopped with state retained; the local reference quorum and read-only explorer remain available. The new testnet does not import old balances or claim an automatic upgrade of that history.

The project sponsor uses a persistent attempt budget of 10,000 and waits roughly 300 seconds between successful tasks. Each attempt is charged before reservation; restarts cannot reset it. A failed lease can burn 2 NEURO collateral. Budget, collateral, stage availability and actual round progress require monitoring. Operator fallback workers supply both stages and yield to public workers. Independent sponsors may apply different selection/budget policies without changing consensus.

The provider serves jobs addressed to its configured public key and writes a local availability heartbeat. The gateway advertises that key and status. A customer can choose another provider through the CLI; the release has no automatic provider marketplace or routing guarantee.

The data collector uses a separate mode-0600 environment file under `/etc/neuroshard/` with the normal AWS credential variables. The daily timer publishes at most 128 records per invocation and 4,096 total records for its pinned source identity. It advances durable progress only after immutable S3 publication. Its `collection_budget_complete` status is expected at the cap. See [data pipeline](DATA_PIPELINE.md). Old uploaders remain disabled; do not start both writers against the old mutable namespace.

## HTTP gateways

The node exposes the [documented API](API.md) independently of the project website. Keep administrative and consensus RPC listeners on loopback. An operator exposing a public HTTP gateway is responsible for TLS, bounded request bodies and its reverse-proxy configuration. Frontend source and site-specific hosting configuration are outside this repository.

A proxy must preserve signed POST bodies and overwrite client-address headers used for rate limits. Allow up to 100 seconds for signed submission requests: neural verification can outlast a normal short HTTP timeout. Customers must recover unknown outcomes by transaction/request ID, not automatically sign another spend. Curated public model assets use content-addressed mirrors with model/dataset attribution; never expose local archives, keys, raw legacy S3 recovery objects or environment files.

The public node ID and chain/genesis, not a reused IP address or website account, identify a network. Verify both HTTP behavior and actual native settlement after changing a gateway configuration. Changes to website publishing must not replace node state or signing identities.

## Operations and rollback

Monitor native height progression, model round and serving root, worker receipts, finalized rewards, provider heartbeat, pending inference/expiry, sponsor collateral/budget, collection cursor/status, disk and process RSS. A systemd process marked active is not evidence of training progress or inference settlement. A healthy explorer also does not prove independent consensus ownership.

The launch installs [log rotation](../config/neuroshard-logrotate.conf) on both hosts: daily, or at 25 MiB, retaining seven compressed rotations. Adapt the supplied paths for other homes. Rotate `logs/*.log` with copy-truncate or coordinated process reopening; do not rotate or edit native signing state. Native blocks and application databases are authoritative. `explorer.sqlite` is only an index and can be rebuilt from retained blocks after stopping that gateway. Keep historical model/source/data artifacts for replay.

Restart validators sequentially and confirm continued blocks. Back up the chain home consistently; never start two copies of a validator's keys. Restore only code compatible with that genesis. An incompatible protocol change requires a separately specified migration or new network, not a hidden balance reset.

Before a production claim, the remaining [research and release requirements](RESEARCH_ROADMAP.md) include independent ownership, longer load/failure trials, security review, improved evaluation, data availability, checkpoint governance and a defensible economic policy. The present release is a public experimental testnet with finite operating and issuance budgets.
