# Deploy the native application

One public checkout contains the node, worker, gateway, website, docs, experiments, and paper. The website is static; it needs a full-node gateway and an optional sponsor. No private application or registration database is required.

## Build from a pinned release

Use Linux x86_64, Python 3.10–3.12, and Node 22.12 or newer in the Node 22 line.

```bash
git clone --branch v0.3.0a1 https://github.com/neuroshard-ai/neuroshard.git
cd neuroshard
bash scripts/install_native.sh
cd website
npm ci
npm run build
cd ../docs-site
npm ci
npm run build
```

The website output is `website/dist`; docs output is `docs-site/.vitepress/dist`. Docs are generated from canonical repository markdown. `legacy/` is excluded from both builds. Install each release under a distinct directory and run services from that pinned directory. Follow [node initialization](PUBLIC_TESTNET.md) separately; building a website does not create a chain.

## HTTPS routes

Copy the builds to `/var/www/neuroshard/releases/v0.3.0a1/site` and `.../docs`, then atomically switch `/var/www/neuroshard/current` to that release directory. A temporary symlink followed by `mv -T` keeps the switch atomic. Preserve the previous target for rollback.

Include [native-site.nginx.conf](../config/native-site.nginx.conf) in the root domain's TLS server and [native-docs.nginx.conf](../config/native-docs.nginx.conf) in the docs server. Supply your existing certificate paths in those outer server blocks. Change loopback gateway/sponsor ports to match your node. The launch deployment uses 38659/38660; the templates use normal node defaults 26659/26660.

The proxy overwrites `X-Real-IP`. Keep native RPC and application gRPC on loopback. Expose TCP 26656 for inbound peers when desired. HTTPS permits outbound-only worker connections. Public endpoints have bounded sizes/rates; a sponsor cannot bypass consensus acceptance.

Run `nginx -t` before reloading. Check the homepage, model/checkpoint/history, ledger/account/validator views, docs links, source and genesis downloads, mobile navigation, unavailable-service behavior, and a real worker trial. Existing `/native-preview/` paths preserve native POST semantics through an internal rewrite. Old login/signup routes lead to `/join`; old authentication APIs return 410.

## Node and sponsor services

Adapt [the native systemd template](../config/neuroshard-native.service) for your user, pinned release, and initialized node home. Preserve account/consensus keys, databases, and signing state. Never run two validator processes with the same consensus identity. Restart validators sequentially and verify continued height advancement.

Sponsor sessions have a finite attempt budget and stop on failure. Do not use an unconditional restart policy that silently resets that budget. The launch session waits for workers and offers at most 100 attempts; availability can stop at any time. Independent sponsors can provide their own endpoints. Keep their keys on the sponsoring machine.

The gateway's `explorer.sqlite` is an auxiliary index. It may be rebuilt from retained blocks after stopping its gateway; it is not the chain database. Missing history pauses the index and reports lag. Preserve chain history and plan disk capacity for long-running nodes. Rotate application/consensus logs without touching signing state.

## Retiring the old deployment

Back up the old database privately and verify the dump is readable before stopping the registration stack. Preserve Docker volumes; never use `down -v`. Switch and verify the static/native routes first, then stop only the old NeuroShard containers and disable their restart policies. Remove alternate public access to their old ports. Preserve existing user records outside Git. No automatic identity or balance migration is defined.

Rollback web routing by restoring the previous static symlink/config and reloading validated nginx configuration. Rolling back gateway/UI code does not require replacing chain state. A consensus-incompatible release requires an explicit migration decision, not a website deployment.

## Public operation gates

This deployment is an experimental testnet. Independent operator ownership, sustained-load measurements, adversarial network testing, release signing/checkpoint policy, economic policy, and independent security review remain necessary before a production network claim. See [testnet gates](TESTNET_GATES.md).
