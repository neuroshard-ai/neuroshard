# NeuroShard native website

The supported interface is a React/Vite application in `src/native`, mounted by `src/App.tsx`. It provides a network overview, ledger, validators, model/checkpoints, training history, node/worker onboarding, and protocol documents. It reads the bounded native gateway and does not require the legacy registration API.

## Develop and build

Use Node 22.12 or newer in the Node 22 line, npm, and Python 3 for the documentation-copy step:

```bash
npm ci
npm run dev
npm run build
```

`vite.config.ts` proxies `/api`, `/network`, `/rpc`, and `/healthz` to the local native gateway at `127.0.0.1:38659`. Adjust the target to match your node. The static page renders connection failures when no gateway is available.

For the existing preview path:

```bash
npm run build -- --base=/native-preview/
```

For a root-domain deployment, build without the base override and configure the reverse proxy for the native API, genesis, signed-transaction relay, and optional sponsor `/work/` routes. The browser does not hold validator or account private keys.

`npm run build` copies public protocol documents and the manuscript from `docs/`. The Join page installs the matching version tag from the public repository; versioned source archives and checksums are attached to its GitHub release. `release.json` can be copied from release assets into the deployed site's root for machine-readable release identity.

The public root application is [neuroshard.com](https://neuroshard.com). The docs site at [docs.neuroshard.com](https://docs.neuroshard.com) builds from the canonical markdown in this same repository. Historical authentication code is in `legacy/website` and is not required by the native interface.

Run browser checks with `npx playwright install chromium && npm test`. The tests exercise real components against deterministic API fixtures; deployment verification additionally checks the live gateway and a real worker settlement.

See [deployment](../docs/DEPLOYMENT.md), [the operator guide](../docs/PUBLIC_TESTNET.md), and [the migration record](../docs/OPEN_SOURCE_TRANSITION.md).
