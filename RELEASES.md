# Native releases

## 0.3.0a1 — experimental public testnet

This release replaces the observer-ledger and account-registration deployment with native consensus, local keys, two-stage CPU training, and verified-work issuance. The application, explorer, model view, documentation, paper, experiments, and supported operator tools are developed in one public repository.

The Python package exposes `neuroshard-chain`, `neuroshard-work`, and `neuroshard` (native chain alias). The installer supplies the pinned CPU environment and CometBFT 0.38.26. Wheels include the corpus and protocol definitions. Install pinned CPU dependencies first; generic dependency resolution does not establish the genesis numerical profile. Earlier `0.2.x` PyPI packages and `--token` commands are unsupported for this chain.

Compatible network: `neuroshard-stage-8i5ghxq5`.

- Genesis SHA-256: `c3af664ca708318286c098d146dc962e781b55ff878a8a03a508f8f889898be6`.
- Manifest hash: `012fdc5f493a6bccf782629fc96dee48be15155e887323281aebb82471bd43d1`.
- CometBFT 0.38.26; Linux x86_64; Python 3.10–3.12; pinned CPU arithmetic.
- Consensus-bound source is unchanged by the public consolidation.

Release assets include source, wheel, genesis, network declaration, and SHA-256 checksums. Build a source release from its public revision with `python3 scripts/build_release.py --ref v0.3.0a1 --output dist/release`. Git snapshots exclude untracked keys and local databases.

The website needs no authentication backend. Old signup/login/download URLs lead to participation instructions; retired API endpoints return a retirement response. Existing user records are preserved operationally outside source control. No conversion of old balances or identities is defined.

Limits remain explicit: 34,976 parameters, full replay at every validator, concentrated launch ownership, bounded sponsorship, unresolved fair assignment, no inclusion proofs/state sync, and no established production monetary policy or sustained-load envelope. Accepted work does not prove model improvement or profitability. The five-page paper remains the earlier formulation; new experiments and protocol details are documented separately.

## Release procedure

1. Run native tests, wheel installation/conformance, website browser checks, docs build, and secret scan.
2. Build from the exact public revision. Compare manifest and genesis. Change chain/release when consensus compatibility changes.
3. Publish a named tag, checksummed assets, and concrete validation results. Mark experimental releases as prereleases.
4. Deploy from pinned directories. Upgrade validators sequentially, preserving signing state. Never auto-deploy from `main`.
5. Verify a fresh public checkout and node/worker trial. Retain web/service rollback records.
