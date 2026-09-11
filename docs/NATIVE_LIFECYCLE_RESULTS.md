# Native lifecycle integration evidence — September 11, 2026

These are isolated integration experiments for the [native lifecycle](NATIVE_LIFECYCLE.md). They do not upgrade the public 0.4.0 chain. All machines and test identities belong to one operator, LZ. Experimental balances are separate from public testnet balances.

## Reproduction and source identity

Use the pinned [numerical dependencies](evolution-requirements.txt), CometBFT 0.38.26, a fresh directory and the commands in the lifecycle guide. Run `review_native_cohort.py` with your own source policy before admitting data. The real trial uses four HTTP workers across two CPU hosts: Intel Xeon Platinum 8259CL and 8175M, each with four virtual CPUs and approximately 16 GiB RAM. It uses float32 eager CPU operations, one numerical thread, default ATen capability, SSE4.2 MKL instructions and no MKLDNN.

The real trial's driver is at [865f085](https://github.com/neuroshard-ai/neuroshard/commit/865f08563e9f442811211445dd5b3a0a0f8adbaa). Its package source commitment is `80949c30b6d901ad54a648917680776e7c295211d81e67d7d4611fd52ac24077`. Later reviewer/topology/documentation changes do not change that package commitment. The source commitment binds all package Python files, including unused modules; it is distinct from the Git commit and genesis hashes.

Four isolated validators run on the first host. A non-voting full node on the second host replays their native blocks from genesis using the same source profile. Worker transport uses operator-managed authentication and SSH tunnels. This layout tests different hardware and processes; it does not establish independent ownership or tolerate losing the validator host.

## Fresh-data evidence

Both cohorts use the imported SmolLM2 text contract `b797b60c9203884ec7d519b50dfd16e4375b239d249ac7377aa3f6c646b56e85` and Smol-SmolTalk revision `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`. The source policy assigns its train split to training and its test split to a document-hash partition of retention, fresh evaluation and protected test data. These are previously unused source rows, not evidence of recent world knowledge.

| Check | First cohort | Second cohort |
| --- | --- | --- |
| Train source interval, end excluded | 896–1,152 | 1,152–1,664 |
| Held-out source interval, end excluded | 0–384 | 384–896 |
| Selected documents: train / retention / fresh | 32 / 32 / 32 | 32 / 32 / 32 |
| Response windows: train / retention / fresh | 117 / 69 / 75 | 111 / 89 / 77 |
| Original documents checked against pinned upstream rows | 96 | 96 |
| Raw selected document evidence | 276,572 bytes | 237,889 bytes |

The first cohort root is `6cf45c8a79ce52a634aca120cc8831acff3f7091eeffd7214800c9fac3005091`; the second is `4b85ef30b90f4224bb655fc6f16f97bc22daca5e9531bc51bfdf9d86adafe102`. Review reconstructs every proposed token window from the committed raw messages and the exact tokenizer. It also checks source revision/file hashes through the upstream repository service. The checks do not independently establish license rights, truth, usefulness, or absence of historical semantic contamination.

An initial one-window policy discarded most eligible multi-turn responses. The final format preserves up to four windows while counting each complete document once in evaluation. Longer evaluation documents remain excluded and training omissions remain explicit. This selection bias is a material limitation, not a reason to count chunks as extra independent observations.

## Synthetic protocol checks

The [afe6078](https://github.com/neuroshard-ai/neuroshard/commit/afe6078d0b80f24b6f63b1ebfed33a1302d7f288) synthetic run uses a 10,384-parameter model and package commitment `21cbc74d8c0422444d1dbf71d720d64bda54f3fe0071e433776859acdac01434`. It completes two data admissions, four training settlements, 107 independent stage replays, a passing synthetic serving decision, a forged-head dispute and a valid paid generation. Training issues exactly 4,000,000 atoms. Inference transfers 1,000 atoms and refunds 6,000; it issues nothing. The head dispute publishes 4,520 bytes.

One stopped validator leaves the other three progressing. Half the voting power stops finality. Restarting the same validators resumes progress and yields matching app hashes in block 446: `4EED463523EE7815F67DDC97FE928D0579BE09CE0F1AD61BD532AA05E117563E`. This successful synthetic quality decision is not evidence of LLM improvement.

The bounded continuation reopens those same node homes and trains the second cohort. It advances round 4 to round 8 and total issuance from 4,000,000 to 8,000,000 atoms. Three fresh batches and one replay batch settle after eight further stage replays. The serving root stays unchanged pending the second evaluation. All four validators agree in block 498 (`A1B3FC3894FD8F5ABAB8A7E28A701E22679F320481C069C288349A5B1C3CBA3C`). A repeated continuation invocation is refused before starting nodes or issuing work.

## Release checks and limits

The extracted source distribution passes 220 tests outside the checkout. The tests include proposal expiry, fresh-data reuse and protected-role rejection, reserved-batch binding, finite training budgets, complete document scoring, score timeout, rejected-model recovery, forged generation, escrow conservation, collector writer exclusion, independent token/source review, and declared validator topology. GitHub checks cover Python 3.10 and 3.12, the native consensus build, repository links, and package contents. Tests do not substitute for an independent security review.

The topology check applied to the existing public ledger finds 40 units of voting power: 20 on each host and all 40 under one operator. Losing either host leaves only half the power. This fails the declared single-host and single-operator failure checks. The isolated four-validator experiment checks process failures within its own topology; it does not repair that public deployment.

The remaining public-cutover requirements are listed in the [lifecycle guide](NATIVE_LIFECYCLE.md#requirements-before-public-cutover). No result here establishes an economical audit market, sustained quality improvement, production chat latency, or automatic scaling from an arbitrary additional peer.
