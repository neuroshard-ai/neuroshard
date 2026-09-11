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
| Response windows: train / retention / fresh | 117 / 69 / 75 | 111 / 89 / 76 |
| Original documents checked against pinned upstream rows | 96 | 96 |
| Raw selected document evidence | 276,572 bytes | 239,684 bytes |

The first cohort root is `6cf45c8a79ce52a634aca120cc8831acff3f7091eeffd7214800c9fac3005091`; the admitted second cohort is `2a3bed0d15fefa3a14505b1c1a6378f7d4d587a718045af6073d67aebf286dec`. Review reconstructs every proposed token window from the committed raw messages and the exact tokenizer. It also checks source revision/file hashes through the upstream repository service. The checks do not independently establish license rights, truth, usefulness, or absence of historical semantic contamination.

An initial one-window policy discarded most eligible multi-turn responses. The final format preserves up to four windows while counting each complete document once in evaluation. Longer evaluation documents remain excluded and training omissions remain explicit. This selection bias is a material limitation, not a reason to count chunks as extra independent observations.

The first trial's four training steps select four of its 117 admitted training windows. Admission does not mean every window was trained. In this reference, the next cohort's replay draws from the previous admitted pool, including windows not selected by the earlier schedule. The two evaluation groups also come from the same dataset family. These choices test protocol mechanics and response loss; they are not a validated lifelong-memory policy or broad capability benchmark.

## Synthetic protocol checks

The [afe6078](https://github.com/neuroshard-ai/neuroshard/commit/afe6078d0b80f24b6f63b1ebfed33a1302d7f288) synthetic run uses a 10,384-parameter model and package commitment `21cbc74d8c0422444d1dbf71d720d64bda54f3fe0071e433776859acdac01434`. It completes two data admissions, four training settlements, 107 independent stage replays, a passing synthetic serving decision, a forged-head dispute and a valid paid generation. Training issues exactly 4,000,000 atoms. Inference transfers 1,000 atoms and refunds 6,000; it issues nothing. The head dispute publishes 4,520 bytes.

One stopped validator leaves the other three progressing. Half the voting power stops finality. Restarting the same validators resumes progress and yields matching app hashes in block 446: `4EED463523EE7815F67DDC97FE928D0579BE09CE0F1AD61BD532AA05E117563E`. This successful synthetic quality decision is not evidence of LLM improvement.

The bounded continuation reopens those same node homes and trains the second cohort. It advances round 4 to round 8 and total issuance from 4,000,000 to 8,000,000 atoms. Three fresh batches and one replay batch settle after eight further stage replays. The serving root stays unchanged pending the second evaluation. All four validators agree in block 498 (`A1B3FC3894FD8F5ABAB8A7E28A701E22679F320481C069C288349A5B1C3CBA3C`). A repeated continuation invocation is refused before starting nodes or issuing work.

## Real model and native quality decision

The seed has 134,515,008 parameters. Four identity blocks expand the learning candidate to 148,675,392 parameters, requiring four partitions under the declared 48M-parameter capacity. Growth settles without issuance. Four full-model training steps settle and issue exactly 4,000,000 experimental atoms. This declared partition limit is not a claim that either physical host cannot hold the complete model.

All 288 evaluation windows settle through 74 native score claims: 69 retention and 75 fresh windows for each of the baseline and candidate. Scores are target-weighted into exactly 32 paired document observations per group.

| Group | Baseline mean response loss | Candidate mean response loss | Conservative upper bound on change | Required bound | Decision |
| --- | --- | --- | --- | --- | --- |
| Retention | 1.429667 | 1.429275 | +0.000311 nats | Below +0.020000 | Pass |
| Fresh | 1.307629 | 1.306425 | −0.000322 nats | Below −0.001000 | Fail |

The native decision at height 2,865 **rejects promotion**. The observed fresh mean improves by approximately 0.001205 nats, but its bound does not establish the prescribed minimum improvement. The original 135M model remains the serving model and becomes the next learning parent. The larger candidate and its measurements are retained; verified training payments remain valid. Display values are rounded from the integer decision, which includes the two-micronat allowance. This is one public cohort, with the statistical and data-policy limitations described in the protocol.

## Real inference, admission failure and recovery

A deliberately forged output head is rejected by native replay. The dispute publishes **113,334,000 bytes**, including the real tied embedding/head, final norm and boundary activation. Valid one-token generation then pays 1,000 atoms and refunds 6,000. The completed original driver path covers 353 independent stage checks, reconstructed from its finalized honest claims and reached execution checkpoints after its interruption. The native ledger records 74 accepted score claims, four accepted training claims, one accepted growth claim, and the refuted and accepted inference claims.

The initial second proposal (`4b85ef30b90f4224bb655fc6f16f97bc22daca5e9531bc51bfdf9d86adafe102`) fails native admission: two distinct documents contain token windows already admitted in the first cohort. Its original upstream/token review checked individual provenance and within-cohort uniqueness, but was not supplied prior admission history. The native guard works; the preparer was incomplete. The experiment driver exits and stops its validators at that point, after the inference payment has completed.

Preparation and review now accept prior admitted proposals through `--exclude-cohort`. The corrected second proposal selects replacement documents without collecting new source rows, passes source/token review with the first cohort excluded, and activates after a native supermajority vote. The recovery driver reopens the original keys, genesis, blocks and application databases; no payment, balance or counter is reset. After one-validator progress, half-power halt and restart checks, all four validators agree in block 3,163 on `4C8316E6A288703B012BC52F9377802A8FAA2122095E922415D7918994CEB3CF`.

The second-host non-voting full node independently replays from genesis and matches that hash, including the fraud dispute and payment. Its SSH peer connection needed a restart after the outage because the default peer reconnect backoff outlasted the short recovery run; the isolated follower was then configured with a five-second maximum redial period. This is a recovered experiment, not an uninterrupted service availability claim.

The subsequent real continuation reopens the same ledger and trains four steps on the corrected second cohort: three fresh windows and one window from the preceding admitted pool. Twelve additional stage replays pass. The round advances from 4 to 8 and total issuance from 4,000,000 to 8,000,000 atoms in 185.73 seconds. Because the first candidate was rejected, these updates start from the original 134,515,008-parameter serving model. The serving root stays unchanged pending the second candidate's evaluation; the experiment stops before that evaluation.

All four validators and the second-host follower match the application hash in block 3,346: `C85B7363D9E0511A00ADCCF4F26D170B5C87B34EE72BA4C6F778CA90F9E5B50C`. The test processes are stopped after verification, with their data retained for inspection. This demonstrates continued paid training after data admission and recovery, not an unattended public continual-learning service.

## Release checks and limits

The final extracted source distribution passes **221 tests in 124.36 seconds outside the checkout**. Wheel/source asset checks and package-metadata validation pass. The tests include proposal expiry, fresh-data reuse and protected-role rejection, reserved-batch binding, finite training budgets, complete document scoring, score timeout, rejected-model recovery, forged generation, escrow conservation, collector writer exclusion, independent token/source review, cross-cohort token-window exclusion, and declared validator topology. GitHub checks cover Python 3.10 and 3.12, the native consensus build, repository links, and package contents. Tests do not substitute for an independent security review.

The topology check applied to the existing public ledger finds 40 units of voting power: 20 on each host and all 40 under one operator. Losing either host leaves only half the power. This fails the declared single-host and single-operator failure checks. The isolated four-validator experiment checks process failures within its own topology; it does not repair that public deployment.

The remaining public-cutover requirements are listed in the [lifecycle guide](NATIVE_LIFECYCLE.md#requirements-before-public-cutover). No result here establishes an economical audit market, sustained quality improvement, production chat latency, or automatic scaling from an arbitrary additional peer.
