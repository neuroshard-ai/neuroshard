# Frozen-feature GPU results

Four model owners reproduced all eight optimizer updates exactly in both tested layouts. Stored prefix activations produced identical losses, gradient norms, weights and Adam states to ordinary distributed training. Each owner held its assigned layers; the learner also held a frozen output-head replica.

| Layout | Exact updates | Distributed update time | Cached update time | Ratio | Initial feature production |
| --- | ---: | ---: | ---: | ---: | ---: |
| append | 8/8 | 27.25 s | 5.80 s | 4.70× | 14.62 s |
| tail-control | 8/8 | 27.11 s | 5.75 s | 4.71× | 14.42 s |

Times above measure optimizer execution over four unique padded batches, repeated twice. Feature production and artifact preservation are additional costs. The result enables measuring longer training on unchanged representations; it does not demonstrate new knowledge or improved answers.

The first attempt failed before training because the agreement declaration mistakenly required equal host names. The corrected source compares numerical settings and records each host separately. Its three-process regression test accepts distinct hosts and rejects a changed numerical runtime. The failed logs are retained in the evidence archive.

The probe reused the completed growth comparison’s four A10G workers after its committed selection and exact recovery. All workers were operated by one operator. Both experiments were preserved before requesting allocation cleanup. No tokens were issued and serving was unchanged.

The [machine-readable results](../config/experiments/frozen-feature-results.json) bind the source, parent, preparation, all intermediate checks, final tail/Adam states and fully verified S3 evidence. See [the method](FROZEN_FEATURES.md) for its ownership and verification limits.
