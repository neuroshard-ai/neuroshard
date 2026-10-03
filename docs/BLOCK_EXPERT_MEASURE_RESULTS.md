# Block-expert measurement: completed, candidate rejected

The [measurement contract](BLOCK_EXPERT_MEASURE.md) ran on September 22, 2026.
Parent retention did not block training. The added blocks and the matched
in-place control both trained. **The candidate failed. Nothing is admitted.**

| Automatic arm | New modular-addition answers | Ordinary-addition retention |
| --- | ---: | ---: |
| Parent | 0/64 | 8/64 |
| Added identity blocks | 10/64 | 0/64 |
| Existing blocks, same CPU budget | 11/64 | 0/64 |

The parent left all 64 new replies unfinished. The added blocks gained 10 new
answers over the parent and lost all 8 protected retention answers. The control
gained 11 new answers and also lost all 8. The required competence bar was 24/64
and a gain over both arms. Cost, latency, memory, and frozen parent weights
passed. Those checks do not offset the answer result.

The local study is `.neuroshard/block-expert-measure-20260922`. Wall time was
3,265 seconds. No GPU was used. The opened cases stay closed. This result does
not authorize another in-place training run, a selector on these replies, a
GPU, or checklist credit.

The next contract is [append-only growth](APPEND_ONLY_GROWTH.md): the parent
remains the default answer, and a new shard can be used only where the parent
missed.
