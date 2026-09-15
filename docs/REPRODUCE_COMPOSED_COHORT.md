# Reproduce the five-owner continual-learning experiment

Download `neuroshard-cohort-evidence.tar.gz` and `SHA256SUMS` from the
[research release](https://github.com/neuroshard-ai/neuroshard/releases/tag/research-cohorts-20260915).

This bundle restores a second learned expert alongside the earlier expert and
checks new answers, retention and the earlier model's operation after the new
expert exits. `numerical-result.json` contains the closed decision. The original
560 training updates are separate from this read-only final evaluation.

Three owners hold disjoint partitions of both the 1.7B trained parent and the
preserved 1.7B interpreter. A fourth holds the first 134M learned expert; a fifth
holds the new 134M expert. The largest owner holds 1,208,033,280 parameters.
No owner holds a complete 1.7B model. The additional experts replace the last
two layers on explicitly routed request paths. They do not make a single dense
3.7B model. This is an operated research reproduction with one operator;
the Gloo process group does not implement public peer admission.

Use five Linux x86_64 machines with NVIDIA A10G GPUs, Python 3.10.12 and a
private network connecting all five processes. The original run used five
`g5.xlarge` instances, each with 16 GiB host RAM and a 24 GB GPU. Reserve disk
for the pinned CUDA environment and each owner's assigned objects. The frozen
runtime and generation settings are in `plans/composed-cohort.json`.

Clone `https://github.com/neuroshard-ai/neuroshard` and check out the exact
`selection_commit` in `manifest.json`. Install
`docs/learning-reference-requirements.txt` in a virtual environment. That commit
contains the committed prepared record and selection; its neural files remain
pinned to `numerical_source`. Do not run this frozen experiment from an arbitrary
newer checkout. Check the bundle's published SHA-256 before extracting it on
each host. `manifest.json` also lists every included file's SHA-256.

## Download each owner's objects

From the extracted bundle on each host, change the rank below to 0 through 4:

```bash
python prepare_owner.py --rank 0
python fetch_objects.py < objects-request.json
python fetch_objects.py < interpreter-request.json
python fetch_objects.py < second-request.json
python prepare_owner.py --rank 0 --link-parent
```

The helper uses only public release URLs in `models.json`. The downloader
streams and checks every complete object's length and SHA-256 before replacing
its destination. Correct existing objects are reused. Rank 4 also links its
original parent tail into the restored checkpoint directory, as required by
the checkpoint loader's provenance checks. The other ranks need no second
expert tensors. No AWS account or GitHub authentication is needed.

## Run the published final

Set the appropriate rank and local interface on each host, using the same
private rank-0 address and port. All five processes must start and their private
network must permit Gloo connections. Use absolute paths and a fresh output
directory on every host. Capture each process's real exit status.

```bash
export RANK=0 WORLD_SIZE=5
export MASTER_ADDR=RANK_ZERO_PRIVATE_ADDRESS MASTER_PORT=29927
export GLOO_SOCKET_IFNAME=YOUR_PRIVATE_INTERFACE
export PYTHONPATH=/absolute/checkout/src
export ATEN_CPU_CAPABILITY=default MKL_ENABLE_INSTRUCTIONS=SSE4_2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python /absolute/checkout/scripts/run_composed_cohort.py final \
  --inputs /absolute/bundle/inputs \
  --parent /absolute/bundle/state/parent.json \
  --first /absolute/bundle/state/first.json \
  --first-objects /absolute/bundle/state/objects \
  --second /absolute/bundle/state/second/selected-checkpoint.json \
  --objects /absolute/bundle/state/objects \
  --interpreter /absolute/bundle/state/interpreter \
  --seed /absolute/bundle/tokenizer \
  --home /absolute/new-output-directory
```

After the complete evaluation, rank 4 exits. Only after observing its actual
successful process exit, write the following receipt on each remaining host
within five minutes. The atomic `save` helper prevents a concurrent reader from
seeing a partially written receipt.

```python
import json
from pathlib import Path
from neuroshard.evolution.reference_data import save

home = Path('/absolute/new-output-directory')
result = json.loads((home / 'result.json').read_bytes())
binding = {key: result[key] for key in
           ('plan', 'prepared', 'job', 'previous_graph', 'retention_cache')}
save(home / 'new-expert-exited.json',
     {'binding': binding, 'exit_code': 0, 'process_exit_observed': True})
```

The four remaining owners then produce earlier answers and `survival.json`.
Writing a receipt without observing the exit does not establish availability.
Keep all five exit statuses with the reports. Compare each owner's graph,
answer identity and complete decision with `outcomes/final`. This reproduces
a published result; it is not another independent quality final.

The new domain uses known NeuroShard documentation facts, held-out wording and
held-out pairs. Each paired request executes two real neural calls through the
second expert and joins their unchanged answers. The grammar and domain routing
are fixed; this result does not establish general planning or ChatGPT quality.
Earlier answers are exposed retention probes and must remain token-exact,
including earlier incorrect answers. They are not new independent test data.

## Training and verification evidence

`training-inputs` records the original 896 training examples and the fixed
560-update schedule. `training-outcomes` includes the original training metrics
and checkpoint metadata at updates 0, 280 and 560. The second expert's actual
560-update Adam state is included in its model objects; the parent retains its
own 448-update state and the first expert retains its 1,024-update state.

`verification` contains the independently executed prefix and update replay
reports. The prefix replay rebuilt the committed feature bank one parent
partition at a time. The update replay matched all 560 metrics and all 28
previously recorded checkpoints in 1,188.22 seconds, producing 140 bounded
four-update windows. These executions used separate machines under the same
operator, not independent operators or cryptographic proofs.

The optional training feature bank is a separate tar asset listed under
`feature_bank` in `models.json`; verify its complete SHA-256 before extracting.
The ordinary five-owner final does not need it. Intermediate replay checkpoints
contain exact commitments, but their intermediate weight and Adam payloads were
not durably published. Prefix and replay reports identify their own frozen
source commits, distinct from the final evaluation checkout.

These artifacts issue no tokens and do not change the public serving model.
Native settlement of this expert, graph promotion and paid inference through
this graph require their own completed integration. Source and upstream terms
are included in `LICENSE`, `NOTICE` and `THIRD_PARTY.md`.
