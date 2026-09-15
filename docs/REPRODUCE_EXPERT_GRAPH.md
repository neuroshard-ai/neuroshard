# Reproduce the four-owner model experiment

These are research checkpoints and numerical evidence. They do not change the
public 0.4.0 client or activate a native serving graph. The completed score is
in `numerical-result.json`; a failed gate remains a failed gate.

The parent contains 1,711,376,384 parameters and the learned expert replaces
the last two layers on its own request path. The original 1.7B interpreter is
preserved separately. Three owners each hold disjoint portions of both 1.7B
models; the fourth holds the 134,225,920-parameter learned expert. Model objects
include the actual Adam state where training occurred. No owner needs every
model object in this release.

Use four Linux x86_64 hosts with NVIDIA A10G GPUs, Python 3.10.12 and a private
network connecting all four processes. The frozen runtime is in
`plans/preserved-interpreter.json`. The experiment used four `g5.xlarge`
instances, with 16 GiB host RAM and a 24 GB GPU each. Allow sufficient disk for
the pinned CUDA environment and the assigned objects. The largest owner has
about 8.5 GB of model objects. This is an operated reproduction procedure;
the Gloo process group is not public peer admission.

Clone `https://github.com/neuroshard-ai/neuroshard` and check out the
`selection_commit` in `manifest.json`. That commit contains the prepared record
and selected graph; its numerical files are pinned to `numerical_source`.
Install `docs/learning-reference-requirements.txt` in a virtual environment.
Extract this evidence bundle on each host. Check its published SHA-256 before
using any contents. Source and upstream notices are in the repository and the
included `LICENSE`, `NOTICE` and `THIRD_PARTY.md` files.

## Download only one owner's objects

Run the following from the extracted bundle, changing `rank` to 0, 1, 2 or 3.
It writes two download requests and the checkpoint metadata. It does not load
any neural weights or contact AWS. `models.json` supplies public release URLs.

```python
import json
from pathlib import Path

rank = 0
models = json.loads(Path('models.json').read_bytes())
plan = json.loads(Path('plans/preserved-interpreter.json').read_bytes())
state = Path('state')
state.mkdir(exist_ok=True)

def owner(name):
    if name in ('model.embed_tokens.weight', 'model.norm.weight'):
        return 0
    layer = int(name.split('.')[2])
    bounds = plan['parent_layout']
    return next(i for i, (a, b) in enumerate(zip(bounds, bounds[1:])) if a <= layer < b)

records = models['parent']['tensors'] if rank < 3 else models['expert']['tensors']
names = [name for name in records if owner(name) == rank] if rank < 3 else [
    name for name in records
    if name.startswith('model.layers.') and int(name.split('.')[2]) >= plan['split']]
trained = {records[name]['sha256'] for name in names}
original = {spec['sha256'] for spec in models['interpreter']['partitions'][str(rank)]['tensors'].values()} if rank < 3 else set()
for label, digests in [('objects', trained), ('interpreter', original)]:
    request = {'destination': str(state / label), 'objects': [
        {'file': sha + '.safetensors', 'sha256': sha,
         'bytes': models['objects'][sha]['bytes'], 'url': models['objects'][sha]['url']}
        for sha in sorted(digests)]}
    Path(label + '-request.json').write_text(json.dumps(request))
for label in ('parent', 'expert'):
    (state / (label + '.json')).write_text(json.dumps(models[label]))
```

Run the included downloader for both requests. It streams each object, verifies
its complete length and SHA-256, and replaces the destination only after the
check passes. An already correct object is reused.

```bash
python fetch_objects.py < objects-request.json
python fetch_objects.py < interpreter-request.json
```

## Execute the selected final

On each host, use the same private address for rank 0 and the appropriate local
interface name. Set `RANK` differently on each host. All four processes must
start and their private network must permit the Gloo connections. Use absolute
paths for the checkout and extracted bundle. Choose an output directory that
does not already exist.

```bash
export RANK=0 WORLD_SIZE=4
export MASTER_ADDR=RANK_ZERO_PRIVATE_ADDRESS MASTER_PORT=29917
export GLOO_SOCKET_IFNAME=YOUR_PRIVATE_INTERFACE
export PYTHONPATH=/absolute/checkout/src
export ATEN_CPU_CAPABILITY=default MKL_ENABLE_INSTRUCTIONS=SSE4_2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python /absolute/checkout/scripts/run_preserved_interpreter.py final \
  --inputs /absolute/bundle/composition-inputs \
  --parent /absolute/bundle/state/parent.json \
  --expert /absolute/bundle/state/expert.json \
  --objects /absolute/bundle/state/objects \
  --expert-objects /absolute/bundle/state/objects \
  --interpreter /absolute/bundle/state/interpreter \
  --seed /absolute/bundle/tokenizer \
  --home /absolute/new-output-directory
```

The processes produce the complete answers and evaluation reports. Rank 3 then
exits; the other three wait for the operator to observe that exit. After actually
observing rank 3 exit with code 0, write this receipt on each remaining host
within five minutes, using that host's output directory:

```python
import json
from pathlib import Path
home = Path('/absolute/new-output-directory')
binding = json.loads((home / 'waiting-for-expert-exit.json').read_bytes())
receipt = {'binding': binding, 'expert_exit_code': 0,
           'process_exit_observed': True}
from neuroshard.evolution.reference_data import save
save(home / 'expert-exited.json', receipt)
```

Use the atomic `save` helper above for every control receipt. The original
release-bundled guide used `write_text` directly; a concurrent reader could see
an empty or partial file. This corrected publication method leaves the frozen
neural source, model and evaluation inputs unchanged.

Each remaining owner then reproduces an established answer through the
three-owner parent group and writes `survival.json`. This receipt records the
operator's observation; writing it without observing the exit does not establish
availability. Keep the real exit statuses with the reports.

Compare each owner's `graph`, `answer_identity` and complete decision with the
published final reports. Every owner should agree. The input and model manifests
contain the hashes needed to identify any difference. This reproduction checks
an already published result; it is not another independent quality final.

`training-inputs` contains the fixed supervision used for the first expert. Its
training source is separately identified in `manifest.json`. The model objects
retain the parent at its actual update 448 and the expert at its actual 1,024
tail updates; the frozen parent was not retrained 1,024 more times.
