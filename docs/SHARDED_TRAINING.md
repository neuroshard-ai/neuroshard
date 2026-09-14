# Persistent training across model shards

This operated implementation trains a single model whose layer weights and AdamW state reside on different workers. Workers exchange forward activations and backward gradients; no worker or coordinator constructs the whole model. The tied embedding, output head and final norm have one owner, so their two gradient contributions accumulate on the same parameter. Each optimizer step uses one batch, target normalization and globally reduced clipping norm across all shards.

The implementation is in `src/neuroshard/evolution/sharded/`; the driver is [run_sharded_training.py](../scripts/run_sharded_training.py). It uses pinned Hugging Face Llama blocks with PyTorch Gloo transport and synchronous microbatches. CUDA retains FP32 weights, gradients and Adam moments, with BF16 autocast for computation. This numerical profile is separate from the released native CPU protocol. Fixed logical shard slots can move to another physical host; automatic repartition is not implemented.

Checkpoints are written only between complete optimizer updates. Each worker saves its own parameters, both Adam moments, per-parameter optimizer cursor and Python/NumPy/PyTorch RNG states. A common manifest binds all shard manifests, the job's data schedule, numerical sources, model configuration, previous committed checkpoint and completed step. Workers acknowledge the same manifest before advancing their local HEAD. Recovery requires an explicitly chosen common checkpoint and its complete shard artifacts; independently choosing each worker's latest local file is forbidden. A partially prepared shard checkpoint cannot be combined with another model version.

The controller must preserve a complete checkpoint outside a departing host before replacing it. A replacement restores only its logical shard. Uncommitted work after the selected checkpoint is replayed; the learning process continues from the committed optimizer state. This is coordinated crash recovery over operated hosts. Manifest agreement and hash verification do not prove honest computation by malicious workers, independent storage ownership or a permissionless quorum.

The [bounded lifecycle plan](../config/experiments/persistent-shards.json) starts from the previously learned 1.7B compressed checkpoint. That older run retained model weights only, so the new phase initializes AdamW once. Subsequent recovery and the second data cohort preserve AdamW. The first 32 updates train 768 new generated tasks plus 256 previously trained public conversations; the next 32 train 512 new tasks, replay 256 task documents actually trained in phase one, and retain the conversation replay. New examples are drawn from public task families; the retention documents were previously exposed. This is not a broad assistant benchmark.

Two 24 GB A10G GPUs hold 838,907,904 and 872,468,480 parameters respectively. The full FP32 weights/gradients/Adam state would require 27,382,022,144 bytes before activations, exceeding either GPU. Peak actual allocation and resident parameter coverage must be recorded; no CPU optimizer offload or full-model loading is allowed on the workers. Seed conversion streams and verifies one source file/tensor at a time, then delivers only the assigned tensors to each worker.

The lifecycle compares an uninterrupted checkpoint at step 32 with restoration from step 16 after a worker disappears during step 20. The replacement host must receive its state from preserved artifacts, then reproduce the complete step-32 manifest, including optimizer and RNG state. The recovered group then learns the second cohort and performs inference through its shards. Inference currently recomputes the prefix and serves a fixed checkpoint; KV-cache and throughput optimization are separate work.

Before the GPU run, a small-model test compares shard-local AdamW against monolithic autograd with unequal sequence lengths, target weights, a partial microbatch, tied weights and active global clipping. Fresh worker processes restore an earlier common checkpoint and must reproduce the exact final manifest and generated tokens. Corrupted tensor bytes and inconsistent optimizer cursors must be rejected. The monolithic model exists only in this small reference test.

Run that reference test in the pinned CPU development environment:

```bash
ATEN_CPU_CAPABILITY=default MKL_ENABLE_INSTRUCTIONS=SSE4_2 \
  python -m pytest -q tests/evolution/test_sharded_training.py
```

For a GPU run, install [learning-reference-requirements.txt](learning-reference-requirements.txt) in a separate environment on each worker. Each worker needs the same prepared job and data files, its own seed manifest and tensor files, and the pinned configuration/tokenizer, including `chat_template.jinja`. Copy the complete tokenizer asset set: copying only its JSON files can omit the saved template and correctly fails tokenizer identity checks. The [input commitment](../config/experiments/persistent-shards-inputs.json) records their hashes and the complete schedule. Large seed/checkpoint objects and the exact preparation scripts are retained with the experiment artifacts in operator storage; the input commitment alone is not a downloadable model distribution.

Start one process per worker, setting `RANK` to its logical slot, `WORLD_SIZE=2`, `MASTER_ADDR` to the first worker's reachable private address, `MASTER_PORT` to the same free port, and `GLOO_SOCKET_IFNAME` to the peer network interface:

```bash
PYTHONPATH=src python scripts/run_sharded_training.py train \
  --prepared /data/inputs/prepared.json --seed /data/owned-seed \
  --home /data/run --until 32
```

To recover, first stop the old group and deliver the selected common `commit-000016.json` plus each owner's `shard-000016/` directory to the appropriate worker. Start a fresh group with the same logical slots and add `--resume /data/restored/commit-000016.json`, using a new output directory. Both workers must select the identical common manifest. Use `--until 64` with the step-32 checkpoint to continue the second cohort; use `evaluate` with a committed checkpoint to score and generate through the shards. Rendezvous, membership changes and storage replication are controller responsibilities; this driver does not discover or trust arbitrary public peers.

The implementation issues no NEURO, promotes no native serving model and establishes no model growth. The [completed lifecycle](SHARDED_TRAINING_RESULTS.md) records exact physical-host recovery, continued optimizer state, distributed inference and the observed model-quality regression.
