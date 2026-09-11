# Runtime map

- `client`: supported `neuroshard` CLI, local keys, native joining and paid inference.
- `inference`: the public 0.4.0 adapter-training and paid-inference application.
- `dataflow`: immutable data ingestion, source collection and recovery.
- `evolution`: experimental full-model training, growth, evaluation and native disputes.
- `publicnet`: inherited bootstrap, API, explorer index and worker transport.
- `lab`: native state machine, bonded membership, storage, and experiments.
- `demo`: reference execution, wire primitives, ABCI bindings, and verification.
- `core/model/llm.py`, `core/crypto/ecdsa.py`: model and signature primitives used by the native profile.

Other modules retain historical prototypes and may require unsupported dependencies. They are not the public entry points. The current evolution source commitment includes every Python module in this package, and released profiles also bind specific inherited files. Removing or moving them is a compatibility change, separate from archiving manuscripts and publishing projects. The repository cleanup preserves their bytes.

Use the commands and installation instructions in the [project README](../../README.md). The tiny corpus at `docs/eval/data/input.txt` and its packaged copy are compatibility fixtures used by reference conformance checks.
