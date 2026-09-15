# Preserve expert checkpoints in native metadata

The existing native portable profile describes a single model with a shared
update cursor. The learned experts retain parent parameters at Adam age 448 while
their own tails reach independent ages such as 1024 or 560. Rewriting those ages
would misrepresent the computation.

`expert_checkpoint` encodes only the active tail and a reference to the immutable
parent. It derives every frozen parameter record and every shard commitment, then
requires the reconstructed checkpoint and learned-state roots to match the exact
numerical checkpoint. A claim cannot supply a changed frozen tensor. The codec
validates parameter coverage, shapes, optimizer recipe, actual ages, ownership and
bounded update windows without importing PyTorch, Transformers or Safetensors.

The [archived-checkpoint check](../config/experiments/native-expert-checkpoint-results.json)
round-trips the first expert at step 1024 and the second at steps 0, 280 and 560.
Each complete checkpoint has 218 tensor references; the compact representation has
18. The terminal second checkpoint shrinks from 39,463 to 3,683 JSON bytes while
reconstructing its original identity. Tensor payloads are not compressed by this
operation.

Update identity binds the consumed numerical batch, parameter object identities,
optimizer recipe, actual cursor and numerical execution profile. It excludes job,
worker and serving-graph labels, so renaming a job cannot make the same prescribed
update payable again. The execution auditor must derive the batch root from the
canonical numerical inputs, excluding descriptive document metadata. This is not
a test for arbitrary mathematical equivalence across serialization or numerical
profiles.

The existing portable checkpoint validator was moved unchanged into a module
without neural imports and remains available through its previous import path.
Native portable settlement and lifecycle regression tests pass.

This change is the checkpoint representation only. It adds no transaction type,
job activation, graph admission, issuance, inference payment or serving promotion.
Those transitions still need to connect to funded execution audits and separate
quality approval. A metadata match is not a proof that training ran correctly.
The second expert currently has durable checkpoints at 0/280/560; settling windows
of at most four steps requires reconstructing and verifying the intervening
checkpoints, not treating a 280-step jump as one accepted window.

The [window replay plan](../config/experiments/expert-window-replay.json) freezes a non-issuing reconstruction of all 560 updates from the archived feature bank. Each intermediate weight/Adam commitment uses the same Safetensors bytes as a normal checkpoint; the replay records 140 windows of at most four updates and checks the actual saved 0/280/560 roots. Its CPU test reproduced the full five-owner training result. The cached prefix still needs its own execution audit before native acceptance.

The [first GPU attempt](../config/experiments/expert-window-replay-attempt-0.json)
matched the first 25 updates but was stopped early because checkpoint recording
could not finish the full trajectory within its frozen deadline. Its observations
are preserved and the instance, disk and security group retired. The replacement
uses a streaming encoder for the same restricted float32 Safetensors bytes,
avoiding large temporary byte strings. Compatibility tests compare its hashes and
lengths with the pinned writer, including ordinary complete weight/Adam
checkpoints. The numerical training kernel, data, update identity and runtime are
unchanged. The new plan additionally requires all 25 recorded intermediate GPU
checkpoints to match. A completed GPU replay is still pending.
