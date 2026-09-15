# Several retained transformer experts

`sharded.branch_groups` shares three parent partitions across separately owned
transformer tails. Each path has its own process group. Logical layer positions
map to actual members, including noncontiguous groups such as `[0, 1, 2, 4]`.
A worker holds only its parent partition or its expert tail.

Explicit routing rules are ordered. An extension must preserve all old rules
and their precedence, so a new overlapping domain cannot redirect an existing
expert's questions. The selector consumes user text only. This is an explicit
domain policy; learned general routing and dynamic process admission are not
implemented by this module.

The five-process CPU test compares all three paths against complete reference
calculations. The controller observes the second expert's process exit before
the first expert answers again. It then observes the first expert's exit before
the three parent owners generate again. All retained outputs match exactly.
These are transport and composition checks with small initialized models, not
a second real-model learning result.

`sharded.cohort_features` prepares the new expert's exact padded training
features through its own four-member group. The three original owners retain
their complete parent partitions. After feature production, the new owner can
reuse those inputs for local AdamW updates while the earlier paths answer
through their separate groups. A read-only output-head replica is needed by
the learner; it has no optimizer and is outside the uniquely trained weights.

The five-process learning test checks the cached prefix and reference tensors
against full-model evaluation, then compares all eight updates, clipping norms,
weights and Adam state against independent full-model autograd. Earlier paths
answer between the new learner's first and last update, retain exactly the
same answers, and keep all their parameters unchanged. This establishes a
small-model execution mechanism; GPU learning quality, concurrent service
performance, arbitrary peer admission and native settlement remain separate
measurements.

The first real-model [branch experiment](BRANCH_GROWTH.md) remains independently
frozen. The multiple-expert module does not change its numerical source, graph,
questions or selection. A later cohort requires its own committed training and
quality contract before any training or settlement.
