"""The accepted assistant served across Granite owners: A2's update, not the round-4 addition.

A4's serving execution with the arm the assistant actually accepted: the round-4 update
of layers 32-39, held by owner 2 as plain projections and switched per episode. The
served development episodes must reproduce the single-host development result of A2's
third attempt token for token.
"""

from neuroshard.evolution import granite_shard_execution as shard
from neuroshard.evolution import granite_shard_serving as serving
from neuroshard.evolution.modular_reference_execution import ROOT

PLAN = 'config/experiments/granite-shard-serving-update.json'
SCRIPT = 'scripts/run_granite_shard_serving_update.py'
PROFILE = 'granite-shard-serving-update'
PHASES = serving.PHASES
UPLOADED = serving.UPLOADED


def committed_sources(root=ROOT):
    return shard.committed_sources(root, PLAN)


def configure():
    shard.configure()


def freeze():
    return shard.freeze(PLAN)


def owner(rank, address, port, phase, home, store, index=0):
    return serving.owner(rank, address, port, phase, home, store, index, plan_path=PLAN)


def assess_phases(plan, fetches, phases):
    return serving.assess_phases(plan, fetches, phases)
