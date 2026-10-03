"""Selection that cannot read the hidden answer.

The append-only execution passed only by consulting correctness flags. This
contract refuses those inputs. It does not train or authorize a GPU.
"""
import json
from pathlib import Path

from neuroshard.evolution.reference_data import identity


FORMAT = "neuroshard-observable-selection-v1"
CONTRACT_IDENTITY = "bb2fb79e531c0ed425bbb8a0c0e5db0f18cf8b7af7afd0487628baedb5583b71"
PLAN = "config/experiments/observable-selection.json"
FORBIDDEN = {"gold_answer", "passed", "parsed_answer_equals_gold"}


def spec_path():
    marker = Path(PLAN)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / marker
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("observable-selection.json is not next to this source tree")


def load_spec():
    spec = json.loads(spec_path().read_text())
    if identity(spec) != CONTRACT_IDENTITY:
        raise ValueError("Observable-selection contract changed; declare a separate experiment")
    if spec.get("train") is not False or spec.get("gpu_launch_authorized") is not False:
        raise ValueError("This specification does not authorize training or a GPU launch")
    if spec.get("later_execution_authorized") is not False:
        raise ValueError("A later execution freeze is required before any run")
    if set(spec.get("selector_must_not_read") or []) != FORBIDDEN:
        raise ValueError("The selector must not read the hidden answer")
    return spec


def choose(observation):
    """Pick parent or added from observable fields only."""
    load_spec()
    leaked = FORBIDDEN & set(observation)
    if leaked:
        raise ValueError("Selector received the hidden answer: " + ", ".join(sorted(leaked)))
    required = {"question_text", "parent_generation", "added_generation"}
    if not required <= set(observation):
        raise ValueError("Selector is missing the question or a generated reply")
    if observation.get("model_internal_scores", {}).get("added", 0) > observation.get(
            "model_internal_scores", {}).get("parent", 0):
        return "added"
    return "parent"
