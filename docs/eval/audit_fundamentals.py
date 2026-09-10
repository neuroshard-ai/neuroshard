#!/usr/bin/env python3
"""Reproduce analytical examples and check recorded evidence for the theory review.

These are calculations and artifact checks, not new training/security experiments.
Run from any directory; no third-party dependencies are needed.
"""

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def calculate():
    convergence = json.loads(
        (ROOT / "docs/eval/results/e1_convergence.json").read_text()
    )
    baseline = next(r for r in convergence["runs"] if r["mode"] == "sync")["curve"][-1]
    recorded = []
    for run in convergence["runs"]:
        last = run["curve"][-1]
        payload = last["bytes_per_worker"]
        recorded.append({
            "mode": run["mode"],
            "final_validation_loss": last["val_loss"],
            "loss_increase_over_sync_percent": 100 * (
                last["val_loss"] / baseline["val_loss"] - 1
            ),
            "outbound_update_payload_bytes_per_worker": payload,
            "payload_reduction_vs_sync": (
                baseline["bytes_per_worker"] / payload if payload else None
            ),
        })

    pipelines = []
    for alpha in (0.05, 0.10, 0.20, 0.30):
        for stages in (1, 4, 8):
            pipelines.append({
                "adversarial_node_fraction": alpha,
                "stages": stages,
                "probability_at_least_one_bad_stage_independent_placement": (
                    1 - (1 - alpha) ** stages
                ),
            })

    # A fixed population, sampled without replacement, illustrates that the
    # independent-placement approximation is not the security assumption.
    population, bad_nodes, stages = 100, 10, 8
    finite_bad_probability = 1 - (
        math.comb(population - bad_nodes, stages) / math.comb(population, stages)
    )

    sampling = []
    transitions, checked = 100, 5
    for invalid in (1, 10, 100):
        unobserved = (
            math.comb(transitions - invalid, checked)
            / math.comb(transitions, checked)
            if transitions - invalid >= checked else 0.0
        )
        sampling.append({
            "transitions": transitions,
            "invalid_transitions": invalid,
            "uniform_checks_without_replacement": checked,
            "detection_probability": 1 - unobserved,
        })

    boundary_bytes = 2 * 1 * 2048 * 4096 * 2
    bandwidth_bytes_per_second = 100_000_000 / 8
    checkpoint_bytes = 8_000_000_000 * 12  # fp32 weights and two Adam moments
    audit_probability = 0.05
    return {
        "scope": "Analytical illustrations and existing artifact checks; not measured network security.",
        "recorded_convergence": recorded,
        "pipeline_contamination": {
            "assumption": "One Byzantine stage can invalidate an unverified pipeline contribution.",
            "independent_placement": pipelines,
            "finite_population_example": {
                "population": population,
                "adversarial_nodes": bad_nodes,
                "stages": stages,
                "probability_at_least_one_bad_stage": finite_bad_probability,
            },
        },
        "identity_splitting": [
            {"identities": k, "total_batches": 100,
             "sum_sqrt_batches": k * math.sqrt(100 / k),
             "influence_multiplier": math.sqrt(k)}
            for k in (1, 4, 16, 100)
        ],
        "idealized_bandwidth": {
            "batch": 1, "sequence_length": 2048, "hidden_width": 4096,
            "bytes_per_activation_element": 2,
            "forward_plus_backward_bytes_per_boundary": boundary_bytes,
            "link_bits_per_second": 100_000_000,
            "serialized_transfer_seconds_per_boundary": (
                boundary_bytes / bandwidth_bytes_per_second
            ),
            "note": "Shared bidirectional bandwidth budget; excludes compute, latency, framing and overlap.",
            "fp32_8b_weights_and_adam_checkpoint_bytes": checkpoint_bytes,
            "checkpoint_fetch_seconds": checkpoint_bytes / bandwidth_bytes_per_second,
        },
        "audits": {
            "transition_sampling": sampling,
            "independent_full_claim_audit_probability": audit_probability,
            "probability_all_100_frauds_escape_if_detection_certain_when_audited": (
                (1 - audit_probability) ** 100
            ),
            "expected_unaudited_frauds_out_of_100": 100 * (1 - audit_probability),
            "strict_penalty_reward_ratio_lower_bound_for_zero_cost_fraud_vs_abstention": (
                (1 - audit_probability) / audit_probability
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Also write the calculation as JSON.")
    args = parser.parse_args()
    result = calculate()
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
