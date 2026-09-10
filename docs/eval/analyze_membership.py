#!/usr/bin/env python3
"""Resource-concentration and reservation-cost scenarios, not market forecasts."""
import argparse
import json
from pathlib import Path


def concentration(worker_share, epochs=2000, activation_delay=4):
    honest, adversarial = 90.0, 10.0
    pending = []
    first_crossing = None
    points = []
    for epoch in range(1, epochs + 1):
        if len(pending) >= activation_delay:
            a, h = pending.pop(0)
            adversarial += a
            honest += h
        weight_share = adversarial / (honest + adversarial)
        # One unit per epoch: 80% to work and 20% to consensus, before fees/missed votes.
        reward_share = 0.8 * worker_share + 0.2 * weight_share
        pending.append((reward_share, 1.0 - reward_share))
        if weight_share >= 1 / 3 and first_crossing is None:
            first_crossing = epoch
        if epoch in (1, 25, 50, 100, 200, 500, epochs):
            points.append({"epoch": epoch, "adversarial_voting_share": weight_share})
    return {"adversarial_work_share": worker_share, "first_epoch_at_one_third": first_crossing,
            "voting_share_at_end": adversarial / (honest + adversarial), "trajectory": points}


def run():
    return {"type": "Analytical scenarios with explicit assumptions; not an observed network attack",
            "assumptions": ["Initial voting stake 100, adversarial stake 10", "One unit of rewards each epoch",
                "All rewards are bonded after four epochs", "Worker reward share 80%, consensus share 20%",
                "Fixed adversarial compute allocation; no transfers, prices, missed votes, or penalties before threshold",
                "The actor performs valid work while accumulating stake"],
            "concentration": [concentration(fraction) for fraction in (0.1, 0.3, 0.5, 0.8)],
            "inference": "Correct work and delayed activation do not preserve a less-than-one-third adversarial stake bound when issuance ownership concentrates.",
            "reservation_budget_example": {"budget_atoms": 10_000_000, "bond_atoms": 2_000_000,
                "fee_atoms": 1000, "maximum_fully_forfeited_reservations": 10_000_000 // 2_001_000,
                "maximum_reserved_block_intervals": (10_000_000 // 2_001_000) * 17,
                "assumptions": "Each reservation expires, no rewards or transfers replenish the attacker, and the next task is available immediately after expiry"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results/protocol_economics.json")
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([{k: row[k] for k in ("adversarial_work_share", "first_epoch_at_one_third")}
                      for row in result["concentration"]], indent=2))
