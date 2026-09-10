#!/usr/bin/env python3
"""Generate paper figures/tables from experiment JSONs."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "..", "figures")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 200,
    # IEEE PDF eXpress rejects Type 3 fonts; embed TrueType (Type 42) instead
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

LABELS = {
    "single": "Single node",
    "sync": "Synchronous DP (sync every step)",
    "diloco_h50": "DiLoCo $H{=}50$",
    "diloco_h250": "DiLoCo $H{=}250$",
    "diloco_h50_comp": "DiLoCo $H{=}50$ + Top-$K$ compression",
}
STYLE = {
    "single": dict(color="gray", ls=":"),
    "sync": dict(color="black", ls="-"),
    "diloco_h50": dict(color="tab:blue", ls="-"),
    # This recorded run has one evaluation point; a marker keeps it visible.
    "diloco_h250": dict(color="tab:orange", ls="--", marker="o", ms=3),
    "diloco_h50_comp": dict(color="tab:green", ls="-."),
}


def fig_convergence():
    with open(os.path.join(RES, "e1_convergence.json")) as f:
        e1 = json.load(f)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.05))
    for run in e1["runs"]:
        mode = run["mode"]
        xs = [p["step"] for p in run["curve"]]
        ys = [p["val_loss"] for p in run["curve"]]
        ax1.plot(xs, ys, label=LABELS.get(mode, mode), **STYLE.get(mode, {}))
        if mode != "single":
            bs = [max(p["bytes_per_worker"], 1) / 1e6 for p in run["curve"]]
            ax2.plot(bs, ys, label=LABELS.get(mode, mode), **STYLE.get(mode, {}))
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Validation loss")
    ax1.set_title("(a) Loss vs. steps")
    ax1.legend(frameon=False)
    ax2.set_xscale("log")
    ax2.set_xlabel("Outbound update payload (MB per worker, log)")
    ax2.set_title("(b) Loss vs. update payload")
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIGS, "eval_convergence.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)

    # print communication summary for the paper text
    for run in e1["runs"]:
        last = run["curve"][-1]
        print(f"{run['mode']:18s} final={last['val_loss']:.3f} "
              f"bytes/worker={last['bytes_per_worker']/1e6:.2f} MB")


def fig_byzantine():
    with open(os.path.join(RES, "e2_byzantine.json")) as f:
        e2 = json.load(f)
    runs = e2["runs"]
    clean = next(r for r in runs if r["attack"] == "none")

    fig, ax = plt.subplots(figsize=(3.2, 1.75))
    ax.plot([p["round"] for p in clean["curve"]],
            [p["val_loss"] for p in clean["curve"]],
            color="gray", ls=":", label="No attack (mean)")
    colors = {"mean": "tab:red", "trimmed_mean": "tab:blue",
              "median": "tab:orange", "multi_krum": "tab:green"}
    names = {"mean": "Mean (no defense)", "trimmed_mean": "Trimmed Mean",
             "median": "Coordinate-wise median", "multi_krum": "Multi-Krum"}
    for r in runs:
        if r["attack"] == "signflip" and r["n_byz"] == 3:
            ax.plot([p["round"] for p in r["curve"]],
                    [p["val_loss"] for p in r["curve"]],
                    color=colors[r["strategy"]], label=names[r["strategy"]])
    ax.set_yscale("log")
    ax.set_xlabel("Outer round")
    ax.set_ylabel("Validation loss (log)")
    ax.legend(frameon=False, ncol=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIGS, "eval_byzantine.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)

    # table numbers
    print("\nByzantine final losses:")
    print(f"clean mean: {clean['curve'][-1]['val_loss']:.3f}")
    for atk in ["signflip", "noise"]:
        for r in runs:
            if r["attack"] == atk and r["n_byz"] == 3:
                print(f"{atk:9s} {r['strategy']:13s} {r['curve'][-1]['val_loss']:.3f}")
    print("\nTrimmed-mean breakdown sweep (signflip):")
    for r in runs:
        if r["strategy"] == "trimmed_mean" and r["attack"] == "signflip":
            print(f"  f={r['n_byz']}/10 -> {r['curve'][-1]['val_loss']:.3f}")


def print_micro():
    with open(os.path.join(RES, "e3_compression.json")) as f:
        e3 = json.load(f)
    print("\nCompression:")
    for row in e3["per_topk"]:
        print(f"  k={row['top_k_ratio']}: {row['ratio']:.1f}x cosine={row['mean_cosine']:.3f}")
    with open(os.path.join(RES, "e4_ponw.json")) as f:
        e4 = json.load(f)
    print("\nPoNW:", json.dumps(e4, indent=1))


if __name__ == "__main__":
    fig_convergence()
    fig_byzantine()
    print_micro()
