#!/usr/bin/env python3
"""
Experiment harness for the FINE 2026 NeuroShard paper revision.

Runs small-scale but *real* experiments against the actual NeuroShard
implementation (src/neuroshard/core/...):

  e1: DiLoCo convergence vs fully-synchronous data parallelism
      (communication-loss trade-off), incl. compressed variant.
  e2: Byzantine robustness of aggregation strategies under attack.
  e3: Gradient compression microbenchmark (ratio / fidelity / speed).
  e4: PoNW microbenchmark (sign/verify/proof size/recompute cost).

Results are written as JSON to docs/eval/results/.

Usage:
    venv_build/bin/python docs/eval/run_experiments.py e1 [--smoke]
    venv_build/bin/python docs/eval/run_experiments.py all
"""

import argparse
import copy
import gc
import json
import math
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch

from neuroshard.core.model.llm import NeuroLLMConfig, NeuroLLMForCausalLM
from neuroshard.core.swarm.aggregation import (
    RobustAggregator,
    AggregationConfig,
    AggregationStrategy,
)
from neuroshard.core.swarm.diloco import OuterOptimizer
from neuroshard.core.training.distributed import GradientCompressor

RESULTS_DIR = os.path.join(ROOT, "docs", "eval", "results")
DATA_PATH = os.path.join(ROOT, "docs", "eval", "data", "input.txt")

torch.set_num_threads(int(os.environ.get("EVAL_THREADS", "2")))


# --------------------------------------------------------------------------
# Data: byte-level LM on TinyShakespeare
# --------------------------------------------------------------------------

def load_data():
    with open(DATA_PATH, "rb") as f:
        raw = f.read()
    data = torch.tensor(list(raw), dtype=torch.long)
    n_val = int(len(data) * 0.05)
    return data[:-n_val], data[-n_val:]


def get_batch(data, batch_size, seq_len, gen):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,), generator=gen)
    x = torch.stack([data[i : i + seq_len] for i in ix])
    return x


def make_model(seed, hidden=128, layers=4, heads=4, kv=2, inter=256, seq=128):
    torch.manual_seed(seed)
    cfg = NeuroLLMConfig(
        vocab_size=256,
        hidden_dim=hidden,
        num_layers=layers,
        num_heads=heads,
        num_kv_heads=kv,
        intermediate_dim=inter,
        max_seq_len=seq,
        tie_word_embeddings=True,
    )
    return NeuroLLMForCausalLM(cfg)


@torch.no_grad()
def eval_loss(model, val_data, seq_len, n_batches=8, batch_size=8):
    gen = torch.Generator().manual_seed(1234)
    model.eval()
    losses = []
    for _ in range(n_batches):
        x = get_batch(val_data, batch_size, seq_len, gen)
        out = model(x, labels=x)
        losses.append(out["loss"].item())
    model.train()
    return sum(losses) / len(losses)


def param_bytes(model):
    return sum(p.numel() for p in model.parameters()) * 4  # fp32


def state_clone(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


# --------------------------------------------------------------------------
# E1: convergence / communication
# --------------------------------------------------------------------------

def run_sync_dp(train_shards, val_data, steps, seq_len, batch_size, lr, eval_every, seed,
                model_fn=None):
    """Fully synchronous data parallelism: gradients averaged every step.

    Simulated with K workers computing grads on their own shard batch;
    the averaged gradient is applied to a single shared model.
    """
    model_fn = model_fn or make_model
    K = len(train_shards)
    model = model_fn(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    gens = [torch.Generator().manual_seed(seed * 100 + i) for i in range(K)]
    pbytes = param_bytes(model)
    curve = []
    for step in range(1, steps + 1):
        opt.zero_grad()
        total_loss = 0.0
        for k in range(K):
            x = get_batch(train_shards[k], batch_size, seq_len, gens[k])
            out = model(x, labels=x)
            (out["loss"] / K).backward()
            total_loss += out["loss"].item() / K
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_every == 0 or step == steps:
            vl = eval_loss(model, val_data, seq_len)
            # each worker transmits its gradient once per step
            comm = step * pbytes
            curve.append({"step": step, "val_loss": vl, "bytes_per_worker": comm})
            print(f"  [sync] step {step}/{steps} val_loss={vl:.4f}", flush=True)
    return {"mode": "sync", "workers": K, "curve": curve, "param_bytes": pbytes}


def run_diloco(train_shards, val_data, steps, seq_len, batch_size, lr, H,
               eval_every, seed, compress=False, top_k_ratio=0.1, model_fn=None):
    """DiLoCo: K workers train H local steps, then outer Nesterov update.

    Uses the repo's OuterOptimizer (outer lr 0.7, momentum 0.9) and, when
    compress=True, the repo's GradientCompressor with error feedback.
    """
    model_fn = model_fn or make_model
    K = len(train_shards)
    global_model = model_fn(seed)
    pbytes = param_bytes(global_model)

    workers = []
    for k in range(K):
        m = model_fn(seed)  # same init as global
        m.load_state_dict(global_model.state_dict())
        o = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
        workers.append({"model": m, "opt": o,
                        "gen": torch.Generator().manual_seed(seed * 100 + k)})

    outer_opt = OuterOptimizer(lr=0.7, momentum=0.9)
    compressor = GradientCompressor(top_k_ratio=top_k_ratio, bits=8) if compress else None
    residuals = [dict() for _ in range(K)] if compress else None

    curve = []
    comm_bytes = 0.0
    rounds = steps // H
    step_count = 0
    for r in range(rounds):
        w0 = state_clone(global_model)
        gparams = dict(global_model.named_parameters())
        deltas = []
        round_payload = 0
        for k, w in enumerate(workers):
            w["model"].load_state_dict(w0)
            for _ in range(H):
                x = get_batch(train_shards[k], batch_size, seq_len, w["gen"])
                out = w["model"](x, labels=x)
                w["opt"].zero_grad()
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(w["model"].parameters(), 1.0)
                w["opt"].step()
            # pseudo-gradient: delta = w_H - w_0 (direction of local progress)
            delta = {}
            for name, p in w["model"].named_parameters():
                d = p.data - gparams[name].data
                if compress:
                    resid = residuals[k].get(name)
                    if resid is not None:
                        d = d + resid
                    blob = compressor.compress(d)
                    round_payload += len(blob)
                    d_hat = compressor.decompress(blob)
                    residuals[k][name] = d - d_hat
                    delta[name] = d_hat
                else:
                    delta[name] = d.clone()
            if not compress:
                round_payload += pbytes
            deltas.append(delta)
        # aggregate (plain average across workers - honest cohort)
        agg = {}
        for name in deltas[0]:
            agg[name] = torch.stack([d[name] for d in deltas]).mean(dim=0)
        outer_opt.step(global_model, agg)
        comm_bytes += round_payload / K  # per-worker average
        step_count += H
        if (step_count % eval_every == 0) or (r == rounds - 1):
            vl = eval_loss(global_model, val_data, seq_len)
            curve.append({"step": step_count, "val_loss": vl,
                          "bytes_per_worker": comm_bytes})
            print(f"  [diloco H={H}{' +comp' if compress else ''}] "
                  f"round {r+1}/{rounds} (step {step_count}) val_loss={vl:.4f}",
                  flush=True)
    return {"mode": f"diloco_h{H}" + ("_comp" if compress else ""),
            "workers": K, "H": H, "compressed": compress,
            "curve": curve, "param_bytes": pbytes}


def run_single(train_data, val_data, steps, seq_len, batch_size, lr, eval_every, seed,
               model_fn=None):
    model_fn = model_fn or make_model
    model = model_fn(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(seed * 7)
    curve = []
    for step in range(1, steps + 1):
        x = get_batch(train_data, batch_size, seq_len, gen)
        out = model(x, labels=x)
        opt.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_every == 0 or step == steps:
            vl = eval_loss(model, val_data, seq_len)
            curve.append({"step": step, "val_loss": vl, "bytes_per_worker": 0})
            print(f"  [single] step {step}/{steps} val_loss={vl:.4f}", flush=True)
    return {"mode": "single", "workers": 1, "curve": curve,
            "param_bytes": param_bytes(model)}


def experiment_e1(smoke=False):
    print("=== E1: convergence / communication ===", flush=True)
    train, val = load_data()
    K = 3
    steps = 120 if smoke else 500
    eval_every = 40 if smoke else 100
    H_list = [(40 if smoke else 50), (120 if smoke else 250)]
    seq_len, batch, lr, seed = 64, 8, 3e-4, 42

    shard = len(train) // K
    shards = [train[i * shard:(i + 1) * shard] for i in range(K)]

    results = {"config": {"workers": K, "steps": steps, "seq_len": seq_len,
                          "batch_size": batch, "lr": lr,
                          "model": "NeuroLLM 128d/4L (0.62M params)",
                          "dataset": "TinyShakespeare (byte-level)"},
               "runs": []}

    t0 = time.time()
    results["runs"].append(run_single(train, val, steps, seq_len, batch, lr, eval_every, seed))
    results["runs"].append(run_sync_dp(shards, val, steps, seq_len, batch, lr, eval_every, seed))
    for H in H_list:
        results["runs"].append(run_diloco(shards, val, steps, seq_len, batch, lr, H, eval_every, seed))
        gc.collect()
    results["runs"].append(run_diloco(shards, val, steps, seq_len, batch, lr, H_list[0],
                                      eval_every, seed, compress=True))
    results["wall_seconds"] = time.time() - t0

    out = os.path.join(RESULTS_DIR, "e1_convergence.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"E1 done in {results['wall_seconds']:.0f}s -> {out}", flush=True)


# --------------------------------------------------------------------------
# E2: Byzantine robustness
# --------------------------------------------------------------------------

def make_small_model(seed):
    return make_model(seed, hidden=64, layers=2, heads=4, kv=2, inter=128, seq=128)


def attack_delta(delta, kind, gen):
    """Construct an adversarial pseudo-gradient."""
    out = {}
    for name, d in delta.items():
        if kind == "signflip":
            out[name] = -5.0 * d          # reversed and amplified direction
        elif kind == "noise":
            out[name] = torch.randn(d.shape, generator=gen) * (d.norm() * 10 / max(1, d.numel()) ** 0.5)
        else:
            out[name] = d
    return out


def run_byzantine(train_shards, val_data, strategy, attack, n_byz, rounds, H,
                  seq_len, batch_size, lr, seed):
    K = len(train_shards)
    global_model = make_small_model(seed)
    workers = []
    for k in range(K):
        m = make_small_model(seed)
        m.load_state_dict(global_model.state_dict())
        o = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
        workers.append({"model": m, "opt": o,
                        "gen": torch.Generator().manual_seed(seed * 100 + k)})
    outer_opt = OuterOptimizer(lr=0.7, momentum=0.9)
    atk_gen = torch.Generator().manual_seed(999)

    strat_map = {
        "mean": AggregationStrategy.MEAN,
        "trimmed_mean": AggregationStrategy.TRIMMED_MEAN,
        "median": AggregationStrategy.MEDIAN,
        "multi_krum": AggregationStrategy.MULTI_KRUM,
    }
    curve = []
    for r in range(rounds):
        w0 = state_clone(global_model)
        agg = RobustAggregator(AggregationConfig(
            strategy=strat_map[strategy],
            trim_fraction=0.3,
            num_byzantine=n_byz,
            use_freshness_weights=False,
        ))
        gparams = dict(global_model.named_parameters())
        for k, w in enumerate(workers):
            w["model"].load_state_dict(w0)
            for _ in range(H):
                x = get_batch(train_shards[k], batch_size, seq_len, w["gen"])
                out = w["model"](x, labels=x)
                w["opt"].zero_grad()
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(w["model"].parameters(), 1.0)
                w["opt"].step()
            delta = {name: p.data - gparams[name].data
                     for name, p in w["model"].named_parameters()}
            if k < n_byz:  # first n_byz workers are adversarial
                delta = attack_delta(delta, attack, atk_gen)
            agg.add_contribution(f"worker_{k}", delta, validate=False)
        aggregated = agg.aggregate()
        outer_opt.step(global_model, aggregated)
        vl = eval_loss(global_model, val_data, seq_len, n_batches=4)
        curve.append({"round": r + 1, "step": (r + 1) * H, "val_loss": vl})
    print(f"  [byz {strategy} atk={attack} f={n_byz}/{K}] final={curve[-1]['val_loss']:.4f}",
          flush=True)
    return {"strategy": strategy, "attack": attack, "n_byz": n_byz,
            "workers": K, "curve": curve}


def experiment_e2(smoke=False):
    print("=== E2: Byzantine robustness ===", flush=True)
    train, val = load_data()
    K = 10
    rounds = 3 if smoke else 12
    H = 5 if smoke else 12
    seq_len, batch, lr, seed = 128, 8, 1e-3, 7
    shard = len(train) // K
    shards = [train[i * shard:(i + 1) * shard] for i in range(K)]

    results = {"config": {"workers": K, "rounds": rounds, "H": H,
                          "byz_fraction": 0.3, "trim_fraction": 0.3,
                          "model": "NeuroLLM 64d/2L (~0.1M params)"},
               "runs": []}
    t0 = time.time()
    # clean baseline
    results["runs"].append(run_byzantine(shards, val, "mean", "none", 0,
                                         rounds, H, seq_len, batch, lr, seed))
    for attack in ["signflip", "noise"]:
        for strategy in ["mean", "trimmed_mean", "median", "multi_krum"]:
            results["runs"].append(run_byzantine(shards, val, strategy, attack, 3,
                                                 rounds, H, seq_len, batch, lr, seed))
            gc.collect()
    # fraction sweep for trimmed mean under the strongest attack:
    # probes the f < 0.3N breakdown point empirically
    for n_byz in [1, 2, 4]:
        results["runs"].append(run_byzantine(shards, val, "trimmed_mean", "signflip",
                                             n_byz, rounds, H, seq_len, batch, lr, seed))
        gc.collect()
    results["wall_seconds"] = time.time() - t0
    out = os.path.join(RESULTS_DIR, "e2_byzantine.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"E2 done in {results['wall_seconds']:.0f}s -> {out}", flush=True)


# --------------------------------------------------------------------------
# E3: compression microbenchmark
# --------------------------------------------------------------------------

def experiment_e3(smoke=False):
    print("=== E3: compression microbenchmark ===", flush=True)
    train, val = load_data()
    seq_len, batch, lr, seed = 128, 8, 3e-4, 42
    model = make_model(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(3)
    w0 = state_clone(model)
    steps = 10 if smoke else 50
    for _ in range(steps):
        x = get_batch(train, batch, seq_len, gen)
        out = model(x, labels=x)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    gparams = dict(model.named_parameters())
    deltas = {n: gparams[n].data - w0[n] for n in gparams}

    results = {"per_topk": [], "model_params": sum(p.numel() for p in model.parameters())}
    for tk in [0.05, 0.1, 0.2]:
        comp = GradientCompressor(top_k_ratio=tk, bits=8)
        raw_b, comp_b, cos_sum, n_t = 0, 0, 0.0, 0
        t0 = time.time()
        for name, d in deltas.items():
            blob = comp.compress(d)
            d_hat = comp.decompress(blob)
            raw_b += d.numel() * 4
            comp_b += len(blob)
            c = torch.nn.functional.cosine_similarity(
                d.flatten().unsqueeze(0), d_hat.flatten().unsqueeze(0)).item()
            cos_sum += c
            n_t += 1
        dt = time.time() - t0
        results["per_topk"].append({
            "top_k_ratio": tk,
            "raw_bytes": raw_b,
            "compressed_bytes": comp_b,
            "ratio": raw_b / comp_b,
            "mean_cosine": cos_sum / n_t,
            "wall_seconds": dt,
        })
        print(f"  top_k={tk}: ratio={raw_b/comp_b:.1f}x cosine={cos_sum/n_t:.3f}",
              flush=True)
    out = os.path.join(RESULTS_DIR, "e3_compression.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"E3 done -> {out}", flush=True)


# --------------------------------------------------------------------------
# E4: PoNW microbenchmark
# --------------------------------------------------------------------------

def experiment_e4(smoke=False):
    print("=== E4: PoNW microbenchmark ===", flush=True)
    import hashlib
    from neuroshard.core.crypto.ecdsa import derive_keypair_from_token, ecdsa_sign, ecdsa_verify
    from neuroshard.core.consensus.verifier import CohortSyncProof, get_verification_rate

    kp = derive_keypair_from_token("a" * 64)

    proof = CohortSyncProof(
        node_id="f" * 40,
        layer_range=(0, 4),
        sync_round=17,
        batches_processed=500,
        pseudo_gradient_hash=hashlib.sha256(b"pg").digest(),
        cohort_members=["a" * 40, "b" * 40, "c" * 40],
        aggregated_gradient_hash=hashlib.sha256(b"agg").digest(),
        initial_weights_hash=hashlib.sha256(b"w0").digest(),
        final_weights_hash=hashlib.sha256(b"wH").digest(),
    )
    payload = json.dumps(proof.to_dict(), sort_keys=True)
    proof_size = len(payload.encode())

    n_iter = 20 if smoke else 200
    t0 = time.time()
    for _ in range(n_iter):
        sig = ecdsa_sign(payload, kp.private_key_bytes)
    sign_ms = (time.time() - t0) / n_iter * 1000
    t0 = time.time()
    for _ in range(n_iter):
        ok = ecdsa_verify(payload, sig, kp.public_key_bytes)
    verify_ms = (time.time() - t0) / n_iter * 1000
    assert ok

    # challenge recompute: one training batch fwd+bwd on the eval model
    train, _ = load_data()
    model = make_model(1)
    gen = torch.Generator().manual_seed(5)
    x = get_batch(train, 8, 128, gen)
    reps = 3 if smoke else 20
    t0 = time.time()
    for _ in range(reps):
        out = model(x, labels=x)
        out["loss"].backward()
        model.zero_grad()
    recompute_ms = (time.time() - t0) / reps * 1000

    rates = {str(n): get_verification_rate(n) for n in [10, 50, 200, 1000]}

    results = {
        "proof_size_bytes": proof_size,
        "sign_ms": sign_ms,
        "verify_ms": verify_ms,
        "recompute_batch_ms": recompute_ms,
        "verification_rates": rates,
        "escape_probability_100_proofs": {
            str(n): (1 - get_verification_rate(n)) ** 100 for n in [10, 50, 200, 1000]
        },
    }
    out = os.path.join(RESULTS_DIR, "e4_ponw.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps(results, indent=1), flush=True)
    print(f"E4 done -> {out}", flush=True)


# --------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["e1", "e2", "e3", "e4", "all"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0)
    if args.exp in ("e1", "all"):
        experiment_e1(args.smoke)
    if args.exp in ("e2", "all"):
        experiment_e2(args.smoke)
    if args.exp in ("e3", "all"):
        experiment_e3(args.smoke)
    if args.exp in ("e4", "all"):
        experiment_e4(args.smoke)


if __name__ == "__main__":
    main()
