#!/usr/bin/env python3
"""Compare CPU reference vs Ascend NPU forward pass outputs.

Loads data saved by cpu_ref.py and diag_think.py save mode, then:
1. Compares token 101 (<channel|>) logit
2. Compares top-5 token overlap
3. Compares hidden state statistics at the LM head input
4. Reports whether divergence warrants Phase 2 (layer-by-layer drill-down)

Usage:
    python scripts/compare_cpu_ascend.py \
        --cpu /tmp/cpu_ref \
        --ascend /tmp/ascend_ref \
        --output /tmp/comparison_report.json
"""

import argparse
import json
import os
import sys

import torch


def load_cpu_data(cpu_dir: str) -> dict:
    """Load CPU reference data."""
    print(f"Loading CPU data from {cpu_dir} ...")
    data = {}

    # Metadata
    with open(os.path.join(cpu_dir, "meta.json")) as f:
        data["meta"] = json.load(f)

    # Stats
    with open(os.path.join(cpu_dir, "stats.json")) as f:
        data["stats"] = json.load(f)

    # Logits
    data["logits"] = torch.load(os.path.join(cpu_dir, "logits.pt"), map_location="cpu")
    # hs_final
    data["hs_final"] = torch.load(os.path.join(cpu_dir, "hs_final.pt"), map_location="cpu")

    # Per-layer hidden states (only load stats for now, load full tensors in Phase 2)
    data["layer_hidden_available"] = os.path.exists(
        os.path.join(cpu_dir, "layer_000_hidden.pt")
    )

    print(f"  Prompt: {data['meta']['prompt']}")
    print(f"  Seq len: {data['meta']['seq_len']}")
    print(f"  Logits shape: {list(data['logits'].shape)}")
    return data


def load_ascend_data(ascend_dir: str) -> dict:
    """Load Ascend NPU data (saved by diag_think.py save mode)."""
    print(f"Loading Ascend data from {ascend_dir} ...")
    data = {}

    # Find the step directory (usually step000000 for prefill)
    step_dirs = sorted(
        [d for d in os.listdir(ascend_dir) if d.startswith("step") and os.path.isdir(os.path.join(ascend_dir, d))]
    )
    if not step_dirs:
        print("  ERROR: No step directories found. Was save mode active?")
        sys.exit(1)

    step_dir = os.path.join(ascend_dir, step_dirs[0])
    data["step"] = step_dirs[0]
    print(f"  Using {step_dirs[0]} (prefill)")

    # Stats
    with open(os.path.join(step_dir, "stats.json")) as f:
        data["stats"] = json.load(f)

    # Logits
    logits_path = os.path.join(step_dir, "logits.pt")
    if os.path.exists(logits_path):
        data["logits"] = torch.load(logits_path, map_location="cpu")
    else:
        print("  WARNING: logits.pt not found")
        data["logits"] = None

    # hs_final
    hs_path = os.path.join(step_dir, "hs_final.pt")
    if os.path.exists(hs_path):
        data["hs_final"] = torch.load(hs_path, map_location="cpu")
    else:
        print("  WARNING: hs_final.pt not found")
        data["hs_final"] = None

    if data["logits"] is not None:
        print(f"  Logits shape: {list(data['logits'].shape)}")
    return data


def compare_logits(cpu: dict, ascend: dict) -> dict:
    """Compare final logits between CPU and Ascend."""
    cpu_logits = cpu["logits"]
    asc_logits = ascend["logits"]

    if cpu_logits is None or asc_logits is None:
        return {"error": "logits missing from one side"}

    # Handle shape differences
    # CPU: [1, 1, vocab_size] or [1, vocab_size]
    # Ascend: [1, seq_len, vocab_size] or [1, 1, vocab_size]

    cpu_l = cpu_logits.float().squeeze()
    asc_l = asc_logits.float().squeeze()

    # If Ascend has multiple positions, use the last one
    if asc_l.dim() > 1:
        asc_l = asc_l[-1, :]
    if cpu_l.dim() > 1:
        cpu_l = cpu_l[-1, :]

    # Align vocab sizes (Ascend may have padding)
    min_vocab = min(cpu_l.shape[-1], asc_l.shape[-1])
    cpu_l = cpu_l[:min_vocab]
    asc_l = asc_l[:min_vocab]

    # Token 101
    cpu_101 = float(cpu_l[101])
    asc_101 = float(asc_l[101])
    abs_diff_101 = abs(cpu_101 - asc_101)

    # Top 5
    cpu_top5_vals, cpu_top5_ids = cpu_l.topk(5)
    asc_top5_vals, asc_top5_ids = asc_l.topk(5)
    top5_overlap = len(set(cpu_top5_ids.tolist()) & set(asc_top5_ids.tolist()))

    # Distribution stats
    abs_diff = (cpu_l - asc_l).abs()
    mean_abs_diff = float(abs_diff.mean())
    max_abs_diff = float(abs_diff.max())
    cos_sim = float(
        torch.nn.functional.cosine_similarity(cpu_l.unsqueeze(0), asc_l.unsqueeze(0), dim=-1)[0]
    )

    return {
        "token_101": {
            "cpu": round(cpu_101, 4),
            "ascend": round(asc_101, 4),
            "abs_diff": round(abs_diff_101, 4),
            "significant": abs_diff_101 > 1.0,
        },
        "top5_overlap": f"{top5_overlap}/5",
        "top5_cpu": list(zip(cpu_top5_ids.tolist(), [round(float(v), 2) for v in cpu_top5_vals])),
        "top5_ascend": list(zip(asc_top5_ids.tolist(), [round(float(v), 2) for v in asc_top5_vals])),
        "distribution": {
            "mean_abs_diff": round(mean_abs_diff, 6),
            "max_abs_diff": round(max_abs_diff, 6),
            "cosine_similarity": round(cos_sim, 6),
        },
    }


def compare_hidden_states(cpu: dict, ascend: dict) -> dict:
    """Compare final hidden states (before LM head)."""
    cpu_hs = cpu.get("hs_final")
    asc_hs = ascend.get("hs_final")

    if cpu_hs is None or asc_hs is None:
        return {"error": "hidden states missing from one side"}

    cpu_h = cpu_hs.float().squeeze()
    asc_h = asc_hs.float().squeeze()

    # Align dimension
    min_dim = min(cpu_h.shape[-1], asc_h.shape[-1])
    cpu_h = cpu_h[..., :min_dim]
    asc_h = asc_h[..., :min_dim]

    abs_diff = (cpu_h - asc_h).abs()
    cos_sim = float(
        torch.nn.functional.cosine_similarity(
            cpu_h.reshape(1, -1), asc_h.reshape(1, -1), dim=-1
        )[0]
    )

    return {
        "cpu_stats": {
            "mean": round(float(cpu_h.mean()), 6),
            "std": round(float(cpu_h.std()), 6),
            "min": round(float(cpu_h.min()), 6),
            "max": round(float(cpu_h.max()), 6),
        },
        "ascend_stats": {
            "mean": round(float(asc_h.mean()), 6),
            "std": round(float(asc_h.std()), 6),
            "min": round(float(asc_h.min()), 6),
            "max": round(float(asc_h.max()), 6),
        },
        "mean_abs_diff": round(float(abs_diff.mean()), 6),
        "max_abs_diff": round(float(abs_diff.max()), 6),
        "cosine_similarity": round(cos_sim, 6),
    }


def print_report(logit_cmp: dict, hs_cmp: dict, cpu: dict, ascend: dict):
    """Print a human-readable comparison report."""
    print()
    print("=" * 72)
    print("CPU vs Ascend Forward Pass Comparison")
    print("=" * 72)
    print(f"Prompt: {cpu['meta']['prompt']}")
    print(f"Seq len: {cpu['meta']['seq_len']} tokens")
    print(f"Ascend step: {ascend['step']}")
    print()

    # Logit comparison
    print("--- Logits Comparison ---")
    if "error" in logit_cmp:
        print(f"  ERROR: {logit_cmp['error']}")
    else:
        t101 = logit_cmp["token_101"]
        sig = " *** SIGNIFICANT ***" if t101["significant"] else ""
        print(f"  Token 101 (<channel|>):")
        print(f"    CPU:    {t101['cpu']:.4f}")
        print(f"    Ascend: {t101['ascend']:.4f}")
        print(f"    Δ:      {t101['abs_diff']:.4f}{sig}")
        print(f"  Top-5 overlap: {logit_cmp['top5_overlap']}")
        print(f"  Top-5 CPU:     {logit_cmp['top5_cpu']}")
        print(f"  Top-5 Ascend:  {logit_cmp['top5_ascend']}")
        dist = logit_cmp["distribution"]
        print(f"  Mean abs diff: {dist['mean_abs_diff']:.6f}")
        print(f"  Max abs diff:  {dist['max_abs_diff']:.6f}")
        print(f"  Cosine sim:    {dist['cosine_similarity']:.6f}")

    print()

    # Hidden states comparison
    print("--- Final Hidden States Comparison ---")
    if "error" in hs_cmp:
        print(f"  ERROR: {hs_cmp['error']}")
    else:
        print(f"  CPU stats:    mean={hs_cmp['cpu_stats']['mean']:.6f}  std={hs_cmp['cpu_stats']['std']:.6f}")
        print(f"  Ascend stats: mean={hs_cmp['ascend_stats']['mean']:.6f}  std={hs_cmp['ascend_stats']['std']:.6f}")
        print(f"  Mean abs diff: {hs_cmp['mean_abs_diff']:.6f}")
        print(f"  Max abs diff:  {hs_cmp['max_abs_diff']:.6f}")
        print(f"  Cosine sim:    {hs_cmp['cosine_similarity']:.6f}")

    print()

    # Verdict
    print("--- Verdict ---")
    token101_sig = logit_cmp.get("token_101", {}).get("significant", False)
    cos_sim_logits = logit_cmp.get("distribution", {}).get("cosine_similarity", 1.0)
    cos_sim_hs = hs_cmp.get("cosine_similarity", 1.0)

    if token101_sig:
        print("*** Token 101 divergence detected! Phase 2 (layer-by-layer) recommended. ***")
    elif cos_sim_logits < 0.999 or cos_sim_hs < 0.999:
        print("Numerical divergence detected (cos_sim < 0.999). Phase 2 recommended.")
    else:
        print("No significant divergence in prefill logits.")
        print("The token-101 suppression likely emerges during decode steps,")
        print("not in the prefill computation. Consider comparing decode-step outputs.")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Compare CPU vs Ascend forward pass")
    parser.add_argument("--cpu", required=True, help="CPU reference output directory")
    parser.add_argument("--ascend", required=True, help="Ascend save directory")
    parser.add_argument("--output", default=None, help="JSON report output path")
    args = parser.parse_args()

    cpu = load_cpu_data(args.cpu)
    ascend = load_ascend_data(args.ascend)

    logit_cmp = compare_logits(cpu, ascend)
    hs_cmp = compare_hidden_states(cpu, ascend)

    report = {
        "cpu_dir": args.cpu,
        "ascend_dir": args.ascend,
        "logits": logit_cmp,
        "hidden_states": hs_cmp,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {args.output}")

    print_report(logit_cmp, hs_cmp, cpu, ascend)


if __name__ == "__main__":
    main()
