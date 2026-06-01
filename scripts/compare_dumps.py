#!/usr/bin/env python3
"""Compare eager vs graph mode dump data to identify precision divergences.

Usage:
  # Enable dump by setting env:
  export VLLM_ASCEND_DUMP_DIR=/tmp/vllm_dump

  # Run model in eager mode
  vllm serve ... --enforce-eager
  # Send same test prompts, kill after some tokens

  # Run model in graph mode
  vllm serve ... --compilation-config '{"cudagraph_mode": "PIECEWISE"}'
  # Send identical test prompts, kill after some tokens

  # Analyze
  python compare_dumps.py /tmp/vllm_dump/eager /tmp/vllm_dump/graph
"""
import json
import math
import sys
from pathlib import Path
from collections import defaultdict


def load_all_dumps(dump_dir: str) -> dict[str, list[dict]]:
    """Load all JSON dump files, grouped by tag."""
    base = Path(dump_dir)
    grouped = defaultdict(list)
    for f in sorted(base.rglob("*.json")):
        if f.name.startswith("."):
            continue
        try:
            data = json.loads(f.read_text())
            tag = data.get("tag", str(f.parent.name))
            grouped[tag].append((int(f.parent.name.rsplit("_", 1)[-1].replace("step", "")), data))
        except (json.JSONDecodeError, ValueError, KeyError):
            continue
    # Sort by step within each group
    return {tag: [d for _, d in sorted(entries)] for tag, entries in grouped.items()}


def cmp_tensor_stats(tag: str, eager_stats: dict, graph_stats: dict,
                     threshold_mean: float = 1e-4,
                     threshold_std: float = 1e-3) -> list[str]:
    """Compare two tensor stat dicts, return warnings for significant differences."""
    warnings = []
    for key in ["mean", "std", "min", "max"]:
        if key not in eager_stats or key not in graph_stats:
            continue
        e_val = eager_stats[key]
        g_val = graph_stats[key]
        abs_diff = abs(e_val - g_val)
        rel_diff = abs_diff / max(abs(e_val), abs(g_val), 1e-12)
        thresh = threshold_mean if key == "mean" else threshold_std
        if abs_diff > thresh:
            warnings.append(
                f"  {tag} [{key}] eager={e_val:.6e}  graph={g_val:.6e}  "
                f"Δ={abs_diff:.2e}  rel={rel_diff:.2%}"
            )
    return warnings


def cmp_logits(eager_dump: dict, graph_dump: dict,
               key_token_ids: dict[str, int] | None = None,
               threshold: float = 0.01) -> list[str]:
    """Compare logits dumps. Return warnings for key token divergence."""
    warnings = []

    # Compare key token logit values
    ek = eager_dump.get("key_tokens", {})
    gk = graph_dump.get("key_tokens", {})
    for token_name in ek:
        if token_name in gk:
            e_val = ek[token_name]
            g_val = gk[token_name]
            if e_val is not None and g_val is not None:
                diff = abs(e_val - g_val)
                if diff > threshold:
                    warnings.append(
                        f"  KEY_TOKEN [{token_name}] eager={e_val:.6f}  "
                        f"graph={g_val:.6f}  Δ={diff:.6f}"
                    )

    # Compare top-k overlap
    etop = {t for t, _ in eager_dump.get("topk", [])}
    gtop = {t for t, _ in graph_dump.get("topk", [])}
    overlap = len(etop & gtop)
    if overlap < 7:
        warnings.append(
            f"  TOPK_OVERLAP: {overlap}/10  eager_unique={etop-gtop}  graph_unique={gtop-etop}"
        )

    # Compare tensor stats
    warnings.extend(cmp_tensor_stats("logits", eager_dump.get("stats", {}),
                                      graph_dump.get("stats", {}), threshold=1e-5))
    return warnings


def compare_runs(eager_dir: str, graph_dir: str,
                 gemma_key_tokens: dict[str, int] | None = None):
    """Main comparison entry point."""
    print(f"Loading eager  dumps from: {eager_dir}")
    eager = load_all_dumps(eager_dir)
    print(f"Loading graph  dumps from: {graph_dir}")
    graph = load_all_dumps(graph_dir)

    if gemma_key_tokens is None:
        gemma_key_tokens = {}

    all_warnings: list[str] = []

    # ── 1. Compare logits ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("LOGITS COMPARISON")
    print("=" * 70)
    for prefix in ["logits/capture", "logits/passthrough", "logits/replay"]:
        e_list = eager.get(prefix, [])
        g_list = graph.get(prefix, [])
        if not e_list and prefix == "logits/capture":
            e_prefix = "logits/passthrough"
            if e_prefix in eager:
                e_list = eager[e_prefix]
                g_list = graph.get(prefix, g_list)
        for i in range(min(len(e_list), len(g_list))):
            warnings = cmp_logits(e_list[i], g_list[i], gemma_key_tokens)
            if warnings:
                print(f"\n[{prefix}] step {i}:")
                for w in warnings:
                    print(w)
                    all_warnings.append(w)

    # ── 2. Compare RMSNorm outputs layer by layer ──────────────────
    print("\n" + "=" * 70)
    print("LAYER RMSNORM COMPARISON")
    print("=" * 70)
    for tag_prefix in ["rmsnorm_out", "gemma_rmsnorm_out"]:
        e_layers: dict[int, list[dict]] = {}
        g_layers: dict[int, list[dict]] = {}
        for tag, entries in eager.items():
            if tag.startswith(tag_prefix):
                for d in entries:
                    lidx = d.get("layer_idx")
                    if lidx is not None:
                        e_layers.setdefault(lidx, []).append(d)
        for tag, entries in graph.items():
            if tag.startswith(tag_prefix):
                for d in entries:
                    lidx = d.get("layer_idx")
                    if lidx is not None:
                        g_layers.setdefault(lidx, []).append(d)

        for lidx in sorted(set(e_layers) | set(g_layers)):
            e_stats_list = [d["stats"] for d in e_layers.get(lidx, [])]
            g_stats_list = [d["stats"] for d in g_layers.get(lidx, [])]
            if not e_stats_list or not g_stats_list:
                continue
            # Compare last occurrence
            warnings = cmp_tensor_stats(f"{tag_prefix} layer={lidx}",
                                        e_stats_list[-1], g_stats_list[-1])
            if warnings:
                for w in warnings:
                    print(w)
                    all_warnings.append(w)

    # ── 3. Compare RoPE outputs layer by layer ──────────────────
    print("\n" + "=" * 70)
    print("ROPE OUTPUT COMPARISON")
    print("=" * 70)
    for tag_prefix in ["rope_q_out", "rope_k_out"]:
        e_layers = {}
        g_layers = {}
        for tag, entries in eager.items():
            if tag.startswith(tag_prefix):
                for d in entries:
                    lidx = d.get("layer_idx")
                    if lidx is not None:
                        e_layers.setdefault(lidx, []).append(d)
        for tag, entries in graph.items():
            if tag.startswith(tag_prefix):
                for d in entries:
                    lidx = d.get("layer_idx")
                    if lidx is not None:
                        g_layers.setdefault(lidx, []).append(d)
        for lidx in sorted(set(e_layers) | set(g_layers)):
            e_stats_list = [d["stats"] for d in e_layers.get(lidx, [])]
            g_stats_list = [d["stats"] for d in g_layers.get(lidx, [])]
            if not e_stats_list or not g_stats_list:
                continue
            warnings = cmp_tensor_stats(f"{tag_prefix} layer={lidx}",
                                        e_stats_list[-1], g_stats_list[-1])
            if warnings:
                for w in warnings:
                    print(w)
                    all_warnings.append(w)

    # ── 4. Summary ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(all_warnings)} total warnings")
    print("=" * 70)
    if not all_warnings:
        print("No significant differences found between eager and graph mode.")
    else:
        print("Key divergences by category:")
        categories = defaultdict(int)
        for w in all_warnings:
            if "KEY_TOKEN" in w:
                categories["key_token_logit"] += 1
            elif "TOPK_OVERLAP" in w:
                categories["topk_overlap"] += 1
            elif "rmsnorm" in w:
                categories["rmsnorm_stats"] += 1
            elif "gemma_rmsnorm" in w:
                categories["gemma_rmsnorm_stats"] += 1
            elif "rope" in w:
                categories["rope_stats"] += 1
            elif "logits" in w:
                categories["logits_stats"] += 1
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")

    return len(all_warnings)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    # Optional: pass key token IDs as "name=id" pairs
    key_tokens = {}
    for arg in sys.argv[3:]:
        if "=" in arg:
            name, tid = arg.split("=", 1)
            key_tokens[name] = int(tid)

    n_warnings = compare_runs(sys.argv[1], sys.argv[2], key_tokens)
    sys.exit(0 if n_warnings == 0 else 1)
