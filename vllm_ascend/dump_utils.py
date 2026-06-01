"""
Precision comparison dump utilities for eager vs graph mode debugging.
Controlled by VLLM_ASCEND_DUMP_DIR env var. Set to a directory path to enable.
"""
import os
import json
import time
import hashlib
from pathlib import Path
from typing import Any

import torch


def _env_dir() -> str | None:
    d = os.environ.get("VLLM_ASCEND_DUMP_DIR", "").strip()
    return d or None


def _is_enabled() -> bool:
    return _env_dir() is not None


def _dump_path(tag: str, step: int, layer_idx: int | None = None) -> Path | None:
    d = _env_dir()
    if d is None:
        return None
    p = Path(d)
    dirname = tag.replace("/", "_").replace(" ", "_")
    if layer_idx is not None:
        dirname = f"{dirname}_layer{layer_idx:03d}"
    prefix = f"step{step:06d}"
    return p / dirname / prefix


def _tensor_stats(t: torch.Tensor) -> dict:
    """Lightweight stats for a tensor."""
    t_f32 = t.detach().float()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "mean": round(t_f32.mean().item(), 8),
        "std": round(t_f32.std().item(), 8),
        "min": round(t_f32.min().item(), 8),
        "max": round(t_f32.max().item(), 8),
        "norm": round(t_f32.norm().item(), 4),
    }


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".json", "w") as f:
        json.dump(data, f, indent=2, default=str)


def _save_pt(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, str(path) + ".pt")


def dump_tensor(tag: str, step: int, tensor: torch.Tensor, *,
                layer_idx: int | None = None,
                save_full: bool = False,
                extra: dict | None = None) -> None:
    """Dump tensor stats (and optionally full tensor) for comparison."""
    if not _is_enabled():
        return
    if torch.compiler.is_compiling():
        return
    path = _dump_path(tag, step, layer_idx)
    if path is None:
        return
    payload = {
        "tag": tag,
        "step": step,
        "layer_idx": layer_idx,
        "timestamp": time.time(),
        "stats": _tensor_stats(tensor),
    }
    if extra:
        payload["extra"] = extra
    _save_json(payload, path)
    if save_full:
        _save_pt(tensor.detach().cpu(), Path(str(path) + "_full"))


def dump_token_probs(tag: str, step: int, logprobs_dict: dict, *,
                     layer_idx: int | None = None,
                     key_token_ids: dict[str, int] | None = None) -> None:
    """Dump key token probabilities for each decode step."""
    if not _is_enabled():
        return
    if torch.compiler.is_compiling():
        return
    path = _dump_path(tag, step, layer_idx)
    if path is None:
        return
    payload = {
        "tag": tag,
        "step": step,
        "timestamp": time.time(),
        "top5": sorted(logprobs_dict.items(), key=lambda x: float(x[1]), reverse=True)[:5],
    }
    if key_token_ids:
        payload["key_tokens"] = {
            name: logprobs_dict.get(str(tid), logprobs_dict.get(tid, None))
            for name, tid in key_token_ids.items()
        }
    _save_json(payload, path)


def dump_logit_scan(tag: str, step: int, logits: torch.Tensor, *,
                    key_token_ids: dict[str, int] | None = None,
                    topk: int = 10) -> None:
    """Dump logits with key token values and top-k for one decode step."""
    if not _is_enabled():
        return
    if torch.compiler.is_compiling():
        return
    path = _dump_path(tag, step)
    if path is None:
        return
    l = logits.detach().float().squeeze()
    topk_vals, topk_ids = torch.topk(l, k=min(topk, l.numel()))
    payload = {
        "tag": tag,
        "step": step,
        "timestamp": time.time(),
        "stats": _tensor_stats(l),
        "topk": [(int(tid), round(float(v), 8)) for tid, v in zip(topk_ids.tolist(), topk_vals.tolist())],
    }
    if key_token_ids:
        payload["key_tokens"] = {
            name: round(float(l[int(tid)]), 8) for name, tid in key_token_ids.items()
        }
    _save_json(payload, path)
    _save_pt(logits.detach().cpu(), Path(str(path) + "_logits_full"))


def dump_mark(tag: str, step: int, *, extra: dict | None = None) -> None:
    """Log a marker event (capture start, replay start, etc.)."""
    if not _is_enabled():
        return
    if torch.compiler.is_compiling():
        return
    path = _dump_path(tag, step)
    if path is None:
        return
    payload = {
        "tag": tag,
        "step": step,
        "timestamp": time.time(),
    }
    if extra:
        payload["extra"] = extra
    _save_json(payload, path)


# ── Global step counter ──────────────────────────────────────────
_step_counter: int = 0


def reset_step():
    global _step_counter
    _step_counter = 0


def next_step() -> int:
    global _step_counter
    _step_counter += 1
    return _step_counter


def get_step() -> int:
    return _step_counter
