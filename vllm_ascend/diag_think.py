"""Diagnostic helpers for VLLM_ASCEND_DIAG_THINK debugging.

Set VLLM_ASCEND_DIAG_THINK=1 to enable per-layer diagnostics that track
hidden state statistics, token-101 (<channel|>) logits, and help identify
which Ascend op introduces logit suppression during long thinking sequences.
"""
import functools
import json
import os
import time

import torch


def _diag_enabled() -> bool:
    return os.environ.get("VLLM_ASCEND_DIAG_THINK", "0") == "1"


def _safe_capture() -> bool:
    """Return True inside compilation/tracing — skip diagnostics.

    torch.compile (Dynamo) and NPU graph capture both cannot handle
    .item() calls. Check Dynamo first, then NPU stream state.
    """
    # Check Dynamo (torch.compile) tracing
    try:
        if torch.compiler.is_compiling():
            return True
    except Exception:
        pass
    # Check NPU aclgraph capture via forward context
    try:
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        return getattr(_EXTRA_CTX, 'capturing', False)
    except Exception:
        return True


def _safe_stats(t: torch.Tensor) -> str:
    """Stats that won't crash during graph capture."""
    try:
        f = t.detach().float()
        return (
            f"shape={list(t.shape)} mean={f.mean().item():.6f} "
            f"std={f.std().item():.6f} min={f.min().item():.4f} max={f.max().item():.4f}"
        )
    except Exception:
        return f"shape={list(t.shape)} [stats unavailable]"


# ── save mode (VLLM_ASCEND_DIAG_SAVE_DIR) ────────────────────────
_save_dir: str | None = os.environ.get("VLLM_ASCEND_DIAG_SAVE_DIR", "").strip() or None
_save_current_step: int = 0
_save_buffer: dict = {}  # step -> {name -> tensor}
_save_enabled: bool = _save_dir is not None


def _get_tp_rank() -> int:
    """Return TP rank (0 = primary) for filesystem-safe saving."""
    try:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:
        pass
    return 0


def _is_save_rank() -> bool:
    """Only TP rank 0 should write to disk to avoid multi-process conflicts."""
    return _get_tp_rank() == 0


def _ensure_save_step_dir() -> str:
    path = os.path.join(_save_dir, f"step{_save_current_step:06d}")
    os.makedirs(path, exist_ok=True)
    return path


def _save_tensor(name: str, t: torch.Tensor) -> None:
    """Save a tensor to the save directory for the current step.

    Only saves on TP rank 0 to avoid filesystem conflicts.
    """
    global _save_dir
    if _save_dir is None:
        return
    if not _is_save_rank():
        return
    _save_buffer[name] = t.detach().cpu()


def _flush_save_buffer() -> None:
    """Write all buffered tensors to disk and clear.

    Only flushes on TP rank 0 to avoid filesystem conflicts.
    """
    global _save_dir, _save_buffer
    if _save_dir is None:
        return
    if not _is_save_rank():
        _save_buffer = {}
        return
    step_dir = _ensure_save_step_dir()
    for name, tensor in _save_buffer.items():
        torch.save(tensor, os.path.join(step_dir, f"{name}.pt"))
    # Also write a stats.json with per-tensor statistics
    stats = {}
    for name, tensor in _save_buffer.items():
        f = tensor.float()
        stats[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "mean": float(f.mean()),
            "std": float(f.std()),
            "min": float(f.min()),
            "max": float(f.max()),
        }
    with open(os.path.join(step_dir, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    _save_buffer = {}


# ── per-layer counters ──────────────────────────────────────────
_step: int = 0
_gemma_rmsnorm_count: int = 0
_rotary_count: int = 0
_silu_count: int = 0
_moe_count: int = 0
_linear_count: int = 0


def next_step() -> int:
    global _step
    _step += 1
    return _step


# ── per-layer hidden state capture ────────────────────────────────
_per_layer_hooks_installed: bool = False
_per_layer_step_done: bool = False  # Only capture prefill (step 0)


def _make_patched_decoder_forward(original_forward):
    """Create a patched forward that captures per-layer hidden states."""

    @functools.wraps(original_forward)
    def patched_forward(self, positions, hidden_states, residual,
                        per_layer_input=None, **kwargs):
        if (_diag_enabled() and not _safe_capture()
                and _save_dir is not None
                and not _per_layer_step_done):
            layer_idx = self.layer_idx
            if layer_idx == 0:
                _save_tensor("embed_hidden", hidden_states.detach())
            else:
                _save_tensor(f"layer_{layer_idx - 1:03d}_hidden",
                             hidden_states.detach())

        result = original_forward(
            self, positions, hidden_states, residual,
            per_layer_input=per_layer_input, **kwargs)

        if (_diag_enabled() and not _safe_capture()
                and _save_dir is not None
                and not _per_layer_step_done):
            _save_tensor(f"layer_{self.layer_idx:03d}_hidden",
                         result[0].detach())

        return result

    return patched_forward


def install_per_layer_hooks():
    """Monkey-patch Gemma4DecoderLayer.forward to capture per-layer states.

    Safe to call multiple times — only applies the patch once.
    Captures embed_hidden (layer-0 input), layer_XXX_hidden (each layer
    output), and final_norm_hidden (saved by log_lm_head as hs_final).

    Called at module import time (bottom of this file) plus as a fallback
    from log_gemma_rmsnorm (the earliest diag hook that fires).
    """
    global _per_layer_hooks_installed
    if _per_layer_hooks_installed:
        return

    try:
        from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer

        Gemma4DecoderLayer.forward = _make_patched_decoder_forward(
            Gemma4DecoderLayer.forward)
        _per_layer_hooks_installed = True
        print("[DIAG] Per-layer hooks installed on Gemma4DecoderLayer.forward",
              flush=True)
    except Exception as e:
        print(f"[DIAG] Failed to install per-layer hooks: {e}", flush=True)


# Try to install hooks at import time (before first forward pass).
# This works if gemma4 module is already loaded, which it should be
# by the time diag_think is first imported by a running op.
if _diag_enabled():
    install_per_layer_hooks()


def _layer_label(module) -> str:
    """Best-effort layer name from a PyTorch module."""
    for attr in ('layer_name', 'prefix', '_diag_name'):
        val = getattr(module, attr, None)
        if val is not None:
            return str(val)
    # Try to extract layer index from prefix like "model.layers.42."
    prefix = getattr(module, 'prefix', '')
    if 'layers.' in prefix:
        try:
            parts = prefix.split('layers.')[1].split('.')
            return f"L{parts[0]}"
        except Exception:
            pass
    return f"id{id(module)}"


def _stats(t: torch.Tensor) -> str:
    """One-line stats string for a tensor."""
    f = t.detach().float()
    return (
        f"shape={list(t.shape)} mean={f.mean().item():.6f} "
        f"std={f.std().item():.6f} min={f.min().item():.4f} max={f.max().item():.4f}"
    )


# ── per-op log functions ────────────────────────────────────────

def _safe_log(fn):
    """Decorator: catch ALL exceptions so diag never crashes the server."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except BaseException:
            pass
    return wrapper


@_safe_log
def log_gemma_rmsnorm(module, inp: torch.Tensor, out: torch.Tensor) -> None:
    if not _diag_enabled() or _safe_capture():
        return
    # Fallback: install per-layer hooks on the first RMSNorm call.
    # This fires inside layer 0's input layernorm, which is the earliest
    # hookable point. Layer 0 itself won't be captured, but layers 1-59
    # will use the patched forward. (The module-level install at import
    # time should already have run — this is a safety net.)
    install_per_layer_hooks()
    global _gemma_rmsnorm_count
    _gemma_rmsnorm_count += 1
    c = _gemma_rmsnorm_count
    if False:  # disabled to avoid I/O flood; enable selectively
        print(
            f"[DIAG][RMSNorm#{c} {_layer_label(module)}] "
            f"IN {_safe_stats(inp)}",
            flush=True,
        )


@_safe_log
def log_rotary(module, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor) -> None:
    if not _diag_enabled() or _safe_capture():
        return
    global _rotary_count
    _rotary_count += 1
    c = _rotary_count
    if False:  # disabled to avoid I/O flood
        pos_vals = positions.detach().float().squeeze()
        pos_str = f"pos=[{pos_vals.min().item():.0f}..{pos_vals.max().item():.0f}]"
        print(
            f"[DIAG][RoPE#{c} {_layer_label(module)}] {pos_str} "
            f"q_in {_safe_stats(query)}",
            flush=True,
        )


@_safe_log
def log_silu_and_mul(module, inp: torch.Tensor, out: torch.Tensor) -> None:
    if not _diag_enabled() or _safe_capture():
        return
    global _silu_count
    _silu_count += 1
    c = _silu_count
    if False:  # disabled to avoid I/O flood
        print(
            f"[DIAG][SiLU#{c} {_layer_label(module)}] "
            f"IN {_safe_stats(inp)} OUT {_safe_stats(out)}",
            flush=True,
        )


@_safe_log
def log_moe(module, hidden_states: torch.Tensor, out: torch.Tensor) -> None:
    if not _diag_enabled() or _safe_capture():
        return
    global _moe_count
    _moe_count += 1
    c = _moe_count
    if c % 100 == 0 or c <= 3 or c >= 3070:
        print(
            f"[DIAG][MoE#{c} {_layer_label(module)}] "
            f"IN {_safe_stats(hidden_states)} OUT {_safe_stats(out)}",
            flush=True,
        )


@_safe_log
def log_linear(module, inp: torch.Tensor, out: torch.Tensor) -> None:
    if not _diag_enabled() or _safe_capture():
        return
    global _linear_count
    _linear_count += 1
    c = _linear_count
    if False:  # disabled to avoid I/O flood
        print(
            f"[DIAG][Linear#{c} {_layer_label(module)}] "
            f"IN {_safe_stats(inp)} OUT {_safe_stats(out)}",
            flush=True,
        )


@_safe_log
def log_lm_head(hidden_states: torch.Tensor, logits: torch.Tensor) -> None:
    """Log final hidden states and token-101 logit."""
    if not _diag_enabled():
        return
    global _save_current_step
    step = next_step()

    # Save mode: capture hidden states + logits for offline comparison
    if _save_dir is not None:
        # Trigger per-layer hooks on first call (model must be loaded by now)
        install_per_layer_hooks()
        _save_tensor("hs_final", hidden_states)
        _save_tensor("final_norm_hidden", hidden_states)
        _save_tensor("logits", logits)
        _flush_save_buffer()
        _save_current_step += 1
        # Only capture per-layer data for the prefill step
        global _per_layer_step_done
        _per_layer_step_done = True

    if step % 10 == 0 or step <= 20:
        l = logits.detach().float().squeeze()
        vocab_sz = l.shape[-1] if l.ndim > 0 else l.numel()
        if vocab_sz <= 101:
            # TP-sharded: index 101 not in this partition, skip
            return
        tok101 = float(l[101])
        top5_vals, top5_ids = l.topk(min(5, l.numel()))
        rank101 = int((l > tok101).sum().item())

        # Status flag for thinking
        if rank101 <= 3:
            status = "THINK_OK"       # <channel|> in top-3, model can close thinking
        elif tok101 > 0:
            status = "THINK_MARGINAL" # positive but not top-3, risky
        else:
            status = "THINK_BLOCKED"  # negative logit, thinking stuck!

        print(
            f"[DIAG][LM_HEAD step={step}] {status} "
            f"tok101(<channel|>)={tok101:.4f} rank={rank101} "
            f"top5={list(zip(top5_ids.tolist(), [round(float(v), 2) for v in top5_vals.tolist()]))}",
            flush=True,
        )

        # Highlight when thinking is blocked
        if status == "THINK_BLOCKED":
            print(
                f"[DIAG][LM_HEAD step={step}] *** WARNING: <channel|> logit is "
                f"NEGATIVE ({tok101:.2f}), thinking may not close!",
                flush=True,
            )
