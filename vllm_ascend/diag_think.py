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
    """Return True inside compilation/tracing — skip diagnostics."""
    try:
        if torch.compiler.is_compiling():
            return True
    except Exception:
        pass
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
    try:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:
        pass
    return 0


def _is_save_rank() -> bool:
    return _get_tp_rank() == 0


def _ensure_save_step_dir() -> str:
    path = os.path.join(_save_dir, f"step{_save_current_step:06d}")
    os.makedirs(path, exist_ok=True)
    return path


def _save_tensor(name: str, t: torch.Tensor) -> None:
    """Save a tensor to the save directory, with NPU sync before CPU copy."""
    global _save_dir
    if _save_dir is None:
        return
    if not _is_save_rank():
        return
    if t.device.type in ('privateuseone', 'npu', 'privateuse1'):
        torch.npu.synchronize()
    saved = t.detach().cpu().clone()
    f = saved.float()
    print(f"[DIAG][SAVE] {name} shape={list(t.shape)} device={str(t.device)} "
          f"mean={f.mean().item():.6f} max={f.max().item():.4f} nz={(saved != 0).sum().item()}",
          flush=True)
    _save_buffer[name] = saved


def _flush_save_buffer() -> None:
    global _save_dir, _save_buffer
    if _save_dir is None:
        return
    if not _is_save_rank():
        _save_buffer = {}
        return
    step_dir = _ensure_save_step_dir()
    for name, tensor in _save_buffer.items():
        torch.save(tensor, os.path.join(step_dir, f"{name}.pt"))
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
_per_layer_step_done: bool = False


# ── sub-op level hooks (for precision debugging) ─────────────────
_subop_hooks_registered: set = set()  # layer idx that already have sub-op hooks


def _make_subop_hook(layer_idx: int, op_name: str):
    """Create a forward hook that captures sub-op output to save buffer."""
    def hook(module, args, output):
        if _safe_capture() or _per_layer_step_done:
            return
        if isinstance(output, torch.Tensor):
            if output.count_nonzero() > 0:
                _save_tensor(f"layer_{layer_idx:03d}_{op_name}", output.detach())
        elif isinstance(output, (list, tuple)) and len(output) > 0:
            if isinstance(output[0], torch.Tensor) and output[0].count_nonzero() > 0:
                _save_tensor(f"layer_{layer_idx:03d}_{op_name}", output[0].detach())
    return hook


def _install_subop_hooks(layer_module, layer_idx: int):
    """Register forward hooks on sub-modules of one decoder layer,
    including attention-internal and MLP-internal ops."""
    if layer_idx in _subop_hooks_registered:
        return
    _subop_hooks_registered.add(layer_idx)

    # DecoderLayer sub-ops
    dl_modules = [
        ("dl_input_layernorm", getattr(layer_module, "input_layernorm", None)),
        ("dl_self_attn", getattr(layer_module, "self_attn", None)),
        ("dl_post_attention_layernorm", getattr(layer_module, "post_attention_layernorm", None)),
        ("dl_pre_feedforward_layernorm", getattr(layer_module, "pre_feedforward_layernorm", None)),
        ("dl_mlp", getattr(layer_module, "mlp", None)),
        ("dl_post_feedforward_layernorm", getattr(layer_module, "post_feedforward_layernorm", None)),
    ]
    for op_name, mod in dl_modules:
        if mod is not None:
            mod.register_forward_hook(_make_subop_hook(layer_idx, op_name))

    # Attention-internal sub-ops
    attn = getattr(layer_module, "self_attn", None)
    if attn is not None:
        attn_modules = [
            ("attn_qkv_proj", getattr(attn, "qkv_proj", None)),
            ("attn_q_norm", getattr(attn, "q_norm", None)),
            ("attn_k_norm", getattr(attn, "k_norm", None)),
            ("attn_rotary_emb", getattr(attn, "rotary_emb", None)),
            ("attn_v_norm", getattr(attn, "v_norm", None)),
            ("attn_kernel", getattr(attn, "attn", None)),
            ("attn_o_proj", getattr(attn, "o_proj", None)),
        ]
        for op_name, mod in attn_modules:
            if mod is not None:
                mod.register_forward_hook(_make_subop_hook(layer_idx, op_name))

    # MLP-internal sub-ops
    mlp = getattr(layer_module, "mlp", None)
    if mlp is not None:
        mlp_modules = [
            ("mlp_gate_up_proj", getattr(mlp, "gate_up_proj", None)),
            ("mlp_act_fn", getattr(mlp, "act_fn", None)),
            ("mlp_down_proj", getattr(mlp, "down_proj", None)),
        ]
        for op_name, mod in mlp_modules:
            if mod is not None:
                mod.register_forward_hook(_make_subop_hook(layer_idx, op_name))

    is_full = getattr(layer_module, 'is_full_attention', False)
    head_dim = getattr(attn, 'head_dim', '?') if attn else '?'
    print(f"[DIAG][SUBOP] Layer_{layer_idx:03d} hooks: decoder+attn+mlp "
          f"(full_attn={is_full}, head_dim={head_dim})", flush=True)


def _make_patched_decoder_forward(original_forward):
    """Create a patched forward that captures per-layer + sub-op hidden states."""

    @functools.wraps(original_forward)
    def patched_forward(self, positions, hidden_states, residual,
                        per_layer_input=None, **kwargs):
        layer_idx = self.layer_idx
        _diag = _diag_enabled()
        _capture = _safe_capture()
        _save_ok = _save_dir is not None
        _step_ok = not _per_layer_step_done

        # Skip warmup/profile passes
        if _diag and _save_ok and _step_ok and not _capture:
            if hidden_states.count_nonzero() == 0:
                if layer_idx == 0:
                    print("[DIAG][HOOK] Skipping warmup (all-zero hidden_states)",
                          flush=True)
            else:
                # Register sub-op hooks for all layers
                _install_subop_hooks(self, layer_idx)

                if layer_idx == 0:
                    _save_tensor("embed_hidden", hidden_states.detach())
                else:
                    _save_tensor(f"layer_{layer_idx - 1:03d}_hidden",
                                 hidden_states.detach())

        result = original_forward(
            self, positions, hidden_states, residual,
            per_layer_input=per_layer_input, **kwargs)

        if (_diag and not _capture and _save_ok
                and not _per_layer_step_done
                and hidden_states.count_nonzero() > 0):
            _save_tensor(f"layer_{self.layer_idx:03d}_hidden",
                         result[0].detach())

        return result

    return patched_forward


def install_per_layer_hooks():
    """Monkey-patch Gemma4DecoderLayer.forward to capture per-layer states."""
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


if _diag_enabled():
    install_per_layer_hooks()


def _layer_label(module) -> str:
    for attr in ('layer_name', 'prefix', '_diag_name'):
        val = getattr(module, attr, None)
        if val is not None:
            return str(val)
    prefix = getattr(module, 'prefix', '')
    if 'layers.' in prefix:
        try:
            parts = prefix.split('layers.')[1].split('.')
            return f"L{parts[0]}"
        except Exception:
            pass
    return f"id{id(module)}"


def _stats(t: torch.Tensor) -> str:
    f = t.detach().float()
    return (
        f"shape={list(t.shape)} mean={f.mean().item():.6f} "
        f"std={f.std().item():.6f} min={f.min().item():.4f} max={f.max().item():.4f}"
    )


# ── per-op log functions ────────────────────────────────────────

def _safe_log(fn):
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
    install_per_layer_hooks()
    global _gemma_rmsnorm_count
    _gemma_rmsnorm_count += 1


@_safe_log
def log_rotary(module, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor) -> None:
    if not _diag_enabled() or _safe_capture():
        return
    global _rotary_count
    _rotary_count += 1


@_safe_log
def log_silu_and_mul(module, inp: torch.Tensor, out: torch.Tensor) -> None:
    if not _diag_enabled() or _safe_capture():
        return
    global _silu_count
    _silu_count += 1


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


@_safe_log
def log_lm_head(hidden_states: torch.Tensor, logits: torch.Tensor) -> None:
    """Log final hidden states and token-101 logit."""
    if not _diag_enabled():
        return
    global _save_current_step
    step = next_step()

    if _save_dir is not None:
        install_per_layer_hooks()
        if hidden_states.count_nonzero() > 0:
            _save_tensor("hs_final", hidden_states)
            _save_tensor("final_norm_hidden", hidden_states)
            _save_tensor("logits", logits)
            _flush_save_buffer()
            _save_current_step += 1
            global _per_layer_step_done
            _per_layer_step_done = True
        else:
            print("[DIAG][LM_HEAD] Skipping warmup save (all-zero hidden_states)",
                  flush=True)

    if step % 10 == 0 or step <= 20:
        l = logits.detach().float().squeeze()
        vocab_sz = l.shape[-1] if l.ndim > 0 else l.numel()
        if vocab_sz <= 101:
            return
        tok101 = float(l[101])
        top5_vals, top5_ids = l.topk(min(5, l.numel()))
        rank101 = int((l > tok101).sum().item())

        if rank101 <= 3:
            status = "THINK_OK"
        elif tok101 > 0:
            status = "THINK_MARGINAL"
        else:
            status = "THINK_BLOCKED"

        print(
            f"[DIAG][LM_HEAD step={step}] {status} "
            f"tok101(<channel|>)={tok101:.4f} rank={rank101} "
            f"top5={list(zip(top5_ids.tolist(), [round(float(v), 2) for v in top5_vals.tolist()]))}",
            flush=True,
        )

        if status == "THINK_BLOCKED":
            print(
                f"[DIAG][LM_HEAD step={step}] *** WARNING: <channel|> logit is "
                f"NEGATIVE ({tok101:.2f}), thinking may not close!",
                flush=True,
            )
