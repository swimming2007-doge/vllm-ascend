"""SD² monkey-patch for Gemma4 MTP draft model — PAPER-ALIGNED VERSION.

Patches Gemma4MLP.forward() at CLASS level to inject steering bias
at the correct location per the SD² paper (Section 3.1-3.2):

  out = W_d( (W_up·x + W_s·g) ⊙ σ(W_gate·x) )
               ^^^^^^^^^^^^^^^^
               bias added to up-projection BEFORE the GELU gate

This is fundamentally different from our previous attempts:
  - BEFORE:  out = W_d(W_up·x ⊙ σ(W_gate·x)) + bias  (after down_proj) ❌
  - NOW:     out = W_d((W_up·x + bias) ⊙ σ(W_gate·x))  (before gate)  ✅

Also patches the draft model's forward to distribute steering_biases kwarg.

Usage:
    from sd2_patch import install_sd2
    steering_module, compute_fn = install_sd2(draft_model)
    biases = compute_fn(backbone_hidden_states)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm_ascend.ops.sd2_steering import SD2Steering

_STEERING_MODULE: SD2Steering | None = None
_SAVE_PATH = "/tmp/sd2_data.pt"

# Collection state
_COLLECT_OUTPUTS = False
_RAW_UP_PROJ: list[list[torch.Tensor]] = []        # [layer][step] = up_proj on-device (8192-dim)
_RAW_BACKBONE: list[torch.Tensor] = []               # on-device backbone_hs from draft fwd (5376-dim)
_RAW_FULL_BACKBONE: list[torch.Tensor] = []          # on-device full backbone_hs (16128-dim, from proposer)
_COLLECTED_UP_PROJ: list[list[torch.Tensor]] = []    # CPU after export
_COLLECTED_BACKBONE: list[torch.Tensor] = []         # CPU after export
_COLLECTED_FULL_BACKBONE: list[torch.Tensor] = []    # CPU after export

_PATCHED = False


def record_full_backbone(hidden_states: torch.Tensor):
    """Record the FULL 16128-dim multi-layer backbone hidden states.
    Call this from the proposer BEFORE truncation (before draft forward).
    Safe during graph capture (tensors stay on-device until flush)."""
    global _RAW_FULL_BACKBONE
    if _COLLECT_OUTPUTS:
        _RAW_FULL_BACKBONE.append(hidden_states)


def _flush_to_cpu():
    """Move collected device tensors to CPU. Call OUTSIDE graph capture."""
    global _RAW_UP_PROJ, _RAW_BACKBONE, _RAW_FULL_BACKBONE
    global _COLLECTED_UP_PROJ, _COLLECTED_BACKBONE, _COLLECTED_FULL_BACKBONE
    for layer_idx, raw_list in enumerate(_RAW_UP_PROJ):
        for t in raw_list:
            _COLLECTED_UP_PROJ[layer_idx].append(t.detach().cpu())
    for t in _RAW_BACKBONE:
        _COLLECTED_BACKBONE.append(t.detach().cpu())
    for t in _RAW_FULL_BACKBONE:
        _COLLECTED_FULL_BACKBONE.append(t.detach().cpu())
    _RAW_UP_PROJ = [[] for _ in range(4)]
    _RAW_BACKBONE = []
    _RAW_FULL_BACKBONE = []


def clear_collected_data():
    global _RAW_UP_PROJ, _RAW_BACKBONE, _RAW_FULL_BACKBONE
    global _COLLECTED_UP_PROJ, _COLLECTED_BACKBONE, _COLLECTED_FULL_BACKBONE
    _RAW_UP_PROJ = [[] for _ in range(4)]
    _RAW_BACKBONE = []
    _RAW_FULL_BACKBONE = []
    _COLLECTED_UP_PROJ = [[] for _ in range(4)]
    _COLLECTED_BACKBONE = []
    _COLLECTED_FULL_BACKBONE = []


def save_collected_data(path: str):
    """Save collected CPU data to disk (atomic: write to tmp then rename)."""
    import os
    data = {
        "backbone_hs": _COLLECTED_BACKBONE,              # 5376-dim (from draft fwd)
        "full_backbone_hs": _COLLECTED_FULL_BACKBONE,    # 16128-dim (from proposer)
        "up_proj_outputs": _COLLECTED_UP_PROJ,            # 8192-dim per layer
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(data, tmp_path)
    os.replace(tmp_path, path)
    import sys
    print(f"[SD2_CALIB] SAVED to {path}: {len(_COLLECTED_BACKBONE)} draft + "
          f"{len(_COLLECTED_FULL_BACKBONE)} full backbone samples, "
          f"up_proj={[len(x) for x in _COLLECTED_UP_PROJ]}",
          file=sys.stderr, flush=True)


def _install_mlp_gate_patch():
    """Patch Gemma4MLP.forward to inject bias at up-projection BEFORE GELU gate."""
    global _PATCHED
    if _PATCHED:
        return
    from vllm.model_executor.models.gemma4 import Gemma4MLP

    _orig_mlp_forward = Gemma4MLP.forward

    def patched_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: gate_up_proj (merged: gate | up)
        gate_up, _ = self.gate_up_proj(x)  # [batch, 2 * intermediate_size]

        # Step 2: Split into gate and up halves
        inter = gate_up.shape[-1] // 2
        gate = gate_up[..., :inter]       # W_gate * x, [batch, intermediate_size]
        up = gate_up[..., inter:]          # W_up * x,   [batch, intermediate_size]

        # Step 3: SD² — inject steering bias to up-projection BEFORE gate
        bias = getattr(self, "_steering_bias", None)
        if bias is not None:
            up = up + bias

        # Step 4: GELU gate × modified up (match Gemma4's gelu_pytorch_tanh)
        out = F.gelu(gate, approximate="tanh") * up  # [batch, intermediate_size]

        # Step 5: down_proj back to hidden_size
        out, _ = self.down_proj(out)  # [batch, hidden_size]

        # Calibration: buffer up_proj output on-device (safe during graph capture)
        if getattr(self, "_sd2_collect", False):
            layer_idx = getattr(self, "_sd2_layer_idx", 0)
            global _RAW_UP_PROJ
            if layer_idx < len(_RAW_UP_PROJ):
                _RAW_UP_PROJ[layer_idx].append(up)

        return out

    Gemma4MLP.forward = patched_mlp_forward
    _PATCHED = True


def install_sd2(
    draft_model: nn.Module,
    backbone_hidden_size: int = 16128,     # 3 layers × 5376
    intermediate_size: int = 8192,          # draft MLP intermediate_size
    num_draft_layers: int = 4,
    bottleneck_dim: int = 128,
    device: torch.device | None = None,
    collect_outputs: bool = False,
):
    """Install SD² steering: patch MLP forward + create steering module.

    Returns (steering_module, compute_fn) where compute_fn maps
    backbone_hidden_states → list of per-layer biases.
    """
    global _STEERING_MODULE, _COLLECT_OUTPUTS

    _install_mlp_gate_patch()

    _STEERING_MODULE = SD2Steering(
        backbone_hidden_size=backbone_hidden_size,
        intermediate_size=intermediate_size,
        num_draft_layers=num_draft_layers,
        bottleneck_dim=bottleneck_dim,
    )
    _STEERING_MODULE.eval()
    if device is not None:
        _STEERING_MODULE.to(device)

    if collect_outputs:
        _COLLECT_OUTPUTS = True
        clear_collected_data()
        layers = draft_model.model.layers
        for i, layer in enumerate(layers):
            if hasattr(layer, "mlp"):
                layer.mlp._sd2_collect = True
                layer.mlp._sd2_layer_idx = i

    # Patch draft model forward to distribute steering biases
    _orig_draft_forward = draft_model.forward

    def patched_draft_forward(self, *args, **kwargs):
        # Buffer backbone hidden states for calibration
        _hidden_states = kwargs.get("hidden_states", args[2] if len(args) > 2 else None)
        if _COLLECT_OUTPUTS and _hidden_states is not None:
            _RAW_BACKBONE.append(_hidden_states)

        # Distribute steering biases to each layer's MLP (TP-aware slicing)
        steering_biases = kwargs.pop("steering_biases", None)
        layers = self.model.layers
        if steering_biases is not None:
            for i, layer in enumerate(layers):
                if i < len(steering_biases) and hasattr(layer, "mlp"):
                    bias = steering_biases[i]  # [batch, full_intermediate (8192)]
                    # TP-slice: each rank's MLP only sees its shard of the up-proj
                    # Full up-proj is [batch, 8192], split across TP ranks.
                    gate_up = layer.mlp.gate_up_proj
                    if hasattr(gate_up, 'tp_rank') and hasattr(gate_up, 'tp_size'):
                        tp_rank = gate_up.tp_rank
                        tp_size = gate_up.tp_size
                        shard_size = bias.shape[-1] // tp_size  # 8192 // 4 = 2048
                        start = tp_rank * shard_size
                        end = (tp_rank + 1) * shard_size
                        layer.mlp._steering_bias = bias[..., start:end]
                    else:
                        # Non-TP or TP=1: use full bias
                        layer.mlp._steering_bias = bias
                elif hasattr(layer, "mlp"):
                    layer.mlp._steering_bias = None
        else:
            for layer in layers:
                if hasattr(layer, "mlp"):
                    layer.mlp._steering_bias = None

        result = _orig_draft_forward(*args, **kwargs)

        # NOTE: _flush_to_cpu() is now called from _run_merged_draft()
        # (Python side) instead of here, to avoid executing during
        # dummy_run / warmup.  Only real inference paths that set
        # _sd2_aux_hidden_states trigger collection.

        return result

    draft_model.forward = patched_draft_forward.__get__(draft_model, type(draft_model))

    def compute_biases(backbone_hidden_states: torch.Tensor) -> list[torch.Tensor]:
        if _STEERING_MODULE is None:
            return []
        device = backbone_hidden_states.device
        _STEERING_MODULE.to(device)
        dtype = backbone_hidden_states.dtype
        biases = _STEERING_MODULE(backbone_hidden_states.to(dtype))
        return [b.to(dtype=dtype) for b in biases]

    return _STEERING_MODULE, compute_biases
