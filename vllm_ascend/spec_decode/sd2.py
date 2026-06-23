"""SD² Runtime — target hidden states → draft MLP steering biases.

Architecture:
  target 3-layer hs [batch, 16128] → encoder(128) → SiLU → 4×decoder(8192)
  → 4 biases injected at each draft MLP's up-projection BEFORE GELU gate.

All computation stays in bfloat16 (Ascend native dtype) — no float() casts.
Normalization (if used) temporarily goes to float32 for numerical stability.

Environment variables:
  VLLM_ASCEND_SD2_COLLECT=1      — enable calibration data collection
  VLLM_ASCEND_SD2_WEIGHTS=<path> — load pre-trained steering weights for inference
  VLLM_ASCEND_SD2_LAYERS=<a,b,c> — override target layer indices (default: 15,30,55)
"""

import os
import torch
import torch.nn as nn
from typing import Tuple


# ── Constants ──
_DEFAULT_LAYERS = (15, 30, 55)
_BACKBONE_HIDDEN = 5376       # Gemma4 31B target hidden_size
_DRAFT_INTERMEDIATE = 8192    # Draft MLP intermediate_size
_NUM_DRAFT_LAYERS = 4
_BOTTLENECK = 128


def get_sd2_layers() -> Tuple[int, ...]:
    val = os.environ.get("VLLM_ASCEND_SD2_LAYERS", "")
    if val:
        return tuple(int(x.strip()) for x in val.split(","))
    return _DEFAULT_LAYERS


def is_sd2_collect() -> bool:
    return os.environ.get("VLLM_ASCEND_SD2_COLLECT", "") == "1"


def is_sd2_enabled() -> bool:
    return is_sd2_collect() or os.environ.get("VLLM_ASCEND_SD2_WEIGHTS", "") != ""


class SD2Runtime:
    """Manages SD² steering: weight loading, bias computation, TP sharding."""

    def __init__(self):
        self._enabled = is_sd2_enabled()
        self._collect = is_sd2_collect()
        self._weights_path = os.environ.get("VLLM_ASCEND_SD2_WEIGHTS", "")
        self._layers = get_sd2_layers()
        self._steering_module = None
        self._X_mean = None
        self._X_std = None
        import sys
        print(f"[SD2_INIT] enabled={self._enabled} collect={self._collect} "
              f"weights={self._weights_path!r} layers={self._layers} "
              f"COLLECT_ENV={os.environ.get('VLLM_ASCEND_SD2_COLLECT','?')!r}",
              file=sys.stderr, flush=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def collect(self) -> bool:
        return self._collect

    @property
    def layers(self) -> Tuple[int, ...]:
        return self._layers

    def get_layer_indices(self, num_target_layers: int = 60) -> Tuple[int, ...]:
        for idx in self._layers:
            if idx < 0 or idx >= num_target_layers:
                raise ValueError(
                    f"SD2 layer index {idx} out of range [0, {num_target_layers})"
                )
        return self._layers

    def _init_steering_module(self, device=None, dtype=None):
        if self._steering_module is not None:
            return

        from vllm_ascend.ops.sd2_steering import SD2Steering

        self._steering_module = SD2Steering(
            backbone_hidden_size=len(self._layers) * _BACKBONE_HIDDEN,
            intermediate_size=_DRAFT_INTERMEDIATE,
            num_draft_layers=_NUM_DRAFT_LAYERS,
            bottleneck_dim=_BOTTLENECK,
        )
        self._steering_module.eval()

        if self._weights_path:
            ckpt = torch.load(self._weights_path, map_location="cpu", weights_only=True)
            state = ckpt.get("model_state", ckpt)
            self._steering_module.load_state_dict(state, strict=False)
            self._X_mean = ckpt.get("X_mean", None)
            self._X_std = ckpt.get("X_std", None)

        if device is not None:
            self._steering_module.to(device)
        if dtype is not None:
            self._steering_module.to(dtype)

    def install_on_draft(self, draft_model: nn.Module, device=None, dtype=None):
        from vllm_ascend.ops.sd2_patch import install_sd2

        self._init_steering_module(device, dtype)
        install_sd2(
            draft_model,
            backbone_hidden_size=len(self._layers) * _BACKBONE_HIDDEN,
            intermediate_size=_DRAFT_INTERMEDIATE,
            num_draft_layers=_NUM_DRAFT_LAYERS,
            bottleneck_dim=_BOTTLENECK,
            device=device,
            collect_outputs=self._collect,
        )

    def compute_biases(self, aux_hidden_states: list) -> list:
        if self._steering_module is None:
            return []

        # Keep bfloat16 — the native Ascend dtype.
        x = torch.cat(aux_hidden_states, dim=-1)
        device = x.device
        work_dtype = x.dtype  # bf16 on Ascend

        self._steering_module.to(device=device, dtype=work_dtype)

        # Normalize in float32 for numerical stability, then cast back.
        if self._X_mean is not None and self._X_std is not None:
            x = (x.float() - self._X_mean.to(device)) / self._X_std.to(device)
            x = torch.clamp(x, min=-10, max=10)
            x = x.to(work_dtype)

        biases = self._steering_module(x)
        return [b.to(dtype=work_dtype) for b in biases]

    def get_steering_biases_for_kwargs(self, aux_hidden_states: list) -> list:
        return self.compute_biases(aux_hidden_states)
