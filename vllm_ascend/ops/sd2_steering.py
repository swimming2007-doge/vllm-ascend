"""SD² Steering module — injects bias at up-projection BEFORE GELU gate (paper Section 3.1-3.2).

Paper formula:  out = W_d( (W_up·x + W_s·g) ⊙ σ(W_gate·x) )

Architecture:
  Input:  concatenated target hidden states from 3 layers (low, mid, high)
          = 3 * backbone_hidden_size = 16128 for Gemma4 31B
  Output: 4 per-layer bias vectors added to up-projection before gate
          = num_draft_layers * intermediate_size = 4 * 8192

  backbone_hs [batch, 16128]
    → encoder Linear(16128 → 128)
    → SiLU
    → 4 × decoder Linear(128 → 8192)
    → 4 × steering bias [batch, 8192]
"""

import torch
import torch.nn as nn


class SD2Steering(nn.Module):
    def __init__(
        self,
        backbone_hidden_size: int = 16128,   # 3 layers × 5376
        intermediate_size: int = 8192,        # draft MLP intermediate_size
        num_draft_layers: int = 4,
        bottleneck_dim: int = 128,
    ):
        super().__init__()
        self.encoder = nn.Linear(backbone_hidden_size, bottleneck_dim, bias=False)
        self.decoders = nn.ModuleList([
            nn.Linear(bottleneck_dim, intermediate_size, bias=False)
            for _ in range(num_draft_layers)
        ])

        # Xavier-like init scaled for large input/output
        nn.init.normal_(self.encoder.weight, std=0.01 / backbone_hidden_size**0.5)
        for dec in self.decoders:
            nn.init.normal_(dec.weight, std=0.01 / bottleneck_dim**0.5)

    def forward(self, backbone_hidden_states: torch.Tensor) -> list[torch.Tensor]:
        """Compute per-layer up-projection biases from target hidden states.

        Args:
            backbone_hidden_states: [batch, backbone_hidden_size] concatenated
                                    target activations from 3 layers.

        Returns:
            List of 4 tensors [batch, intermediate_size] — biases added to
            each draft layer's up-projection before the GELU gate.
        """
        latent = torch.nn.functional.silu(self.encoder(backbone_hidden_states))
        return [decoder(latent) for decoder in self.decoders]
