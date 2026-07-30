"""WideBind: mlp module."""

import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import WideBindConfig

class GroupedMLP(nn.Module):
    """
    Grouped bottleneck MLP with per-group expansion (SwiGLU optional).

    Instead of D → D → D (rank-bounded by D), splits D into G groups
    and gives each group internal expansion (d → expand*d → d).
    Total rank still ≤ D, but each group learns richer features
    within its d-dim subspace.

    G=32, d=128, expand=4 → 4× per-group expansion.
    With SwiGLU: gate and up projections both d → expand*d, down expand*d → d.
    """
    def __init__(self, D, expand, groups, swiglu=True):
        super().__init__()
        assert D % groups == 0
        self.D = D
        self.G = groups
        self.d = D // groups
        d = self.d
        e = expand
        self.swiglu = swiglu
        if swiglu:
            hidden = e * d
            up_std = (2.0 / (d + hidden)) ** 0.5
            down_std = (2.0 / (hidden + d)) ** 0.5
            self.W_gate = nn.Parameter(torch.randn(groups, d, hidden) * up_std)
            self.W_up = nn.Parameter(torch.randn(groups, d, hidden) * up_std)
            self.W_down = nn.Parameter(torch.randn(groups, hidden, d) * down_std)
        else:
            up_std = (2.0 / (d + e * d)) ** 0.5
            down_std = (2.0 / (e * d + d)) ** 0.5
            self.W_up = nn.Parameter(torch.randn(groups, d, e * d) * up_std)
            self.W_down = nn.Parameter(torch.randn(groups, e * d, d) * down_std)
        self.norm_w = nn.Parameter(torch.ones(D))

    def forward(self, h):
        B, L, D = h.shape
        h = F.rms_norm(h, (D,), self.norm_w)
        h = h.reshape(B, L, self.G, self.d)
        if self.swiglu:
            gate = F.silu(torch.einsum('blgd,gdf->blgf', h, self.W_gate))
            up = torch.einsum('blgd,gdf->blgf', h, self.W_up)
            h = gate * up
        else:
            h = F.silu(torch.einsum('blgd,gdf->blgf', h, self.W_up))
        h = torch.einsum('blgf,gfd->blgd', h, self.W_down)
        self._cached_group_out = h  # (B, L, G, d)
        return h.reshape(B, L, D)


# ─── BottleneckBind (Fibonacci-twisted) ────────────────────────────────
