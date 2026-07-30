"""WideBind: embedding module."""

import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import WideBindConfig
from .vsa_utils import zeckendorf_codes, sparse_block_codes


class RotaryEmbedding(nn.Module):
    def __init__(self, D, theta=1000000.0, scaling=1.0, max_len=65536):
        super().__init__()
        self.D = D
        self.theta = theta
        self.scaling = scaling
        half = D // 2
        freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
        self.register_buffer('_freqs', freqs)
        self._max_cached = 0

    def _build_cache(self, L):
        if L <= self._max_cached:
            return
        t = torch.arange(L, dtype=torch.float32, device=self._freqs.device) / self.scaling
        angles = t[:, None] * self._freqs[None, :]
        self._cos_cached = angles.cos()
        self._sin_cached = angles.sin()
        self._max_cached = L

    def forward(self, x):
        B, L, D = x.shape
        self._build_cache(L)
        cos = self._cos_cached[:L].to(x.dtype).to(x.device)
        sin = self._sin_cached[:L].to(x.dtype).to(x.device)
        x0 = x[..., 0::2].contiguous()
        x1 = x[..., 1::2].contiguous()
        out0 = x0 * cos.unsqueeze(0) - x1 * sin.unsqueeze(0)
        out1 = x0 * sin.unsqueeze(0) + x1 * cos.unsqueeze(0)
        out = torch.empty_like(x)
        out[..., 0::2] = out0
        out[..., 1::2] = out1
        return out

class ZeckendorfEmbedding(nn.Module):
    """Token -> D-space via Zeckendorf codes + learned projection.
    
    Legacy: проекция K→D через Linear. Ранг матрицы эмбеддингов ≤ K=23.
    """
    def __init__(self, cfg):
        super().__init__()
        codes = zeckendorf_codes(cfg.vocab)
        K = codes.shape[1]
        self.register_buffer('codes', codes)
        self.proj = nn.Linear(K, cfg.D, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)
    
    def forward(self, tokens):
        return self.proj(self.codes[tokens])



class PartitionedEmbedding(nn.Module):
    """Token -> D-space via partitioned sparse codes.
    
    D делится на K сегментов, K = D // seg_size (точное деление).
    Каждый бит кода получает свой сегмент: h = Σ z_k · w_k.
    
    K=32, S=6: C(32,6)=906192 ≥ V=50000. Ровно 6 активных бит на токен.
    Per-token: 6 × d = 6×112 = 672 dims (18.8%), детерминированно.
    
    Математические свойства:
      - rank(E) = 3584 (полный ранг)
      - Segment ↔ mirror group: 1:1 alignment (32×112)
      - Равномерная частота бит: ~19% каждый
      - K=32 → bind compression 32→16: ровно 2 сегмента на bind-канал
    """
    def __init__(self, cfg):
        super().__init__()
        codes = sparse_block_codes(cfg.vocab, K=cfg.code_dim, S=cfg.code_sparsity)
        self.K = codes.shape[1]
        self.register_buffer('codes', codes)
        
        D = cfg.D
        assert D % self.K == 0, f'D={D} must be divisible by K={self.K}'
        d = D // self.K
        
        # Rank expansion: mixing matrix M (K×K) с ортогональной инициализацией
        # codes → sigmoid(M·codes) даёт плотные коэффициенты, каждый бит влияет на все сегменты
        self.embed_mix = nn.Parameter(torch.zeros(self.K, self.K))
        nn.init.orthogonal_(self.embed_mix)
        self.register_buffer('_mix_scale', torch.tensor(2.0), persistent=False)
        
        self.basis = nn.Parameter(torch.randn(self.K, d))
        nn.init.xavier_uniform_(self.basis, gain=0.5)
        self._rope_theta = getattr(cfg, 'rope_theta', 1000000.0)
        self._rope_scaling = getattr(cfg, 'rope_scaling', 1.0)
        self.rope = RotaryEmbedding(D, theta=self._rope_theta, scaling=self._rope_scaling)
    
    def forward(self, tokens):
        codes = self.codes[tokens]  # (B, L, K), sparse binary
        # Dense mixing: sigmoid(scale · M · codes) → каждый бит влияет на все сегменты
        codes = torch.sigmoid(torch.einsum('blk,kj->blj', codes, self.embed_mix) * self._mix_scale)
        B, L = tokens.shape
        out = torch.einsum('blk,kd->blkd', codes, self.basis).reshape(B, L, -1)
        out = self.rope(out)
        return out



class LmHead(nn.Module):
    """D-space -> vocab logits via Zeckendorf code projection (legacy)."""
    def __init__(self, cfg):
        super().__init__()
        codes = zeckendorf_codes(cfg.vocab)
        K = codes.shape[1]
        self.register_buffer('codes', codes)
        self.proj = nn.Linear(cfg.D, K, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)
    
    def forward(self, h):
        return self.proj(h) @ self.codes.T



class PartitionedHead(nn.Module):
    """D-space -> vocab logits via segment-addressed readout + per-token bias.
    
    h ∈ ℝᴰ → split по тем же K сегментам, что и в PartitionedEmbedding.
    Каждый сегмент h_k сравнивается со своим readout r_k:
        logit_v = Σ_k z_{vk} · ⟨h_k, r_k⟩ + b_v
    
    b_v — learnable per-token bias (token frequency prior).
    K=32: каждый сегмент выровнен с mirror group (1:1).
    
    Если embed_basis передан (PartitionedEmbedding.basis), readout делится с ним
    (weight tying encode/decode). Иначе — собственный readout.
    """
    def __init__(self, cfg, embed_basis=None):
        super().__init__()
        codes = sparse_block_codes(cfg.vocab, K=cfg.code_dim, S=cfg.code_sparsity)
        self.K = codes.shape[1]
        self.register_buffer('codes', codes)
        
        D = cfg.D
        assert D % self.K == 0
        d = D // self.K
        
        if embed_basis is not None:
            self.readout = embed_basis  # shared reference
        else:
            self.readout = nn.Parameter(torch.randn(self.K, d))
            nn.init.xavier_uniform_(self.readout, gain=0.5)
        self.token_bias = nn.Parameter(torch.zeros(cfg.vocab))
    
    def forward(self, h):
        B, L, D = h.shape
        h_g = h.reshape(B, L, self.K, -1)  # (B, L, K, d)
        scores = torch.einsum('blkd,kd->blk', h_g, self.readout)
        return scores @ self.codes.T + self.token_bias.unsqueeze(0).unsqueeze(0)


# ─── Grouped Cognitive Mirror (32 эксперта) ────────────────────────────
