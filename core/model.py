"""
WideBind: hybrid D-space LM with VSA memory + bottleneck bind.
"""

import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import WideBindConfig


# ─── Utilities ──────────────────────────────────────────────────────────

def dct_basis(n):
    """DCT-II basis vectors of shape (n, n) — orthogonal rows."""
    k = torch.arange(n, dtype=torch.float32)
    v = k.unsqueeze(1) * (k.unsqueeze(0) + 0.5)
    basis = torch.cos(v * math.pi / n)
    basis[0, :] = basis[0, :] / math.sqrt(2)
    return basis * math.sqrt(2.0 / n)


def zeckendorf_codes(vocab=50000):
    """Fibonacci Zeckendorf binary codes for vocab tokens.
    Возвращает (V, K≈23) — длина кода зависит от vocab.
    """
    fib = [1, 2]
    while fib[-1] <= vocab:
        fib.append(fib[-1] + fib[-2])
    fib = fib[:-1]
    K = len(fib)
    codes = torch.zeros(vocab, K)
    for i in range(vocab):
        n = i + 1
        for j in range(K - 1, -1, -1):
            if n >= fib[j]:
                codes[i, j] = 1.0
                n -= fib[j]
    return codes


def sparse_block_codes(vocab=50000, K=32, S=6):
    """Sparse block codes: ровно S единиц из K на каждый токен.
    
    Использует комбинаторную систему счисления (combinadic) с
    фиксированной случайной перестановкой, чтобы все K бит были
    равномерно представлены среди vocab токенов.
    
    Гарантии:
      - C(K, S) ≥ vocab     (C(32,6)=906192 ≥ 50000 ✓)
      - Ровно S=6 активных бит на каждый токен
      - Каждый бит активен у ≈ vocab·S/K токенов (≈ 9375)
      - Детерминированность (seed=42)
    """
    from math import comb
    total = comb(K, S)
    # Фиксированная случайная перестановка всех C(K, S) индексов
    perm = torch.randperm(total, generator=torch.Generator().manual_seed(42))
    codes = torch.zeros(vocab, K)
    for v in range(vocab):
        idx = int(perm[v].item())
        n = idx
        for i in range(S, 0, -1):
            c = i - 1
            while comb(c + 1, i) <= n:
                c += 1
            codes[v, c] = 1.0
            n -= comb(c, i)
    return codes


# ─── VSA Prefix Scan ───────────────────────────────────────────────────

def vsa_prefix_scan(a, b, state=None):
    """Associative parallel prefix scan for VSA memory.
    mem[t] = a[t] * mem[t-1] + b[t]  (element-wise)
    
    a: (B, L, D) or (B, L) — decay factors
    b: (B, L, D) — input increments
    state: (B, D) — initial state or None
    
    Returns: (B, L, D) full scan, (B, D) final state
    """
    B, L, D = b.shape
    if a.dim() == 2:
        a = a.unsqueeze(-1).expand(-1, -1, D)
    
    if state is not None:
        a_state = torch.ones(B, 1, D, device=b.device, dtype=b.dtype)
        b_state = state.unsqueeze(1)
        a = torch.cat([a_state, a], dim=1)
        b = torch.cat([b_state, b], dim=1)
    
    n = a.shape[1]
    a_curr, b_curr = a, b
    step = 1
    while step < n:
        a_left = a_curr[:, :step]
        a_step = a_curr[:, step:]
        a_prev = a_curr[:, :-step]
        b_left = b_curr[:, :step]
        b_step = b_curr[:, step:]
        b_prev = b_curr[:, :-step]
        a_curr = torch.cat([a_left, a_step * a_prev], dim=1)
        b_curr = torch.cat([b_left, b_prev * a_step + b_step], dim=1)
        step *= 2
    
    if state is not None:
        return b_curr[:, 1:], b_curr[:, -1]
    return b_curr, b_curr[:, -1]


# ─── Embedding ──────────────────────────────────────────────────────────

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
        self.dims = [d] * self.K
        offsets = list(range(0, D + 1, d))
        self.register_buffer('_offsets', torch.tensor(offsets))
        
        self.basis = nn.Parameter(torch.randn(self.K, d))
        nn.init.normal_(self.basis, std=0.02)
    
    def forward(self, tokens):
        codes = self.codes[tokens]
        B, L = tokens.shape
        D = self._offsets[-1].item()
        parts = []
        for k in range(self.K):
            o = int(self._offsets[k].item())
            d = self.dims[k]
            parts.append(codes[:, :, k:k+1] * self.basis[k, :d])
        return torch.cat(parts, dim=-1)


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
    """D-space -> vocab logits via segment-addressed readout.
    
    h ∈ ℝᴰ → split по тем же K сегментам, что и в PartitionedEmbedding.
    Каждый сегмент h_k сравнивается со своим readout r_k:
        logit_v = Σ_k z_{vk} · ⟨h_k, r_k⟩
    
    K=32: каждый сегмент выровнен с mirror group (1:1).
    """
    def __init__(self, cfg):
        super().__init__()
        codes = sparse_block_codes(cfg.vocab, K=cfg.code_dim, S=cfg.code_sparsity)
        self.K = codes.shape[1]
        self.register_buffer('codes', codes)
        
        D = cfg.D
        assert D % self.K == 0
        d = D // self.K
        self.dims = [d] * self.K
        offsets = list(range(0, D + 1, d))
        self.register_buffer('_offsets', torch.tensor(offsets))
        
        self.readout = nn.Parameter(torch.randn(self.K, d))
        nn.init.normal_(self.readout, std=0.02)
    
    def forward(self, h):
        scores = []
        for k in range(self.K):
            o = int(self._offsets[k].item())
            d = self.dims[k]
            h_k = h[:, :, o:o+d]
            r_k = self.readout[k, :d]
            scores.append((h_k * r_k).sum(dim=-1))
        scores = torch.stack(scores, dim=-1)
        return scores @ self.codes.T


# ─── Grouped Cognitive Mirror (32 эксперта) ────────────────────────────

class GroupedCognitiveMirror(nn.Module):
    """
    Ансамбль из 32 экспертов-зеркал, каждый в своём d=112 подпространстве.
    
    Каждый эксперт:
      - Имеет свой K-space (k=8) внутри своего d=112
      - Вычисляет 4 сигнала коррекции: temp, pred, smooth, sym
      - lo half k: temp + pred (медленные/долгоживущие ошибки)
      - hi half k: smooth + sym (быстрые/локальные ошибки)
      - Имеет свой tanh_bias + skip_connection + log_scale
      - Имеет meta-gate: учится доверять/игнорировать эксперта
    
    Predictive mirror:
      - W_pred: линейный предсказатель K-space (t-1 → t)
      - pred_error = hp_t - pred(hp_{t-1}) — ошибка предсказания
      - Обучает зеркало динамике VSA-состояния
    
    Frequency-Adaptive K:
      - Первые k/2 направлений K-space: temp + pred (медленные)
      - Последние k/2 направлений: smooth + sym (быстрые)
      - Естественная специализация, 0 дополнительных параметров
    
    Gradient-Adaptive Gate:
      - delta_var: running EMA variance дельты K-space
      - Эксперт с высокой variance активен, с низкой — прижат
      - Дополняет внешний grad_norm сигнал внутренней метрикой
    
    Внешний сигнал подкрепления:
      - prev_grad_norm: норма градиента по подпространству (c предыдущего backward)
      - Устанавливается извне через cache_grad_norms(grad_h) после backward
    
    Skip connection (alpha=0.1):
      - mirror = tanh(linear + bias) + alpha * linear
      - Обеспечивает per-dim градиент для log_scale даже при насыщении tanh
    """
    def __init__(self, D, G=32, k=8, w_pred_scale_init=0.1, log_scale_init_std=0.05,
                 gate_pred_scale_init=-1.0, skip_alpha=None):
        super().__init__()
        assert D % G == 0
        self.D = D
        self.G = G
        self.k = k
        self.d = D // G
        
        proj_std = 1.0 / (self.d * k) ** 0.25
        
        self.W_proj = nn.Parameter(torch.randn(G, self.d, k) * proj_std)
        self.W_out = nn.Parameter(torch.randn(G, k, self.d) * proj_std)
        
        self.w_temp = nn.Parameter(torch.randn(G, k))
        self.w_global = nn.Parameter(torch.randn(G, k))
        
        # Depthwise conv per group in K-space
        self.conv_smooth = nn.Conv1d(G * k, G * k, 3, padding=2,
                                      groups=G * k, bias=False)
        with torch.no_grad():
            # dirac_ bug: doesn't fill all channels for grouped convs
            self.conv_smooth.weight.zero_()
            self.conv_smooth.weight[:, :, 1] = 1.0  # all channels get center dirac
        
        self.w_sym_u = nn.Parameter(torch.randn(G, k))
        self.w_sym_v = nn.Parameter(torch.randn(G, k))
        
        # Predictive mirror: K-space prediction from previous step
        pred_std = 1.0 / k ** 0.5
        self.W_pred = nn.Parameter(torch.randn(G, k, k) * pred_std)
        self.w_pred_scale = nn.Parameter(torch.ones(G, k) * w_pred_scale_init)
        self.tanh_bias = nn.Parameter(torch.zeros(G, k))
        self.log_scale = nn.Parameter(torch.randn(G, self.d) * log_scale_init_std)
        
        # ─── K-space gate (per-token, per-expert from hp) ───
        # w_gate: (G, k) — maps K-state (k=8) to gate logit per expert
        gate_std = 1.0 / (self.k + 1) ** 0.5
        self.w_gate = nn.Parameter(torch.randn(G, self.k) * gate_std)
        self.b_gate = nn.Parameter(torch.zeros(G))
        # Gate coupling with W_pred: β = σ(gate_pred_scale), grows with W_pred
        self.gate_pred_scale = nn.Parameter(torch.tensor(gate_pred_scale_init))
        
        # External gradient cache (устанавливается после backward)
        self.register_buffer('_prev_grad_norm', torch.zeros(G), persistent=False)
        self.register_buffer('_delta_var', torch.zeros(G), persistent=False)  # running EMA of delta var
        self.register_buffer('_last_magnitude', torch.zeros(1), persistent=False)
        self.register_buffer('_last_gates', torch.zeros(G), persistent=False)
        self.register_buffer('_last_h_pool', torch.zeros(G, self.d), persistent=False)
        
        # ─── Per-expert learned modulation ───
        self.log_dvar_mod_scale = nn.Parameter(torch.full((G,), math.log(0.1)))
        self.dvar_mod_bias = nn.Parameter(torch.full((G,), -0.01))
        self.log_grad_mod_scale = nn.Parameter(torch.full((G,), math.log(0.1)))
        self.grad_mod_bias = nn.Parameter(torch.full((G,), -0.01))
        self.log_skip_alpha = nn.Parameter(torch.full((G,), math.log(0.1)))
    
    def forward(self, h, mem_all, global_state=None, diff=None):
        B, L, D = h.shape
        G, d, k = self.G, self.d, self.k
        
        # Split into subspaces
        h_g = h.reshape(B, L, G, d)           # (B, L, G, d)
        mem_g = mem_all.reshape(B, L, G, d)
        mc_g = mem_g.mean(dim=1, keepdim=True)  # (B, 1, G, d)
        
        # Project each group to its K-space
        hp = torch.einsum('blgd,gdk->blgk', h_g, self.W_proj)    # (B, L, G, k)
        mc_k = torch.einsum('b l gd,gdk->b l gk', mc_g, self.W_proj)
        
        # hp_prev shared by sym_k and pred_error
        hp_prev = torch.cat([torch.zeros_like(hp[:, 0:1]), hp[:, :-1]], dim=1)
        
        # ─── Slow signals (lo half of K-space) ───
        # Temporal: deviation from memory centroid
        temp_k = (hp - mc_k) * self.w_temp  # (B, L, G, k)
        
        # Global: deviation from cross-layer state
        if global_state is not None:
            gs_k = torch.einsum('b l gd,gdk->b l gk',
                                global_state.reshape(1, 1, G, d), self.W_proj)
            temp_k = temp_k + (hp - gs_k) * self.w_global
        
        # Predictive: error in K-space self-prediction (t-1 -> t)
        pred_k = torch.einsum('blgk,gkk->blgk', hp_prev, self.W_pred)
        pred_error = (hp - pred_k) * self.w_pred_scale  # (B, L, G, k)
        
        # ─── Fast signals (hi half of K-space) ───
        # Smoothness: local coherence in K-space
        hp_perm = hp.permute(0, 2, 3, 1).reshape(B, G * k, L)  # (B, G*k, L)
        hp_smooth = self.conv_smooth(hp_perm)[:, :, :L]
        hp_smooth = hp_smooth.reshape(B, G, k, L).permute(0, 3, 1, 2)  # (B, L, G, k)
        smooth_k = hp - hp_smooth
        
        # Symmetry: bilinear temporal interaction
        sym_k = (hp * self.w_sym_u) * (hp_prev * self.w_sym_v)
        
        # ─── Frequency-Adaptive merge (lo/hi split) ───
        k_lo = k // 2
        delta_lo = temp_k[..., :k_lo] + pred_error[..., :k_lo]
        delta_hi = smooth_k[..., k_lo:] + sym_k[..., k_lo:]
        delta = torch.cat([delta_lo, delta_hi], dim=-1)
        
        delta = F.rms_norm(delta, (delta.shape[-1],))  # norm over k
        delta = delta + self.tanh_bias  # break zero-mean symmetry in K-space
        
        # Linear projection + skip connection
        linear = torch.einsum('blgk,gkd->blgd', delta, self.W_out)  # (B, L, G, d)
        skip_alpha = torch.exp(self.log_skip_alpha).view(1, 1, G, 1)
        mirror = torch.tanh(linear) + skip_alpha * linear
        mirror = mirror * torch.exp(self.log_scale)  # per-dim scale
        
        # ─── K-Space Gate (per-token, per-expert) ───
        # Gate signal: hp + β·pred_error where β = σ(gate_pred_scale)
        # hp: (B, L, G, k) — expert's own K-state (directly reflects output quality)
        # pred_error: (B, L, G, k) — prediction quality (W_pred dynamics)
        gate_beta = torch.sigmoid(self.gate_pred_scale)
        gate_signal = hp + gate_beta * pred_error  # (B, L, G, k)
        gate_logits = torch.einsum('blgk,gk->blg', gate_signal, self.w_gate) + self.b_gate
        grad_mod = torch.exp(self.log_grad_mod_scale) * torch.tanh(self._prev_grad_norm + self.grad_mod_bias)
        gate_logits = gate_logits + grad_mod.unsqueeze(0).unsqueeze(0)
        
        # Internal delta variance: expert with high variance is actively correcting
        with torch.no_grad():
            dvar = delta.var(dim=(0, 1), unbiased=False).mean(dim=-1)  # (G,)
            if diff is not None:
                ema_alpha = 0.8 + diff * 0.19  # diff∈[0,1] → alpha∈[0.8, 0.99]
            else:
                ema_alpha = 0.9
            self._delta_var.mul_(ema_alpha).add_(dvar * (1.0 - ema_alpha))
        dvar_mod = torch.exp(self.log_dvar_mod_scale) * torch.tanh(self._delta_var + self.dvar_mod_bias)
        gate_logits = gate_logits + dvar_mod.unsqueeze(0).unsqueeze(0)
        
        expert_gate = torch.sigmoid(gate_logits)  # (B, L, G)
        
        mirror = mirror * expert_gate.unsqueeze(-1)  # (B, L, G, d) * (B, L, G, 1)
        mirror = mirror.reshape(B, L, D)
        
        self._last_magnitude.fill_(mirror.abs().mean().item())
        self._last_gates.copy_(expert_gate.detach().mean(dim=(0, 1)))
        self._last_h_pool.copy_(h_g.detach().mean(dim=(0, 1)))  # (G, d) for live_inference
        
        return mirror
    
    def cache_grad_norms(self, grad_h):
        """Call after backward: store per-subspace gradient norm."""
        with torch.no_grad():
            g_norms = grad_h.reshape(-1, self.G, self.d).norm(dim=-1).mean(dim=0)
            self._prev_grad_norm.copy_(g_norms)


# ─── Grouped MLP ──────────────────────────────────────────────────────

class GroupedMLP(nn.Module):
    """
    Grouped bottleneck MLP with per-group expansion.

    Instead of D → D → D (rank-bounded by D), splits D into G groups
    and gives each group internal expansion (d → expand*d → d).
    Total rank still ≤ D, but each group learns richer features
    within its d-dim subspace.

    G=8, d=112, expand=4 → 4× per-group expansion at half the params
    of a full 896→896→896 MLP.
    """
    def __init__(self, D, expand, groups):
        super().__init__()
        assert D % groups == 0
        self.D = D
        self.G = groups
        self.d = D // groups
        d = self.d
        e = expand

        up_std = (2.0 / (d + e * d)) ** 0.5
        down_std = (2.0 / (e * d + d)) ** 0.5
        self.W_up = nn.Parameter(torch.randn(groups, d, e * d) * up_std)
        self.W_down = nn.Parameter(torch.randn(groups, e * d, d) * down_std)
        self.norm_w = nn.Parameter(torch.ones(D))

    def forward(self, h):
        B, L, D = h.shape
        h = F.rms_norm(h, (D,), self.norm_w)
        h = h.reshape(B, L, self.G, self.d)
        h = F.silu(torch.einsum('blgd,gdf->blgf', h, self.W_up))
        h = torch.einsum('blgf,gfd->blgd', h, self.W_down)
        return h.reshape(B, L, D)


# ─── WideBind Block ────────────────────────────────────────────────────

class WideBindBlock(nn.Module):
    """
    Hybrid block: D -> K (bottleneck bind) + VSA memory + Conv + Spectral + MLP.
    
    Key design decisions:
    - Pre-LN: RMS norm at block start
    - Bind: D->K projection, bilinear in K, K->D projection
    - Memory: VSA vector superposition (not covariance matrix)
    - Gates: per-dim element-wise
    - Conv: depthwise 48-tap
    - Spectral: DCT basis scaling
    - MLP: D -> bottleneck -> D with residual
    """
    
    def __init__(self, cfg: WideBindConfig, layer_idx: int):
        super().__init__()
        self.D = cfg.D
        self.K = cfg.bind_K
        self.layer_idx = layer_idx
        
        # Pre-LN weight
        self.register_buffer('pre_ln_w', torch.ones(cfg.D))
        
        # ─── Bind: D -> K -> bilinear -> D ───
        proj_std = 1.0 / (cfg.D * cfg.bind_K) ** 0.25
        self.W_proj = nn.Parameter(torch.randn(cfg.D, cfg.bind_K) * proj_std)
        self.w_u = nn.Parameter(torch.randn(cfg.bind_K))
        self.w_v = nn.Parameter(torch.randn(cfg.bind_K))
        self.W_out = nn.Parameter(torch.randn(cfg.bind_K, cfg.D) * proj_std)
        
        # Cognitive Mirror (32 эксперта, grouped K-space)
        self.mirror = GroupedCognitiveMirror(cfg.D, G=cfg.mlp_groups, k=cfg.mirror_k,
            w_pred_scale_init=cfg.w_pred_scale_init, log_scale_init_std=cfg.log_scale_init_std,
            gate_pred_scale_init=cfg.gate_pred_scale_init)
        
        # ─── VSA Memory (gates) ───
        self.w_i = nn.Parameter(torch.randn(cfg.D))          # content-dependent write gate
        self.w_d = nn.Parameter(torch.randn(cfg.D) * cfg.w_d_init_std)    # content-dependent decay
        self.w_q = nn.Parameter(torch.full((cfg.D,), 1.0 / math.sqrt(cfg.D)))  # warm read: mem_read ≈ mem_all at init
        self.w_mem2v = nn.Parameter(torch.randn(cfg.D))
        # Linear decay across layers: shallow → short memory, deep → long
        layer_frac = layer_idx / max(cfg.n_layers - 1, 1)
        b_d_init = 2.0 + 3.0 * layer_frac  # L0: τ≈7, L23: τ≈400
        self.b_i = nn.Parameter(torch.full((cfg.D,), -2.5))   # i_gate ~0.08 init (was -3.0, ~0.05)
        self.b_d = nn.Parameter(torch.full((cfg.D,), b_d_init))

        # First moment
        self.w_k_mu = nn.Parameter(torch.randn(cfg.D))
        self.w_q_mu = nn.Parameter(torch.randn(cfg.D))
        self.w_mu_mem = nn.Parameter(torch.randn(cfg.D))
        
        # ─── Conv ───
        self.conv = nn.Conv1d(cfg.D, cfg.D, kernel_size=cfg.conv_kernel,
                              padding=cfg.conv_kernel - 1, groups=cfg.D, bias=False)
        nn.init.normal_(self.conv.weight, std=cfg.conv_init_std)
        
        # ─── Spectral ───
        self.register_buffer('V_dct', dct_basis(cfg.D))
        lam = torch.full((cfg.D,), 0.5 + layer_idx / max(cfg.n_layers - 1, 1))
        self.lambda_k = nn.Parameter(lam)
        
        # ─── MLP (grouped: per-group 4× expansion, half params) ───
        self.mlp = GroupedMLP(cfg.D, expand=cfg.mlp_expand, groups=cfg.mlp_groups)
    
    def forward(self, h, state=None, global_state=None,
                mem2v_scale=1.0, diff=None, noise_scale=0.0):
        mem_state = mu_state = conv_state = None
        if state is not None:
            mem_state, mu_state, conv_state = state
        B, L, D = h.shape
        K = self.K
        device = h.device
        
        # ─── Pre-LN ───
        h = F.rms_norm(h, (D,), self.pre_ln_w)
        
        # ─── Conv ───
        if conv_state is None:
            conv_state = torch.zeros(B, D, self.conv.padding[0], device=device, dtype=h.dtype)
        h_perm = h.transpose(1, 2)
        h_conv = self.conv(torch.cat([conv_state, h_perm], dim=-1))
        h_conv = h_conv[..., :L].transpose(1, 2)
        conv_state_out = h_perm[:, :, -(self.conv.padding[0]):]
        h = h + h_conv
        
        # ─── Hybrid Bind: D -> K -> bilinear -> D ───
        hp = h @ self.W_proj          # (B, L, K)
        u = hp * self.w_u             # (B, L, K)
        v = hp * self.w_v             # (B, L, K)
        bind_out = (u * v) @ self.W_out  # (B, L, D)
        
        # ─── VSA Memory (adaptive gates) ───
        i_gate = F.softplus(h * self.w_i + self.b_i)        # (B, L, D) bounded [0, ∞)
        decay = torch.sigmoid(h * self.w_d + self.b_d)      # (B, L, D)
        if noise_scale > 0 and self.training:
            noise = 1.0 + noise_scale * torch.randn_like(i_gate)
            i_gate = i_gate * noise
        
        mem_all, mem_state_out = vsa_prefix_scan(decay, h * i_gate, mem_state)
        mem_read = mem_all * self.w_q                    # (B, L, D)
        
        # First moment
        mu_all, mu_state_out = vsa_prefix_scan(
            decay, h * i_gate * self.w_k_mu, mu_state)
        mu_read = mu_all * self.w_q_mu
        mem_read = mem_read + mu_read * self.w_mu_mem
        
        # ─── Mirror (self-consistency: local + global) ───
        mirror = self.mirror(h, mem_all, global_state=global_state, diff=diff)
        
        # ─── Output (adaptive memory scale) ───
        enhanced = bind_out + mem_read * self.w_mem2v * mem2v_scale + mirror
        h = h + enhanced
        
        # ─── Spectral ───
        h_dct = h @ self.V_dct.T
        h = h + (h_dct * self.lambda_k) @ self.V_dct
        
        # ─── MLP ───
        h = h + self.mlp(h)
        
        return h, (mem_state_out, mu_state_out, conv_state_out)


# ─── WideBind Stack ────────────────────────────────────────────────────

class WideBindStack(nn.Module):
    """Stack of WideBindBlock layers with embedding and lm_head."""
    
    def __init__(self, cfg: WideBindConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = PartitionedEmbedding(cfg)
        self.lm_head = PartitionedHead(cfg)
        
        self.layers = nn.ModuleList([
            WideBindBlock(cfg, i) for i in range(cfg.n_layers)
        ])
        
        self.register_buffer('final_norm_w', torch.ones(cfg.D))
    
    def forward(self, h, state=None, global_state=None):
        """h: (B, L, D) — pre-embedded tokens
           state: per-layer memory states from previous forward (or None)
           global_state: cross-layer EMA self-model (or None, created fresh)
        """
        if state is None:
            state = [None] * len(self.layers)
        B, L, D = h.shape
        
        # ─── Adaptive gate biases from mirror stats ───
        with torch.no_grad():
            expl, diff = AdaptiveController.stats(self.layers,
                expl_thresh=self.cfg.exploration_threshold,
                diff_thresh=self.cfg.differentiation_threshold)
            mem2v_scale = AdaptiveController.w_mem2v_scale(self.layers,
                min_val=self.cfg.w_mem2v_scale_min, max_val=self.cfg.w_mem2v_scale_max)
            ema_alpha = AdaptiveController.ema_alpha(self.layers,
                min_val=self.cfg.ema_alpha_min, max_val=self.cfg.ema_alpha_max)
            noise_scale = AdaptiveController.noise_scale(self.layers,
                min_val=self.cfg.noise_scale_min, max_val=self.cfg.noise_scale_max)
            n = len(self.layers)
            for i, layer in enumerate(self.layers):
                layer_frac = i / max(n - 1, 1)
                b_i_val = -3.0 + expl * 1.5
                b_d_val = (2.0 + 3.0 * layer_frac) + expl * (5.0 - (2.0 + 3.0 * layer_frac))
                layer.b_i.fill_(b_i_val)
                layer.b_d.fill_(b_d_val)
        
        # Global self-model: running EMA of layer memory centroids
        if global_state is None:
            global_state = torch.zeros(1, 1, D, device=h.device, dtype=h.dtype)
        
        new_state = []
        for layer, s in zip(self.layers, state):
            h, s_out = layer(h, s, global_state=global_state, mem2v_scale=mem2v_scale, diff=diff, noise_scale=noise_scale)
            if s_out is not None:
                mem_out = s_out[0]  # (B, D) — layer's final memory state
                # Update global state: adaptive EMA aggregation
                mem_avg = mem_out.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, D)
                global_state = (ema_alpha * global_state +
                                (1.0 - ema_alpha) * mem_avg)
                s_out = tuple(t.detach() for t in s_out)
            new_state.append(s_out)
        
        return F.rms_norm(h, (self.cfg.D,), self.final_norm_w), new_state, global_state.detach()
    
    def embed_tokens(self, tokens):
        """Token indices -> D-space vectors."""
        return self.embed(tokens)
    
    def compute_loss(self, h, targets):
        """h: (B, L, D) -> logits -> cross-entropy loss"""
        logits = self.lm_head(h)
        return F.cross_entropy(logits.reshape(-1, self.cfg.vocab),
                               targets.reshape(-1), reduction='mean')
    
    def param_count(self):
        return sum(p.numel() for p in self.parameters())
    
    def param_groups(self, lr=None, weight_decay=None, gate_lr_mult=None, gate_pred_scale_mult=None):
        """Optimizer parameter groups with weight decay.
        Gate biases (b_d, b_i) are excluded — set adaptively by AdaptiveController.
        Gate weight params get increased lr for faster adaptation.
        gate_pred_scale gets separate high lr for rapid β growth."""
        cfg = self.cfg
        lr = lr or cfg.lr
        wd = weight_decay or cfg.weight_decay
        gate_lr_mult = cfg.gate_lr_mult if gate_lr_mult is None else gate_lr_mult
        gate_pred_scale_mult = cfg.gate_pred_scale_mult if gate_pred_scale_mult is None else gate_pred_scale_mult
        
        decay = []
        no_decay = []
        gate_decay = []
        gate_no_decay = []
        gate_pred_scale_params = []
        for name, p in self.named_parameters():
            if '.b_d' in name or '.b_i' in name:
                continue  # adaptive controller handles these
            if 'gate_pred_scale' in name:
                gate_pred_scale_params.append(p)
                continue
            is_gate = any(g in name for g in ['.w_i', '.w_d', '.w_q', '.w_mem2v',
                                               '.w_k_mu', '.w_q_mu', '.w_mu_mem',
                                               '.w_u', '.w_v',
                                               '.tanh_bias', '.log_scale',
                                               '.mirror.W_proj', '.mirror.W_out',
                                               '.mirror.w_temp', '.mirror.w_global',
                                               '.mirror.W_pred', '.mirror.w_pred_scale',
                                               '.mirror.w_gate', '.mirror.b_gate',
                                               '.log_dvar_mod_scale', '.dvar_mod_bias',
                                               '.log_grad_mod_scale', '.grad_mod_bias',
                                               '.log_skip_alpha'])
            if is_gate:
                if p.ndim < 2:
                    gate_no_decay.append(p)
                else:
                    gate_decay.append(p)
            else:
                if p.ndim < 2:
                    no_decay.append(p)
                else:
                    decay.append(p)
        
        groups = [
            {'params': decay, 'lr': lr, 'weight_decay': wd},
            {'params': no_decay, 'lr': lr, 'weight_decay': 0},
        ]
        if gate_decay:
            groups.append({'params': gate_decay, 'lr': lr * gate_lr_mult, 'weight_decay': wd})
        if gate_no_decay:
            groups.append({'params': gate_no_decay, 'lr': lr * gate_lr_mult, 'weight_decay': 0})
        if gate_pred_scale_params:
            groups.append({'params': gate_pred_scale_params,
                          'lr': lr * gate_pred_scale_mult, 'weight_decay': 0})
        return groups


# ─── Adaptive Controller ──────────────────────────────────────────────

class AdaptiveController:
    """
    Computes all adaptive hyperparameters from cognitive mirror state.

    Two fundamental signals drive every parameter:
    ──────────────────────────────────────────────────────────
    exploration = min(1, |mirror| / 0.3)
        How much correction is the mirror applying.
        High → model is actively adjusting, needs aggressive learning.
        Low → model is stable, needs conservative parameters.

    differentiation = min(1, var(log_scale) / 0.1)
        How specialized has the mirror become (per-dim scaling).
        High → mirror has learned which dims to trust/suppress.
        Low → mirror hasn't differentiated, still exploring.

    Mathematically derived ranges:
    ──────────────────────────────
    b_d  ∈ [2.0 + 3.0*layer_frac, 5.0] per layer
         L0: τ≈[7, 150], L31: τ≈[150, 150]
    b_i  ∈ [-3.0, -1.5] → i_gate ≈ [0.049, 0.27] (write rate, softplus)
    w_mem2v_scale ∈ [0.5, 1.0]  (memory contribution)
    ema_alpha ∈ [0.90, 0.99]  (cross-layer memory aggregation)
    noise_scale ∈ [0.001, 0.05]  (parameter noise for exploration)
    """
    @staticmethod
    def stats(blocks, expl_thresh=0.25, diff_thresh=0.08):
        """Return (exploration, differentiation) from all blocks' mirrors."""
        var_sum = mag_sum = 0.0
        for layer in blocks:
            m = layer.mirror
            ls = m.log_scale.data
            var_sum += ls.var().item()
            mag_sum += m._last_magnitude.item()
        n = len(blocks)
        avg_var = var_sum / n
        avg_mag = mag_sum / n
        exploration = min(1.0, avg_mag / expl_thresh)
        differentiation = min(1.0, avg_var / diff_thresh)
        return exploration, differentiation

    @staticmethod
    def b_d(blocks):
        """Decay bias (average across layers). High exploration → shorter memory.

        τ = -1/ln(sigmoid(b_d))
        Per-layer init: L0=2.0(τ≈7), L31=5.0(τ≈150), capped at 5.0 by exploration.
        """
        expl, _ = AdaptiveController.stats(blocks)
        return 5.0 - expl * 2.0

    @staticmethod
    def b_i(blocks):
        """Write gate bias. High exploration → more writing (capture corrections).

        i_gate = softplus(b_i) — note: softplus, not sigmoid.
        b_i=-3.0 → i_gate≈0.049 (low write, model stable)
        b_i=-1.5 → i_gate≈0.27 (high write, exploring, bounded)
        """
        expl, _ = AdaptiveController.stats(blocks)
        return -3.0 + expl * 1.5

    @staticmethod
    def w_mem2v_scale(blocks, min_val=0.5, max_val=1.0):
        """Memory contribution scaling. High diff → trust mirror, reduce memory.

        Scale applied to w_mem2v in the enhanced output.
        min_val = memory cut in half (mirror dominates)
        max_val = full memory (mirror hasn't specialized)
        """
        _, diff = AdaptiveController.stats(blocks)
        return max_val - diff * (max_val - min_val)

    @staticmethod
    def ema_alpha(blocks, min_val=0.90, max_val=0.99):
        """Cross-layer global EMA rate. High diff → stable self-model, slow EMA.

        min_val = fast update (model still learning)
        max_val = slow update (model converged)
        """
        _, diff = AdaptiveController.stats(blocks)
        return min_val + diff * (max_val - min_val)

    @staticmethod
    def noise_scale(blocks, min_val=0.001, max_val=0.05):
        """Parameter noise for exploration. Inversely proportional to diff.

        Applied as Gaussian noise to VSA gates during training.
        max_val = high noise (model exploring, low diff)
        min_val = low noise (model converged, high diff)
        """
        _, diff = AdaptiveController.stats(blocks)
        return max_val - diff * (max_val - min_val)


class MirrorLRScheduler:
    """LR scheduler modulated by cognitive mirror state dynamics.

    Two signals:
    1. var(log_scale) — per-dim amplitude divergence. Starts at 0, grows
       as mirror learns which dimensions to amplify/suppress. LR decays
       proportionally to var / target_var.
    2. |mirror| — mean absolute mirror correction magnitude. Large when
       model is unstable, shrinks at convergence. Caps LR.

    LR_mult = max(0.05, (1 - var/target) * min(1, mag/threshold))
    """
    def __init__(self, model, optimizer, base_lr, warmup=1000,
                 target_var=0.1, mag_threshold=0.3, lr_min_ratio=0.05,
                 max_decay_steps=50000, var_min_for_lr_decay=0.001):
        self.model = model
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup = warmup
        self.target_var = target_var
        self.mag_threshold = mag_threshold
        self.lr_min_ratio = lr_min_ratio
        self.max_decay_steps = max_decay_steps
        self.var_min_for_lr_decay = var_min_for_lr_decay
        self._step = 0
        self._last_log = 0

    def _mirror_stats(self):
        var_sum = 0.0
        mag_sum = 0.0
        for layer in self.model.layers:
            ls = layer.mirror.log_scale.data
            var_sum += ls.var().item()
            mag_sum += layer.mirror._last_magnitude.item()
        n = len(self.model.layers)
        return var_sum / n, mag_sum / n

    def step(self):
        self._step += 1
        if self._step < self.warmup:
            mult = self._step / max(self.warmup, 1)
        else:
            var, mag = self._mirror_stats()

            # Mirror hasn't differentiated yet — don't cut LR on noise
            if var < self.var_min_for_lr_decay:
                mirror_mult = 1.0
            else:
                decay = 1.0 - min(1.0, var / max(self.target_var, 1e-10))
                mag_factor = min(1.0, max(self.lr_min_ratio, mag / max(self.mag_threshold, 1e-10)))
                mirror_mult = max(self.lr_min_ratio, decay * mag_factor)

            # Fallback: forced cosine over full schedule (floored at lr_min_ratio)
            post_warmup = self._step - self.warmup
            progress = post_warmup / max(self.max_decay_steps, 1)
            forced = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            forced = max(self.lr_min_ratio, forced)
            mult = min(mirror_mult, forced)

            if self._step - self._last_log >= 500:
                self._last_log = self._step
                print(f'  lr_adapt: var(ls)={var:.6f} |mirror|={mag:.4f} '
                      f'mirror_mult={mirror_mult:.4f} forced={forced:.4f} '
                      f'mult={mult:.4f} lr={self.base_lr*mult:.2e}')

        for pg in self.optimizer.param_groups:
            pg['lr'] = self.base_lr * mult

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]

    def state_dict(self):
        return {'step': self._step, 'last_log': self._last_log, 'type': 'MirrorLRScheduler'}

    def load_state_dict(self, sd):
        self._step = sd.get('step', 0)
        self._last_log = sd.get('last_log', 0)


# ─── Verify ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    cfg = WideBindConfig(n_layers=24, D=896, bottleneck=896, bind_K=32, mlp_groups=8)
    model = WideBindStack(cfg).to(device)
    n = model.param_count()
    print(f'  D=896 G=8: params={n:,} ({n/1e6:.2f}M)')
    
    print()
    cfg = WideBindConfig(n_layers=4, D=896, bottleneck=896, bind_K=32)
    model = WideBindStack(cfg).to(device)
    
    x = torch.randint(0, cfg.vocab, (2, 16), device=device)
    h = model.embed_tokens(x)
    out, state, _ = model(h)
    loss = model.compute_loss(out[:, :-1], x[:, 1:])
    loss.backward()
    
    total_grad = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    out_std = out.std().item()
    print(f'Output: {out.shape}  std={out_std:.4f}')
    print(f'Loss: {loss.item():.4f}  Grad: {total_grad:.4f}')
    print('OK' if not math.isnan(loss.item()) and total_grad > 0 else 'FAIL')
