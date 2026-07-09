"""
WideBind: hybrid D-space LM with VSA memory + bottleneck bind.
"""

import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from wbconfig import WideBindConfig


# ─── Utilities ──────────────────────────────────────────────────────────

def dct_basis(n):
    """DCT-II basis vectors of shape (n, n) — orthogonal rows."""
    k = torch.arange(n, dtype=torch.float32)
    v = k.unsqueeze(1) * (k.unsqueeze(0) + 0.5)
    basis = torch.cos(v * math.pi / n)
    basis[0, :] = basis[0, :] / math.sqrt(2)
    return basis * math.sqrt(2.0 / n)


def zeckendorf_codes(vocab=50000):
    """Fibonacci Zeckendorf binary codes for vocab tokens."""
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


def compute_timescales(cfg):
    """Timescale biases for multi-timescale decay."""
    tau_min, tau_max = cfg.cov_tau_lo, cfg.cov_tau_hi
    n = cfg.n_layers
    tau = torch.exp(torch.linspace(math.log(tau_min), math.log(tau_max), n))
    return tau


def compute_spectrum(cfg):
    """Spectral weight vector for DCT mixing."""
    n = cfg.n_layers
    lo, hi = cfg.spec_lo, cfg.spec_hi
    lam = torch.linspace(lo, hi, n)
    return lam, lam


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
    a_curr, b_curr = a.clone(), b.clone()
    step = 1
    while step < n:
        a_prev, b_prev = a_curr.clone(), b_curr.clone()
        a_curr[:, step:] = a_prev[:, step:] * a_prev[:, :-step]
        b_curr[:, step:] = b_prev[:, :-step] * a_prev[:, step:] + b_prev[:, step:]
        step *= 2
    
    if state is not None:
        return b_curr[:, 1:], b_curr[:, -1]
    return b_curr, b_curr[:, -1]


# ─── Embedding ──────────────────────────────────────────────────────────

class ZeckendorfEmbedding(nn.Module):
    """Token -> D-space via Zeckendorf codes + learned projection."""
    def __init__(self, cfg):
        super().__init__()
        codes = zeckendorf_codes(cfg.vocab)
        K = codes.shape[1]
        self.register_buffer('codes', codes)
        self.proj = nn.Linear(K, cfg.D, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)
    
    def forward(self, tokens):
        return self.proj(self.codes[tokens])


class LmHead(nn.Module):
    """D-space -> vocab logits via Zeckendorf code projection."""
    def __init__(self, cfg):
        super().__init__()
        codes = zeckendorf_codes(cfg.vocab)
        K = codes.shape[1]
        self.register_buffer('codes', codes)
        self.proj = nn.Linear(cfg.D, K, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)
    
    def forward(self, h):
        return self.proj(h) @ self.codes.T


# ─── Cognitive Mirror ─────────────────────────────────────────────────

class CognitiveMirror(nn.Module):
    """
    Unified self-consistency mirror with three bounded correction paths.
    
    Local (per-layer) paths:
      1. Temporal:   h[t] - memory centroid    (VSA prediction error)
      2. Smoothness: h[t] - conv1x3(h[t])      (local coherence)
      3. Symmetry:   h[t] . h[t-1]              (bilinear trajectory)
    
    Global (cross-layer) path:
      - h[t] - global_state                     (deviation from global self-model)
    
    All paths in K-space -> rms_norm -> tanh(W_out) -> exp(log_scale).
    tanh guarantees bounded correction; exp(log_scale) gives per-dim amplitude.
    """
    def __init__(self, D, K):
        super().__init__()
        proj_std = 1.0 / (D * K) ** 0.25
        
        self.W_proj = nn.Parameter(torch.randn(D, K) * proj_std)
        self.W_out = nn.Parameter(torch.randn(K, D) * proj_std)
        
        self.w_temp = nn.Parameter(torch.randn(K))
        self.w_global = nn.Parameter(torch.randn(K))
        
        self.conv_smooth = nn.Conv1d(K, K, 3, padding=2, groups=K, bias=False)
        nn.init.dirac_(self.conv_smooth.weight)
        
        self.w_sym_u = nn.Parameter(torch.randn(K))
        self.w_sym_v = nn.Parameter(torch.randn(K))
        
        self.log_scale = nn.Parameter(torch.zeros(D))
    
    def forward(self, h, mem_all, global_state=None):
        B, L, D = h.shape
        K = self.W_proj.shape[1]
        
        hp = h @ self.W_proj  # (B, L, K)
        
        # 1. Temporal: deviation from local memory centroid
        mem_centroid = mem_all.mean(dim=1, keepdim=True)
        mc_k = mem_centroid @ self.W_proj
        temp_k = (hp - mc_k) * self.w_temp
        
        # 1b. Global: deviation from cross-layer state
        if global_state is not None:
            gs_k = global_state @ self.W_proj
            temp_k = temp_k + (hp - gs_k) * self.w_global
        
        # 2. Smoothness: local coherence via conv1x3
        hp_perm = hp.transpose(1, 2)
        hp_smooth = self.conv_smooth(hp_perm)[:, :, :L].transpose(1, 2)
        smooth_k = hp - hp_smooth
        
        # 3. Symmetry: h[t] . h[t-1] bilinear (zero at t=0, no predecessor)
        hp_prev = torch.cat([torch.zeros_like(hp[:, 0:1]), hp[:, :-1]], dim=1)
        sym_k = (hp * self.w_sym_u) * (hp_prev * self.w_sym_v)
        
        delta = temp_k + smooth_k + sym_k
        delta = F.rms_norm(delta, (K,))
        
        mirror = torch.tanh(delta @ self.W_out)
        mirror = mirror * torch.exp(self.log_scale)
        
        self._last_magnitude = mirror.abs().mean().item()
        
        return mirror


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
        
        # Cognitive Mirror (self-consistency correction)
        self.mirror = CognitiveMirror(cfg.D, cfg.bind_K)
        
        # ─── VSA Memory (gates) ───
        self.w_i = nn.Parameter(torch.randn(cfg.D))          # content-dependent write gate
        self.w_d = nn.Parameter(torch.randn(cfg.D) * 0.1)    # content-dependent decay
        self.w_q = nn.Parameter(torch.randn(cfg.D))
        self.w_mem2v = nn.Parameter(torch.randn(cfg.D))
        self.b_i = nn.Parameter(torch.full((cfg.D,), -3.0))  # sigmoid init → i_gate ≈ 0.047
        self.b_d = nn.Parameter(torch.full((cfg.D,), 5.0))    # high init → τ ≈ 150

        # First moment
        
        # First moment
        self.w_k_mu = nn.Parameter(torch.randn(cfg.D))
        self.w_q_mu = nn.Parameter(torch.randn(cfg.D))
        self.w_mu_mem = nn.Parameter(torch.randn(cfg.D))
        
        # ─── Conv ───
        self.conv = nn.Conv1d(cfg.D, cfg.D, kernel_size=cfg.conv_kernel,
                              padding=cfg.conv_kernel - 1, groups=cfg.D, bias=False)
        nn.init.normal_(self.conv.weight, std=0.01)
        
        # ─── Spectral ───
        self.register_buffer('V_dct', dct_basis(cfg.D))
        lam = torch.full((cfg.D,), 0.5 + layer_idx / max(cfg.n_layers - 1, 1))
        self.lambda_k = nn.Parameter(lam)
        
        # ─── MLP (grouped: per-group 4× expansion, half params) ───
        self.mlp = GroupedMLP(cfg.D, expand=cfg.mlp_expand, groups=cfg.mlp_groups)
    
    def forward(self, h, state=None, global_state=None,
                mem2v_scale=1.0):
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
        i_gate = torch.sigmoid(h * self.w_i + self.b_i)     # (B, L, D) bounded [0, 1]
        decay = torch.sigmoid(h * self.w_d + self.b_d)      # (B, L, D)
        
        mem_all, mem_state_out = vsa_prefix_scan(decay, h * i_gate, mem_state)
        mem_read = mem_all * self.w_q                    # (B, L, D)
        
        # First moment
        mu_all, mu_state_out = vsa_prefix_scan(
            decay, h * i_gate * self.w_k_mu, mu_state)
        mu_read = mu_all * self.w_q_mu
        mem_read = mem_read + mu_read * self.w_mu_mem
        
        # ─── Mirror (self-consistency: local + global) ───
        mirror = self.mirror(h, mem_all, global_state=global_state)
        
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
        self.embed = ZeckendorfEmbedding(cfg)
        self.lm_head = LmHead(cfg)
        
        self.layers = nn.ModuleList([
            WideBindBlock(cfg, i) for i in range(cfg.n_layers)
        ])
        
        self.register_buffer('final_norm_w', torch.ones(cfg.D))
    
    def forward(self, h, state=None):
        """h: (B, L, D) — pre-embedded tokens"""
        if state is None:
            state = [None] * len(self.layers)
        B, L, D = h.shape
        
        # ─── Adaptive gate biases from mirror stats ───
        with torch.no_grad():
            expl, _ = AdaptiveController.stats(self.layers)
            b_d_val = 6.0 - expl * 3.0
            b_i_val = -5.0 + expl * 4.0
            mem2v_scale = AdaptiveController.w_mem2v_scale(self.layers)
            ema_alpha = AdaptiveController.ema_alpha(self.layers)
            for layer in self.layers:
                layer.b_i.fill_(b_i_val)
                layer.b_d.fill_(b_d_val)
        
        # Global self-model: running EMA of layer memory centroids
        global_state = torch.zeros(1, 1, D, device=h.device, dtype=h.dtype)
        
        new_state = []
        for layer, s in zip(self.layers, state):
            h, s_out = layer(h, s, global_state=global_state, mem2v_scale=mem2v_scale)
            if s_out is not None:
                mem_out = s_out[0]  # (B, D) — layer's final memory state
                # Update global state: adaptive EMA aggregation
                mem_avg = mem_out.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, D)
                global_state = (ema_alpha * global_state +
                                (1.0 - ema_alpha) * mem_avg)
                s_out = tuple(t.detach() for t in s_out)
            new_state.append(s_out)
        
        return F.rms_norm(h, (self.cfg.D,), self.final_norm_w), new_state
    
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
    
    def param_groups(self, lr=None, weight_decay=None):
        """Optimizer parameter groups with weight decay.
        Gate biases (b_d, b_i) are excluded — they are set adaptively by AdaptiveController."""
        cfg = self.cfg
        lr = lr or cfg.lr
        wd = weight_decay or cfg.weight_decay
        
        decay = []
        no_decay = []
        for name, p in self.named_parameters():
            if '.b_d' in name or '.b_i' in name:
                continue  # adaptive controller handles these
            if p.ndim < 2:
                no_decay.append(p)
            else:
                decay.append(p)
        
        return [
            {'params': decay, 'lr': lr, 'weight_decay': wd},
            {'params': no_decay, 'lr': lr, 'weight_decay': 0},
        ]


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
    b_d  ∈ [3.0, 6.0]  →  τ ≈ [20, 400]  (decay timescale)
    b_i  ∈ [-5.0, -1.0] → i_gate ≈ [0.007, 0.269] (write rate)
    w_mem2v_scale ∈ [0.5, 1.0]  (memory contribution)
    ema_alpha ∈ [0.90, 0.99]  (cross-layer memory aggregation)
    noise_scale ∈ [0.001, 0.05]  (parameter noise for exploration)
    """
    @staticmethod
    def stats(blocks):
        """Return (exploration, differentiation) from all blocks' mirrors."""
        var_sum = mag_sum = 0.0
        for layer in blocks:
            m = layer.mirror
            ls = m.log_scale.data
            var_sum += ls.var().item()
            if hasattr(m, '_last_magnitude'):
                mag_sum += m._last_magnitude
        n = len(blocks)
        avg_var = var_sum / n
        avg_mag = mag_sum / n
        exploration = min(1.0, avg_mag / 0.3)
        differentiation = min(1.0, avg_var / 0.1)
        return exploration, differentiation

    @staticmethod
    def b_d(blocks):
        """Decay bias. High exploration → shorter memory (model needs fresh signal).

        τ = -1/ln(sigmoid(b_d))
        b_d=3.0 → τ≈20  (short memory, high exploration)
        b_d=6.0 → τ≈400 (long memory, low exploration)
        """
        expl, _ = AdaptiveController.stats(blocks)
        return 6.0 - expl * 3.0

    @staticmethod
    def b_i(blocks):
        """Write gate bias. High exploration → more writing (capture corrections).

        i_gate = sigmoid(b_i)
        b_i=-5.0 → i_gate≈0.007 (low write, model stable)
        b_i=-1.0 → i_gate≈0.269 (high write, exploring)
        """
        expl, _ = AdaptiveController.stats(blocks)
        return -5.0 + expl * 4.0

    @staticmethod
    def w_mem2v_scale(blocks):
        """Memory contribution scaling. High diff → trust mirror, reduce memory.

        Scale applied to w_mem2v in the enhanced output.
        0.5 = memory cut in half (mirror dominates)
        1.0 = full memory (mirror hasn't specialized)
        """
        _, diff = AdaptiveController.stats(blocks)
        return 1.0 - diff * 0.5

    @staticmethod
    def ema_alpha(blocks):
        """Cross-layer global EMA rate. High diff → stable self-model, slow EMA.

        0.90 = fast update (model still learning)
        0.99 = slow update (model converged)
        """
        _, diff = AdaptiveController.stats(blocks)
        return 0.90 + diff * 0.09

    @staticmethod
    def noise_scale(blocks):
        """Parameter noise for exploration. Inversely proportional to diff.

        Applied as Gaussian noise to VSA gates during training.
        0.05 = high noise (model exploring, low diff)
        0.001 = low noise (model converged, high diff)
        """
        _, diff = AdaptiveController.stats(blocks)
        return 0.05 - diff * 0.049


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
                 max_decay_steps=50000):
        self.model = model
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup = warmup
        self.target_var = target_var
        self.mag_threshold = mag_threshold
        self.lr_min_ratio = lr_min_ratio
        self.max_decay_steps = max_decay_steps
        self._step = 0
        self._last_log = 0

    def _mirror_stats(self):
        var_sum = 0.0
        mag_sum = 0.0
        for layer in self.model.layers:
            ls = layer.mirror.log_scale.data
            var_sum += ls.var().item()
            if hasattr(layer.mirror, '_last_magnitude'):
                mag_sum += layer.mirror._last_magnitude
        n = len(self.model.layers)
        return var_sum / n, mag_sum / n

    def step(self):
        self._step += 1
        if self._step < self.warmup:
            mult = self._step / max(self.warmup, 1)
        else:
            var, mag = self._mirror_stats()

            # Mirror hasn't differentiated yet — don't cut LR on noise
            if var < 0.001:
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
    
    cfg = WideBindConfig(n_layers=24, D=896, bottleneck=896, bind_K=16, mlp_groups=8)
    model = WideBindStack(cfg).to(device)
    n = model.param_count()
    print(f'  D=896 G=8: params={n:,} ({n/1e6:.2f}M)')
    
    print()
    cfg = WideBindConfig(n_layers=4, D=896, bottleneck=896, bind_K=16)
    model = WideBindStack(cfg).to(device)
    
    x = torch.randint(0, cfg.vocab, (2, 16), device=device)
    h = model.embed_tokens(x)
    out, state = model(h)
    loss = model.compute_loss(out[:, :-1], x[:, 1:])
    loss.backward()
    
    total_grad = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    out_std = out.std().item()
    print(f'Output: {out.shape}  std={out_std:.4f}')
    print(f'Loss: {loss.item():.4f}  Grad: {total_grad:.4f}')
    print('OK' if not math.isnan(loss.item()) and total_grad > 0 else 'FAIL')
