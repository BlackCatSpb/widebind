"""
WideBind: hybrid D-space LM with VSA memory + bottleneck bind.
"""

import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import WideBindConfig


# ─── Utilities ──────────────────────────────────────────────────────────

def dct_basis(n):
    """DCT-II basis vectors of shape (n, n)."""
    k = torch.arange(n, dtype=torch.float32)
    v = k.unsqueeze(1) * (k.unsqueeze(0) + 0.5)
    basis = torch.cos(v * math.pi / n)
    basis[:, 0] = basis[:, 0] / math.sqrt(2)
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
        
        # Mirror bind
        self.W_proj_m = nn.Parameter(torch.randn(cfg.D, cfg.bind_K) * proj_std)
        self.w_u_m = nn.Parameter(torch.randn(cfg.bind_K))
        self.w_v_m = nn.Parameter(torch.randn(cfg.bind_K))
        self.W_out_m = nn.Parameter(torch.randn(cfg.bind_K, cfg.D) * proj_std)
        self.mirror_scale = nn.Parameter(torch.tensor(0.1))
        
        # ─── VSA Memory (gates) ───
        self.w_i = nn.Parameter(torch.randn(cfg.D) * 0.1)
        self.w_d = nn.Parameter(torch.randn(cfg.D) * 0.1)
        self.w_q = nn.Parameter(torch.randn(cfg.D))
        self.w_mem2v = nn.Parameter(torch.randn(cfg.D))
        self.b_i = nn.Parameter(torch.full((cfg.D,), 1.0))
        self.b_d = nn.Parameter(torch.zeros(cfg.D))
        
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
        
        # ─── MLP ───
        self.mlp_up = nn.Linear(cfg.D, cfg.bottleneck, bias=False)
        self.mlp_down = nn.Linear(cfg.bottleneck, cfg.D, bias=False)
        nn.init.xavier_uniform_(self.mlp_up.weight)
        nn.init.xavier_uniform_(self.mlp_down.weight)
        self.register_buffer('mlp_norm_w', torch.ones(cfg.D))
    
    def forward(self, h, state=None):
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
        
        # ─── VSA Memory ───
        i_gate = torch.exp(h * self.w_i + self.b_i)     # (B, L, D)
        decay = torch.sigmoid(h * self.w_d + self.b_d)  # (B, L, D)
        
        mem_all, mem_state_out = vsa_prefix_scan(decay, h * i_gate, mem_state)
        mem_read = mem_all * self.w_q                    # (B, L, D)
        
        # First moment
        mu_all, mu_state_out = vsa_prefix_scan(
            decay, h * i_gate * self.w_k_mu, mu_state)
        mu_read = mu_all * self.w_q_mu
        mem_read = mem_read + mu_read * self.w_mu_mem
        
        # ─── Mirror ───
        h_centered = h - h.mean(dim=1, keepdim=True)
        hp_m = h_centered @ self.W_proj_m
        mirror_u = (h @ self.W_proj_m) * self.w_v_m
        mirror = ((hp_m * self.w_u_m) * mirror_u) @ self.W_out_m
        mirror = mirror * self.mirror_scale
        
        # ─── Output ───
        enhanced = bind_out + mem_read * self.w_mem2v + mirror
        h = h + enhanced
        
        # ─── Spectral ───
        h_dct = h @ self.V_dct.T
        h = h + (h_dct * self.lambda_k) @ self.V_dct
        
        # ─── MLP ───
        h_mlp = F.rms_norm(h, (D,), self.mlp_norm_w)
        h_mlp = F.silu(self.mlp_up(h_mlp))
        h = h + self.mlp_down(h_mlp)
        
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
        new_state = []
        for layer, s in zip(self.layers, state):
            h, s_out = layer(h, s)
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
        """Optimizer parameter groups with weight decay."""
        cfg = self.cfg
        lr = lr or cfg.lr
        wd = weight_decay or cfg.weight_decay
        
        decay = []
        no_decay = []
        for name, p in self.named_parameters():
            if p.ndim < 2:
                no_decay.append(p)
            else:
                decay.append(p)
        
        return [
            {'params': decay, 'lr': lr, 'weight_decay': wd},
            {'params': no_decay, 'lr': lr, 'weight_decay': 0},
        ]


# ─── Verify ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for bottleneck in [896, 3584]:
        cfg = WideBindConfig(n_layers=24, bottleneck=bottleneck, bind_K=16)
        model = WideBindStack(cfg).to(device)
        n = model.param_count()
        print(f'  bottleneck={bottleneck:>5}: params={n:,} ({n/1e6:.2f}M)')
    
    print()
    cfg = WideBindConfig(n_layers=4, bottleneck=896, bind_K=16)
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
