"""WideBind: bind module."""

import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import WideBindConfig
from .vsa_utils import dct_basis, fib_sigmoid_init

def migrate_bind_state_dict(sd, n_layers, mode="off", S=1):
    """Convert old (pre-BottleneckBind) state dict keys to new format.
    Old: layers.N.W_proj (D,K)  layers.N.W_out (K,D)  layers.N.w_u (K,)  layers.N.w_v (K,)
    New: layers.N.bind.W_proj.weight (K,D)  layers.N.bind.W_out (K,D|S,K,D)  layers.N.bind.w_u (S,K)  layers.N.bind.w_v (S,K)
    """
    import re
    map_sd = {}
    for key, val in sd.items():
        m = re.match(r'layers\.(\d+)\.(W_proj|W_out|w_u|w_v|w_bind_bias)$', key)
        if not m:
            map_sd[key] = val
            continue
        lidx, param = m.groups()
        new_key = f'layers.{lidx}.bind.{param}'
        if param == 'W_proj':
            new_key = f'layers.{lidx}.bind.{param}.weight'
            map_sd[new_key] = val.t().contiguous()
        elif param == 'w_u' or param == 'w_v':
            map_sd[new_key] = val.unsqueeze(0)
        else:
            map_sd[new_key] = val
    # Remove old keys that were transformed; keep everything else
    return map_sd


# ─── Grouped MLP ──────────────────────────────────────────────────────


def _golden_shifts(K: int, S: int) -> list:
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    shifts, used = [], set()
    s = 1
    while len(shifts) < S:
        sh = int(math.floor(s * K / phi)) % K
        while sh in used or sh == 0:
            sh = (sh + 1) % K
        shifts.append(sh); used.add(sh); s += 1
    return shifts



class BottleneckBind(nn.Module):
    """Bilinear cross-mixing with Fibonacci/golden-angle shifts.

    Modes (cfg.bind_twist_mode):
      "off"     — legacy diagonal bind (u⊙v), no shifts. Exact regression.
      "shift"   — simple sum of S shifted bilinear products.
      "cascade" — Fibonacci-nested cascade, monomials up to order F_S.

    Ocular (cfg.bind_twist_ocular):
      "tied"   — shared W_out = W_proj^T for all shifts. rank(M) ≤ K.
      "multi"  — per-shift W_outˢ (S, K, D). rank(M) ≤ min(S·K, D).

    Critical: w_u, w_v init std=1.0. Bilinear gradient scales as std³;
    default 0.02 kills it (8e-6 vs 1.0).
    """

    def __init__(self, D: int, K: int, cfg):
        super().__init__()
        self.D, self.K = D, K
        self.mode = getattr(cfg, "bind_twist_mode", "off")
        self.S = int(getattr(cfg, "bind_twist_S", 4))
        if self.mode == "off":
            self.S = 1
        self.ocular = getattr(cfg, "bind_twist_ocular", "tied")
        self.gated = bool(getattr(cfg, "bind_twist_gate", False)) and self.mode != "off"
        scheme = getattr(cfg, "bind_twist_scheme", "golden")
        tie_bind = bool(getattr(cfg, "tie_bind", True))

        # Bias for projection
        self.w_bind_bias = nn.Parameter(torch.zeros(K))

        # --- objective lens ---
        self.W_proj = nn.Linear(D, K, bias=False)
        if getattr(cfg, "bind_qk_norm", False):
            self.hp_norm = nn.RMSNorm(K)
        else:
            self.hp_norm = nn.Identity()

        # --- deterministic shifts ---
        shifts = _golden_shifts(K, self.S) if scheme == "golden" else _fibonacci_shifts(K, self.S)
        self.register_buffer("shifts", torch.tensor(shifts, dtype=torch.long), persistent=False)

        # --- bilinear weights: init std=1.0 is CRITICAL ---
        self.w_u = nn.Parameter(torch.empty(self.S, K))
        self.w_v = nn.Parameter(torch.empty(self.S, K))
        nn.init.normal_(self.w_u, 0.0, 1.0)
        nn.init.normal_(self.w_v, 0.0, 1.0)

        # For shift mode with tie_bind, separate W_out per shift is needed for full rank S*K
        if self.mode == "shift" and tie_bind and self.S > 1:
            self.ocular = "multi"

        # --- ocular(s) ---
        if self.mode != "off" and self.ocular == "multi" and self.S > 1:
            self.W_out = nn.Parameter(torch.empty(self.S, K, D))
            nn.init.xavier_uniform_(self.W_out, gain=0.5)
            self._tied = False
        else:
            self.W_out = nn.Parameter(torch.empty(K, D))
            nn.init.xavier_uniform_(self.W_out, gain=0.5)
            self._tied = tie_bind
            if self._tied:
                self.W_proj.register_forward_pre_hook(self._tie_hook)

        # --- gated aperture ---
        if self.gated:
            self.w_gate_proj = nn.Linear(K, self.S, bias=True)
            nn.init.xavier_uniform_(self.w_gate_proj.weight, gain=0.5)
            nn.init.zeros_(self.w_gate_proj.bias)

        # --- cascade mix ---
        if self.mode == "cascade":
            self.mix_logit = nn.Parameter(fib_sigmoid_init(self.S).log() - (1 - fib_sigmoid_init(self.S)).log())

    def _tie_hook(self, module, inp):
        with torch.no_grad():
            self.W_out.data.copy_(self.W_proj.weight.data)

    def _cross(self, left, right, shift):
        return left * torch.roll(right, shifts=int(shift), dims=-1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        hp = self.hp_norm(self.W_proj(h) + self.w_bind_bias)  # (B, L, K)

        if self.gated:
            g = torch.sigmoid(self.w_gate_proj(hp)).unsqueeze(-1)  # (B,L,S,1)
        else:
            g = None

        # --- OFF: legacy diagonal ---
        if self.mode == "off":
            prod = (hp * self.w_u[0]) * (hp * self.w_v[0])
            return prod @ self.W_out

        # --- SHIFT: simple sum of S shifted products ---
        if self.mode == "shift":
            if not self._tied and self.ocular == "multi":
                out = None
                for s in range(self.S):
                    prod = self._cross(hp * self.w_u[s], hp * self.w_v[s], self.shifts[s])
                    if g is not None:
                        prod = prod * g[:, :, s]
                    term = prod @ self.W_out[s]
                    out = term if out is None else out + term
                return out
            else:
                acc = None
                for s in range(self.S):
                    prod = self._cross(hp * self.w_u[s], hp * self.w_v[s], self.shifts[s])
                    if g is not None:
                        prod = prod * g[:, :, s]
                    acc = prod if acc is None else acc + prod
                return acc @ self.W_out

        # --- CASCADE: Fibonacci-nested ---
        if self.mode == "cascade":
            a = [None] * (self.S + 1)
            a[1] = hp * self.w_u[0]
            a[2] = hp * self.w_v[0] if self.S >= 2 else a[1]
            seed_norm = a[1].norm(dim=-1, keepdim=True).detach()
            for n in range(3, self.S + 1):
                crossed = self._cross(a[n-1] * self.w_u[n-1], a[n-2] * self.w_v[n-1], self.shifts[n-1])
                a[n] = F.normalize(crossed + 1e-10, dim=-1) * seed_norm

            mix = torch.softmax(self.mix_logit, dim=0)
            if not self._tied and self.ocular == "multi":
                out = None
                for n in range(1, self.S + 1):
                    w = mix[n-1]
                    if g is not None:
                        w = w * g[:, :, n-1]
                    term = a[n] * w.unsqueeze(-1) @ self.W_out[n-1]
                    out = term if out is None else out + term
                return out
            else:
                m = None
                for n in range(1, self.S + 1):
                    w = mix[n-1]
                    if g is not None:
                        w = w * g[:, :, n-1]
                    term = a[n] * w.unsqueeze(-1)
                    m = term if m is None else m + term
                return m @ self.W_out



def _fibonacci_shifts(K: int, S: int) -> list:
    shifts, used, a, b = [], set(), 1, 1
    guard = 0
    while len(shifts) < S and guard < 10 * S:
        sh = b % K
        if sh not in used and sh != 0:
            shifts.append(sh); used.add(sh)
        a, b = b, a + b; guard += 1
    if len(shifts) < S:
        for g in _golden_shifts(K, S):
            if g not in used:
                shifts.append(g); used.add(g)
            if len(shifts) == S:
                break
    return shifts


class SpiralBind(nn.Module):
    def __init__(self, D, K, cfg):
        super().__init__()
        self.D, self.K = D, K
        self.S = int(getattr(cfg, "bind_twist_S", 4))
        self.W_proj = nn.Linear(D, K, bias=True)
        self.w_bind_bias = nn.Parameter(torch.zeros(K))
        if getattr(cfg, "bind_qk_norm", False):
            self.hp_norm = nn.RMSNorm(K)
        else:
            self.hp_norm = nn.Identity()
        self.w_u_re = nn.Parameter(torch.randn(self.S, K) * 0.3)
        self.w_u_im = nn.Parameter(torch.zeros(self.S, K))
        self.w_v_re = nn.Parameter(torch.randn(self.S, K) * 0.3)
        self.w_v_im = nn.Parameter(torch.zeros(self.S, K))
        tau_init = torch.log(torch.arange(1, K + 1, dtype=torch.float32) / K).unsqueeze(0).expand(self.S, -1).clone()
        self.W_freq = nn.Parameter(tau_init)
        self.W_phase = nn.Parameter(torch.randn(self.S, K) * 0.1)
        self.W_out = nn.Parameter(torch.empty(2 * K, D))
        nn.init.xavier_uniform_(self.W_out, gain=0.5)
        self._tied = False

    def forward(self, h):
        hp = self.hp_norm(self.W_proj(h) + self.w_bind_bias)
        K = self.K
        out_acc = None
        for s in range(self.S):
            freq = torch.exp(self.W_freq[s]).unsqueeze(0).unsqueeze(0)
            phase = self.W_phase[s].unsqueeze(0).unsqueeze(0)
            theta = freq * hp + phase
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            u_re = hp * self.w_u_re[s].unsqueeze(0).unsqueeze(0)
            u_im = hp * self.w_u_im[s].unsqueeze(0).unsqueeze(0)
            v_re = hp * self.w_v_re[s].unsqueeze(0).unsqueeze(0)
            v_im = hp * self.w_v_im[s].unsqueeze(0).unsqueeze(0)
            vr_re = v_re * cos_t - v_im * sin_t
            vr_im = v_re * sin_t + v_im * cos_t
            prod_re = u_re * vr_re - u_im * vr_im
            prod_im = u_re * vr_im + u_im * vr_re
            out_s = torch.cat([prod_re, prod_im], dim=-1)
            out_acc = out_s if out_acc is None else out_acc + out_s
        return out_acc @ self.W_out


def _fib_sequence(n_max):
    fibs = [1, 2]
    while fibs[-1] + fibs[-2] < n_max:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs


def _zeckendorf_levels(t, max_levels=6):
    fibs = _fib_sequence(100)[:max_levels]
    weights = []
    prev = -2
    remaining = t
    for f in fibs:
        if f <= remaining and f > prev + 1:
            weights.append(1.0 / f)
            prev = f
            remaining -= f
        else:
            weights.append(0.0)
    w = torch.tensor(weights[:max_levels], dtype=torch.float32)
    return w / (w.sum() + 1e-8)


class TrajectorySpiralBind(nn.Module):
    def __init__(self, D, K, cfg):
        super().__init__()
        self.D, self.K = D, K
        self.S = int(getattr(cfg, "bind_twist_S", 4))
        self.n_dims = int(getattr(cfg, "bind_traj_dims", 3))
        self.W_proj = nn.Linear(D, K, bias=True)
        self.w_bind_bias = nn.Parameter(torch.zeros(K))
        if getattr(cfg, "bind_qk_norm", False):
            self.hp_norm = nn.RMSNorm(K)
        else:
            self.hp_norm = nn.Identity()
        self.w_u_re = nn.Parameter(torch.randn(self.S, self.n_dims, K) * 0.3)
        self.w_u_im = nn.Parameter(torch.zeros(self.S, self.n_dims, K))
        self.w_v_re = nn.Parameter(torch.randn(self.S, self.n_dims, K) * 0.3)
        self.w_v_im = nn.Parameter(torch.zeros(self.S, self.n_dims, K))
        tau_init = torch.log(torch.arange(1, K + 1, dtype=torch.float32) / K).unsqueeze(0).unsqueeze(0).expand(self.S, self.n_dims, -1).clone()
        self.W_freq = nn.Parameter(tau_init)
        self.W_phase = nn.Parameter(torch.randn(self.S, self.n_dims, K) * 0.1)
        self.register_buffer('_step_count', torch.zeros(1, dtype=torch.long))
        self.hybrid_alpha_max = getattr(cfg, 'hybrid_alpha_max', 0.7)
        self.hybrid_alpha_min = getattr(cfg, 'hybrid_alpha_min', 0.3)
        self.W_out = nn.Parameter(torch.empty(self.n_dims * 2 * K, D))
        nn.init.xavier_uniform_(self.W_out, gain=0.5)
        self._tied = False

    def _hybrid_alpha(self):
        t = min(1.0, self._step_count.item() / 5000.0)
        return self.hybrid_alpha_min + (self.hybrid_alpha_max - self.hybrid_alpha_min) * math.exp(-2.0 * t)

    def _hrr_bind(self, a, b):
        fa = torch.fft.rfft(a, dim=-1)
        fb = torch.fft.rfft(b, dim=-1)
        return torch.fft.irfft(fa * fb, n=a.shape[-1], dim=-1)

    def _hybrid_bind(self, a, b):
        alpha = self._hybrid_alpha()
        hrr = self._hrr_bind(a, b)
        ewise = a * b
        return alpha * hrr + (1 - alpha) * ewise

    def forward(self, h, traj_state=None):
        if h.dim() == 2:
            h = h.unsqueeze(0)
        hp = self.hp_norm(self.W_proj(h) + self.w_bind_bias)
        B, L, K = hp.shape
        self._step_count += 1

        # Build trajectory: use traj_state only if sequence length matches
        if traj_state is not None:
            state_L = traj_state[0].shape[1]
            if state_L != L:
                traj_state = None

        if traj_state is None or len(traj_state) < self.n_dims - 1:
            n_have = 0 if traj_state is None else len(traj_state)
            padding = [torch.zeros_like(hp) for _ in range(self.n_dims - 1 - n_have)]
            traj = [hp] + (list(traj_state) if traj_state else []) + padding
        else:
            traj = [hp] + list(traj_state[:self.n_dims - 1])

        out_acc = None
        for s in range(self.S):
            dim_outputs = []
            for d in range(self.n_dims):
                freq = torch.exp(self.W_freq[s, d]).unsqueeze(0).unsqueeze(0)
                phase = self.W_phase[s, d].unsqueeze(0).unsqueeze(0)
                theta = freq * hp + phase
                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)
                u_re = hp * self.w_u_re[s, d].unsqueeze(0).unsqueeze(0)
                u_im = hp * self.w_u_im[s, d].unsqueeze(0).unsqueeze(0)
                v_re = traj[d] * self.w_v_re[s, d].unsqueeze(0).unsqueeze(0)
                v_im = traj[d] * self.w_v_im[s, d].unsqueeze(0).unsqueeze(0)
                vr_re = v_re * cos_t - v_im * sin_t
                vr_im = v_re * sin_t + v_im * cos_t
                prod_re = u_re * vr_re - u_im * vr_im
                prod_im = u_re * vr_im + u_im * vr_re
                hybrid = self._hybrid_bind(u_re, v_re)
                prod_re = prod_re + 0.1 * hybrid
                dim_outputs.append(torch.cat([prod_re, prod_im], dim=-1))
            out_s = torch.cat(dim_outputs, dim=-1)
            out_acc = out_s if out_acc is None else out_acc + out_s
        result = out_acc @ self.W_out
        new_traj = traj[1:]
        return result, new_traj


# ─── WideBind Block ────────────────────────────────────────────────────
