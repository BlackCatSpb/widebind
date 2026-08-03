"""
Collective Concept Layer — ported to WideBind (big model) from the Mini prototype.

Concepts live in their OWN slot bank (S shared slots in K-space), separate from
private_mem and the block signal path. Two phases:

  ACCUMULATION (read_out=False, default):
    - The layer only WRITES (no_grad): it mines confident+novel tokens into its
      slot bank, gated by the layer's OWN maturity signal (mirror residual-var EMA).
    - Nothing is emitted into the block signal -> the training graph is untouched,
      zero learnable parameters, zero impact on loss/other methods.
    - Use this to observe per-layer readiness before enabling readout.

  READOUT (read_out=True):
    - W_o (S*k -> D) readout is created; concepts enter the block signal gated by
      uncertainty gate (main signal weak) and contradiction gate (concept must
      align with current context — the "pink elephant" guard).

Maturity = the model's own quality signal: mirror residual-var EMA settles below
a fraction (maturity_frac) of its post-warmup reference -> the layer is "mature"
and mining turns on. Reference is captured after maturity_warmup steps, NOT at
step 0 when resvar is still initialization noise.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CollectiveConceptLayer(nn.Module):
    def __init__(self, D, k, S=8, write_delay=5000, maturity_warmup=5000,
                 uncert_theta=0.5, uncert_kappa=3.0,
                 contra_thresh=-0.1, contra_gain=6.0,
                 birth_gap=0.55, maturity_frac=0.85,
                 read_out=False, seed=None):
        super().__init__()
        self.D = D
        self.k = k
        self.S = S
        self._write_delay = write_delay
        self._maturity_warmup = maturity_warmup
        self._uncert_theta = uncert_theta
        self._uncert_kappa = uncert_kappa
        self._contra_thresh = contra_thresh
        self._contra_gain = contra_gain
        self._birth_gap = birth_gap
        self._maturity_frac = maturity_frac
        self._read_out = read_out

        g = torch.Generator().manual_seed(seed) if seed is not None else None
        m_init = torch.randn(S, k, generator=g)
        self.register_buffer('M', F.normalize(m_init, dim=-1))
        self.register_buffer('U_s', torch.zeros(S))
        self.register_buffer('N_s', torch.zeros(S, dtype=torch.long))
        self.register_buffer('_step', torch.zeros(1, dtype=torch.long), persistent=True)
        self.register_buffer('_resvar_ref', torch.zeros(1), persistent=True)
        self.register_buffer('_mature', torch.zeros(1), persistent=True)
        self.register_buffer('_gate_u', torch.zeros(1), persistent=False)
        self.register_buffer('_gate_c', torch.zeros(1), persistent=False)
        self._last_write_step = -1

        if read_out:
            self.W_o = nn.Linear(S * k, D, bias=False)
            nn.init.orthogonal_(self.W_o.weight)
            self._read_scale = nn.Parameter(torch.tensor(0.0))
            self._temp = nn.Parameter(torch.tensor(2.0))
        else:
            self.W_o = None
            self._read_scale = None
            self._temp = None

    # ─── diagnostics ───
    def debug(self):
        occ = (self.U_s / (self.U_s.sum() + 1e-8)).tolist()
        counts = self.N_s.tolist()
        return {
            'mature': self._mature.item(),
            'u_gate': self._gate_u.item(),
            'c_gate': self._gate_c.item(),
            'read_scale': torch.sigmoid(self._read_scale).item() if self._read_scale is not None else None,
            'occupied': int((self.N_s > 0).sum().item()),
            'step': self._step.item(),
            'U_s': [round(o, 3) for o in occ],
            'N_s': counts,
        }

    @torch.no_grad()
    def _update_maturity(self, resvar):
        """The layer matures relative to itself: settled residual variance."""
        if resvar is None:
            return
        if self._resvar_ref.item() == 0.0 and self._step.item() >= self._maturity_warmup:
            self._resvar_ref.fill_(max(resvar, 1e-6))
        if self._resvar_ref.item() == 0.0:
            self._mature.fill_(0.0)
            return
        mature = resvar < self._resvar_ref.item() * self._maturity_frac
        self._mature.fill_(1.0 if mature else 0.0)

    @torch.no_grad()
    def _maybe_write(self, hp, pen, allow_write, step=None):
        """Mature-gated, confident+novel slot refinement and birth."""
        if not allow_write:
            return
        # Skip gradient-checkpoint recomputation of the SAME training step:
        # the backward pass re-runs the forward with the same `step`.
        if step is not None and step == self._last_write_step:
            return
        self._last_write_step = step
        self._step += 1
        if self._step.item() < self._write_delay:
            return
        if self._mature.item() < 0.5:
            return

        B, L, G, k = hp.shape
        shared = hp.mean(dim=-2)                      # (B,L,k)
        shared_n = F.normalize(shared, dim=-1)
        M_n = F.normalize(self.M, dim=-1)
        sim = shared_n @ M_n.T                        # (B,L,S)
        best = sim.argmax(dim=-1)
        best_sim = sim.max(dim=-1).values
        d_min = 1.0 - best_sim                        # (B,L)
        conf = torch.sigmoid(-pen)                    # (B,L) low pred_error -> confident

        # refine nearest slot with confident tokens
        for s in range(self.S):
            mask = (best == s) & (conf > 0.3)
            if mask.any():
                upd = F.normalize(shared[mask].mean(dim=0), dim=-1)
                if self.N_s[s].item() < 10:
                    self.M.data[s] = upd
                else:
                    alpha = 0.01
                    self.M.data[s] = F.normalize(
                        self.M[s] * (1 - alpha) + upd * alpha, dim=-1)
                self.N_s[s] += mask.sum().item()

        # birth: empty slot + confident novel tokens
        empty = torch.nonzero(self.N_s == 0)
        novel = (d_min > self._birth_gap) & (conf > 0.5)
        if empty.numel() > 0 and novel.any():
            idx = empty[0].item()
            self.M.data[idx] = F.normalize(shared[novel].mean(dim=0), dim=-1)
            self.N_s[idx] += 1
        elif empty.numel() == 0 and novel.any():
            # eviction: bank full -> least-used slot gets recycled for the novel concept
            evict = int(torch.argmin(self.U_s).item())
            self.M.data[evict] = F.normalize(shared[novel].mean(dim=0), dim=-1)
            self.N_s[evict] = 1
            self.U_s[evict] = 0.0

        # occupancy EMA
        occ = torch.zeros(self.S)
        for s in range(self.S):
            occ[s] = (best == s).float().mean().item()
        self.U_s.mul_(0.99).add_(occ.to(self.U_s), alpha=0.01)

    def forward(self, h, hp, pen, resvar=None, allow_write=None, mature_override=None, step=None):
        """
        h   (B,L,D)   block state (used by the contradiction gate when read_out)
        hp  (B,L,G,k) mirror K-states
        pen (B,L)     mirror pred_error norm (model's own uncertainty)
        resvar float  mirror residual-var EMA (maturity signal from the model itself)
        step int/None training step (dedupes gradient-checkpoint recomputation)

        Returns (B,L,D) concept read when read_out=True, else None.
        """
        self._update_maturity(resvar)
        self._maybe_write(hp, pen, allow_write, step=step)

        if not self._read_out:
            return None

        B, L, G, k = hp.shape
        shared = F.normalize(hp.mean(dim=-2), dim=-1)     # (B,L,k)
        M_n = F.normalize(self.M, dim=-1)
        sim = shared @ M_n.T                              # (B,L,S)
        temp = self._temp.clamp(min=0.5)
        # Независимые per-slot гейты (sigmoid+норм): слоты могут быть
        # соактивными, но сумма по S нормируется для сохранения масштаба readout.
        a = torch.sigmoid(sim * temp)                      # (B,L,S) независимые гейты
        a = a / (a.sum(dim=-1, keepdim=True) + 1e-8)       # нормировка бленда
        occ_w = (self.U_s / (self.U_s.max() + 1e-8)).clamp(0, 1)
        blend = (a.unsqueeze(-1) * occ_w.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                 * M_n.unsqueeze(0).unsqueeze(0))
        read = self.W_o(blend.reshape(B, L, -1))          # (B,L,D)

        with torch.no_grad():
            # uncertainty gate: open when the main-layer signal is weak
            u_gate = torch.sigmoid(self._uncert_kappa * (pen.unsqueeze(-1) - self._uncert_theta))
            # contradiction gate: concept must align with current context
            out_n = F.normalize(read, dim=-1)
            h_n = F.normalize(h.detach(), dim=-1)
            cos_c = (out_n * h_n).sum(dim=-1, keepdim=True)
            c_gate = torch.sigmoid(self._contra_gain * (cos_c - self._contra_thresh))
            self._gate_u.fill_(u_gate.mean().item())
            self._gate_c.fill_(c_gate.mean().item())

        scale = torch.sigmoid(self._read_scale)
        return read * u_gate * c_gate * scale
