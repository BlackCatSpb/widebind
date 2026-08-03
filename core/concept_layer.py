"""Collective Concept Layer — memory of the expert collective.

This is a MEMORY, not a network. Design rules (the "no direct gradient"
requirement):

  * Accumulation / writing — every backed op runs under ``torch.no_grad()``
    and mutates plain buffers in place (slotted EMA).  The layer owns ZERO
    ``nn.Parameter`` objects, so it is structurally impossible for backprop
    to turn it into a signal path.

  * Maturity — a per-layer reference level of the mirror residual variance,
    captured during the first 100 steps.  Writes are gated until the current
    residual variance drops below ``maturity_frac`` of that level (i.e. the
    experts have started to self-organise); until then the bank is a passive
    observer.

  * Mining (U_s) — a K-state is committed to a slot only when it is both
    *confident* (uncertainty below ``uncert_theta``) and *novel* (activation
    vs the bank below ``uncert_kappa``).  ``write_delay`` skips mining during
    warmup.

  * Conflict resolution — a freshly mined K-state either refines the slot it
    best activates (close pair), contrastively rewrites it (mid similarity),
    or starts a new slot when familiarity is near zero (birth).  Writes are
    allowed only after ``maturity_warmup`` steps.

  * Read-out (``read_out=True``) — the collective context contributed to the
    block signal is computed entirely under ``no_grad`` and added to the
    residual output as a *constant* (autograd does not see the read).  A
    frozen orthonormal projection maps the slot blend back to D-space, so a
    read adds structure to the signal without opening any route for the main
    loss gradient to reach the K-states or the mirror.

All state (M, U_s, N_s, step / maturity counters) is registered persistent so
resuming a checkpoint continues warm-ups where they left off.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CollectiveConceptLayer(nn.Module):
    def __init__(
        self,
        D: int,
        k: int,
        S: int = 8,
        write_delay: int = 5000,
        maturity_warmup: int = 5000,
        uncert_theta: float = 0.005,
        uncert_kappa: float = 0.10,
        contrast_thresh: float = 0.92,
        contrast_gain: float = 1.0,
        birth_gap: float = 0.55,
        maturity_frac: float = 0.85,
        read_out: bool = False,
        seed: int = 0,
    ):
        super().__init__()
        self.D = D
        self.k = k
        self.S = int(S)
        self._write_delay = int(write_delay)
        self._maturity_warmup = int(maturity_warmup)
        self._uncert_theta = float(uncert_theta)
        self._uncert_kappa = float(uncert_kappa)
        self._contrast_thresh = float(contrast_thresh)
        self._contrast_gain = float(contrast_gain)
        self._birth_gap = float(birth_gap)
        self._maturity_frac = float(maturity_frac)
        self._read_out = bool(read_out)
        self._last_write_step = -1
        self._seen_step = -1  # dedupe gradient-checkpoint recomputes

        # ─── Shared slot bank (persistent) ───
        g = torch.Generator().manual_seed(seed)
        self.register_buffer('M', F.normalize(torch.randn(S, k, generator=g), dim=-1))
        self.register_buffer('U_s', torch.zeros(S))                    # slot familiarity (EMA)
        self.register_buffer('N_s', torch.zeros(S, dtype=torch.long))  # slot writes
        # ── counters (persistent so warm-ups survive resume) ──
        self.register_buffer('_step', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_resvar_ref', torch.zeros(1))
        self.register_buffer('_mature', torch.zeros(1))
        if self._read_out:
            # Frozen orthonormal readout projector: slot blend (k,) -> D-space.
            g2 = torch.Generator().manual_seed(seed + 1)
            wr = torch.randn(D, k, generator=g2)
            q, _ = torch.linalg.qr(wr)
            self.register_buffer('W_read', q.contiguous())  # (D, k) orthonormal columns
        else:
            self.register_buffer('W_read', torch.zeros(D, k))

    # ── bookkeeping ──────────────────────────────────────────────────
    @torch.no_grad()
    def _bump_and_mature(self, resvar):
        self._step += 1
        if resvar is None:
            return
        s = self._step.item()
        if s <= self._maturity_warmup:
            # running mean of the early residual-variance level (maturity reference)
            self._resvar_ref.mul_((s - 1) / max(s, 1)).add_(resvar, alpha=1.0 / max(s, 1))
        else:
            ref = self._resvar_ref.item()
            if self._mature.item() == 0.0 and ref > 0.0:
                # mature once current residual variance has dropped below
                # maturity_frac of its early level (experts self-organised)
                if resvar <= ref * self._maturity_frac:
                    self._mature.fill_(1.0)

    # ── mining / writes ──────────────────────────────────────────────
    @torch.no_grad()
    def _maybe_write(self, hp, pen):
        step = self._step.item()
        if step < self._write_delay or not self._mature.item():
            self._last_write_step = -1
            return 0
        hp_d = hp.detach()
        pen_d = pen.detach()
        B, L, G, k = hp_d.shape
        S = self.S
        if pen_d.shape != (B, L):
            pen_d = pen_d.view(B, L)

        hd = F.normalize(hp_d.permute(2, 0, 1, 3).reshape(G, B * L, k), dim=-1)  # (G, BL, k)
        pen_g = pen_d.unsqueeze(-1).expand(-1, -1, G).permute(2, 0, 1).reshape(G, B * L)
        M = F.normalize(self.M, dim=-1)

        writes = 0
        cap = min(2, S)
        for gidx in range(G):
            if writes >= cap:
                break
            hg = hd[gidx]                                # (BL, k)
            pg = pen_g[gidx]
            sim = hg @ M.T                              # (BL, S)
            a, ids = sim.max(dim=1)
            ok = (pg < self._uncert_theta) & (a < self._uncert_kappa) & (a > 0.0)
            ids_ok = ids[ok]
            if ids_ok.numel() == 0:
                continue
            idx = ids_ok[torch.randint(ids_ok.numel(), (1,))].item()
            best_a = a[idx].item()
            best_s = ids[idx].item()
            proto = hg[idx]                             # (k,)

            if best_a >= self._contrast_thresh:
                # refine nearest slot
                m = F.normalize((M[best_s] + proto * best_a), dim=0)
                self.M.data[best_s].copy_(m)
                self.U_s.data[best_s].mul_(0.9).add_(best_a, alpha=0.1)
                self.N_s.data[best_s].add_(1)
                self._last_write_step = step
                writes += 1
            elif self.U_s[best_s].item() <= 0.01:
                # birth into an untouched slot
                self.M.data[best_s].copy_(proto)
                self.U_s.data[best_s].add_(best_a)
                self.N_s.data[best_s].add_(1)
                self._last_write_step = step
                writes += 1
            elif a[idx].item() <= self._birth_gap:
                # low-similarity with an occupied slot -> contrastive rewrite
                up = F.normalize((M[best_s] - self._contrast_gain * proto), dim=0)
                self.M.data[best_s].copy_(up)
                self.U_s.data[best_s].mul_(0.95)
                self._last_write_step = step
                writes += 1
            # else: defer (the pair is too similar/noise to act on)
        return writes

    # ── read-out ─────────────────────────────────────────────────────
    def forward(self, h, hp, pen, resvar=None, allow_write=None, step=None):
        """Accumulate expert states into the bank; optionally return a
        DETACHED memory read (B, L, D) or None.

        ``hp`` : (B, L, G, k) cached noisy K-states from the mirror.
        ``pen``: (B, L) residual-prediction errors.
        Returns None when read is disabled; otherwise a constant tensor with
        ``requires_grad=False`` that blocks can ADD to their residual output
        without opening an autograd path through the memory.
        """
        if step is not None:
            step = int(step)
            if step == self._seen_step:
                # backward-recompute of a checkpointed forward: skip all side effects
                if not self._read_out:
                    return None
            else:
                self._seen_step = step
                self._bump_and_mature(resvar)
                allow = self.training and (allow_write if allow_write is not None else True)
                if allow:
                    self._maybe_write(hp, pen)
        else:
            self._bump_and_mature(resvar)
            allow = self.training and (allow_write if allow_write is not None else True)
            if allow:
                self._maybe_write(hp, pen)
        if not self._read_out:
            return None
        with torch.no_grad():
            B, L = hp.shape[0], hp.shape[1]
            hp_d = hp.detach()
            S = self.S
            k = self.k
            M = F.normalize(self.M, dim=-1)
            # collective student: mean over experts' confident K-states
            col = F.normalize(hp_d.mean(dim=2), dim=-1)     # (B, L, k)
            sim = col @ M.T                                  # (B, L, S)
            a = F.softmax(sim * 2.0, dim=-1)                 # (B, L, S)
            occ = (self.U_s / (self.U_s.max() + 1e-8)).clamp(0, 1)
            blend = (a.unsqueeze(-1) * occ.view(1, 1, S, 1) * M.view(1, 1, S, k)).sum(dim=2)
            read = blend @ self.W_read.T                     # (B, L, D)
            w = 0.2 * (0.5 + 0.5 * math.tanh((self._step.item() - 0.6 * self._maturity_warmup) / 2000.0))
            return read * w

    # ── summary ─────────────────────────────────────────────────────
    @torch.no_grad()
    def info(self):
        return {
            'step': int(self._step.item()),
            'mature': bool(self._mature.item()),
            'writes': int(self.N_s.sum().item()),
            'act': float(self.U_s.mean().item()),
            'occ': float((self.U_s > 0.01).float().mean().item()),
        }