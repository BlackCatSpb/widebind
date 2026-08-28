"""WideBind: unified per-layer maturation gate.

Replaces the ad-hoc wake-up crutches (hard pm_coh threshold on mlp_mod std,
fixed pm_write_delay, bridge-injection scale=0 hack) with ONE principled
mechanism: each layer l has a maturity M_l(t) in [0,1] that gates EVERY
"wake-up" signal (live BridgeGLU modulation, private-memory write, semantic
bridge injection, intent bus). M_l is the product of two smooth, principled
factors:

  readiness_l(t) : expert saturation on layer l. Experts are "ripe" when their
                   prediction-error has DROPPED relative to the initial (random)
                   regime. readiness = sigmoid((sat_l - r0)/rs),
                   sat_l = 1 - EMA(pred_err_l) / pred_err_init_l.
  geometry_l(t)  : deeper layers (larger tau in the VSA ladder) engage LATER,
                   unified with the model's tau geometry (log-normalised so the
                   ladder spans [0,1] instead of 5 decades of raw tau):
                   geom_l = sigmoid((t - alpha*tau_norm_l*T_delay)/delta_t).

  M_l = readiness_l * geometry_l.

At M_l ~ 0 every wake-up branch is ~closed, but the FROZEN base MLP gate
(sigmoid(mod_scale_mlp) ~ 0.667) stays OPEN — so the model learns from step 0
(the old, proven-stable regime), experts saturate, readiness rises, and only
THEN do the live modulation / memory-write / bridge-inject / intent engage,
smoothly, layer-by-layer. No divergence (residual gain rho ~ 1 at start), no
deadlock (the base MLP always provides the learning signal).
"""

import math

import torch
import torch.nn as nn


class MaturationController(nn.Module):
    def __init__(self, n_layers, tau_min, tau_max, cfg):
        super().__init__()
        self.n_layers = int(n_layers)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        lf = torch.linspace(0.0, 1.0, self.n_layers)
        self.register_buffer("_lf", lf)
        self.register_buffer("tau_norm", torch.zeros(self.n_layers))
        self.register_buffer("readiness", torch.zeros(self.n_layers))
        self.register_buffer("gate", torch.zeros(self.n_layers))
        self.register_buffer("pen_init", torch.full((self.n_layers,), 1.0))
        self.register_buffer("pen_ema", torch.full((self.n_layers,), 1.0))
        self.register_buffer("_warm_done", torch.zeros(1))

        self.alpha = float(getattr(cfg, "matur_alpha", 1.0))
        self.T_delay = float(getattr(cfg, "matur_T_delay", 20000.0))
        self.delta_t = float(getattr(cfg, "matur_delta", 4000.0))
        self.r0 = float(getattr(cfg, "matur_r0", 0.6))
        self.rs = float(getattr(cfg, "matur_rs", 0.3))
        self.ema = float(getattr(cfg, "matur_ema", 0.999))
        self.warm = int(getattr(cfg, "matur_warm", 300))
        self._update_tau_norm(torch.zeros(self.n_layers))  # dev = 0 initially

    def _update_tau_norm(self, dev):
        # dev: (n_layers,) per-layer tau deviation (log-scale param _tau_l_dev).
        # log-lerp the VSA ladder, then normalise to [0,1] across the ladder.
        log_tau = math.log(self.tau_min) + (
            math.log(self.tau_max) - math.log(self.tau_min)
        ) * self._lf * (1.0 + 0.1 * dev)
        denom = math.log(self.tau_max) - math.log(self.tau_min)
        if denom <= 0:
            denom = 1.0
        self.tau_norm.copy_(((log_tau - math.log(self.tau_min)) / denom).clamp(0.0, 1.0))

    def step_gate(self, step, tau_dev=None):
        """Gate for THIS step, computed from the previous-step readiness EMA.

        Call BEFORE the layer loop (pred_err for this step is not known yet).
        Returns a (n_layers,) tensor in [0,1].
        """
        if tau_dev is not None:
            self._update_tau_norm(tau_dev)
        t = float(step)
        geom = torch.sigmoid((t - self.alpha * self.tau_norm * self.T_delay) / self.delta_t)
        gate = self.readiness * geom
        self.gate.copy_(gate)
        return gate

    def update(self, step, pred_err):
        """Update readiness EMA from this step's per-layer pred_err (detached, (n_layers,))."""
        with torch.no_grad():
            pe = pred_err.detach().float().clamp(min=1e-6)
            if self._warm_done.item() < 0.5:
                # Warm phase: capture the RANDOM-regime baseline. Keep pen_ema == pen_init
                # so sat=0 (readiness ~ 0) until we know the baseline and the model has
                # actually started learning. Opening gates during warm is the bug we fix.
                self.pen_init.copy_(torch.maximum(self.pen_init, pe))
                self.pen_ema.copy_(self.pen_init)
                if int(step) >= self.warm:
                    self._warm_done.fill_(1.0)
            else:
                self.pen_ema.lerp_(pe, 1.0 - self.ema)
            sat = (1.0 - self.pen_ema / self.pen_init.clamp(min=1e-6)).clamp(0.0, 1.0)
            self.readiness.copy_(torch.sigmoid((sat - self.r0) / self.rs))
