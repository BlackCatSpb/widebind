"""WideBind: unified per-layer maturation gate.

Replaces the ad-hoc wake-up crutches (hard pm_coh threshold on mlp_mod std,
fixed pm_write_delay, bridge-injection scale=0 hack) with ONE principled
mechanism: each layer l has a maturity M_l(t) in [0,1] that gates EVERY
"wake-up" signal (live BridgeGLU modulation, private-memory write, semantic
bridge injection, intent bus).

  M_l(t) = sigmoid((t - (T0 + alpha*tau_norm_l*T_delay)) / delta_t)

i.e. a SMOOTH TIME/τ RAMP. It starts at ~0 for every layer (so the trunk is
never perturbed by untrained wake-up branches at init -> no divergence), then
opens gradually on a schedule: shallow layers (small tau in the VSA ladder)
open first, deeper layers later, unified with the model's τ geometry
(log-normalised so the ladder spans [0,1] instead of 5 decades of raw tau).

Why a time ramp and not expert-saturation (readiness)? At scale the base model
does NOT learn LM on its own (ce ~ ln(vocab), random baseline), so a
readiness trigger (pred_err must DROP first) deadlocks: gate closed -> no
learning -> pred_err never drops -> gate stays closed forever -> bridge never
engages. The time ramp breaks that by opening the wake-up branches on a fixed
schedule, giving the bridge a chance to supply the learning signal while
staying smooth enough to avoid the original divergence.

The FROZEN base MLP gate (sigmoid(mod_scale_mlp) ~ 0.667) stays OPEN regardless
(no deadlock). `readiness` is still tracked (see update) and exposed for
diagnostics, but it no longer blocks the gate.
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
        self.T0 = float(getattr(cfg, "matur_T0", 20000.0))
        self.T_delay = float(getattr(cfg, "matur_T_delay", 20000.0))
        self.delta_t = float(getattr(cfg, "matur_delta", 6000.0))
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
        """Gate for THIS step: a smooth TIME/τ ramp (no learning-dependence).

        gate_l(t) = sigmoid((t - (T0 + alpha*tau_norm_l*T_delay)) / delta_t)

        Starts at ~0 for every layer (so the trunk is never perturbed by untrained
        wake-up branches at init -> no divergence), then opens SMOOTHLY on a schedule:
        shallow layers (small tau in the VSA ladder) open first, deeper layers later,
        unified with the model's τ geometry. This replaces the readiness-gated design,
        which deadlocked at scale: readiness required pred_err to DROP, but the base
        model does not learn LM at this scale, so pred_err never dropped and the gate
        stayed ~0 forever (bridge never engaged -> no learning -> ...).

        `readiness` is still tracked (see update) and exposed for diagnostics, but it
        no longer blocks the gate.
        """
        if tau_dev is not None:
            self._update_tau_norm(tau_dev)
        t = float(step)
        # Top-down ramp: глубокие (глобальные) слои (tau_norm->1) открываются
        # первыми — модель сначала опирается на грубую глобальную структуру, затем
        # подключает локальную детализацию мелких слоёв. Инверсия исходного
        # bottom-up порядка (где открывались сначала мелкие).
        gate = torch.sigmoid(
            (t - (self.T0 + self.alpha * (1.0 - self.tau_norm) * self.T_delay)) / self.delta_t)
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
