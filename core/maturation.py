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

INTELLIGENT WARMUP (after resume):
  When resuming from a checkpoint, the time ramp is frozen for
  `matur_warmup_steps` (default 2000). During this period, only
  bridge_readiness gates the maturation. This prevents "unfreezing shock"
  where deep layers open too fast before the bridge has adapted to the
  new learning rate. After warmup, the time ramp gradually takes over
  with a smooth blend: gate = (1-α)*readiness + α*time_ramp, where α
  ramps from 0 to 1 over warmup_steps.
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
        self.T0 = float(getattr(cfg, "matur_T0", 8000.0))
        self.T_delay = float(getattr(cfg, "matur_T_delay", 8000.0))
        self.delta_t = float(getattr(cfg, "matur_delta", 4000.0))
        self.r0 = float(getattr(cfg, "matur_r0", 0.3))
        self.rs = float(getattr(cfg, "matur_rs", 0.2))
        self.ema = float(getattr(cfg, "matur_ema", 0.999))
        self.warm = int(getattr(cfg, "matur_warm", 300))

        # Intelligent warmup after resume
        self.warmup_steps = int(getattr(cfg, "matur_warmup_steps", 2000))
        self.register_buffer("_resume_step", torch.tensor(0, dtype=torch.long))
        self._is_resumed = False

        self._update_tau_norm(torch.zeros(self.n_layers))  # dev = 0 initially

    def set_resume_step(self, step):
        """Call this when resuming from a checkpoint to activate warmup."""
        self._resume_step.fill_(int(step))
        self._is_resumed = True

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

    def step_gate(self, step, tau_dev=None, bridge_readiness=None):
        """Gate for THIS step: max(TIME/τ ramp, bridge-readiness).

        INTELLIGENT WARMUP: After resume, the time ramp is frozen for
        `warmup_steps`. Only bridge_readiness is used during this period.
        After warmup, the time ramp gradually blends in:
          α = (step - resume_step - warmup_steps) / warmup_steps  (clamped to [0,1])
          gate = max((1-α)*readiness + α*time_ramp, readiness)

        This prevents the "unfreezing shock" where deep layers open too fast
        before the bridge has adapted to the new learning rate.
        """
        if tau_dev is not None:
            self._update_tau_norm(tau_dev)
        t = float(step)

        # Compute time ramp (always, for diagnostics)
        time_ramp = torch.sigmoid(
            (t - (self.T0 + self.alpha * (1.0 - self.tau_norm) * self.T_delay)) / self.delta_t)

        # Intelligent warmup: blend readiness and time_ramp after resume
        if self._is_resumed and self.warmup_steps > 0:
            steps_since_resume = t - self._resume_step.item()
            if steps_since_resume < 0:
                # Before resume step (shouldn't happen, but safety)
                alpha = 0.0
            elif steps_since_resume < self.warmup_steps:
                # During warmup: only readiness (time ramp frozen)
                alpha = 0.0
            else:
                # After warmup: gradually blend in time ramp over another warmup_steps
                blend_steps = steps_since_resume - self.warmup_steps
                alpha = min(1.0, blend_steps / self.warmup_steps)

            if alpha < 1.0:
                # Blend: (1-α)*readiness + α*time_ramp, then max with readiness
                if bridge_readiness is not None:
                    br = bridge_readiness.to(time_ramp.device).reshape(1)
                    blended = (1.0 - alpha) * br.expand_as(time_ramp) + alpha * time_ramp
                    gate = torch.maximum(blended, br.expand_as(time_ramp))
                else:
                    gate = (1.0 - alpha) * time_ramp + alpha * time_ramp  # just time_ramp
            else:
                # Full time ramp
                gate = time_ramp
                if bridge_readiness is not None:
                    br = bridge_readiness.to(gate.device).reshape(1)
                    gate = torch.maximum(gate, br.expand_as(gate))
        else:
            # No resume or warmup_steps=0: original behavior
            gate = time_ramp
            if bridge_readiness is not None:
                br = bridge_readiness.to(gate.device).reshape(1)
                gate = torch.maximum(gate, br.expand_as(gate))

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
