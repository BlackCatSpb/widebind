"""training_guard.py — anti-collapse training regime for WideBind.

Implements the "intelligent" stabilisation the user asked for:

  * Layer-wise LR decay (LLRD) — deeper blocks get a *smaller* LR, damping the
    residual-stream growth that blew up the logits around step ~1000-1200.
  * Progressive / readiness-driven layer unfreezing — only a subset of blocks is
    optimised at a time; deeper blocks come online as the meta-cognitive core
    MATURES (AdaptiveController `differentiation`), or on a fixed step schedule
    as a backstop. A block is "ready" when the meta-layer has stabilised.
  * grad_clip — bounds the per-step update, cutting gradient-runaway at the source.
  * Watchdog — if CE explodes, roll back to the last healthy checkpoint
    (best.pt), rebuild a FRESH Adam (no momentum), and cut the LR.

The forward pass always runs ALL blocks; frozen blocks just don't receive
gradients (they stay near-identity at init until unfrozen).
"""

import gc
import math
import os

import torch
import torch.nn as nn

from .stack import AdaptiveController


# ─────────────────────────────────────────────────────────────────────────────
# Progressive unfreezing
# ─────────────────────────────────────────────────────────────────────────────

def set_active_depth(model, k):
    """Freeze every block with index >= k (0-based); keep [0, k) trainable."""
    k = max(0, min(int(k), len(model.layers)))
    for i, layer in enumerate(model.layers):
        requires = (i < k)
        for p in layer.parameters():
            p.requires_grad_(requires)
    model._active_depth = k
    return k


def meta_maturity(model):
    """Meta-cognitive readiness proxy: global `differentiation`
    (variance of expert log_scale across mirror layers).

    Grows as the meta-core learns which dims to trust/suppress — i.e. as the
    meta-cognitive layer matures. Returns 0.0 on any failure.
    """
    try:
        _, diff = AdaptiveController.stats(
            model.layers,
            expl_thresh=getattr(model.cfg, 'exploration_threshold', 0.296),
            diff_thresh=getattr(model.cfg, 'differentiation_threshold', 0.087))
        return float(diff)
    except Exception:
        return 0.0


class ReadinessActivator:
    """Unfreezes blocks progressively.

    Block i (beyond ``init_active_layers``) activates when EITHER
      - meta maturity >= its threshold  (readiness mode, primary), OR
      - step >= its fixed unlock step    (schedule backstop)
    Once activated a block stays active (monotonic).
    """

    def __init__(self, model, init_active_layers=8, n_layers=None,
                 stage_steps=15000, unfreeze_inc=4, readiness_full=0.6,
                 mode='readiness', readiness_interval=200):
        self.model = model
        self.n = int(n_layers if n_layers is not None else len(model.layers))
        self.init_k = min(int(init_active_layers), self.n)
        self.stage_steps = int(stage_steps)
        self.unfreeze_inc = int(unfreeze_inc)
        self.readiness_full = float(readiness_full)
        self.mode = mode
        self.readiness_interval = int(readiness_interval)
        self.active = self.init_k
        self._last_step = -10 ** 9
        set_active_depth(model, self.active)

    def _thresholds(self):
        taus = []
        remaining = max(self.n - self.init_k, 1)
        for i in range(self.n):
            if i < self.init_k:
                taus.append(-1.0)
            else:
                frac = (i - self.init_k + 1) / remaining
                taus.append(frac * self.readiness_full)
        return taus

    def update(self, step):
        if self.mode == 'fixed':
            # Staircase unfreeze: unlock `unfreeze_inc` blocks every `stage_steps`.
            # Pure function of step (cheap), so we do NOT throttle it — the
            # depth target must stay correct even if called sparsely.
            stages = (step // self.stage_steps) if self.stage_steps > 0 else 0
            target = min(self.init_k + stages * self.unfreeze_inc, self.n)
            if target > self.active:
                self.active = target
                set_active_depth(self.model, self.active)
                print(f'  [Activator] step={step} (schedule) '
                      f'-> active_depth={self.active}/{self.n}')
            return self.active

        # readiness mode: throttle the (relatively expensive) meta_maturity call.
        if step - self._last_step < self.readiness_interval and step > 0:
            return self.active
        self._last_step = step
        taus = self._thresholds()
        maturity = meta_maturity(self.model)
        progressed = False
        for i in range(self.n):
            if i < self.active:
                continue
            if maturity >= taus[i]:
                self.active = i + 1
                progressed = True
        if progressed:
            set_active_depth(self.model, self.active)
            print(f'  [Activator] step={step} maturity={maturity:.3f} '
                  f'-> active_depth={self.active}/{self.n}')
        return self.active


# ─────────────────────────────────────────────────────────────────────────────
# Layer-wise LR decay (LLRD) optimizer
# ─────────────────────────────────────────────────────────────────────────────

def _layer_index_of(name):
    if name.startswith('layers.'):
        try:
            return int(name.split('.')[1])
        except (IndexError, ValueError):
            return -1
    return -1


def _role_lr_mult(name, lam):
    if '.b_d' in name or '.b_i' in name or '.scale_w' in name:
        return lam ** (-2)            # vsa scales
    if name.startswith('embed.') or name.startswith('lm_head.readout') \
            or name.startswith('lm_head.proj'):
        return lam ** (-2)            # embeddings / readout
    if any(g in name for g in ['.mirror.alpha_diag', '.log_skip_alpha',
                               '.mirror.W_proj', '.mirror.W_out', '.mirror.w_temp',
                               '.mirror.w_global', '.mirror.log_scale',
                               '.mirror.tanh_bias', '.log_dvar_mod_scale',
                               '.dvar_mod_bias', '.log_grad_mod_scale',
                               '.grad_mod_bias']):
        return lam ** (1)             # mirror projections / gates
    if '.mlp.' in name or '.bind.W_proj.weight' in name \
            or name.endswith('.W_out') or name.endswith('.W_proj'):
        return lam ** (-1)            # MLP cores / bind
    if 'reasoning_gate' in name:
        return lam ** (1)
    if any(g in name for g in ['.w_gate', '.b_gate', '.w_delta_gate', '.b_delta_gate',
                               '.w_i', '.w_d', '.w_q', '.w_q_leaf', '.w_q_ctx',
                               '.w_mem2v', '.w_k_mu', '.w_q_mu', '.w_mu_mem',
                               '.w_u', '.w_v']):
        return lam ** (1)             # gating / memory
    return 1.0


def build_optimizer(model, base_lr, llrd_decay=0.9, weight_decay=0.01,
                    betas=(0.9, 0.95), lam=None):
    """AdamW with LLRD: lr = base_lr * role_mult * (llrd_decay ** layer_idx).

    Includes ALL params (frozen blocks have requires_grad=False and are simply
    not updated until ``set_active_depth`` unfreezes them).
    """
    if lam is None:
        from .lambda_utils import lambda_d
        lam = lambda_d(model.cfg.lambda_d)
    groups = {}
    for name, p in model.named_parameters():
        li = _layer_index_of(name)
        role_mult = _role_lr_mult(name, lam)
        depth_mult = llrd_decay ** max(li, 0)
        lr = base_lr * role_mult * depth_mult
        wd = weight_decay if p.ndim >= 2 else 0.0
        key = (round(role_mult, 4), round(depth_mult, 6), round(wd, 6))
        g = groups.get(key)
        if g is None:
            g = {'params': [], 'lr': lr, 'weight_decay': wd}
            groups[key] = g
        g['params'].append(p)
    return torch.optim.AdamW([g for g in groups.values() if g['params']],
                             betas=betas)


# ─────────────────────────────────────────────────────────────────────────────
# Cosine warmup LR (rebuild-safe, preserves LLRD base lrs)
# ─────────────────────────────────────────────────────────────────────────────

class CosineWarmup:
    def __init__(self, optimizer, warmup, max_steps):
        self.optimizer = optimizer
        self.warmup = int(warmup)
        self.max_steps = int(max_steps)
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self._step = 0

    def set_step(self, n):
        self._step = int(n)

    def _mult(self, step):
        if step < self.warmup:
            return step / max(self.warmup, 1)
        prog = (step - self.warmup) / max(self.max_steps - self.warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    def step(self):
        self._step += 1
        m = self._mult(self._step)
        for pg, bl in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = bl * m

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]

    def report_val_loss(self, val):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, sd):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Watchdog: CE-explosion auto-recovery
# ─────────────────────────────────────────────────────────────────────────────

class Watchdog:
    """On ``ce > watchdog_ce``: reload best.pt, rebuild a FRESH Adam (LLRD) with
    ``base_lr *= recover_lr_mult``, keep depth progress, and signal the caller
    to rebind ``optimizer``/``scheduler`` and skip this step's update.
    """

    def __init__(self, model, make_optimizer_fn, best_path, base_lr,
                 watchdog_ce=15.0, recover_lr_mult=0.5, recover_max=20, cooldown=50):
        self.model = model
        self.make_optimizer_fn = make_optimizer_fn
        self.best_path = best_path
        self.base_lr = float(base_lr)
        self.watchdog_ce = float(watchdog_ce)
        self.recover_lr_mult = float(recover_lr_mult)
        self.recover_max = int(recover_max)
        self.cooldown = int(cooldown)
        self._cooldown = 0
        self.recover_count = 0
        self.optimizer = None

    def check(self, ce, step):
        if self._cooldown > 0:
            self._cooldown -= 1
            return False
        if ce is None or not (ce > self.watchdog_ce):
            return False
        if not os.path.exists(self.best_path):
            print(f'  [Watchdog] ce={ce:.2f} > {self.watchdog_ce} but no best.pt yet '
                  f'— skipping rollback')
            self._cooldown = self.cooldown
            return False
        print(f'  [Watchdog] CE EXPLOSION ce={ce:.2f} > {self.watchdog_ce} at step {step} '
              f'-> rollback to {self.best_path}')
        ckpt = torch.load(self.best_path, map_location='cpu')
        self.model.load_state_dict(ckpt['model'], strict=False)
        if getattr(self.model, '_active_depth', None) is not None:
            set_active_depth(self.model, self.model._active_depth)
        self.recover_count += 1
        new_lr = self.base_lr * (self.recover_lr_mult ** self.recover_count)
        self.optimizer = self.make_optimizer_fn(new_lr)
        self.base_lr = new_lr
        # Release the just-loaded checkpoint (CPU) and, crucially, let the OLD
        # optimizer's CUDA momentum buffers be collected BEFORE the next forward
        # allocates — otherwise we transiently hold two optimizers' worth of GPU
        # memory and OOM on the very next step after rollback.
        del ckpt
        gc.collect()
        torch.cuda.empty_cache()
        self._cooldown = self.cooldown
        if self.recover_count > self.recover_max:
            raise RuntimeError(
                f'Watchdog: {self.recover_count} recoveries exceeded max '
                f'{self.recover_max}; aborting')
        return True
