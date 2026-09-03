"""TauConfig — единое τ-поле EVA.

Все τ-зависимые величины архитектуры выводятся из одного класса:
- VSA tau ladder (пер-слойный временной масштаб)
- Maturation delay (когда слой просыпается)
- Gate temperatures (diversity vs precision)
- Intent alpha ladder (свежий vs перенесённый контекст)
- Memory bank temperatures (L1=быстрый, L3=медленный)
- LLRD (learning rate по глубине)

Формула:
  tau_l = tau_min * (tau_max / tau_min) ^ (phi(l) * (1 + 0.1 * dev_l))
  phi(l) = l / (n_layers - 1)  — монотонная функция глубины
  dev_l = tanh(_tau_dev[l])     — обучаемое отклонение, двустороннее [-1, 1]

Из tau_l выводятся ВСЕ остальные τ-зависимые величины:
  mat_delay_l = T0 + (1 - tau_norm_l) * T_delay
  gate_tau_l  = tau_max_gate * (tau_min_gate / tau_max_gate) ^ mat_gate_l
  alpha_l     = clamp(1 - c_ema / tau_l, 0, 1)
  mem_tau_l   = tau_l / tau_ref (для memory bank temperatures)
  lr_mult_l   = (tau_l / tau_ref) ^ (-gamma) (для LLRD)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TauConfig(nn.Module):
    """Единое τ-поле: один набор параметров → все τ-зависимые величины.

    Parameters:
        n_layers: количество слоёв
        tau_min: минимальный τ (самый быстрый слой, shallow)
        tau_max: максимальный τ (самый медленный слой, deep)
        tau_init_spread: начальный разброс τ по слоям ( fraction of log range)
        dev_max: максимальное отклонение dev (clip range)
        T0: базовая задержка созревания (steps)
        T_delay: дополнительная задержка для shallow слоёв (steps)
        delta_t: ширина рампы созревания (steps)
        gate_tau_min: минимальная температура гейтов
        gate_tau_max: максимальная температура гейтов
        intent_c_ema: целевая EMA скорость intent
        mem_tau_ref: референсный τ для memory bank temperatures
        llrd_gamma: показатель степени для LLRD (∝ τ^{-gamma})
    """

    def __init__(
        self,
        n_layers: int = 24,
        tau_min: float = 8.0,
        tau_max: float = 512.0,
        tau_init_spread: float = 0.5,
        dev_max: float = 0.3,
        T0: float = 8000.0,
        T_delay: float = 8000.0,
        delta_t: float = 4000.0,
        gate_tau_min: float = 0.3,
        gate_tau_max: float = 5.0,
        intent_c_ema: float = 1.0,
        mem_tau_ref: float = 64.0,
        llrd_gamma: float = 0.3,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.dev_max = dev_max
        self.T0 = T0
        self.T_delay = T_delay
        self.delta_t = delta_t
        self.gate_tau_min = gate_tau_min
        self.gate_tau_max = gate_tau_max
        self.intent_c_ema = intent_c_ema
        self.mem_tau_ref = mem_tau_ref
        self.llrd_gamma = llrd_gamma

        # ─── Learnable parameters ───
        # _tau_dev: per-layer deviation from base tau ladder
        # Initialized to small spread around 0
        init_devs = torch.linspace(-tau_init_spread, tau_init_spread, n_layers)
        self._tau_dev = nn.Parameter(init_devs)

        # ─── Buffers (non-learnable, for diagnostics) ───
        self.register_buffer('_tau_l_cache', torch.zeros(n_layers))
        self.register_buffer('_tau_norm_cache', torch.zeros(n_layers))
        self.register_buffer('_mat_delay_cache', torch.zeros(n_layers))
        self.register_buffer('_gate_tau_cache', torch.zeros(n_layers))
        self.register_buffer('_alpha_cache', torch.zeros(n_layers))
        self.register_buffer('_lr_mult_cache', torch.zeros(n_layers))
        self.register_buffer('_mem_tau_cache', torch.zeros(3))  # L1, L2, L3

        # Live tensors for gradient flow (set in update())
        self._tau_l_live = None
        self._tau_norm_live = None
        self._mat_delay_live = None
        self._alpha_live = None
        self._lr_mult_live = None
        self._mem_tau_live = None

    def _compute_tau_ladder(self) -> torch.Tensor:
        """Compute per-layer tau values from dev parameters.

        Returns:
            tau_l: (n_layers,) — per-layer temporal scale
        """
        # Base ladder: tau_min * (tau_max/tau_min) ^ phi(l)
        # phi(l) = l / (n_layers - 1) ∈ [0, 1]
        lf = torch.arange(self.n_layers, device=self._tau_dev.device, dtype=self._tau_dev.dtype)
        lf = lf / max(self.n_layers - 1, 1)

        # Deviation: bilateral, clipped to [-dev_max, +dev_max]
        dev = torch.tanh(self._tau_dev) * self.dev_max

        # tau_l = tau_min * (tau_max/tau_min) ^ (lf * (1 + 0.1 * dev))
        # Using log-space for numerical stability
        log_tau_min = math.log(self.tau_min)
        log_tau_range = math.log(self.tau_max / self.tau_min)
        log_tau = log_tau_min + log_tau_range * lf * (1.0 + 0.1 * dev)
        tau_l = torch.exp(log_tau)

        return tau_l

    def _compute_tau_norm(self, tau_l: torch.Tensor) -> torch.Tensor:
        """Normalize tau to [0, 1] per layer.

        tau_norm = 0 for fastest (shallow) layer
        tau_norm = 1 for slowest (deep) layer
        """
        tau_range = self.tau_max - self.tau_min
        return ((tau_l - self.tau_min) / tau_range).clamp(0.0, 1.0)

    def _compute_mat_delay(self, tau_norm: torch.Tensor) -> torch.Tensor:
        """Per-layer maturation delay in steps.

        Deep layers (tau_norm~1): open at T0
        Shallow layers (tau_norm~0): open at T0 + T_delay
        """
        return self.T0 + (1.0 - tau_norm) * self.T_delay

    def _compute_gate_tau(self, mat_gate: torch.Tensor) -> torch.Tensor:
        """Per-layer gate temperature from maturation gate.

        Immature (mat_gate~0): gate_tau = gate_tau_max (diversity)
        Mature (mat_gate~1): gate_tau = gate_tau_min (precision)
        Uses log-space interpolation for symmetry.
        """
        log_min = math.log(self.gate_tau_min)
        log_max = math.log(self.gate_tau_max)
        log_tau = log_max + (log_min - log_max) * mat_gate
        return torch.exp(log_tau)

    def _compute_intent_alpha(self, tau_l: torch.Tensor) -> torch.Tensor:
        """Per-layer intent EMA alpha from tau.

        alpha = clamp(1 - c_ema / tau_l, 0, 1)
        Deep layers (high tau): alpha → 1 (mostly carried context)
        Shallow layers (low tau): alpha → 0 (mostly fresh probe)
        """
        return torch.clamp(1.0 - self.intent_c_ema / tau_l, min=0.0, max=1.0)

    def _compute_lr_mult(self, tau_l: torch.Tensor) -> torch.Tensor:
        """Per-layer LR multiplier from tau.

        lr_mult = (tau_l / tau_ref) ^ (-gamma)
        Deep layers (high tau): lower lr
        Shallow layers (low tau): higher lr
        """
        return (tau_l / self.mem_tau_ref) ** (-self.llrd_gamma)

    def _compute_mem_tau(self, tau_l: torch.Tensor) -> torch.Tensor:
        """Memory bank temperatures per level (L1, L2, L3).

        L1: fastest (tau_min end)
        L2: middle (tau_mid)
        L3: slowest (tau_max end)
        """
        # Use percentiles of the tau ladder
        sorted_tau, _ = torch.sort(tau_l)
        n = len(sorted_tau)
        l1_tau = sorted_tau[max(0, n // 6)]      # ~17th percentile
        l2_tau = sorted_tau[max(0, n // 2)]       # median
        l3_tau = sorted_tau[min(n - 1, 5 * n // 6)]  # ~83rd percentile
        return torch.stack([l1_tau, l2_tau, l3_tau])

    @property
    def tau_l(self) -> torch.Tensor:
        """Per-layer temporal scale (live during training for gradient flow)."""
        return getattr(self, '_tau_l_live', self._tau_l_cache)

    @property
    def tau_norm(self) -> torch.Tensor:
        """Normalized tau [0,1] per layer (live during training for gradient flow)."""
        return getattr(self, '_tau_norm_live', self._tau_norm_cache)

    @property
    def mat_delay(self) -> torch.Tensor:
        """Per-layer maturation delay in steps (cached, non-differentiable)."""
        return self._mat_delay_cache

    @property
    def gate_tau(self) -> torch.Tensor:
        """Per-layer gate temperature (cached, non-differentiable)."""
        return self._gate_tau_cache

    @property
    def intent_alpha(self) -> torch.Tensor:
        """Per-layer intent EMA alpha (live during training for gradient flow)."""
        return getattr(self, '_alpha_live', self._alpha_cache)

    @property
    def lr_mult(self) -> torch.Tensor:
        """Per-layer LR multiplier (cached, non-differentiable)."""
        return self._lr_mult_cache

    @property
    def mem_tau(self) -> torch.Tensor:
        """Memory bank temperatures [L1, L2, L3] (cached, non-differentiable)."""
        return self._mem_tau_cache

    def update(self, mat_gate: torch.Tensor | None = None):
        """Recompute all τ-derived values. Call once per forward.

        Args:
            mat_gate: (n_layers,) maturation gate values [0,1], or None for initial values
        """
        tau_l = self._compute_tau_ladder()
        tau_norm = self._compute_tau_norm(tau_l)

        # Store LIVE tensors for gradient flow (used in forward)
        self._tau_l_live = tau_l
        self._tau_norm_live = tau_norm
        self._mat_delay_live = self._compute_mat_delay(tau_norm)
        self._alpha_live = self._compute_intent_alpha(tau_l)
        self._lr_mult_live = self._compute_lr_mult(tau_l)
        self._mem_tau_live = self._compute_mem_tau(tau_l)

        # Cache DETACHED values for diagnostics
        self._tau_l_cache.copy_(tau_l.detach())
        self._tau_norm_cache.copy_(tau_norm.detach())
        self._mat_delay_cache.copy_(self._mat_delay_live.detach())
        self._alpha_cache.copy_(self._alpha_live.detach())
        self._lr_mult_cache.copy_(self._lr_mult_live.detach())
        self._mem_tau_cache.copy_(self._mem_tau_live.detach())

        # Gate tau depends on maturation (needs current gate values)
        if mat_gate is not None:
            self._gate_tau_cache.copy_(self._compute_gate_tau(mat_gate).detach())
        else:
            # Default: all immature → max diversity
            self._gate_tau_cache.fill_(self.gate_tau_max)

    def get_tau_for_layer(self, layer_idx: int) -> float:
        """Get tau for a specific layer (for inference)."""
        return self._tau_l_cache[layer_idx].item()

    def get_diagnostics(self) -> dict:
        """Return diagnostic metrics."""
        return {
            'tau_l_mean': self._tau_l_cache.mean().item(),
            'tau_l_std': self._tau_l_cache.std().item(),
            'tau_l_min': self._tau_l_cache.min().item(),
            'tau_l_max': self._tau_l_cache.max().item(),
            'tau_dev_mean': torch.tanh(self._tau_dev).mean().item(),
            'tau_dev_std': torch.tanh(self._tau_dev).std().item(),
            'tau_dev_max': torch.tanh(self._tau_dev).abs().max().item(),
            'gate_tau_mean': self._gate_tau_cache.mean().item(),
            'intent_alpha_mean': self._alpha_cache.mean().item(),
            'lr_mult_mean': self._lr_mult_cache.mean().item(),
        }
