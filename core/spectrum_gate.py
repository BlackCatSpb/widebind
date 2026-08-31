"""WideBind: SpectrumGate — sigmoid-softmax continuum.

Три режима на одном спектре, управляемом tau:

  tau→∞  : pure sigmoid  → maximum diversity (все фичи независимы)
  tau=1  : balanced      → diversity with precision (сигмоида + softmax boost)
  tau→0  : pure softmax  → final precision (winner-take-all, один выбор)

Формула:
  independent = sigmoid(logits)           # [0,1] per feature, no competition
  relative = softmax(logits / tau)        # sum=1, relative ranking
  gate = independent * (1 + relative)     # combined

Ключевое свойство:
  - Base (independent) ВСЕГДА сохраняется
  - Softmax добавляет EMPHASIS, не подавляя остальные
  - tau = "лампа переключения" между diversity и precision

Применение в WideBind:
  - Diagnostics routing: tau=∞ для всех diagnostics (diversity)
  - Bridge routing: tau≈1 для bridge contribution (balanced)
  - Intent selection: tau≈0 для top-1 intent (final precision)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectrumGate(nn.Module):
    """Sigmoid-Softmax spectrum gate.
    
    Modes:
      diversity:          tau → ∞, gate ≈ sigmoid(logits) * 1.0
      diversity+precision: tau ≈ 1, gate = sigmoid * (1 + softmax)
      final_precision:   tau → 0, gate ≈ sigmoid * (1 + one-hot)
    
    Learnable tau allows the model to transition between modes during training.
    """
    
    def __init__(
        self,
        n_features: int,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
        tau_max: float = 10.0,
        learnable_tau: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.tau_min = tau_min
        self.tau_max = tau_max
        
        if learnable_tau:
            self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))
        else:
            self.register_buffer('log_tau', torch.tensor(math.log(tau_init)))
        
        # Diagnostics
        self.register_buffer('_last_indep_mean', torch.tensor(0.0))
        self.register_buffer('_last_relative_entropy', torch.tensor(0.0))
        self.register_buffer('_last_gate_std', torch.tensor(0.0))
        self.register_buffer('_last_dominance', torch.tensor(0.0))
    
    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau).clamp(self.tau_min, self.tau_max)
    
    @property
    def mode(self) -> str:
        """Current effective mode based on tau."""
        t = self.tau.item()
        if t > 5.0:
            return 'diversity'
        elif t > 0.5:
            return 'diversity+precision'
        else:
            return 'final_precision'
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (*, n_features) — raw feature scores
            
        Returns:
            gate: (*, n_features) — combined gate values
        """
        tau = self.tau
        
        # 1. Independent activation (sigmoid) — ALWAYS active
        independent = torch.sigmoid(logits)  # [0,1] per feature
        
        # 2. Relative emphasis (softmax with temperature)
        relative = F.softmax(logits / tau, dim=-1)  # sum=1
        
        # 3. Combined: sigmoid preserves base, softmax adds emphasis
        gate = independent * (1.0 + relative)
        
        # 4. Update diagnostics
        with torch.no_grad():
            self._last_indep_mean.copy_(independent.mean().detach())
            self._last_gate_std.copy_(gate.std().detach())
            
            # Entropy of relative distribution
            self._last_relative_entropy.copy_(
                -(relative * (relative + 1e-8).log()).sum(dim=-1).mean().detach()
            )
            
            # Dominance: how much does max feature dominate?
            max_vals = relative.max(dim=-1).values
            mean_vals = relative.mean(dim=-1)
            self._last_dominance.copy_((max_vals / (mean_vals + 1e-8)).mean().detach())
        
        return gate
    
    def set_tau(self, tau: float):
        """Manually set tau."""
        with torch.no_grad():
            self.log_tau.fill_(math.log(max(tau, self.tau_min)))
    
    def get_diagnostics(self) -> dict:
        """Return diagnostic metrics."""
        return {
            'tau': self.tau.item(),
            'mode': self.mode,
            'indep_mean': self._last_indep_mean.item(),
            'relative_entropy': self._last_relative_entropy.item(),
            'gate_std': self._last_gate_std.item(),
            'dominance': self._last_dominance.item(),
        }


class SpectrumGateBundle(nn.Module):
    """Multiple SpectrumGates for different WideBind subsystems.
    
    Пример использования:
        bundle = SpectrumGateBundle()
        bundle.add('diagnostics', n_features=6, tau_init=5.0)   # diversity
        bundle.add('bridge', n_features=4, tau_init=1.0)        # balanced
        bundle.add('intent', n_features=8, tau_init=0.3)        # precision
        
        gate_diag = bundle('diagnostics', diag_logits)
        gate_bridge = bundle('bridge', bridge_logits)
        gate_intent = bundle('intent', intent_logits)
    """
    
    def __init__(self):
        super().__init__()
        self.gates = nn.ModuleDict()
    
    def add(
        self,
        name: str,
        n_features: int,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
        tau_max: float = 10.0,
        learnable_tau: bool = True,
    ):
        """Register a new gate."""
        self.gates[name] = SpectrumGate(
            n_features, tau_init, tau_min, tau_max, learnable_tau
        )
    
    def forward(self, name: str, logits: torch.Tensor) -> torch.Tensor:
        """Forward through named gate."""
        return self.gates[name](logits)
    
    def get_all_diagnostics(self) -> dict:
        """Return diagnostics from all gates."""
        diag = {}
        for name, gate in self.gates.items():
            for k, v in gate.get_diagnostics().items():
                diag[f'sg/{name}/{k}'] = v
        return diag
    
    def get_modes(self) -> dict:
        """Return current mode for each gate."""
        return {name: gate.mode for name, gate in self.gates.items()}
