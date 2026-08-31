"""WideBind: AdaptiveGate — sigmoid independence + softmax emphasis.

Киллер-фича: combines the best of both worlds without the zero-sum penalty.

  gate = sigmoid(logits) * (1 + softmax(logits / tau))

Почему это работает:
- sigmoid: каждая фича независима в [0,1] (нет zero-sum)
- softmax: относительный акцент среди активных фичей (winner emphasis)
- multiplicative: sigmoid управляет "участвует ли фича",
                  softmax управляет "насколько сильно относительно других"

Кейсы:
  sigmoid=0 → gate=0  (фича отключена, неважно что делает softmax)
  sigmoid=1 → gate=1+softmax  (фича включена, получает softmax boost)
  все sigmoid=1 → gate = 1 + normalized  (все активны, нет конкуренции)

tau (temperature) управляет sharpness softmax:
  tau→0: winner-take-all (одна фича доминирует)
  tau→∞: равномерное распределение (softmax → 1/n для всех)
  tau=1: default balance
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGate(nn.Module):
    """Sigmoid-Softmax hybrid gate.
    
    Логика:
      independent = sigmoid(logits)        # [0,1] per feature, no competition
      relative = softmax(logits / tau)     # sum=1, relative ranking
      gate = independent * (1 + relative)  # combined effect
      
    Градиенты:
      sigmoid path: прямой gradient для включения/выключения фичей
      softmax path: relative gradient для перераспределения веса
      multiplicative: strong features get stronger (без подавления слабых)
    """
    
    def __init__(self, n_features: int, tau_init: float = 1.0, learnable_tau: bool = True):
        super().__init__()
        self.n_features = n_features
        
        if learnable_tau:
            self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))
        else:
            self.register_buffer('log_tau', torch.tensor(math.log(tau_init)))
        
        # Diagnostics
        self.register_buffer('_last_indep_mean', torch.tensor(0.0))
        self.register_buffer('_last_relative_entropy', torch.tensor(0.0))
        self.register_buffer('_last_gate_std', torch.tensor(0.0))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (*, n_features) — raw feature scores
            
        Returns:
            gate: (*, n_features) — combined gate values
        """
        tau = torch.exp(self.log_tau).clamp(min=0.1, max=10.0)
        
        # 1. Independent activation (sigmoid)
        independent = torch.sigmoid(logits)  # [0,1] per feature
        
        # 2. Relative emphasis (softmax with temperature)
        relative = F.softmax(logits / tau, dim=-1)  # sum=1
        
        # 3. Combined: sigmoid gates participation, softmax adds relative boost
        gate = independent * (1.0 + relative)
        
        # 4. Update diagnostics
        with torch.no_grad():
            self._last_indep_mean.copy_(independent.mean().detach())
            self._last_gate_std.copy_(gate.std().detach())
            # Entropy of relative distribution (high = uniform, low = peaked)
            self._last_relative_entropy.copy_(
                -(relative * (relative + 1e-8).log()).sum(dim=-1).mean().detach()
            )
        
        return gate
    
    def get_diagnostics(self) -> dict:
        """Return diagnostic metrics for logging."""
        return {
            'indep_mean': self._last_indep_mean.item(),
            'relative_entropy': self._last_relative_entropy.item(),
            'gate_std': self._last_gate_std.item(),
            'tau': torch.exp(self.log_tau).item(),
        }


class AdaptiveGateBundle(nn.Module):
    """Multiple AdaptiveGates for different subsystems.
    
    Каждая подсистема (diagnostics, bridge routing, intent, etc.)
    получает свою AdaptiveGate с уникальными n_features.
    """
    
    def __init__(self):
        super().__init__()
        self.gates = nn.ModuleDict()
    
    def add(self, name: str, n_features: int, tau_init: float = 1.0):
        """Register a new gate."""
        self.gates[name] = AdaptiveGate(n_features, tau_init)
    
    def forward(self, name: str, logits: torch.Tensor) -> torch.Tensor:
        """Forward through named gate."""
        return self.gates[name](logits)
    
    def get_all_diagnostics(self) -> dict:
        """Return diagnostics from all gates."""
        diag = {}
        for name, gate in self.gates.items():
            for k, v in gate.get_diagnostics().items():
                diag[f'ag/{name}/{k}'] = v
        return diag
