"""WideBind: per-layer bridge gate (intelligent gradient routing to SemanticBridge).

Каждый слой имеет свою health MLP, которая агрегирует diagnostics в scalar gate.
Gate = sigmoid(health_mlp(diagnostics)) * tau — связан с maturation.
Нормализация: sigmoid + weighted average, NaN/explosion control.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerBridgeGate(nn.Module):
    """Per-layer intelligent gate to SemanticBridge.
    
    Каждый слой получает:
    - Per-layer health MLP (6 diagnostics → 1 scalar)
    - Gate = sigmoid(health) * tau (connected to maturation)
    - Weighted average normalization
    - NaN/explosion protection
    
    Diagnostic features (per layer):
    0. pred_error_norm: предсказание зеркала (низкая = хорошо)
    1. gate_l1: стабильность гейтов экспертов (низкая = стабильно)
    2. mirror_norm: активность зеркала (умеренная = хорошо)
    3. bridge_contribution: вклад в bridge (высокая = помогает)
    4. expert_entropy: разнообразие экспертов (умеренная = хорошо)
    5. diversity: разнообразие представлений (умеренная = хорошо)
    """
    
    def __init__(self, n_layers: int, health_features: int = 6):
        super().__init__()
        self.n_layers = n_layers
        self.health_features = health_features
        
        # Per-layer health MLPs: each layer has its own
        self.health_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(health_features, 16),
                nn.GELU(),
                nn.Linear(16, 8),
                nn.GELU(),
                nn.Linear(8, 1),
            ) for _ in range(n_layers)
        ])
        
        # Per-layer normalization: learnable temperature
        self.layer_temps = nn.Parameter(torch.ones(n_layers) * 2.0)
        
        # NaN/explosion control
        self._nan_count = 0
        self._max_nan = 10  # max NaN before fallback
        
    def compute_health(self, diagnostics: torch.Tensor) -> torch.Tensor:
        """Compute per-layer health scores from diagnostics.
        
        Args:
            diagnostics: (n_layers, health_features) — per-layer diagnostics
            
        Returns:
            health: (n_layers, 1) — health scores in [0, 1]
        """
        health_scores = []
        for l in range(self.n_layers):
            h = self.health_mlps[l](diagnostics[l])  # (1,)
            health_scores.append(h)
        
        health = torch.cat(health_scores, dim=0)  # (n_layers, 1)
        return torch.sigmoid(health)
    
    def forward(
        self, 
        layer_outputs: torch.Tensor,  # (n_layers, B, D)
        diagnostics: torch.Tensor,    # (n_layers, health_features)
        tau: torch.Tensor,            # (n_layers,) — maturation gate
    ):
        """Compute weighted bridge input from layer outputs.
        
        Args:
            layer_outputs: (n_layers, B, D) — per-layer hidden states
            diagnostics: (n_layers, health_features) — per-layer diagnostics
            tau: (n_layers,) — maturation gate values
            
        Returns:
            bridge_input: (B, D) — weighted sum of layer outputs
            gate_weights: (n_layers, 1) — per-layer gate weights (for logging)
            health_scores: (n_layers, 1) — raw health scores (for logging)
        """
        # 1. Compute health scores
        health_scores = self.compute_health(diagnostics)  # (n_layers, 1)
        
        # 2. Gate = health * tau (connected to maturation)
        tau_expanded = tau.unsqueeze(-1)  # (n_layers, 1)
        gates = health_scores * tau_expanded  # (n_layers, 1)
        
        # 3. Temperature scaling (learnable per-layer)
        temps = torch.sigmoid(self.layer_temps).unsqueeze(-1)  # (n_layers, 1)
        gates = gates * temps
        
        # 4. NaN/explosion control
        gates = torch.where(torch.isnan(gates), torch.zeros_like(gates), gates)
        gates = torch.clamp(gates, min=0.0, max=2.0)  # prevent explosion
        
        # 5. Weighted average normalization
        gate_sum = gates.sum(dim=0, keepdim=True)  # (1, 1)
        gate_sum = torch.clamp(gate_sum, min=1e-6)  # prevent division by zero
        normalized_gates = gates / gate_sum  # (n_layers, 1)
        
        # 6. Fallback: if all gates are zero, use equal weights
        all_zero = (gate_sum < 1e-6).all()
        if all_zero:
            normalized_gates = torch.ones_like(gates) / self.n_layers
            self._nan_count += 1
            if self._nan_count > self._max_nan:
                # Reset temperatures if too many NaNs
                with torch.no_grad():
                    self.layer_temps.fill_(2.0)
                self._nan_count = 0
        else:
            self._nan_count = 0
        
        # 7. Weighted sum of layer outputs
        weighted = normalized_gates.unsqueeze(-1) * layer_outputs  # (n_layers, B, D)
        bridge_input = weighted.sum(dim=0)  # (B, D)
        
        return bridge_input, normalized_gates, health_scores
    
    def get_diagnostics(
        self, 
        layers: nn.ModuleList,
        bridge_contribution: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract per-layer diagnostics from model state.
        
        Args:
            layers: list of WideBindBlock modules
            bridge_contribution: (n_layers,) — bridge contribution scores (optional)
            
        Returns:
            diagnostics: (n_layers, health_features) — per-layer diagnostics
        """
        n = self.n_layers
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        diagnostics = torch.zeros(n, self.health_features, device=device, dtype=dtype)
        
        for l in range(n):
            layer = layers[l]
            mir = layer.mirror
            
            # 0. pred_error_norm (низкая = хорошо)
            pe = getattr(mir, '_cached_pred_error_norm', None)
            if pe is not None:
                diagnostics[l, 0] = pe.detach().mean().clamp(0.0, 1.0)
            
            # 1. gate_l1 (низкая = стабильно)
            gl = getattr(mir, '_cached_gate_l1', None)
            if gl is not None:
                diagnostics[l, 1] = gl.detach().clamp(0.0, 1.0)
            
            # 2. mirror_norm (умеренная = хорошо)
            mp = getattr(mir, '_cached_pred_k', None)
            if mp is not None:
                # Normalize to [0, 1] range
                mn = mp.detach().norm()
                diagnostics[l, 2] = (mn / 1000.0).clamp(0.0, 1.0)
            
            # 3. bridge_contribution (высокая = помогает)
            if bridge_contribution is not None:
                diagnostics[l, 3] = bridge_contribution[l].clamp(0.0, 1.0)
            
            # 4. expert_entropy (умеренная = хорошо)
            hp = getattr(mir, '_cached_hp', None)
            if hp is not None:
                # Entropy of expert usage
                hp_det = hp.detach()
                hp_norm = torch.sigmoid(hp_det)
                hp_norm = hp_norm / hp_norm.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                entropy = -(hp_norm * hp_norm.clamp_min(1e-9).log()).sum()
                # Normalize to [0, 1]
                max_entropy = math.log(hp_det.shape[-1])
                diagnostics[l, 4] = (entropy / max_entropy).clamp(0.0, 1.0)
            
            # 5. diversity (умеренная = хорошо)
            # Use gate diversity as proxy
            gate_l1 = getattr(mir, '_cached_gate_l1', None)
            if gate_l1 is not None:
                # Low gate_l1 = high diversity
                diagnostics[l, 5] = (1.0 - gate_l1).clamp(0.0, 1.0)
        
        return diagnostics
