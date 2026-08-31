"""WideBind: per-layer bridge gate with SpectrumGate (sigmoid-softmax hybrid).

Каждый слой имеет SpectrumGate, который агрегирует diagnostics в gate value.
Gate = SpectrumGate(diagnostics) * tau_maturation — связь с maturation.
SpectrumGate = sigmoid(logits) * (1 + softmax(logits/tau)) — оба преимущества.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpectrumGate(nn.Module):
    """Sigmoid-Softmax hybrid gate.
    
    gate = sigmoid(logits) * (1 + softmax(logits / tau))
    
    - sigmoid: independent activation per feature (no zero-sum)
    - softmax: relative emphasis among features
    - tau controls the blend (high=diversity, low=precision)
    """
    
    def __init__(self, n_features: int, tau_init: float = 1.0):
        super().__init__()
        self.n_features = n_features
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        tau = torch.exp(self.log_tau).clamp(0.1, 10.0)
        independent = torch.sigmoid(logits)
        relative = F.softmax(logits / tau, dim=-1)
        return independent * (1.0 + relative)
    
    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau).clamp(0.1, 10.0)


class LayerBridgeGate(nn.Module):
    """Per-layer intelligent gate to SemanticBridge with SpectrumGate.
    
    Каждый слой:
    - Per-layer SpectrumGate (6 diagnostics → 6 gated values → scalar gate)
    - Gate tau TIED to maturation: immature=diversity, mature=precision
    - NaN/explosion protection
    
    Diagnostic features (per layer):
    0. pred_error_norm: предсказание зеркала (низкая = хорошо)
    1. gate_l1: стабильность гейтов экспертов (низкая = стабильно)
    2. mirror_norm: активность зеркала (умеренная = хорошо)
    3. bridge_contribution: вклад bridge (высокая = помогает)
    4. expert_entropy: разнообразие экспертов (умеренная = хорошо)
    5. diversity: разнообразие представлений (умеренная = хорошо)
    """
    
    def __init__(self, n_layers: int, health_features: int = 6):
        super().__init__()
        self.n_layers = n_layers
        self.health_features = health_features
        
        # Per-layer SpectrumGate: each layer decides its own sigmoid/softmax blend
        self.gates = nn.ModuleList([
            SpectrumGate(health_features, tau_init=1.0)
            for _ in range(n_layers)
        ])
        
        # NaN/explosion control
        self._nan_count = 0
        self._max_nan = 10
    
    def forward(
        self,
        layer_outputs: torch.Tensor,  # (n_layers, B, D)
        diagnostics: torch.Tensor,    # (n_layers, health_features)
        tau_maturation: torch.Tensor, # (n_layers,) — maturation gate values
    ):
        """Compute weighted bridge input from layer outputs.
        
        Args:
            layer_outputs: (n_layers, B, D) — per-layer hidden states
            diagnostics: (n_layers, health_features) — per-layer diagnostics
            tau_maturation: (n_layers,) — maturation gate values
            
        Returns:
            bridge_input: (B, D) — weighted sum of layer outputs
            gate_weights: (n_layers, 1) — per-layer gate weights (for logging)
            gate_info: dict — diagnostic info for logging
        """
        n_layers = self.n_layers
        
        # 1. Per-layer SpectrumGate: each layer processes its own diagnostics
        raw_gates = []
        gate_taus = []
        for l in range(n_layers):
            # SpectrumGate: (health_features,) → (health_features,)
            gated_features = self.gates[l](diagnostics[l])
            # Reduce to scalar: mean of gated features
            scalar_gate = gated_features.mean()
            raw_gates.append(scalar_gate)
            gate_taus.append(self.gates[l].tau.item())
        
        raw_gates = torch.stack(raw_gates)  # (n_layers,)
        
        # 2. TIE TO MATURATION: gate = raw_gate * maturation
        #    Immature layers (mat≈0) → gate≈0 (conservative)
        #    Mature layers (mat≈1) → gate=raw_gate (full participation)
        gates = raw_gates * tau_maturation  # (n_layers,)
        
        # 3. NaN/explosion control
        gates = torch.where(torch.isnan(gates), torch.zeros_like(gates), gates)
        gates = torch.clamp(gates, min=0.0, max=2.0)
        
        # 4. Weighted average normalization
        gate_sum = gates.sum()
        gate_sum = torch.clamp(gate_sum, min=1e-6)
        normalized_gates = gates / gate_sum  # (n_layers,)
        
        # 5. Fallback: if all gates are zero, use equal weights
        if gate_sum < 1e-6:
            normalized_gates = torch.ones_like(gates) / n_layers
            self._nan_count += 1
            if self._nan_count > self._max_nan:
                # Reset SpectrumGate tau if too many NaNs
                with torch.no_grad():
                    for g in self.gates:
                        g.log_tau.fill_(0.0)  # tau=1.0
                self._nan_count = 0
        else:
            self._nan_count = 0
        
        # 6. Weighted sum of layer outputs
        weighted = normalized_gates.unsqueeze(-1).unsqueeze(-1) * layer_outputs
        bridge_input = weighted.sum(dim=0)  # (B, D)
        
        # 7. Diagnostic info for logging
        gate_info = {
            'lbg_tau': gate_taus,
            'lbg_raw_mean': raw_gates.mean().item(),
            'lbg_mat_corr': torch.corrcoef(
                torch.stack([raw_gates.detach(), tau_maturation.detach()])
            )[0, 1].item() if n_layers > 1 else 0.0,
        }
        
        return bridge_input, normalized_gates.unsqueeze(-1), gate_info
    
    def get_diagnostics(
        self,
        layers: nn.ModuleList,
        bridge_contribution: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract per-layer diagnostics from model state."""
        n = self.n_layers
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        diagnostics = torch.zeros(n, self.health_features, device=device, dtype=dtype)
        
        for l in range(n):
            layer = layers[l]
            mir = layer.mirror
            
            pe = getattr(mir, '_cached_pred_error_norm', None)
            if pe is not None:
                diagnostics[l, 0] = pe.detach().mean().clamp(0.0, 1.0)
            
            gl = getattr(mir, '_cached_gate_l1', None)
            if gl is not None:
                diagnostics[l, 1] = gl.detach().clamp(0.0, 1.0)
            
            mp = getattr(mir, '_cached_pred_k', None)
            if mp is not None:
                mn = mp.detach().norm()
                diagnostics[l, 2] = (mn / 1000.0).clamp(0.0, 1.0)
            
            if bridge_contribution is not None:
                diagnostics[l, 3] = bridge_contribution[l].clamp(0.0, 1.0)
            
            hp = getattr(mir, '_cached_hp', None)
            if hp is not None:
                hp_det = hp.detach()
                hp_norm = torch.sigmoid(hp_det)
                hp_norm = hp_norm / hp_norm.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                entropy = -(hp_norm * hp_norm.clamp_min(1e-9).log()).sum()
                max_entropy = math.log(hp_det.shape[-1])
                diagnostics[l, 4] = (entropy / max_entropy).clamp(0.0, 1.0)
            
            gate_l1 = getattr(mir, '_cached_gate_l1', None)
            if gate_l1 is not None:
                diagnostics[l, 5] = (1.0 - gate_l1).clamp(0.0, 1.0)
        
        return diagnostics
