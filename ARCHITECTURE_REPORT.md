# EVA / WideBind — Complete Architectural Analysis Report

## Table of Contents
1. [Project Overview](#overview)
2. [Architecture Overview](#architecture-overview)
3. [The Unified τ-System (TauConfig)](#tau-system)
4. [File-by-File Module Map](#module-map)
5. [Forward Pass Flow](#forward-flow)
6. [Training Flow](#training-flow)
7. [Gradient Flow Analysis](#gradient-flow)
8. [Bottlenecks & Issues](#bottlenecks)

---

## 1. PROJECT OVERVIEW

EVA (WideBind) is a PyTorch neural network implementing Vector Symbolic Architecture (VSA) with cognitive mirror experts, adaptive memory, and explicit reasoning. The architecture uses:

- **Sparse block codes** for token embedding (not standard embeddings)
- **Bottleneck bind** (Fibonacci-twisted bilinear cross-mixing) in D→K→D space
- **32-group cognitive mirror** experts operating in independent K-space subspaces
- **Multi-scale VSA memory** (4 fixed τ scales: 8, 32, 128, 512)
- **DCT spectral gating** for frequency-domain modulation
- **Streaming memory bank** (hierarchical L1+L2+L3) with τ-prior initialization
- **Unified TauConfig** — single τ-field replaces scattered _vsa_log_param, _tau_l_dev, _tau_intent_dev
- **Maturation controller** (unified per-layer wake-up gate using tau_config.tau_norm)
- **Semantic bridge** (cross-layer self-supervised probe)
- **Intent bridge** (top-down expert modulation, alpha derived from tau_config)
- **Adaptive reasoning** (chain-of-thought with per-step gating)
- **Triad verification** (re-circulation on low confidence)

Default config: D=4096, n_layers=32, G=32 experts, k=32/16/8 (staircase), vocab=50000, seq_len=128.

---

## 2. ARCHITECTURE OVERVIEW

### 2.1 Component Hierarchy

```
WideBindStack (core/stack.py)
├── PartitionedEmbedding (core/embedding.py)
│   ├── sparse_block_codes → (V, K) binary
│   ├── embed_mix (K×K) orthogonal mixing
│   ├── basis (K, d) outer product embedding
│   └── RotaryEmbedding (RoPE)
├── TauConfig (core/tau_config.py) ← UNIFIED τ-FIELD
│   ├── _tau_dev (n_layers,) ← learnable bilateral deviation
│   ├── tau_l (n_layers,) ← per-layer temporal scale
│   ├── tau_norm (n_layers,) ← normalized [0,1]
│   ├── mat_delay (n_layers,) ← maturation delay in steps
│   ├── gate_tau (n_layers,) ← gate temperature from maturation
│   ├── intent_alpha (n_layers,) ← intent EMA alpha
│   ├── lr_mult (n_layers,) ← LLRD multiplier
│   └── mem_tau (3,) ← [L1, L2, L3] memory temperatures
├── _vsa_log_param (4,) ← block-level VSA memory scales (S=4)
├── StreamingMemoryBank (core/memory_bank.py) [optional]
│   ├── L1Buffer (rolling, fast)
│   ├── L2Bank (learned, medium)
│   └── L3Concepts (emergent, slow)
├── SemanticBridge (core/bridge.py) [optional]
├── LayerBridgeGate (core/layer_bridge_gate.py) [optional]
├── MaturationController (core/maturation.py)
├── WideBindBlock × n_layers (core/block.py)
│   ├── Pre-LN (RMS)
│   ├── Conv1d (depthwise, causal)
│   ├── BottleneckBind / TrajectorySpiralBind
│   ├── VSA Memory (S=4 scales, chunked prefix scan)
│   ├── GroupedCognitiveMirror (core/mirror.py)
│   │   ├── 32 experts × d=128 subspace
│   │   ├── BridgeGLU (optional)
│   │   ├── AdaptiveGate (hybrid_gate)
│   │   └── Private Memory (optional)
│   ├── CollectiveConceptLayer (optional)
│   ├── PrecisionGate + ExactSequenceMemory
│   ├── DCT Spectral Gate
│   └── GroupedMLP (SwiGLU, mirror-conditioned)
├── SigmoidCodedHead / CognitiveCodedHead (core/embedding.py)
├── ReasoningMemory + ReasoningGate (core/reasoning.py)
└── AdaptiveController + MirrorLRScheduler (core/stack.py)
```

### 2.2 Data Flow Summary

```
Token IDs → PartitionedEmbedding → (B,L,D)
  → [StreamingMemoryBank] → (B,L,D)
  → For each layer:
      → [Intent Bridge probe] → intent_i
      → [Semantic Bridge inject] → h += bridge_signal
      → WideBindBlock → (h, state)
      → [Maturation update]
  → Final LN → (B,L,D)
  → [Explicit Reasoning] → h_acc
  → [Triad re-circulation] → h
  → SigmoidCodedHead → (B,L,V) logits
```

---

## 3. THE UNIFIED τ-SYSTEM (TauConfig)

### 3.1 TauConfig Class (core/tau_config.py:32)

The unified τ-field is an `nn.Module` with one learnable parameter set that derives ALL τ-dependent quantities in the architecture.

**Constructor Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_layers` | 24 | Number of layers |
| `tau_min` | 8.0 | Minimum τ (fastest layer, shallow) |
| `tau_max` | 512.0 | Maximum τ (slowest layer, deep) |
| `tau_init_spread` | 0.5 | Initial spread of τ deviation (fraction of log range) |
| `dev_max` | 0.3 | Maximum bilateral deviation (clip range) |
| `T0` | 8000.0 | Base maturation delay (steps) |
| `T_delay` | 8000.0 | Additional delay for shallow layers (steps) |
| `delta_t` | 4000.0 | Maturation ramp width (steps) |
| `gate_tau_min` | 0.3 | Minimum gate temperature (precision) |
| `gate_tau_max` | 5.0 | Maximum gate temperature (diversity) |
| `intent_c_ema` | 1.0 | Target EMA speed for intent |
| `mem_tau_ref` | 64.0 | Reference τ for memory bank temperatures |
| `llrd_gamma` | 0.3 | LLRD exponent: lr_l ∝ (tau_l / tau_ref)^(-gamma) |

**Learnable Parameter:**

```python
self._tau_dev = nn.Parameter(torch.linspace(-tau_init_spread, tau_init_spread, n_layers))
# (n_layers,) — per-layer bilateral deviation from base τ-ladder
```

### 3.2 Core Formula (core/tau_config.py:104-125)

```python
def _compute_tau_ladder(self) -> torch.Tensor:
    lf = torch.arange(self.n_layers) / max(self.n_layers - 1, 1)  # φ(l) ∈ [0, 1]
    dev = torch.tanh(self._tau_dev) * self.dev_max  # bilateral, clipped to [-dev_max, +dev_max]
    log_tau = log_tau_min + log_tau_range * lf * (1.0 + 0.1 * dev)
    tau_l = torch.exp(log_tau)
    return tau_l
```

**Formula:**
```
tau_l = tau_min * (tau_max / tau_min) ^ (phi(l) * (1 + 0.1 * dev_l))
phi(l) = l / (n_layers - 1)  — monotonic depth function
dev_l = tanh(_tau_dev[l])     — learnable bilateral deviation, [-1, 1]
```

### 3.3 Derived Quantities

All derived from `tau_l` in `TauConfig.update()`:

| Derived | Formula | Range | Used By |
|---------|---------|-------|---------|
| `tau_norm` | `(tau_l - tau_min) / (tau_max - tau_min)` | [0, 1] | MaturationController, diagnostics |
| `mat_delay` | `T0 + (1 - tau_norm) * T_delay` | [T0, T0+T_delay] | MaturationController.step_gate |
| `gate_tau` | `exp(log_max + (log_min - log_max) * mat_gate)` | [gate_tau_min, gate_tau_max] | AdaptiveGate temperature |
| `intent_alpha` | `clamp(1 - c_ema / tau_l, 0, 1)` | [0, 1] | Intent bridge EMA blending |
| `lr_mult` | `(tau_l / mem_tau_ref) ^ (-llrd_gamma)` | varying | LLRD per-layer LR scaling |
| `mem_tau` | `[sorted_tau[n/6], sorted_tau[n/2], sorted_tau[5n/6]]` | [tau_min, tau_max] | Memory bank L1/L2/L3 temperatures |

### 3.4 Gradient Flow Through τ

The `_tau_dev` parameter is the ONLY source of gradient for τ-dependent quantities:

```
_tau_dev → tanh → dev → tau_l (live tensor) → tau_norm (live)
                                              → intent_alpha (live)
                                              → lr_mult (cached, non-differentiable)
                                              → mat_delay (cached, non-differentiable)
                                              → mem_tau (cached, non-differentiable)
```

**Live vs Cached:**
- `_tau_l_live`, `_tau_norm_live`, `_alpha_live`, `_lr_mult_live`, `_mem_tau_live`: live during forward for gradient flow
- `_tau_l_cache`, `_tau_norm_cache`, etc.: detached copies for diagnostics

### 3.5 How τ_l Feeds Into Subsystems

#### 3.5.1 Maturation (core/maturation.py:91-119)

```python
def step_gate(self, step, tau_dev=None):
    if self.tau_config is not None:
        self.tau_norm.copy_(self.tau_config.tau_norm)
    gate = torch.sigmoid(
        (step - (self.T0 + self.alpha * (1.0 - self.tau_norm) * self.T_delay)) / self.delta_t)
```

**Effect:** Deep layers (tau_norm≈1) open at T_eff = T0 ≈ 8000 steps. Shallow layers (tau_norm≈0) open at T_eff = T0 + T_delay ≈ 16000 steps.

#### 3.5.2 Gate Temperatures (core/tau_config.py:144-154)

```python
def _compute_gate_tau(self, mat_gate):
    log_tau = log_max + (log_min - log_max) * mat_gate
    return torch.exp(log_tau)
```

Immature layers (mat_gate≈0) → gate_tau = gate_tau_max (diversity). Mature layers (mat_gate≈1) → gate_tau = gate_tau_min (precision).

#### 3.5.3 Intent Alpha (core/tau_config.py:156-163)

```python
def _compute_intent_alpha(self, tau_l):
    return torch.clamp(1.0 - self.intent_c_ema / tau_l, min=0.0, max=1.0)
```

Deep layers (high τ): alpha → 1 (mostly carried context). Shallow layers (low τ): alpha → 0 (mostly fresh probe).

**Used in stack.py:383:**
```python
_alpha_i = self.tau_config.intent_alpha[i].detach()
intent_streams[i] = _alpha_i * _bus_carried[i] + (1.0 - _alpha_i) * fresh_i.detach()
```

#### 3.5.4 LLRD (core/tau_config.py:165-172)

```python
def _compute_lr_mult(self, tau_l):
    return (tau_l / self.mem_tau_ref) ** (-self.llrd_gamma)
```

Deep layers (high τ): lower LR. Shallow layers (low τ): higher LR.

#### 3.5.5 Memory Bank Temperatures (core/tau_config.py:174-187)

```python
def _compute_mem_tau(self, tau_l):
    sorted_tau, _ = torch.sort(tau_l)
    n = len(sorted_tau)
    l1_tau = sorted_tau[max(0, n // 6)]      # ~17th percentile (fastest)
    l2_tau = sorted_tau[max(0, n // 2)]       # median
    l3_tau = sorted_tau[min(n - 1, 5 * n // 6)]  # ~83rd percentile (slowest)
    return torch.stack([l1_tau, l2_tau, l3_tau])
```

L1 = fast (low τ → precision), L2 = medium, L3 = slow (high τ → diversity).

### 3.6 _vsa_log_param vs tau_config

| Parameter | Controls | Scope | Grad Flow |
|-----------|----------|-------|-----------|
| `tau_config._tau_dev` | Per-layer τ-ladder → maturation, gate temps, intent alpha, LLRD, memory temps | Layer-level (n_layers) | Live during forward |
| `_vsa_log_param` (4,) | Block-level VSA memory scales: `vsa_tau = exp(cumsum(softplus(_vsa_log_param))) + 1` | Block-level (S=4 scales) | Always trainable |

**Key distinction:** `_vsa_log_param` controls the 4 exponential decay rates within each block's VSA prefix scan (block.py:274-275). It is NOT replaced by tau_config. Both coexist:
- `tau_config._tau_dev` → inter-layer τ hierarchy (which layer opens when, how fast, etc.)
- `_vsa_log_param` → intra-block VSA memory time constants (how quickly memory decays within a single block's prefix scan)

### 3.7 τ-Dependent Subsystems Status

| Subsystem | τ-Source | Frozen vs Trainable |
|-----------|----------|---------------------|
| Maturation timing | tau_config.mat_delay | Non-differentiable (buffer) |
| Maturation gate | tau_config.tau_norm | Live (gradient through _tau_dev) |
| Gate temperatures | tau_config.gate_tau | Non-differentiable (buffer) |
| Intent alpha | tau_config.intent_alpha | Live (gradient through _tau_dev) |
| Intent EMA blending | intent_alpha[i] | Detached in stack.py (line 383) |
| LLRD multipliers | tau_config.lr_mult | Non-differentiable (buffer, affects optimizer) |
| Memory bank τ-priors | tau_config.mem_tau | Non-differentiable (used for init only) |
| Block-level VSA scales | _vsa_log_param | Always trainable |
| AdaptiveGate.log_tau | Per-gate learnable | Always trainable |
| SpectrumGate.log_tau | Per-gate learnable | Always trainable |
| LayerBridgeGate.gates.log_tau | Per-layer, per-feature | Always trainable |
| Memory bank log_tau | Per-level (L1/L2/L3) | Always trainable |

---

## 4. FILE-BY-FILE MODULE MAP

### core/config.py (409 lines)

**Class: `WideBindConfig`** (dataclass, line 12)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__post_init__` | 380 | Applies λ_d hierarchy if enabled | None | None | `_apply_lambda_d` | All construction | No |
| `_apply_lambda_d` | 384 | Overrides ~20 hyperparameters from generalized golden ratio λ_d | None | None | `LambdaConfig` | `__post_init__` | No |

**Key config values:**
- D=4096, n_layers=32, bind_K=64, vocab=50000, seq_len=128
- mlp_groups=32 (→ d=128 per expert), mlp_expand=4
- mirror_k=32 (staircase: 8/16/32 by depth thirds)
- reasoning_max_steps=8, reasoning_adaptive=True
- bridge_dim=256, bridge_conn=0.1
- maturation_enabled=True, matur_T0=8000, matur_T_delay=8000
- private_mem=True, memory_bank=False (default off)
- bridge_glu=True, bridge_glu_beta=0.25
- intent_bridge=True
- **tau_enabled=True, tau_min=8.0, tau_max=512.0, tau_dev_max=0.3, tau_llrd_gamma=0.3, tau_mem_ref=64.0**

---

### core/tau_config.py (273 lines) — **NEW: Unified τ-field**

**Class: `TauConfig`** (nn.Module, line 32)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 51 | Initialize τ-field with learnable _tau_dev | n_layers, tau_min, tau_max, ... | None | nn.Parameter | WideBindStack | Yes (_tau_dev) |
| `_compute_tau_ladder` | 104 | Compute per-layer τ from dev | None | tau_l (n_layers,) | torch.tanh, torch.exp | update | Yes (grad through _tau_dev) |
| `_compute_tau_norm` | 127 | Normalize τ to [0,1] | tau_l | tau_norm (n_layers,) | — | update | No |
| `_compute_mat_delay` | 136 | Per-layer maturation delay | tau_norm | mat_delay (n_layers,) | — | update | No |
| `_compute_gate_tau` | 144 | Gate temperature from maturation | mat_gate | gate_tau (n_layers,) | torch.exp | update | No |
| `_compute_intent_alpha` | 156 | Intent EMA alpha from τ | tau_l | alpha (n_layers,) | torch.clamp | update | Yes (grad) |
| `_compute_lr_mult` | 165 | LLRD multiplier from τ | tau_l | lr_mult (n_layers,) | — | update | No |
| `_compute_mem_tau` | 174 | Memory bank temperatures | tau_l | mem_tau (3,) | torch.sort | update | No |
| `update` | 224 | Recompute all τ-derived values | mat_gate | None | All _compute_* | WideBindStack.forward | Yes (live tensors) |
| `get_diagnostics` | 260 | Return diagnostic metrics | None | dict | — | logging | No |

---

### core/adaptive_gate.py (130 lines)

**Function: `hybrid_gate`** (line 26) — **Single source of truth for sigmoid-softmax hybrid**

```python
gate = sigmoid(logits) * (1 + softmax(logits / tau))
```

| Variant | Usage |
|---------|-------|
| `hybrid_gate(logits, tau)` | Raw output (AdaptiveGate, SpectrumGate) |
| `hybrid_gate(logits, tau, log=True)` | Log space (SigmoidCodedHead._su) |
| `hybrid_gate(scores, tau, normalize=True)` | With L1 normalization (memory attention) |

**Class: `AdaptiveGate`** (nn.Module, line 75)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 89 | Sigmoid-softmax hybrid gate | n_features, tau_init, learnable_tau | None | nn.Parameter | GroupedCognitiveMirror | Yes (log_tau) |
| `forward` | 103 | gate = sigmoid * (1 + softmax/tau), **accepts tau_prior** | logits, tau_prior | gate | hybrid_gate | mirror.hybrid_gate | Yes |
| `get_diagnostics` | 124 | Return gate metrics | None | dict | — | logging | No |

**Key change:** `forward()` accepts `tau_prior` parameter (line 103). When provided:
```python
tau = tau_prior.clamp(min=0.1, max=10.0) * torch.exp(self.log_tau).clamp(min=0.1, max=10.0)
```
This allows tau_config to provide a baseline, with learnable deviation on top.

---

### core/spectrum_gate.py (113 lines)

**Class: `SpectrumGate`** (nn.Module, line 27)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 38 | Sigmoid-softmax spectrum gate | n_features, tau_init, tau_min, tau_max, learnable_tau | None | — | WideBindStack | Yes (log_tau) |
| `forward` | 76 | **tau = tau_external * exp(log_tau)** | logits, tau_external | gate | hybrid_gate | LayerBridgeGate | Yes |
| `set_tau` | 101 | Manually set tau | tau | None | — | — | No |
| `get_diagnostics` | 105 | Return diagnostic metrics | None | dict | — | logging | No |

**Key change (P0 FIX):** Line 80:
```python
tau = tau_external.clamp(self.tau_min, self.tau_max) * torch.exp(self.log_tau).clamp(self.tau_min, self.tau_max)
```
Was: `tau = tau_external` (ignoring log_tau). Now: external tau * exp(log_tau) — allows learnable deviation from maturation-driven baseline.

---

### core/layer_bridge_gate.py (247 lines)

**Class: `SpectrumGate`** (line 22) — local variant with external tau support

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 36 | Sigmoid-softmax gate | n_features, tau_init | None | — | LayerBridgeGate.gates | Yes (log_tau) |
| `forward` | 41 | **tau = tau_external * exp(log_tau)** | logits, tau_external | gate | hybrid_gate | LayerBridgeGate.forward | Yes |

**Key change (P0 FIX):** Line 45:
```python
tau = tau_external.clamp(0.1, 10.0) * torch.exp(self.log_tau).clamp(0.1, 10.0)
```
Same pattern as spectrum_gate.py — learnable deviation from external tau.

**Class: `LayerBridgeGate`** (nn.Module, line 56)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 74 | Per-layer SpectrumGate for bridge routing | n_layers, health_features, tau_min, tau_max | None | ModuleList(SpectrumGate) | WideBindStack | Yes (gates) |
| `_effective_tau` | 92 | Compute effective tau from maturation | maturation | tensor | — | forward | No |
| `forward` | 103 | Weighted bridge input from layer diagnostics + maturation | layer_outputs, diagnostics, tau_maturation, global_ready | (bridge_input, gate_weights, gate_info) | SpectrumGate | WideBindStack.forward | Yes |
| `get_diagnostics` | 202 | Extract per-layer diagnostics from model state | layers, bridge_contribution | (n_layers, health_features) | — | forward | No |

---

### core/memory_bank.py (640 lines)

**Function: `_memory_attention`** (line 53)

```python
scores = (q @ k.T) / math.sqrt(bridge_dim) * temp
attn = hybrid_gate(scores, temp)  # or F.softmax if not softmax_free
```

**Class: `L1Buffer`** (nn.Module, line 87)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 95 | Rolling buffer of K sentence embeddings | D, bridge_dim, n_slots, softmax_free, tau_prior | None | nn.Linear | StreamingMemoryBank.l1 | Yes (proj, q_proj, out_proj, **log_tau**) |
| `write` | 120 | Write sentence embedding (ring buffer) | embedding: (D,) | None | torch.no_grad | StreamingMemoryBank.forward | No |
| `read` | 138 | Read via hybrid attention | query: (B,L,D) | (B,L,D) | _memory_attention | StreamingMemoryBank.forward | Yes |
| `get_stats` | 157 | Return write statistics | None | dict | — | diagnostics | No |
| `reset` | 163 | Clear buffer | None | None | — | StreamingMemoryBank.reset | No |

**Key change:** Line 112: `self.log_tau = nn.Parameter(torch.tensor(math.log(max(tau_prior, 0.1))))` — initialized from τ-prior instead of frozen=1.0.

**Class: `L2Bank`** (nn.Module, line 170)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 181 | Learned memory bank with N slots | D, bridge_dim, n_slots, softmax_free, tau_prior | None | nn.Linear | StreamingMemoryBank.l2 | Yes (W_k, W_v, W_o, q_proj, val_norm, keys, vals, novelty_gate, **log_tau**, key_log_scale, val_log_scale) |
| `write` | 224 | Write with novelty gating | embedding | slot_idx | torch.no_grad, F.normalize | StreamingMemoryBank.forward | No |
| `mark_consumed` | 272 | Mark slot as consumed by L3 | slot | None | — | StreamingMemoryBank.forward | No |
| `read` | 278 | Read via hybrid attention | query: (B,L,D) | (B,L,D) | _memory_attention | StreamingMemoryBank.forward | Yes |
| `get_stats` | 297 | Return bank statistics | None | dict | — | diagnostics | No |
| `reset` | 308 | Clear all slots | None | None | — | StreamingMemoryBank.reset | No |

**Key change:** Line 209: `self.log_tau = nn.Parameter(torch.tensor(math.log(max(tau_prior, 0.1))))` — initialized from τ-prior.

**Class: `L3Concepts`** (nn.Module, line 319)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 340 | Emergent concept clustering from L2 slots | D, bridge_dim, n_concepts, birth_threshold, ... | None | _memory_attention | StreamingMemoryBank.l3 | Yes (concept_keys, concept_vals, q_proj, out_proj, **log_tau**, val_log_scale) |
| `write` | 375 | Concept birth/update from L2 key+val | l2_key, l2_val, confidence | bool | F.normalize, torch.no_grad | StreamingMemoryBank.forward | No |
| `read` | 436 | Read from concepts via attention | query: (B,L,D) | (B,L,D) | _memory_attention | StreamingMemoryBank.forward | Yes |
| `get_active_concepts` | 455 | Count active concepts | None | int | — | diagnostics | No |
| `get_stats` | 459 | Return concept statistics | None | dict | — | diagnostics | No |
| `reset` | 468 | Clear all concepts | None | None | — | StreamingMemoryBank.reset | No |

**Key change:** Line 362: `self.log_tau = nn.Parameter(torch.tensor(math.log(max(tau_prior, 0.1))))` — initialized from τ-prior.

**Class: `StreamingMemoryBank`** (nn.Module, line 478)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 491 | Combined L1+L2+L3 memory bank | D, bridge_dim, l1_slots, l2_slots, l3_concepts, ... | None | L1Buffer, L2Bank, L3Concepts, nn.Sequential | WideBindStack.memory_bank | Yes (fusion, log_scale) |
| `forward` | 547 | Read from all levels, write at sentence boundaries | h: (B,L,D), tokens, step, mat_gate | (B,L,D) | L1/L2/L3.read, L1/L2/L3.write | WideBindStack.forward | Yes |
| `reset` | 611 | Clear all memory | None | None | — | — | No |
| `get_diagnostics` | 617 | Return diagnostic info | None | dict | — | logging | No |

**Key change:** Lines 506-515: τ-priors computed from tau_config:
```python
if tau_config is not None:
    mem_tau = tau_config.mem_tau  # (3,) — [L1_tau, L2_tau, L3_tau]
    l1_tau_prior = (mem_tau[0] / tau_config.mem_tau_ref).clamp(0.1, 5.0).item()
    l2_tau_prior = (mem_tau[1] / tau_config.mem_tau_ref).clamp(0.1, 5.0).item()
    l3_tau_prior = (mem_tau[2] / tau_config.mem_tau_ref).clamp(0.1, 5.0).item()
```

---

### core/maturation.py (146 lines)

**Class: `MaturationController`** (nn.Module, line 47)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 48 | Unified per-layer maturation gate (time ramp) | n_layers, tau_min, tau_max, cfg, tau_config | None | — | WideBindStack.maturation | No (buffers only) |
| `_update_tau_norm` | 82 | Update tau_norm from per-layer deviation | dev | None | — | step_gate | No |
| `step_gate` | 91 | Per-layer sigmoid time ramp: M_l(t) | step, tau_dev | (n_layers,) | sigmoid | WideBindStack.forward | No |
| `global_ready` | 121 | Property: True when ALL layers above threshold | None | bool | — | WideBindStack.forward | No |
| `global_readiness_ratio` | 132 | Property: fraction of mature layers | None | float | — | diagnostics | No |
| `update` | 136 | Update readiness EMA from per-layer pred_err | step, pred_err | None | torch.no_grad | WideBindStack.forward (end) | No |

**Key change:** Line 53: `self.tau_config = tau_config` — uses tau_config.tau_norm instead of manual computation. Lines 106-108: `self.tau_norm.copy_(self.tau_config.tau_norm)`.

---

### core/stack.py (1949 lines)

**Class: `WideBindStack`** (nn.Module, line 18)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 21 | Full model: embed + layers + head + reasoning + bridge + maturation + tau_config | cfg | None | All modules | train.py, generate.py | Yes |
| `forward` | 156 | Main forward: τ-update → layers → reasoning → triad → output | h, state, global_state, ... | (h, new_state, global_state, ...) | All layers, AdaptiveController, MaturationController, SemanticBridge, StreamingMemoryBank | train.py, generate.py | Yes |
| `_knowledge_signal` | 591 | Compute model confidence features (B,8) | h, state | (B,8) | lm_head | _adaptive_reasoning | No |
| `_last_conf` | 638 | Confidence of head on last position | h | (B,) | lm_head | triad_reason, _adaptive_reasoning | No |
| `_adaptive_reasoning` | 648 | Adaptive-depth reasoning loop | h, s, state, reasoning_buffer, reasoning_count | h_acc | ReasoningMemory, ReasoningGate, _knowledge_signal, _last_conf | forward | Yes (reasoning_gate) |
| `reasoning_scale` | 756 | Property: reasoning influence scale | None | float | reasoning_scale_override, reasoning_ramp_steps | forward | No |
| `embed_tokens` | 769 | Token indices → D-space vectors | tokens | (B,L,D) | embed | generate.py | Yes |
| `compute_losses` | 827 | CE + all auxiliary losses (raw, unweighted) | h, targets, pred_weight, h_emb | (ce_loss, aux_dict) | lm_head, bridge.loss, all layer mirrors | LossBalancer.backward | Yes |
| `param_groups` | 1283 | Optimizer parameter groups with λ_d LR hierarchy | lr, weight_decay, gate_lr_mult | list[dict] | lambda_d | build_optimizer | No |
| `apply_mlp_depth_gradient_boost` | 1234 | Register backward hooks to boost deep MLP gradients | exp | None | register_hook | train.py | No |

**Key τ-integration points in forward:**

1. **Lines 202-215:** tau_config.update(mat_gate_for_tau) — recomputes all τ-derived values
2. **Line 208:** `tau_l = self.tau_config.tau_l` — per-layer temporal scale
3. **Line 210:** `vsa_tau = torch.exp(torch.cumsum(F.softplus(self._vsa_log_param), dim=0)) + 1.0` — block-level VSA scales (separate from tau_config)
4. **Line 335:** `mat_gate = self.maturation.step_gate(step, self._tau_l_dev.detach())` — uses _tau_l_dev (aliased to tau_config._tau_dev)
5. **Line 383:** `_alpha_i = self.tau_config.intent_alpha[i].detach()` — intent alpha from tau_config
6. **Line 513:** `alpha_l = self.tau_config.intent_alpha[i]` — global state EMA rate
7. **Lines 964-967:** w_m2v loss uses `self.tau_config.tau_l[i]` for target computation
8. **Lines 976-983:** intent_tau loss uses `self.tau_config.tau_l[i]` for target alpha

**Legacy params kept for checkpoint compat (lines 115-117):**
```python
self._vsa_log_param = nn.Parameter(torch.tensor([1.7918, 1.2321, 1.1304, 1.1065]))
self._tau_l_dev = self.tau_config._tau_dev  # alias — shared gradient
self._tau_intent_dev = nn.Parameter(torch.zeros(cfg.n_layers))
```

**Class: `AdaptiveController`** (line 1437)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `layer_stats` | 1487 | Per-layer (exploration, differentiation) | layer, expl_thresh, diff_thresh | (float, float) | mirror.log_scale, mirror._last_magnitude | stack.forward | No |
| `stats` | 1496 | Global average (expl, diff) across all layers | blocks, expl_thresh, diff_thresh | (float, float) | layer_stats | stack.forward | No |
| `layer_b_d` | 1509 | Per-layer decay bias from exploration | layer, expl, b_d_max | float | layer_stats | stack.forward | No |
| `layer_b_i` | 1519 | Per-layer write gate bias from exploration+τ | layer, expl, tau_l | float | layer_stats | stack.forward | No |
| `layer_w_mem2v_scale` | 1545 | Per-layer memory contribution | layer, min_val, max_val, diff | float | layer_stats | stack.forward | No |
| `layer_noise_scale` | 1552 | Per-layer parameter noise | layer, min_val, max_val, diff | float | layer_stats | stack.forward | No |
| `layer_ema_alpha` | 1559 | Per-layer EMA rate | layer, min_val, max_val, diff | float | layer_stats | stack.forward | No |
| `pred_weight` | 1568 | Adaptive alpha aux loss weight | blocks, min_val, max_val | float | stats | stack.forward | No |
| `tanh_bias_modulation` | 1579 | Scale tanh_bias by exploration | layer, expl | float | layer_stats | stack.forward | No |
| `spectral_modulation` | 1590 | Modulate spectral lambda_k by differentiation | layer, diff | float | layer_stats | stack.forward | No |
| `pred_scale_mod` | 1603 | Per-expert pred error modulation from delta_var | layer | tensor | mirror._delta_var | stack.forward | No |
| `b_d`, `b_i`, `w_mem2v_scale`, `ema_alpha`, `noise_scale` | 1618-1640 | Global wrappers (backward compat) | blocks | float | stats | — | No |

**Class: `MirrorLRScheduler`** (line 1644)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 1653 | LR scheduler modulated by cognitive mirror state | model, optimizer, base_lr, warmup, ... | None | — | LRController | No |
| `_mirror_stats` | 1686 | Aggregate var(log_scale), |mirror|, alpha, gate_var | None | (var, mag, alpha, gate_var) | step | No |
| `_update_ls_mult` | 1702 | Per-layer LR mult from fast/slow EMA of var(ls) | None | list or None | — | step | No |
| `report_val_loss` | 1739 | Adaptive LR damping on val loss regression | val_loss | None | — | training | No |
| `step` | 1780 | Update LR with warmup + mirror adaptive multiplier | None | None | _mirror_stats, _update_ls_mult | LRController.step | No |
| `get_last_lr` | 1870 | Return current LRs | None | list[float] | — | training | No |
| `state_dict` | 1873 | Serialize scheduler state | None | dict | — | checkpoint | No |
| `load_state_dict` | 1895 | Restore scheduler state | sd | None | — | resume | No |

---

### core/block.py (504 lines)

**Class: `WideBindBlock`** (nn.Module, line 48)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 62 | Full block: bind + mirror + VSA memory + conv + spectral + MLP + collective | cfg, layer_idx | None | BottleneckBind/TrajectorySpiralBind/TrajectoryManifoldBind, GroupedCognitiveMirror, GroupedMLP, PrecisionGate, ExactSequenceMemory, CollectiveConceptLayer, Conv1d | WideBindStack.layers | Yes (pre_ln_w, w_i, w_d, w_q, w_q_leaf, w_q_ctx, w_mem2v, w_q_dyn, w_i_dyn, w_d_pen, w_bind_gate, scale_w, b_i, b_d, gamma_surprisal, bind_coh_gate, w_k_mu, w_q_mu, w_mu_mem, conv, lambda_k, + bind/mirror/mlp/collective/precision/exact submodules) |
| `forward` | 201 | Full block forward: preLN→conv→bind→VSA→mirror→output→spectral→MLP | h, state, global_state, ... | (h, state_out) | All submodules | WideBindStack.forward | Yes |
| `base_parameters` | 493 | Property: all params except mirror | None | list | — | — | No |
| `mirror_parameters` | 498 | Property: all mirror params | None | list | — | — | No |

**Block forward flow (line 201-491):**
1. Pre-LN: RMS normalization
2. Conv: depthwise 1D convolution + residual
3. Bind: D→K→D via BottleneckBind/TrajectorySpiralBind + residual
4. VSA Memory: multi-scale prefix scan (S=4 scales) with content-dependent write/decay
5. Mirror: GroupedCognitiveMirror(h, mem_all, ...) → (mirror, mlp_mod, mem_mod, hp, pred_error_norm)
6. Output: enhanced = bind_gated + mem_modulated + mirror + collective
7. Variable Precision: PrecisionGate + ExactSequenceMemory (optional)
8. Spectral: DCT basis * lambda_k modulation + residual
9. MLP: GroupedMLP with mirror_gate + residual

---

### core/mirror.py (798 lines)

**Class: `BridgeGLU`** (nn.Module, line 11)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 19 | GLU gating: Wg, Wv linear + log_gain | G, k | None | nn.Linear | GroupedCognitiveMirror | Yes (Wg, Wv, log_gain) |
| `forward` | 28 | delta → sigmoid(Wg·flat) * sigmoid(Wv·flat) * sigmoid(log_gain) | delta: (B,L,G,k) | (B,L,G) | torch.sigmoid | GroupedCognitiveMirror.bridge_glu_net | Yes |

**Class: `GroupedCognitiveMirror`** (nn.Module, line 36)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 73 | 32-group mirror expert ensemble | D, G, k, ... | None | AdaptiveGate, BridgeGLU | WideBindBlock.mirror | Yes (W_proj, W_out, w_temp, w_global, conv_smooth, w_sym_u/v, alpha_diag, tanh_bias, log_scale, w_gate, b_gate, w_delta_gate, gate_bias, log_skip_alpha, w_alpha, b_alpha, log_dvar_mod_scale, log_grad_mod_scale, dvar_mod_bias, grad_mod_bias, _signal_log_weights, w_help, w_contra, mod_scale_mlp, mod_scale_mem, w_intent, b_intent, w_sal) |
| `_sync_W_out` | 297 | Sync W_out = W_proj^T for tied mode | None | None | torch.no_grad | forward_pre_hook | No |
| `forward` | 301 | Main mirror: project→K-space→signals→gate→output | h, mem_all, global_state, diff, tanh_bias_mod, pred_scale_mod, context_mem, allow_write, step, intent, salience, maturity | (mirror, mlp_mod, mem_mod, hp, pred_error_norm) | BridgeGLU, AdaptiveGate | WideBindBlock | Yes |
| `cache_grad_norms` | 733 | Store per-subspace gradient norm after backward | grad_h | None | torch.no_grad | training loop | No |
| `debug_mind` | 744 | Return meta-cognitive stats dict | None | dict | torch.no_grad | smart_controller, analyze.py | No |
| `meta_signals` | 786 | GPU-friendly (trust_max, gate_ema_mean) tensors | None | (tensor, tensor) | torch.no_grad | smart_controller.generate | No |

**Mirror forward signal computation (within forward, line 301-731):**

1. **Split h into G subspaces**: h_g = h.reshape(B,L,G,d)
2. **Project to K-space**: hp = einsum(h_g, W_proj) → (B,L,G,k)
3. **Bipolar pos_id binding**: hp *= _pos_id_buf
4. **hp_prev**: shifted by 1 timestep
5. **temp_k**: (hp - mc_k) * w_temp — deviation from memory centroid
6. **pred_error**: hp - hp_prev * alpha_diag — predictive error (per-dim tau)
7. **smooth_k**: hp - conv_smooth(hp) — local coherence
8. **sym_k**: (hp * w_sym_u) * (hp_prev * w_sym_v) — temporal symmetry
9. **help_k**: private memory read via cross-expert attention (when enabled)
10. **Signal EMA normalization**: per-signal RMS normalization
11. **Learnable signal weights**: sigmoid(_signal_log_weights)
12. **Delta merge**: weighted sum of all signals, RMS-normalized
13. **Gate modulation**: grad_mod, dvar_mod, intent_bridge, salience, contradiction
14. **Expert gate**: sigmoid(gate_logits) — per-token, per-expert
15. **MLP mod**: hybrid_gate(usefulness_logits) * BridgeGLU (when enabled)
16. **Mirror output**: tanh(W_out·delta) + skip_alpha * linear, scaled by log_scale
17. **SMF gate**: alpha = sigmoid(cat(h_norm, m_norm) · w_alpha + b_alpha)
18. **Output**: mirror * expert_gate * maturation

---

### core/bind.py (651 lines)

**Class: `_ExpRMSNorm`** (nn.Module, line 11)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 14 | Creates RMSNorm weight | K | None | nn.Parameter | — | Yes (weight) |
| `forward` | 18 | RMSNorm via explicit formula | x: (..., K) | (..., K) | torch.rsqrt | BottleneckBind.hp_norm | Yes |

**Function: `migrate_bind_state_dict`** (line 21)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `migrate_bind_state_dict` | 21 | Converts old bind state dict keys to new format | sd, n_layers, mode, S | dict | re | train.py resume | No |

**Function: `_golden_shifts`** (line 49)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `_golden_shifts` | 49 | Generates S golden-angle circular shifts for bind | K: int, S: int | list[int] | math.floor | BottleneckBind.__init__ | No |

**Class: `BottleneckBind`** (nn.Module, line 62)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 78 | Bilinear cross-mixing with Fibonacci/golden-angle shifts | D, K, cfg | None | _ExpRMSNorm, _golden_shifts, nn.Linear | WideBindBlock.bind | Yes (W_proj, w_u, w_v, W_out, mix_logit, log_tau, w_gate_proj, w_bind_bias) |
| `_tie_hook` | 138 | Copies W_proj to W_out for tied mode | module, inp | None | torch.no_grad | register_forward_pre_hook | No |
| `_cross` | 142 | Bilinear cross-mixing: left * roll(right, shift) | left, right, shift | tensor | torch.roll | forward | No |
| `forward` | 145 | Main bind: D→K bottleneck, bilinear mixing, K→D output | h: (B, L, D) | (B, L, D) | hp_norm, W_proj, w_u, w_v, W_out | WideBindBlock | Yes |

**Bind modes:**
- "off": legacy diagonal (u·v), no shifts
- "shift": sum of S shifted bilinear products
- "cascade": Fibonacci-nested cascade, monomials up to order F_S

**Class: `TrajectorySpiralBind`** (nn.Module, line 303)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 304 | Trajectory-based spiral bind with n_dims trajectory history | D, K, cfg | None | _ExpRMSNorm, nn.Linear | WideBindBlock | Yes (w_u_re/im, w_v_re/im, W_freq, W_phase, W_out, W_proj, freq_scale, w_bind_bias) |
| `_hybrid_alpha` | 348 | Returns hybrid alpha (HRR vs element-wise blend) | None | float | self.training, _step_count | forward | No |
| `_hrr_bind` | 355 | HRR (holographic reduced representation) bind | a, b | tensor | _circ_conv_idx, torch.einsum | _hybrid_bind | No |
| `_hybrid_bind` | 359 | Alpha-weighted blend of HRR and element-wise bind | a, b | tensor | _hrr_bind | forward | No |
| `forward` | 365 | Full trajectory spiral bind with coherence measure | h: (B,L,D), traj_state | (result, new_traj, coherence) | hp_norm, W_proj, cos/sin of trajectory | WideBindBlock | Yes |

**Class: `TrajectoryManifoldBind`** (TrajectorySpiralBind, line 457)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 473 | Extends TrajectorySpiralBind with manifold beam clustering | D, K, cfg | None | super().__init__ | WideBindBlock | Yes (W_man, logit_gain, log_tau + all parent) |
| `_fib_list` | 508 | Static: Fibonacci sequence up to max_n | max_n | list | — | _zeck_weight | No |
| `_zlen` | 514 | Static: Zeckendorf decomposition length | fibs, n | int | — | _zeck_weight | No |
| `_hrr_unbind` | 526 | HRR unbind (correlation) | a, b | tensor | _circ_corr_idx | _hybrid_unbind | No |
| `_hybrid_unbind` | 530 | Alpha-weighted HRR+element-wise unbind | a, b | tensor | _hrr_unbind | _push_transitions | No |
| `_push_transitions` | 536 | Write unbind(hp_t, hp_{t-1}) transitions to ring buffer | hp | None | _hybrid_unbind, _rebuild_beams | forward | No |
| `_rebuild_beams` | 559 | VSA-cluster transitions into beam centers | None | None | cos_threshold | _push_transitions | No |
| `_zeck_weight` | 605 | Zeckendorf-based age decay for beams | age | float | _zlen, _fib_cache | _manifold_read | No |
| `_manifold_read` | 611 | Hybrid attention read from manifold beams | hp: (B,L,K) | (B,L,K) | beam_centers, logit_gain, log_tau | forward | Yes (logit_gain, log_tau) |
| `forward` | 641 | Trajectory + manifold bind combined | h, traj_state | (result, new_traj, coherence) | super().forward, _push_transitions, _manifold_read | WideBindBlock | Yes |

---

### core/mlp.py (80 lines)

**Class: `GroupedMLP`** (nn.Module, line 9)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 21 | Grouped bottleneck MLP with per-group expansion (SwiGLU) | D, expand, groups, swiglu, gate_b_init | None | — | WideBindBlock.mlp | Yes (W_gate/W_up/W_down, norm_w, mlp_gate_a, mlp_gate_b) |
| `forward` | 50 | Grouped SwiGLU: silu(gate) * up → down, with mirror_gate modulation | h: (B,L,D), mirror_gate: (B,L,G) | (B,L,D) | F.silu, torch.matmul | WideBindBlock.forward | Yes |

**MLP flow:**
1. RMSNorm: norm_w * h / sqrt(mean(h²))
2. Reshape to G groups of d dims
3. SwiGLU: silu(W_gate·h) * (a + b·mirror_gate) * W_up·h → W_down
4. Cache _cached_group_out for diversity loss

---

### core/embedding.py (378 lines)

**Class: `RotaryEmbedding`** (nn.Module, line 12)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 13 | Precompute RoPE frequencies | D, theta, scaling, max_len | None | — | PartitionedEmbedding | No |
| `_build_cache` | 23 | Build cos/sin cache for length L | L | None | — | forward | No |
| `forward` | 32 | Apply rotary position embedding | x: (B,L,D) | (B,L,D) | cos, sin | PartitionedEmbedding | No |

**Class: `ZeckendorfEmbedding`** (nn.Module, line 46) — Legacy, not used in default config

**Class: `PartitionedEmbedding`** (nn.Module, line 64)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 79 | Token→D via sparse block codes + dense mixing + RoPE | cfg | None | sparse_block_codes, RotaryEmbedding | WideBindStack.embed | Yes (embed_mix, basis) |
| `forward` | 101 | codes→sigmoid(M·codes)→outer(basis)→RoPE | tokens | (B,L,D) | — | WideBindStack.embed_tokens | Yes |

**Class: `LmHead`** (nn.Module, line 115) — Legacy, not used in default config

**Class: `PartitionedHead`** (nn.Module, line 130) — Alternative head, not used by default

**Class: `SigmoidCodedHead`** (nn.Module, line 167)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 168 | Sigmoid-coded head with hybrid gate | cfg, embed_basis | None | sparse_block_codes | WideBindStack.lm_head | Yes (readout, bit_bias, log_temp, token_bias) |
| `_gates` | 189 | Compute per-bit gate logits | h, temp_factor, bus_bias | (B,L,K) | — | forward, log_probs_for_target | Yes |
| `_su` | 210 | Sigmoid-softmax hybrid: log-odds computation | zt | (u, base) | hybrid_gate(log=True) | forward, log_probs_for_target | Yes |
| `forward` | 214 | Full head: gates→log-odds→token logits | h, bus_bias | (B,L,V) | _gates, _su, codes | WideBindStack.compute_losses | Yes |
| `log_probs_for_target` | 228 | Efficient log-probs for specific target tokens | h, targets, bus_bias | (N,) | _gates, _su, codes | CE computation | Yes |

**Class: `CognitiveCodedHead`** (nn.Module, line 254)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 255 | Cognitive head with prior, social bias, resonance | cfg, embed_basis, k_mirror | None | sparse_block_codes, nn.Embedding | — (alt head) | Yes (readout, log_temp_base, w_res, w_stab, bit_bias, W_q_prior, W_k_prior, alpha_prior, w_prior_scale, beta_social, w_energy, gamma, W_code_mod, token_shift_embed, proj_shift, token_bias) |
| `set_cognitive_state` | 296 | Set external cognitive signals | pred_error, private_mem, trust_matrix, contra_graph, dominance | None | — | WideBindBlock | No |
| `_compute_z` | 304 | Compute z with prior, social bias, resonance, context | h, B, L, device | (z, base) | — | forward, log_probs_for_target | Yes |
| `_shift_all` | 349 | Token shift embedding for all tokens | None | (V,K) | — | forward | Yes |
| `_shift_targets` | 353 | Token shift for specific targets | token_ids | (N,K) | — | log_probs_for_target | Yes |
| `forward` | 358 | Full cognitive head | h | (B,L,V) | _compute_z, _shift_all | — | Yes |
| `log_probs_for_target` | 366 | Efficient log-probs | h, targets | (N,) | _compute_z, _shift_targets | — | Yes |

---

### core/concept_layer.py (279 lines)

**Class: `CollectiveConceptLayer`** (nn.Module, line 16)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 17 | Adaptive maturity concept layer with S slots | D, k, S, uncert_theta, ... | None | nn.Linear, nn.Parameter | WideBindBlock.collective | Yes (_birth_log_scale, W_o, _read_scale, _temp) |
| `_update_maturity` | 77 | Adaptive maturity from residual variance | resvar | None | — | forward | No |
| `_maybe_write` | 102 | Mature-gated concept write/update/birth | hp, pen, allow_write | (write_event, best) | F.normalize, torch.no_grad | forward | No |
| `forward` | 187 | Main: write concepts + read with gate-weighted expert attention | h, hp, pen, resvar, allow_write, mature_override, gate | out: (B,L,D) | F.normalize, torch.sigmoid/einsum | WideBindBlock.forward | Yes |
| `birth_gate_mean` | 264 | Average birth gate weight | None | float | — | diagnostics | No |
| `get_diagnostics` | 269 | Return concept diagnostics | None | dict | — | logging | No |

---

### core/reasoning.py (178 lines)

**Class: `ReasoningTokens`** (line 11) — static token IDs (THINK=65536, STEP=65537, ANSWER=65538, END=65539)

**Class: `ReasoningMemory`** (nn.Module, line 19)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 24 | Reasoning step encoder + attention + output proj | D, max_steps | None | nn.Linear | WideBindStack.reasoning_memory | Yes (step_encoder, step_query, step_key, step_value, output_proj) |
| `forward` | 40 | Encode current step, attend to buffer, update buffer | h, reasoning_buffer, reasoning_count, record | (output, new_buffer, new_count) | torch.sigmoid attention | _adaptive_reasoning | Yes |

**Class: `ReasoningGate`** (nn.Module, line 106)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 121 | Per-step decision gates for adaptive reasoning depth | D, max_steps, know_dim | None | nn.Linear | WideBindStack.reasoning_gate | Yes (proj, know_proj, r_proj) |
| `forward` | 139 | tanh(proj(h) + know_proj(know) + r_proj(r)) | h, know, r | (B,L,max_steps) | torch.tanh | _adaptive_reasoning | Yes |
| `logits` | 154 | Raw logits for straight-through gradient | h, know, r | (B,L,max_steps) | — | _adaptive_reasoning | Yes |

**Class: `ThinkingTokenHead`** (nn.Module, line 164)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 169 | Head predicting thinking tokens | D, num_reasoning_tokens | None | nn.Linear | WideBindStack.thinking_head | Yes (reasoning_proj) |
| `forward` | 173 | linear(h) → reasoning token logits | h: (B,L,D) | (B,L,4) | — | — | Yes |

---

### core/bridge.py (181 lines)

**Class: `SemanticBridge`** (nn.Module, line 31)

| Method | Line | Description | Inputs | Outputs | Dependencies | Consumed By | Trainable |
|--------|------|-------------|--------|---------|--------------|-------------|-----------|
| `__init__` | 32 | Per-layer semantic bridge with stream injection | D, n_layers, bridge_dim, depth, cfg | None | nn.Sequential, nn.Linear | WideBindStack.bridge | Yes (probe, emb_proj, stream_proj, stream_log_scale, stream_log_weights) |
| `readiness` | 79 | Scalar readiness [0,1] from bridge competence | None | tensor | torch.no_grad | MaturationController | No |
| `start_forward` | 93 | Reset predictions list for new forward | None | None | — | WideBindStack.forward | No |
| `probe_layer` | 96 | Emit semantic vector for a layer | h_l: (B,L,D) | (B,L,bridge_dim) | probe | WideBindStack.forward | Yes |
| `inject_layer` | 100 | Add cross-layer stream signal to hidden state | i, h_l, maturity | h_l: (B,L,D) | stream_proj, sigmoid weights | WideBindStack.forward | Yes |
| `update_stream` | 126 | EMA-update persistent stream | i, s_l | None | torch.no_grad | WideBindStack.forward | No |
| `record` | 132 | Record prediction for loss | s_l | None | — | WideBindStack.forward | No |
| `reset_stream` | 138 | Clear bridge stream | None | None | — | — | No |
| `loss` | 141 | Self-supervised bridge loss: predict next token embedding | y, embed_fn | scalar or None | F.cosine_similarity | compute_losses | Yes |

---

## 5. FORWARD PASS FLOW

### 5.1 Complete Data Flow (Training)

```
Token IDs (B, L)
    │
    ▼
PartitionedEmbedding.forward()
    ├─ codes[tokens] → (B,L,K) sparse binary
    ├─ sigmoid(embed_mix · codes) → (B,L,K) dense mixing
    ├─ outer(codes, basis) → (B,L,D) embedding
    └─ RoPE → (B,L,D) positional
    │
    ▼
[TauConfig.update(mat_gate)] ← computes ALL τ-derived values
    │
    ▼
[Optional: StreamingMemoryBank.forward(h, tokens)]
    ├─ L1.read(h) → mem_l1 (B,L,D)
    ├─ L2.read(h) → mem_l2 (B,L,D)
    ├─ L3.read(h) → mem_l3 (B,L,D)
    └─ h = h + scale * fusion(cat(h, mem_l1, mem_l2, mem_l3))
    │
    ▼
For each layer i (0..n_layers-1):
    │
    ├─ [Intent Bridge] intent_probe(h) → intent_i (1,1,G,Kmax)
    │   └─ alpha_i = tau_config.intent_alpha[i] → EMA blend
    │   └─ truncate to layer k, scale by maturation
    │
    ├─ [Semantic Bridge] bridge.inject_layer(i, h, maturity=mat_gate[i])
    │   └─ h += scale * stream_proj(combined_neighbors)
    │
    ├─ [Memory Bank] memory_bank(h, tokens) per layer (if enabled)
    │
    ├─ WideBindBlock.forward(h, state, global_state, ...)
    │   │
    │   ├─ Pre-LN: h = pre_ln_w * h / sqrt(mean(h²))
    │   │
    │   ├─ Conv: h += conv1d(h)  (depthwise, causal)
    │   │
    │   ├─ Bind: bind_out = BottleneckBind(h) or TrajectorySpiralBind(h)
    │   │   ├─ hp = W_proj(h) + bias  → (B,L,K)
    │   │   ├─ bilinear mixing with shifts/trajectory
    │   │   └─ bind_out = mixed @ W_out  → (B,L,D)
    │   │
    │   ├─ VSA Memory (multi-scale, S=4):
    │   │   ├─ tau_s = exp(cumsum(softplus(_vsa_log_param))) + 1  (block-level, NOT tau_config)
    │   │   ├─ i_gate = softplus(h*w_i + b_i + γ·pred_error)
    │   │   ├─ d_mod = sigmoid(h*w_d + b_d)
    │   │   ├─ decay = tau_scale * d_mod  (per-scale per-channel)
    │   │   ├─ mem_input = h * i_gate * dynamic_write_mod
    │   │   ├─ Parallel prefix scan (4 scales, chunked)
    │   │   ├─ Weighted combination: mem_all = Σ sigmoid(scale_w) * mem_per_scale
    │   │   ├─ mem_read = mem_all * w_q + mem_leaf * w_q_leaf + mem_all * w_q_ctx
    │   │   └─ mem_read += mu_read * w_mu_mem
    │   │
    │   ├─ Mirror: mirror, mlp_mod, mem_mod, hp, pred_error_norm
    │   │   = GroupedCognitiveMirror(h, mem_all, global_state, ...)
    │   │   ├─ Split h → h_g (B,L,G,d)
    │   │   ├─ Project: hp = einsum(h_g, W_proj) → (B,L,G,k)
    │   │   ├─ pos_id binding: hp *= pos_id_buf
    │   │   ├─ Signals:
    │   │   │   ├─ temp_k = (hp - mc_k) * w_temp
    │   │   │   ├─ pred_error = (hp - hp_prev*alpha) / hp_norm
    │   │   │   ├─ smooth_k = hp - conv_smooth(hp)
    │   │   │   ├─ sym_k = (hp*w_sym_u) * (hp_prev*w_sym_v)
    │   │   │   └─ [help_k from private memory]
    │   │   ├─ EMA-normalize each signal
    │   │   ├─ Weighted merge: delta = Σ sigmoid(w_sig) * signal_normed
    │   │   ├─ RMS-normalize delta
    │   │   ├─ Usefulness predictor → gate logits
    │   │   ├─ Expert gate: sigmoid(gate_logits + grad_mod + dvar_mod + intent)
    │   │   ├─ Mirror out: tanh(W_out·delta) + skip*linear, scaled by log_scale
    │   │   ├─ SMF gate: alpha = sigmoid(cat(h,m)·w_alpha + b_alpha)
    │   │   ├─ BridgeGLU: glu = sigmoid(Wg·delta) * sigmoid(Wv·delta)
    │   │   └─ MLP mod: base = hybrid_gate(usefulness, tau_prior=gate_tau) * (1 + β*(2*glu-1))
    │   │
    │   ├─ Output:
    │   │   ├─ bind_gated = bind_out * mem_mod * bind_gate
    │   │   ├─ mem_modulated = mem_read * read_mod * mem_mod
    │   │   ├─ enhanced = bind_gated + mem_modulated*w_mem2v + mirror
    │   │   ├─ [CollectiveConceptLayer output added]
    │   │   └─ h += enhanced
    │   │
    │   ├─ Variable Precision Memory (optional):
    │   │   ├─ precision = sigmoid(linear(h))
    │   │   └─ h += precision * exact_memory(h) * gate
    │   │
    │   ├─ Spectral:
    │   │   ├─ h_dct = h @ V_dct.T * lambda_k * spectral_mod
    │   │   └─ h += h_dct @ V_dct
    │   │
    │   └─ MLP:
    │       ├─ h_mlp = GroupedMLP(h, mirror_gate=mlp_mod)
    │       ├─ SwiGLU: silu(W_gate·h) * (a + b*mlp_mod) * W_up·h → W_down
    │       └─ h += h_mlp
    │
    ├─ [Bridge probe] bridge.record(probe(h))
    ├─ [Bridge update] bridge.update_stream(i, probe(h))
    ├─ [Maturation update] pred_errs.append(mirror._cached_pred_error_norm)
    └─ [Global state update] global_state[i] = α * gs + (1-α) * mem_avg
    │
    ▼
After all layers:
    ├─ h = final_norm_w * h * rsqrt(mean(h²) + 1e-7)
    │
    ├─ [Maturation update] maturation.update(step, pred_errs)
    │
    ├─ [Explicit Reasoning] (if enabled and scale > 0):
    │   └─ _adaptive_reasoning(h, s, state, ...):
    │       ├─ know = _knowledge_signal(h)  (B,8)
    │       ├─ For i in 0..K-1:
    │       │   ├─ r_i = ReasoningMemory(h, buf, count)
    │       │   ├─ l_i = ReasoningGate.logits(h_acc, know, r_i)[..., i]
    │       │   ├─ a_i = tanh(l_i)  (gate ∈ (-1,1))
    │       │   ├─ contrib = a_i * normalize(r_i) * w_soft * run_mask
    │       │   └─ h_acc += s * contrib / denom_accum
    │       └─ return h_acc
    │
    └─ [Triad re-circulation] (inference only, if confidence < threshold):
        ├─ _conf = _last_conf(h).mean()
        └─ if _conf < triad_conf_thr:
            h2 = self.forward(h, ..., _triad_depth+1)  (recursive)
            h = 0.5*h + 0.5*h2
    │
    ▼
Output: (B, L, D)
    │
    ▼
SigmoidCodedHead.forward(h, bus_bias):
    ├─ z = (h_g * readout).sum(-1) / T + bit_bias + bus_bias
    ├─ gate = sigmoid(z) * (1 + softmax(z/tau))
    ├─ log_odds = log(gate/(1-gate))
    ├─ logits = log_odds @ codes.T + log(1-gate).sum(-1) + token_bias
    └─ logits = logits - logsumexp(logits)  (normalized)
    │
    ▼
Output: (B, L, vocab) logits
```

### 5.2 Memory Bank Read/Write at Each Stage

**Write** (at sentence boundaries, token == 2):
- L1: overwrite oldest slot with sentence mean
- L2: overwrite oldest/consumed slot with projected key+val
- L3: if L2 key matches existing concept → update; else → birth (mark L2 consumed)

**Read** (every token position):
- L1: hybrid attention over rolling buffer (log_tau from τ-prior)
- L2: hybrid attention over learned slots (log_tau from τ-prior)
- L3: hybrid attention over emergent concepts (log_tau from τ-prior)
- Fusion: cat(h, mem_l1, mem_l2, mem_l3) → MLP → scale * fused → inject into h

### 5.3 Mirror Expert Processing

Each of 32 experts operates in its own d=128 subspace:
1. **Project** to K-space (k=8/16/32)
2. **Compute 4+ signals**: temporal, predictive, smooth, symmetry, [help from private memory]
3. **EMA-normalize** each signal independently
4. **Merge** with learnable Fibonacci-initialized weights
5. **Gate** using usefulness predictor + gradient/delta modulation
6. **Output**: tanh(linear) + skip connection, scaled by log_scale

### 5.4 Reasoning Cycle

1. `_knowledge_signal(h)` → 8-dim vector (confidence, entropy, mem agreement, etc.)
2. For each step i (0..K-1):
   - `r_i = ReasoningMemory(h, buffer)` → encode current, attend to past
   - `l_i = ReasoningGate.logits(h_acc, know, r_i)` → step gate logit
   - `a_i = tanh(l_i)` → gate value (-1, 1)
   - If previous gate closed → step doesn't run (run_mask)
   - `contrib = a_i * normalize(r_i)` → gated, normalized contribution
   - Validate: check if confidence improves → w_soft scales contribution
   - `h_acc += s * accum / Σ|w_i|` → accumulated reasoning
3. Buffer updated only for committed steps

### 5.5 Maturation Gates

```
M_l(t) = sigmoid((t - (T0 + α*(1-τ_norm_l)*T_delay)) / δt)
```

- **Deep layers** (τ_norm≈1): open at t ≈ T0 = 8000 steps
- **Shallow layers** (τ_norm≈0): open at t ≈ T0 + T_delay = 16000 steps
- Gates: BridgeGLU, private-memory write, bridge injection, intent bus

---

## 6. TRAINING FLOW

### 6.1 Loss Computation

**Primary loss:**
- Cross-entropy (CE) on logits vs targets, with PAD/EOS masking and optional surprisal weighting

**Auxiliary losses (all raw, unweighted):**

| Loss | Config Weight | Description |
|------|--------------|-------------|
| `pred` | 0 | MSE(pred_k, hp.detach()) — temporal prediction |
| `gate_l1` | gate_l1_weight=0.0001 | Mean expert gate (sparsity) |
| `reinforce` | reinforce_weight=0.001 | MSE(usefulness, gate) — align gate with self-assessment |
| `balance` | balance_weight=0.026 | HHI-based load balancing (uniform expert usage) |
| `diversity` | diversity_weight=0.001 | ||cov(group_out) - I||² — decorrelate groups |
| `nuc` | nuclear_weight=1e-5 | Stochastic nuclear norm of bind W_proj |
| `orth` | orth_weight=1e-4 | ||Ŵ^TŴ - I||² — orthogonality of bind W_proj |
| `div` | div_weight=50.0 | sigmoid-bounded log_scale divergence |
| `ranking` | ranking_weight=0.01 | Pairwise order loss: ls_mean vs gate_usage |
| `gate_repulse` | gate_repulse_weight=0.3 | Push gate variance up |
| `alpha_novelty` | alpha_novelty_weight=0.05 | Push per-expert alpha apart |
| `decorr` | 0 | Decorrelation of weighted mirror signals |
| `signal_ent` | 0 | Entropy of signal weights (diversity) |
| `ls_reg` | 0 | L2 on log_scale excess above 2.3 |
| `branch` | branch_balance_weight=0.0 | Equalize log-variance of conv/bind/mirror |
| `w_m2v` | w_m2v_hierarchy_weight=0.01 | Drive w_mem2v toward τ-hierarchy target (uses tau_config.tau_l) |
| `intent_tau` | intent_tau_hierarchy_weight=0.01 | Drive intent τ-deviation toward target (uses tau_config.tau_l) |
| `bridge_conn` | bridge_conn=0.1 | Bridge cosine loss: predict next-token embedding |
| `pred_w` | 0 | MSE(pred_w, I) — identity prediction weights |

### 6.2 Gradient Flow

1. **CE loss** → through head → through all layers → parameters
2. **Aux losses** → LossBalancer.backward:
   - Compute `g_CE = ∇ CE`
   - Compute `g_aux = ∇ (Σ aux_i)`
   - `cos = ⟨g_CE, g_aux⟩ / (||g_CE||·||g_aux||)`
   - `scale = clamp(cos, 0, cap) * ||g_CE|| / (||g_aux|| + ε)`
   - `p.grad = g_CE + scale * g_aux` (only when cos > 0)
3. **GradientClipper** (AGC): clip iff `||g|| > c·||θ||`
4. **MLP depth boost**: `p.grad *= exp(mlp_depth_lr_exp * layer_idx)` for MLP params

### 6.3 Optimizer Groups (λ_d hierarchy, default)

| Group | LR multiplier | Parameters |
|-------|--------------|------------|
| embed | λ⁻² ≈ 0.296 | embed.embed_mix, embed.basis |
| embed_wd | λ⁻² | (same with weight_decay) |
| mlp | λ⁻¹ ≈ 0.544 | mlp.W_*, mlp.norm_w, mlp.mlp_gate_*, bind.W_proj |
| mirror | λ¹ ≈ 1.839 | mirror.W_proj, mirror.W_out, mirror.alpha_diag, mirror.log_scale, mirror.tanh_bias, mirror.w_temp, mirror.w_global |
| gate | λ¹ | mirror.w_gate, mirror.b_gate, mirror.w_delta_gate, mirror.gate_bias, w_i, w_d, w_q |
| **tau_config** | λ¹ | **tau_config._tau_dev** (unified τ-field) |
| vsa | λ⁻² | b_d, b_i, scale_w |
| bridge | 1.0 (base) | bridge.*, bridge_glu_net.*, intent_probe, bus_head_proj, layer_bridge_gate.* |
| default | 1.0 | everything else |

**Layer-wise LR decay (LLRD):** `lr_layer_i = lr_group * llrd^i` (llrd=0.9)

### 6.4 What Gets Updated vs Frozen

**Always trainable:**
- Embedding: embed_mix, basis
- All bind params: W_proj, w_u, w_v, W_out, mix_logit, log_tau
- Mirror: W_proj, W_out, alpha_diag, log_scale, tanh_bias, w_temp, w_global, w_gate, b_gate, all signal weights
- MLP: W_gate, W_up, W_down, norm_w, mlp_gate_a, mlp_gate_b
- VSA: w_i, w_d, w_q, w_q_leaf, w_q_ctx, w_mem2v, b_i, b_d, scale_w, w_k_mu, w_q_mu, w_mu_mem
- Spectral: lambda_k
- Head: readout, bit_bias, log_temp, token_bias
- Bridge: probe, emb_proj, stream_proj, stream_log_scale, stream_log_weights
- Intent: intent_probe, bus_head_proj
- Maturation: None (all buffers)
- Reasoning: step_encoder, step_query/key/value, output_proj, reasoning_gate
- **tau_config._tau_dev**: always trainable (the unified τ-field)
- **Memory bank log_tau**: always trainable (initialized from τ-prior)

**Frozen (by DepthController):**
- Blocks with index >= k are frozen (requires_grad=False)
- Default: first 8 blocks active, rest frozen until plateau triggers unfreezing

**Maturation-gated (effectively frozen at init):**
- BridgeGLU gates (live but scaled by maturity → ~0 at start)
- Private memory writes (gated by maturity > threshold)
- Bridge injection (scaled by maturity)
- Intent bus (scaled by maturity)

---

## 7. GRADIENT FLOW ANALYSIS

### 7.1 Parameters with Gradients at Step 0

| Parameter | Grad at Step 0 | Source |
|-----------|----------------|--------|
| `tau_config._tau_dev` | YES | Through intent_alpha, tau_norm → maturation → gate → mirror → head → CE |
| `_vsa_log_param` | YES | Through vsa_tau → VSA memory → head → CE |
| `embed.embed_mix` | YES | Direct path through embedding → head → CE |
| `embed.basis` | YES | Direct path through embedding → head → CE |
| `bind.W_proj.weight` | YES | Through bind → head → CE |
| `bind.w_u, w_v` | YES | Through bilinear mix → head → CE |
| `mirror.W_proj` | YES | Through mirror → head → CE |
| `mirror.alpha_diag` | YES | Through pred_error → mirror → head → CE |
| `mirror.log_scale` | YES | Through mirror output scaling → head → CE |
| `mlp.W_gate, W_up, W_down` | YES | Through MLP → head → CE |
| `lm_head.readout, bit_bias, log_temp` | YES | Direct path through head → CE |
| `w_i, w_d, w_q, scale_w` | YES | Through VSA memory → head → CE |
| `lambda_k` | YES | Through spectral gate → head → CE |

### 7.2 Parameters Activating After Maturation

| Parameter | Activation | Gate |
|-----------|------------|------|
| `mirror.bridge_glu_net.*` | After M_l > 0.1 | bridge_glu * maturity |
| `bridge.stream_proj` | After M_l > 0 | bridge injection * maturity |
| `intent_probe` | After M_l > 0 | intent bus * mat_gate[i] |
| `bus_head_proj` | After M_l > 0 | bus_bias in head |
| `layer_bridge_gate.gates.*` | After global_ready | LayerBridgeGate routing |

### 7.3 Known Gradient Issues

1. **bridge.loss** (bridge.py:141): Uses `h.detach()` for the probe — gradient from bridge loss doesn't flow into the trunk. Only the probe parameters learn. This is intentional (prevents CE divergence).

2. **SemanticBridge.inject_layer** (bridge.py:100): Stream injection has gradient through `stream_proj`, but the stream itself is EMA-updated (detached). No cross-step BPTT through the stream.

3. **Private memory write** (mirror.py:494-512): All writes under `torch.no_grad()`. No gradient through memory updates. Memory is a pure buffer.

4. **MaturationController.update** (maturation.py:136): All updates under `torch.no_grad()`. Maturity is a pure buffer, not learnable.

5. **pred_loss** (stack.py:866-873): `F.mse_loss(pred_k, hp.detach())` — hp is DETACHED, so gradient only flows through pred_k (alpha_diag). The main loss path through hp is severed.

---

## 8. BOTTLENECKS & ISSUES

### 8.1 Dead Code Paths

1. **`ZeckendorfEmbedding`** (embedding.py:46): Legacy, replaced by `PartitionedEmbedding`. Not used in default config.

2. **`LmHead`** (embedding.py:115): Legacy Zeckendorf head. Replaced by `SigmoidCodedHead`. Only used if `head_mode='zeckendorf'`.

3. **`PartitionedHead`** (embedding.py:130): Alternative head, not used by default (`head_mode='sigmoid_coded'`).

4. **`vsa_prefix_scan`** (vsa_utils.py:90): Standalone function, now inlined in `WideBindBlock.forward` with chunked parallel scan. Dead code.

5. **`SpiralBind`** (bind.py:235): Not used in default config (bind_twist_mode='trajectory_spiral'). Only instantiated if `bind_mode == "spiral"`.

6. **`AdaptiveGateBundle`** (adaptive_gate.py:100): Generic bundle, only used as `self.hybrid_gate = AdaptiveGate(...)` inside mirror. The Bundle class is unused.

7. **`SpectrumGateBundle`** (spectrum_gate.py:136): Generic bundle, not used anywhere in the codebase.

8. **`CurriculumTracker`** (curriculum.py:13): Not imported or used in train.py or any other module.

9. **`AmpAdam`** (amp_optim.py:97): Custom optimizer for AMP codec head. Not used with `SigmoidCodedHead` (the default head).

10. **`_tau_intent_dev`** (stack.py:117): Kept for checkpoint compat but superseded by `tau_config.intent_alpha`. Still has its own Parameter but gradient is redundant.

### 8.2 Redundant Calculations

1. **`AdaptiveController.stats`** is called in the outer loop (stack.py:227) AND again inside the per-layer loop (stack.py:342). The outer call computes global averages; the inner call recomputes per-layer stats. Could cache per-layer results.

2. **`_knowledge_signal`** (stack.py:591): Called in `_adaptive_reasoning` at each reasoning step. It runs `self.lm_head(h)` which is a full forward through the head — expensive at 8 reasoning steps × 32 layers.

3. **Memory bank `forward`** (memory_bank.py:547): Runs L1/L2/L3 reads at EVERY token position, but writes only at sentence boundaries. The read is O(n_slots) per position — expensive for long sequences.

4. **`SigmoidCodedHead.forward`** (embedding.py:214): Computes full logits for ALL vocab tokens, then normalizes. `log_probs_for_target` (line 228) is the efficient path for training (only computes target log-probs).

5. **Spectral transform** (block.py:476-479): `h @ V_dct.T` then `result @ V_dct` — two matrix multiplications of (B,L,D) × (D,D). For D=4096, this is 4096² = 16M multiplications per layer per direction.

### 8.3 Performance Bottlenecks

1. **Prefix scan** (block.py:329-374): The VSA memory uses a 2-level chunked prefix scan. For L=128 and CHUNK=32, that's 4 chunks × 2 levels = 8 scan operations per layer per scale. With S=4 scales, that's 32 scan operations per layer. For 32 layers: 1024 prefix scans.

2. **DCT spectral transform** (block.py:476-479): Two D×D matrix multiplications per layer. For D=4096 and 32 layers: 32 × 2 × 4096² ≈ 1B FLOPs.

3. **Private memory cross-attention** (mirror.py:409-428): G×G attention per token (32×32=1024 attention scores). For B=2, L=128: 2 × 128 × 1024 = 262K sigmoid operations.

4. **Reasoning loop** (stack.py:648-754): Up to K=8 iterations, each involving ReasoningMemory (linear + attention) + ReasoningGate (3 linear projections) + _last_conf (full head forward). The head forward is the most expensive part.

5. **Triad re-circulation** (stack.py:569-587): Recursively calls the entire forward pass up to 3 times when confidence is low. Each re-circulation is a full forward pass. Bounded by `triad_max_passes=3`.

### 8.4 Architecture Concerns

1. **32 CollectiveConceptLayers**: When `collective_layer=True` and `collective_layer_idx=None`, a CollectiveConceptLayer is created for EVERY block. Each has S=8 concept slots with attention over K-space. This is 32 × 8 × K parameters just for concept management.

2. **Private memory per expert**: Each of 32 layers has G=32 experts, each with a (k,) private memory vector. Total: 32 × 32 × 32 = 32K private memory entries — small but spread thin.

3. **Dual τ parameters**: `_tau_l_dev` (alias to tau_config._tau_dev) and `_tau_intent_dev` coexist. The intent dev is no longer used for intent alpha (tau_config handles it), but is kept for checkpoint compat.

4. **Memory bank is default off**: `memory_bank=False` in config. When enabled, adds L1+L2+L3 with full attention at every position — significant compute overhead.

---

*Report generated from analysis of all Python files in the WideBind/EVA codebase.*
*Total estimated lines of code: ~10,000+ across core/ and scripts/.*
*Key architectural change: Unified TauConfig replaces scattered τ-parameters with a single source of truth.*
