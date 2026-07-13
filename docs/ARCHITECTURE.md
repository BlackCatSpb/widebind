# WideBind: Architecture & Method Description

> **⚠️ This document describes the 41M architecture (D=896, L=24, K=16, 2026-07 epoch).**
> **Current architecture (2026-07): 221M params, D=3584, L=32, K=32, G=32, bind_K=32.**
> **See `README.md` for the current spec. This file is preserved as historical reference.**
>
> Version: 2.0 (GroupedMLP era)
> Params: 41,246,336 (41.25M)
> Last updated: 2026-07-07

---

## Table of Contents

1. [Motivation & Design Philosophy](#1-motivation--design-philosophy)
2. [Architecture Overview](#2-architecture-overview)
3. [Component Deep Dives](#3-component-deep-dives)
   - 3.1 [Zeckendorf Embedding](#31-zeckendorf-embedding)
   - 3.2 [Pre-LN RMS Norm](#32-pre-ln-rms-norm)
   - 3.3 [Depthwise Conv1d](#33-depthwise-conv1d)
   - 3.4 [Bottleneck Bind (D → K → D)](#34-bottleneck-bind-d--k--d)
   - 3.5 [VSA Vector Memory](#35-vsa-vector-memory)
   - 3.6 [Cognitive Mirror](#36-cognitive-mirror)
   - 3.7 [Grouped MLP](#37-grouped-mlp)
   - 3.8 [DCT Spectral](#38-dct-spectral)
   - 3.9 [LM Head](#39-lm-head)
   - 3.10 [Global Cross-Layer State](#310-global-cross-layer-state)
4. [Training Loop](#4-training-loop)
5. [Checkpoint System & Reporting](#5-checkpoint-system--reporting)
6. [Evolution History](#6-evolution-history)
7. [Known Issues & Limitations](#7-known-issues--limitations)
8. [Hypothesis: Layer Specialization](#8-hypothesis-layer-specialization)
9. [Mathematical Appendix](#9-mathematical-appendix)

---

## 1. Motivation & Design Philosophy

### Why not Transformer?

The Transformer architecture has three structural constraints that are rarely questioned:

1. **O(L²) attention**: Each token attends to every other token. For L=128 this is 16,384 comparisons per head — tolerable. For L=8192 it is 67 million. The quadratic cost is baked into the architecture.

2. **Unbounded KV-cache**: Inference requires caching all key-value pairs for all previous tokens. This grows linearly with sequence length and number of layers. For 24 layers at D=896 and L=128K, this is 24 × 128K × 896 × 2 × 4 bytes ≈ 22 GB — more than most GPUs.

3. **Softmax bottleneck**: Attention weights are normalized via softmax, which produces a probability distribution. This forces the model to distribute attention among tokens, but there is evidence that the softmax operation itself limits representational capacity (the "softmax bottleneck" in Transformers).

WideBind replaces all three with:

- **O(L log L) prefix scan** — parallel associative scan over element-wise VSA memory
- **O(D) state per layer** — 336 KB total for 24 layers, independent of sequence length
- **No normalization bottleneck** — bilinear bind through bottleneck K=16 provides cross-dim mixing without probability distributions

### Design Constraints

1. **MX550 2 GB VRAM** — All architectural decisions constrained by this. 41.25M params in fp32 is ~165 MB for weights, ~330 MB for AdamW states, ~165 MB for gradients. Total ~660 MB, leaving room for activations and CUDA overhead in 2 GB.

2. **No softmax, no attention, no sigmoid gates** — Sigmoid is used only for decay and write gates in VSA memory (where bounded [0, 1] is required by design). The term "sigmoid gates" in the constraint refers to learned sigmoidal gating in the style of LSTM or Highway networks: WideBind does not use sigmoid to gate information flow between sublayers.

3. **Content-dependence through bilinear bind** — The primary mechanism for cross-dim mixing is the bottleneck bind (D → K → D bilinear), not element-wise operations. This is what distinguishes WideBind from pure VSA approaches that collapse beyond 4 layers.

---

## 2. Architecture Overview

### Data Flow

```
tokens: (B, L) — integer token IDs
  │
  ├── ZeckendorfEmbedding
  │     tokens → Fibonacci binary codes → Linear(D) → h: (B, L, D=896)
  │
  └── [WideBindBlock × 24]  (with residual connections)
        │
        ├── Pre-LN (RMS Norm)
        ├── Depthwise Conv1d (k=48, groups=D)
        ├── Bottleneck Bind (D → K=16 → bilinear → D)
        ├── VSA Memory (prefix scan + read)
        ├── Cognitive Mirror (4 paths in K-space)
        ├── h += enhanced (bind + mem_read + mirror)
        ├── DCT Spectral (h_dct * lambda_k)
        ├── Grouped MLP (8 groups × 112→896→112)
        └── h_out: (B, L, D)
  │
  ├── Final RMS Norm
  │
  └── LmHead
        h → Linear(K=23) → @ codes.T → logits: (B, L, 50000)
```

### Parameter Distribution (per layer)

| Component | Parameters | % of Layer |
|---|---|---|
| Grouped MLP | 1,606,528 | 93.6% |
| Bottleneck Bind | 43,024 | 2.5% |
| Cognitive Mirror | 43,025 | 2.5% |
| VSA Memory (gates + moment) | 8,064 | 0.5% |
| Depthwise Conv1d | 43,008 | 2.5% |
| Spectral lambda_k | 896 | 0.05% |
| **Total per layer** | **1,744,545** | 100% |

Embedding + LM Head: 41,216 params (0.1% of total).

### Total: 41,246,336 params (41.25M)

---

## 3. Component Deep Dives

### 3.1 Zeckendorf Embedding

#### What it does
Maps token IDs (0–49999) to D-dimensional vectors using Fibonacci Zeckendorf codes.

#### How it works
Each token is represented by its Zeckendorf representation: a binary code where no two consecutive bits are 1, using Fibonacci numbers as the base. For vocab=50000, this requires K=23 bits.

```
fib = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377,
       610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368]
```

Token 42 → 42 = 34 + 8 → binary: 00000000000000100010000 (bits at positions 7 and 4)
Token 49999 → binary: 11101000101001000001010

The binary code (K=23) is projected to D=896 via learned Linear(K, D):

```python
h = Linear(K, D) @ codes[tokens]
```

#### Why Zeckendorf?
1. **No learned token embeddings** — The code is deterministic, so the model cannot memorize token IDs by index. The projection must learn meaningful structure.
2. **Arithmetic structure** — Zeckendorf codes have a natural ordering and arithmetic: nearby tokens have related codes. Unlike one-hot embeddings, the projection receives structured input.
3. **Compact** — K=23 for 50000 tokens is efficient (one-hot would be 50000).

#### Related Work
Perkins (2023) proposed Fibonacci embeddings for transformers. WideBind uses a simplified version: fixed codes + learned projection, without the multiplicative interactions of the original.

---

### 3.2 Pre-LN RMS Norm

#### What it does
Normalizes the input to each block using Root Mean Square normalization (RMS Norm), without the mean-centering of LayerNorm.

#### Method

```
RMS(x) = sqrt(mean(x²) + ε)
h = x / RMS(x) * w
```

where `w` is a learned scale vector of dimension D, initialized to all ones.

#### Why RMS Norm over LayerNorm?
- LayerNorm subtracts the mean, then divides by std. RMS Norm only divides by RMS.
- For zero-mean activations, RMS Norm = LayerNorm without the mean computation.
- Saves the mean computation (D extra operations per token per layer).
- In practice, no quality difference for language models (Zhang & Sennrich, 2019).

#### Why pre-LN (before sublayers) instead of post-LN?
- Post-LN (original Transformer) causes training instability — the residual path is normalized, which amplifies gradients through the residual.
- Pre-LN normalizes the input to each sublayer, leaving the residual path unnormalized. This is the dominant practice (Xiong et al., 2020).

---

### 3.3 Depthwise Conv1d

#### What it does
Applies a depthwise 1D convolution along the sequence dimension, independently per channel (group = D means each of the 896 channels has its own 48-tap filter).

#### Method

```
h_perm = h.transpose(1, 2)      # (B, L, D) → (B, D, L)
h_conv = conv1d(h_perm)         # Conv1d(in=D, out=D, kernel=48, groups=D)
h_conv = h_conv.transpose(1, 2)  # (B, D, L) → (B, L, D)
h = h + h_conv
```

#### Stateful Conv
The convolution maintains a state buffer of the previous `kernel_size - 1` timesteps for each channel. This enables correct convolution across batch boundaries:

```python
if state is None:
    state = zeros(B, D, padding)  # 47 zeros per channel
h_perm = h.transpose(1, 2)
h_conv = conv(cat([state, h_perm], dim=-1))
h_conv = h_conv[..., :L].transpose(1, 2)
state_out = h_perm[:, :, -(padding):]
```

#### Why depthwise conv?
1. **Local context** — Bind and VSA memory are global (all tokens). Conv provides local n-gram patterns that bind/memory miss.
2. **Efficient** — Depthwise conv has D × kernel params (896 × 48 = 43,008) vs D² × kernel for a full conv (896² × 48 = 38.5M).
3. **Per-channel** — Each channel learns its own temporal filter. This is crucial because different channels represent different semantic features with different temporal dynamics.

#### Why kernel=48?
- 48 tokens at L=128 covers ~37% of the sequence. This is a hyperparameter balancing: too small (k=3) → only adjacent tokens; too large (k=128) → too many params for depthwise (114,688).
- Empirically, k=48 provided the best gradient flow in early experiments.

---

### 3.4 Bottleneck Bind (D → K → D)

#### What it does
Projects D-dimensional vectors down to K=16, applies a bilinear interaction in K-space, and projects back to D. This is the primary mechanism for cross-dim mixing.

#### Method

```
hp = h @ W_proj             # (B, L, D) → (B, L, K=16)
u = hp * w_u                # element-wise scaling
v = hp * w_v                # element-wise scaling
bind_out = (u * v) @ W_out  # (B, L, K) → (B, L, D)
```

Where:
- W_proj: D × K (896 × 16) — projection into bottleneck
- w_u, w_v: K-dimensional vectors — element-wise scaling of the projection
- W_out: K × D (16 × 896) — projection back to D

#### Why this specific form?

The bilinear bind `(u ⊙ v) @ W_out` is the key innovation over pure element-wise VSA.

**Element-wise VSA (diagonal Jacobian):**
```
mem[t] = decay · mem[t-1] + h[t]
```
Forward gradient: ∂mem[t]/∂h[t] = I (identity).
Jacobian is diagonal — each dimension evolves independently.
After 4 layers, gradients vanish because there is no cross-dim mixing: each channel operates in isolation, and the model cannot coordinate across dimensions.

**Bottleneck bind (full Jacobian):**
```
bind_out = (W_proj @ h) ⊙ w_u ⊙ w_v @ W_out
```
Forward gradient: ∂bind_out/∂h = (diag(w_u ⊙ w_v @ W_out @ h) + ...) @ W_proj^T
The Jacobian includes W_proj (D × K) and W_out (K × D) — these are dense matrices, not diagonal. Gradients flow through all D×K paths, providing full cross-dim mixing.

In practice: element-wise VSA dies after 4 layers (grad/param < 1e-4). Bottleneck bind maintains grad/param > 0.4 through all 24 layers at init.

#### Why K=16?

| K | Params (bind) | Grad/param at init | VRAM |
|---|---|---|---|
| 4 | 7,176 | 0.15 | lowest |
| 8 | 14,344 | 0.28 | low |
| **16** | **28,680** | **0.42** | **low** |
| 32 | 57,352 | 0.51 | low |
| 128 | 229,128 | 0.63 | moderate |
| 896 | 1,605,632 | 0.89 | high |

K=16 is the sweet spot: sufficient gradient mixing (>0.4) at minimal param cost (28K vs 1.6M for full projection). The eff_rank of W_proj is typically 12-14/16 at step 1000, confirming the bottleneck is well-utilized.

#### Init

```
proj_std = 1.0 / (D * K) ** 0.25
W_proj = randn(D, K) * proj_std   # std ≈ 0.28
w_u = randn(K) * 1.0               # std=1.0 (critical!)
w_v = randn(K) * 1.0               # std=1.0 (critical!)
W_out = randn(K, D) * proj_std     # std ≈ 0.28
```

**Why std=1.0 for w_u and w_v?** The product w_u * w_v * W_out scales as std³. At std=0.02 (default PyTorch Linear init), the expected gradient flowing through the bind is ∝ 0.02³ = 8e-6, which is effectively zero. At std=1.0, the gradient flow is ∝ 1.0³ = 1.0 — a factor of 125,000× more. This is not a minor tuning; it is the difference between dead and alive at initialization.

---

### 3.5 VSA Vector Memory

#### What it does
Maintains a per-layer vector memory over the sequence using an element-wise recurrence:

```
mem[t] = decay[t] · mem[t-1] + i_gate[t] · h[t]
```

#### Method

**Gates:**

```
decay[t]  = sigmoid(h[t] · w_d + b_d)    ∈ (0, 1)  — how much to forget
i_gate[t] = sigmoid(h[t] · w_i + b_i)    ∈ (0, 1)  — how much to write
```

**Full memory read:**

```
mem_all = prefix_scan(decay, h * i_gate)    # (B, L, D)
mem_read = mem_all * w_q                     # learned query
```

**First moment correction:**

```
mu_all = prefix_scan(decay, h * i_gate * w_k_mu)
mu_read = mu_all * w_q_mu
mem_read = mem_read + mu_read * w_mu_mem
```

#### Associative Parallel Prefix Scan

The recurrence `mem[t] = a[t] · mem[t-1] + b[t]` is not sequential! It is an associative operation that can be parallelized using a tree scan:

```
(a[t], b[t]) ⊗ (a[s], b[s]) = (a[t] · a[s], a[s] · b[t] + b[s])
```

The algorithm:
1. Initialize (a_curr, b_curr) = (decay, h * i_gate) for all t
2. For step = 1, 2, 4, 8, ... while step < L:
   a_curr[t] = a_curr[t] · a_curr[t - step]  (for t >= step)
   b_curr[t] = b_curr[t - step] · a_curr[t] + b_curr[t]
3. Result: mem[t] = b_curr[t]

Complexity: O(L log L) parallel steps, O(L) total operations. Each step is a fused element-wise multiply-add, fully parallelizable on GPU.

#### Decay Timescale τ

The effective timescale τ is determined by the decay bias b_d:

```
τ = -1 / ln(sigmoid(b_d))
  = -1 / ln(1 / (1 + exp(-b_d)))
```

| b_d | sigmoid(b_d) | τ | Effect |
|---|---|---|---|
| 0.0 | 0.50 | 1.4 | Forgets 50% per step. After L=128: retention 2⁻¹²⁸ ≈ 0 |
| 2.0 | 0.88 | 7.8 | Retains ~1M tokens...wait, no. Retention after 128 steps: 0.88¹²⁸ ≈ 1e-7 |
| 5.0 | 0.9933 | **149.8** | Retention after 128 steps: 0.9933¹²⁸ ≈ 0.42 (42% of first token) |

At b_d=5.0 (τ≈150):
- After 1 step: 99.3% retained
- After 50 steps: 71% retained
- After 128 steps: 42% retained
- After 300 steps: 13% retained

This means the model sees the entire 128-token context with moderate decay, and can carry information across batches. This is critical for training stability: with τ≈2, the model effectively sees 2-token windows regardless of seq_len.

#### Write Gate Calibration

`b_i = -3.0` gives i_gate ≈ sigmoid(-3.0) = 0.047 at initialization. This means:
- At step 0, only 4.7% of h[t] is written to memory per step.
- With τ≈150, the total memory content after processing L=128 tokens:
  mem[L] ≈ Σ(decay^t * i_gate * h[L-t]) = i_gate * h / (1 - decay) = 0.047 / 0.0067 ≈ 7.0× h
- A mem_all std of ~5.8 is observed at init, vs h std of 1.0.

**Why not exp(i_gate)?** In earlier versions, i_gate was computed via `exp(h·w_i + b_i)`. With τ≈150, mem_all grew exponentially: 640 std at step 0 → NaN. Sigmoid bounds the gate to [0, 1] and keeps the memory stable.

**Why not learn b_i and b_d?** At step 1000, b_i=-3.000 and b_d=5.000 (no change from init). The gradient for these biases is proportional to the gate output: ∂L/∂b_i = ∂L/∂i_gate · i_gate · (1 - i_gate). At i_gate ≈ 0.047, the factor i_gate · (1 - i_gate) ≈ 0.045, which is small but non-zero. The problem is that the gradient for b_i is dominated by the gradient for w_i (h · ∂L/∂i_gate · i_gate · (1 - i_gate)), and h has std≈1 giving it a larger effective gradient. The biases are frozen not because of architecture but because the optimization landscape favors pulling w_i before b_i.

---

### 3.6 Cognitive Mirror

#### What it does
Computes a bounded, per-dimension correction to h based on four self-consistency signals. The mirror detects when h deviates from expected patterns and provides a corrective push.

#### Method

All four paths operate in the **K=16 bottleneck space** (same K as the bind):

```python
hp = h @ W_proj  # (B, L, D) → (B, L, K) — project to K-space
```

**1. Temporal path (local memory deviation):**
```
mem_centroid = mean(mem_all, dim=1)  # average memory over L
temp_k = (hp - mem_centroid @ W_proj) * w_temp
```
Detects: "Is my current h different from what the VSA memory has accumulated over time?" A large deviation means the current input is unexpected given the context.

**2. Global path (cross-layer deviation):**
```
gs_k = global_state @ W_proj
temp_k += (hp - gs_k) * w_global
```
Detects: "Is my current h different from the aggregated memory of all previous layers?" The global_state is a running EMA (α=0.95) of each layer's final memory centroid:

```python
global_state = 0.95 * global_state + 0.05 * mem_centroid
```

**3. Smoothness path (local coherence):**
```
hp_perm = hp.transpose(1, 2)  # (B, K, L)
hp_smooth = conv1d_k3(hp_perm)  # depthwise Conv1d kernel=3 in K-space
smooth_k = hp - hp_smooth.transpose(1, 2)
```
Detects: "Is h[t] predictable from h[t-1] and h[t+1]?" Sharp transitions (h[t] far from local average) get a correction.

**4. Symmetry path (bilinear self-consistency):**
```
hp_prev = shift_right(hp, 1)  # hp[t-1]
sym_k = (hp * w_sym_u) * (hp_prev * w_sym_v)
```
Detects: "Is the transition from h[t-1] to h[t] consistent?" This is a bilinear form: (h[t] · u) · (h[t-1] · v) captures the similarity between consecutive timesteps in learned directions u, v.

**Combination and output:**

```python
delta = temp_k + smooth_k + sym_k      # sum all deviations
delta = rms_norm(delta, K)              # normalize in K-space
mirror = tanh(delta @ W_out)            # K → D, bounded to [-1, 1]
mirror = mirror * exp(log_scale)        # per-dim learned amplitude
```

Where log_scale is a D-dimensional vector initialized to 0 (so mirror = tanh(...) × 1.0 at init).

#### Why this design?

| Property | Old Mirror | Cognitive Mirror | Benefit |
|---|---|---|---|
| Output range | Unbounded (max=5.05) | tanh → [-1, 1] | No explosion; mirror cannot exceed residual |
| Gate type | Scalar sigmoid | Per-dim exp(log_scale) | Each of 896 dims has its own learned amplitude |
| Gate gradient | σ(1-σ) ≈ 0.02 at σ=0.5 | exp(log_scale) ≈ 1.0 with full grad | 50× stronger gradient |
| Paths | 1 (temporal) | 4 (temporal+global+smooth+symmetry) | Richer correction signals |
| K-space | Same as bind | Same as bind | Reuses same projection |

#### Why tanh(W_out)?
- tanh bounds every dimension of the output to [-1, 1]. This guarantees the mirror correction is never larger than the scale factor exp(log_scale).
- The scale factor (init=1.0) is per-dimension, learned via gradient descent with a clean gradient: ∂mirror/∂log_scale = mirror (not through a sigmoid).
- If the model determines that certain dimensions should not be corrected, it can drive log_scale → -∞ (exp → 0).

---

### 3.7 Grouped MLP

#### What it does
The primary parameter sink (93.6% of all parameters). Replaces the standard D→D→D MLP with a grouped structure where D=896 is split into 8 independent groups of 112, each with 8× internal expansion.

#### Method

```python
# Input: h (B, L, D=896)

# RMS norm (as in flat MLP)
h = rms_norm(h, (D,), norm_w)

# Reshape into groups
h = h.reshape(B, L, G=8, d=112)

# Per-group MLP (independent!)
for g in range(G):
    h_up = silu(h_g @ W_up[g])     # 112 → 896 (8× expansion)
    h_down = h_up @ W_down[g]      # 896 → 112

# Reshape back
h = h.reshape(B, L, D)
```

Implemented as a single einsum:

```python
h = silu(einsum('blgd,gdf->blgf', h, W_up))       # (B, L, 8, 896)
h = einsum('blgf,gfd->blgd', h, W_down)            # (B, L, 8, 112)
h = h.reshape(B, L, D)
```

#### Parameter Math

| Configuration | Formula | Params | Expansion |
|---|---|---|---|
| Flat D→D | W_up + W_down | 896² + 896² = 1,605,632 | 1× |
| **Grouped G=8, expand=8** | **G×(d×expand×d + expand×d×d)** | **8×(112×896 + 896×112) = 1,605,632** | **8× per group** |
| Grouped G=4, expand=4 | 4×(224×448 + 448×224) | 802,816 | 4× per group |
| Grouped G=8, expand=4 | 8×(112×448 + 448×112) | 802,816 | 4× per group |

For G=8, expand=8: total params = 1,605,632 — exactly the same as flat 896→896→896.

#### Why does grouping help?

The standard D→D→D MLP has rank(W_down · SiLU(·) · W_up) ≤ D = 896. But the effective rank in practice is much lower: the old flat MLP had eff_rank ≈ 4 at L23 after 2000 steps, meaning only 4 out of 896 dimensions carried signal.

With grouped MLP:

1. **Each group processes 112 dims independently** — the SiLU activation operates on 896 neurons per group (not 896 total). This means 7,168 active neurons total across all groups at the intermediate layer, vs 896 in the flat case.

2. **Independent optimization** — Each group has its own W_up and W_down (200,704 params each). The gradient for group g is isolated to its 112-dim subspace. If group 3's features are useful for syntax and group 7's for semantics, they don't compete for the same params.

3. **Natural feature grouping** — At step 1000, per-group eff_rank is 60-63/112 for L0-L17 (55% utilization), vs 4/896 (0.5%) for flat MLP. All 8 groups are equally active (std across groups < 5%).

4. **Graceful degradation at deep layers** — At L23, per-group eff_rank is 10-13/112 (10% utilization). This is still 22× better than flat MLP's 4/896 (0.5%). And it is uniform across groups: no single group dies completely.

#### Why not increase D or bottleneck?

Increasing D to, say, 1792 would give 4× params (4 × 41.25M ≈ 165M) — exceeds MX550 2GB VRAM. Increasing bottleneck to 1792 (2× expansion) with D=896 would give 2 × 896 × 1792 × 24 ≈ 77M MLP params alone + 41M base ≈ 118M total — still too much for 2GB.

GroupedMLP gives **the effect of expansion without the cost of expansion** by organizing the same number of params into independent subspaces.

---

### 3.8 DCT Spectral

#### What it does
Applies a learned per-dimension frequency scaling to h using the Discrete Cosine Transform (DCT-II) basis.

#### Method

```python
V = dct_basis(D=896)         # DCT-II basis matrix (896 × 896), orthogonal

h_dct = h @ V.T              # (B, L, 896) → DCT coefficients
h = h + (h_dct * lambda_k) @ V  # scale by lambda_k, inverse DCT
```

**DCT basis construction (DCT-II):**

```
V[n, k] = sqrt(2/D) * cos(π/D * (k + 0.5) * n)  for n > 0
V[0, k] = sqrt(1/D)                                for n = 0
```

The basis is orthonormal: V @ Vᵀ = I. Important: `V[0,:] /= sqrt(2)` (for n=0), NOT `V[:,0] /= sqrt(2)` (which would corrupt the first basis vector's dimension). This was a bug fixed in an earlier version (see §6.2).

**lambda_k** is a learned D-dimensional vector initialized as:

```python
lambda_k = 0.5 + layer_idx / (n_layers - 1)
```

For 24 layers:
- L0: lambda_k = 0.5 (all frequencies attenuated by half)
- L11: lambda_k = 1.0 (neutral — identity-like)
- L23: lambda_k = 1.5 (all frequencies amplified by 1.5×)

#### What lambda_k learns

The spectral block effectively learns a frequency filter: each of the 896 DCT basis components is scaled by its lambda_k value. Since lambda_k is per-dimension, the model can learn to emphasize certain frequency components and suppress others.

At step 1000 (GroupedMLP):
- lambda_k has per-dim std of 0.002 (L0) to 0.015 (L23)
- The variance is small but non-zero — the model is beginning to differentiate dimensions
- The mean increases linearly from 0.50 to 1.50 as expected

#### Layer Specialization Hypothesis

The monotonic growth of lambda_k across layers (0.5→1.5) supports a hierarchical processing hypothesis:

| Layer | lambda_k | Spectral behavior | Likely role |
|---|---|---|---|
| L0-L4 | 0.50-0.67 | All frequencies attenuated | Context integration — building stable representation |
| L5-L12 | 0.72-0.98 | Near-identity (neutral) | Syntax — all frequencies equally important |
| L13-L18 | 1.02-1.28 | Gentle amplification | Lexical — begins emphasizing patterns |
| L19-L22 | 1.33-1.46 | Strong amplification | Detail — emphasizing discriminative features |
| L23 | 1.50 | Maximum amplification | Compression for LM head — emphasizing everything possible |

The DCT basis components with the highest lambda_k at L23 correspond to the most discriminative directions for the LM head classifier.

#### Connection to GroupedMLP

In the flat MLP epoch, lambda_k had near-zero per-dim variance (std ≈ 0.0002 at all layers), meaning the spectral block was effectively a single learned scalar per layer. With GroupedMLP, per-dim std is 7-75× larger (0.002-0.015), suggesting the model has more capacity to differentiate frequency channels.

---

### 3.9 LM Head

#### What it does
Projects the final D-dimensional representation to vocabulary logits.

#### Method

```python
h_final = rms_norm(h, (D,), final_norm_w)       # final RMS norm
z = Linear(D, K=23) @ h_final                    # D → K (same K as embedding)
logits = z @ codes.T                              # K → 50000 via Zeckendorf codes
```

The codes are the same Fibonacci Zeckendorf binary codes used in the embedding (see §3.1). The projection D → K is similar to the embedding's K → D but not tied.

#### Why not tie embeddings?
Tied embeddings (input projection == output projection) save parameters but reduce expressivity. For a model this small (41.25M), the 41,216 params saved by tying (0.1% of total) are negligible. Untied projections allow the model to learn different representations for input and output.

#### Why Zeckendorf codes in LM head?
Standard LM heads use a learned embedding table: h @ Embedding.T gives logits. This is a massive matrix: 896 × 50000 = 44.8M params — larger than the entire model! Zeckendorf codes reduce this to 896 × 23 = 20,608 by sharing the Fibonacci code structure. The downside is that the logits are constrained to the Zeckendorf code space, but in practice this works well (loss 1.44 at step 1000, comparable to learned embeddings).

---

### 3.10 Global Cross-Layer State

#### What it does
Propagates a running estimate of the "self-model" (aggregated layer memories) from early layers to later layers.

#### Method

```python
global_state = zeros(1, 1, D)  # initialized at start of forward pass

for layer in layers:
    h, state_out = layer(h, state, global_state)
    mem_out = state_out[0]                   # (B, D) — layer's final memory state
    mem_avg = mean(mem_out, dim=0)            # average over batch
    global_state = 0.95 * global_state + 0.05 * mem_avg
```

The global state is fed to each layer's Cognitive Mirror as the "global" path input (see §3.6).

#### Design rationale

- **No extra parameters** — The global state is a running EMA, not a learned projection. The 0.95/0.05 ratio is fixed.
- **Leaky aggregation** — α=0.95 gives a timescale of ~20 layers, meaning layers early in the stack influence later layers but are gradually forgotten.
- **Layer 0 sees pure local** — Before the first layer, global_state is all zeros, so the global path contributes nothing. Layer 0's mirror is purely local.
- **Layer 23 sees full context** — By the last layer, global_state contains contributions from all 23 previous layers, plus its own local signals.

#### Why this isn't just attention-in-disguise

Attention computes pairwise interactions: each token attends to every other token. The global state is a single D-dimensional vector representing the centroid of all layer memories. It is not pairwise — it is a pooled representation. The mirror then computes h - global_state, which is a mean deviation, not a weighted sum. This is O(D) per layer (336 KB total for 24 layers), not O(D²) or O(L²).

---

## 4. Training Loop

### Data Loading

Token streams are stored in `.bin` files as uint16 numpy arrays. The trainer loads all streams, then cycles through them sequentially:

```python
streams = [load(f) for f in glob('token_stream_*.bin')]
# load: np.fromfile(path, dtype=np.uint16) → tensor(int64)

# Per stream: offset tracks current position; wraps to 0 when exhausted
x, y, offset = get_batch(stream, seq_len, batch_size, offset)
```

### Optimizer

```python
optimizer = torch.optim.AdamW(
    param_groups, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.01
)
```

Parameter groups:
- **Decay:** All 2D parameters (matrices) — weight_decay = 0.01
- **No decay:** All 1D parameters (biases, gate vectors, scale vectors) — weight_decay = 0

### Learning Rate Schedule

```
if step < warmup:        LR = base_lr * step / warmup          (linear warmup)
if step >= warmup:       LR = 0.5 * (1 + cos(π * progress))    (cosine decay)
```

Implemented as LambdaLR: the scheduler returns a multiplier that is applied to the base LR.

### Gradient Flow

```
W_up grad norm: 4.04  (W_up: 8×112×896)
W_down grad norm: 4.11  (W_down: 8×896×112)
bind W_proj grad norm: 0.89
bind w_u grad norm: 1.12
bind w_v grad norm: 1.05
mirror log_scale grad norm: 0.02
```

All grads non-zero. The bind and MLP receive healthy gradient flow. Mirror log_scale has small gradient (mirror is not yet scaling corrections).

### Batch Size & Sequence Length

B=2, L=128. This is the maximum that fits MX550 2GB at 41.25M params. The effective batch size is 2 × 128 = 256 tokens per step.

---

## 5. Checkpoint System & Reporting

### Save Format

```python
torch.save({
    'step': step,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'best_val_loss': best_val_loss,
    'cfg': cfg,
}, save_path)
```

The config is saved inside each checkpoint, making checkpoints self-contained. `add_safe_globals([WideBindConfig])` enables `weights_only=True` loading.

### Save Triggers

| Trigger | Filename | Frequency |
|---|---|---|
| Time interval | `step_{N}.pt` | Every 5000 steps |
| New best eval | `best.pt` | When val_loss improves |
| Ctrl+C | `interrupt_step_{N}.pt` | On KeyboardInterrupt |

### Auto HTML Report

Every saved checkpoint triggers `analyze_checkpoint.py`, which generates an HTML report with:

- **Overview:** Total params, weight mean/std/min/max, output std from forward pass
- **Weight norms by group:** Top-3 params for MLP/Bind/Gates/Spectral/Mirror with norms and stds
- **Layer analysis (table):**
  - eff_rank(MLP) — per-group average SVD-based effective rank
  - ||W_mlp|| — Frobenius norm of MLP weights
  - eff_rank(bind) — SVD-based effective rank of W_proj
  - ||W_bind|| — Frobenius norm of bind weights
  - log_scale μ and σ — mean and std of mirror scale vector
- **Memory gates:** Mean of w_i, w_d, w_q, b_i, b_d, w_mem2v across all layers
- **Gradient stats (from optimizer state):** mean_abs_grad and RMS grad from AdamW exp_avg

### Compatibility

With the transition from flat MLP to GroupedMLP, checkpoint key names changed from `mlp_up.weight` / `mlp_down.weight` to `mlp.W_up` / `mlp.W_down`. Loading old checkpoints uses `strict=False`:

```python
missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
# missing = GroupedMLP keys (fresh init)
# unexpected = old MLP keys (ignored)
```

---

## 6. Evolution History

### 6.1 epoch 1: Pure VSA (died after 4 layers)

Initial architecture had element-wise VSA memory without the bottleneck bind. The Jacobian was diagonal, and gradients vanished after 4 layers regardless of width or depth. Loss stalled at ~8.

**Fix:** Introduced bottleneck bind (D → K=16 → D). The bilinear projection provides full D×K gradient mixing. Grad/param went from <1e-4 to >0.4.

### 6.2 epoch 2: DCT basis bug

The DCT basis was constructed incorrectly:

```python
# BUG: basis[0, :] /= sqrt(2) — but 0 is a row index
# Correct: basis[0, :] /= sqrt(2) — for the first ROW (n=0)
# BUT the original code had: basis[:, 0] /= sqrt(2) — dividing the first COLUMN

# BUG code:
basis = cos(v * pi / n)  # v has shape (n, n)
basis[:, 0] = basis[:, 0] / sqrt(2)  # WRONG! modifies first basis vector's dimension 0

# FIXED code:
basis[0, :] = basis[0, :] / sqrt(2)  # correct: first basis vector's all dimensions
```

The bug caused all DCT basis vectors to be non-orthogonal. Orthogonality error went from 0.001 (correct) to 1.117 (buggy). The spectral block was effectively injecting noise. After fix, orthogonality returned to 0.001.

### 6.3 epoch 3: τ≈2 (too short memory)

Initial decay bias `b_d=0` gave τ≈2 (sigmoid(0) = 0.5, τ = -1/ln(0.5) ≈ 1.4). After 128 steps, the first token's retention was 0.5¹²⁸ ≈ 2.9e-39 — effectively zero after ~10 tokens. The model was operating as a bigram model regardless of seq_len.

**Fix:** b_d = 5.0 → τ ≈ 150. Retention after 128 steps: 42%. The model now sees the full 128-token context.

### 6.4 epoch 4: exp(i_gate) → NaN

With τ≈150 and `i_gate = exp(h·w_i + b_i)`, the memory grew exponentially:

```
mem[t] = decay · mem[t-1] + exp(h·w_i + b_i) · h[t]
```

At init, mem_all had std ≈ 640 (vs h std = 1). This caused immediate NaN.

**Fix:** Changed to sigmoid(i_gate) with b_i=-3.0. i_gate ≈ 0.047 at init. mem_all std ≈ 5.8 (healthy).

### 6.5 epoch 5: scalar gate for Mirror → barely learned

The original mirror had a single scalar sigmoid gate (gradient ≈ σ(1-σ) ≈ 0.02 when σ=0.5). The mirror correction was unbounded (max observed: 5.05). L23 collapse was severe (eff_rank=4).

**Fix:** Replaced with CognitiveMirror:
- tanh(W_out) → bounded to [-1, 1]
- Per-dim exp(log_scale) → full D=896 gradient for each scale
- 4 paths instead of 1

### 6.6 epoch 6: Flat MLP → GroupedMLP

The old flat MLP (896→896→896) had eff_rank=4 at L23 after 2000 steps. MLP weights barely moved (std=0.033 at step 2000 vs init).

**Fix:** GroupedMLP (G=8, d=112, expand=8×). Same params (1.6M/layer) but structured into 8 independent groups, each with 8× internal expansion. Per-group eff_rank at L23: 12/112 (10% utilization) vs 4/896 (0.5% utilization) — 22× improvement.

### Standard init fix

All element-wise gate and memory vectors are initialized with std=1.0 (not 0.02, the default PyTorch Linear init). The product `w_u * w_v * W_out` scales as std³. At std=0.02, the gradient was 8e-6. At std=1.0, the gradient is 1.0. This is a factor of 125,000× more gradient flow.

---

## 7. Known Issues & Limitations

### 7.1 Last-layer collapse

**Symptom:** At L23 (the layer before LM head), eff_rank drops from ~62/112 (L0-L17) to ~12/112 (L23). This is a structural compression: the LM head (896→23→50000) projects to K=23, then to vocab. The last layer must produce representations that the LM head can classify, and this compression forces rank reduction.

**Mitigations tried:**
- CognitiveMirror (4 paths) — no effect on last-layer collapse
- Global cross-layer state — no effect
- GroupedMLP (8× expansion) — improved from 0.5% to 10% utilization — best so far

**Hypothesis:** The collapse is inherent to the D→K→vocab head structure. The LM head has only K=23 intermediate dimensions. The last layer must align its 896-dim representation to 23 meaningful directions, which naturally compresses rank.

**Potential fix:** Replace or augment the LM head. Options:
- Learned embedding head (896→50000) — 44.8M extra params, exceeds VRAM
- Multi-head LM head: split 896 into 8 heads, each predicting a subset of vocab
- Hierarchical softmax or class-based prediction

### 7.2 Frozen gates (b_i, b_d)

**Symptom:** b_i and b_d stay at their initial values (-3.0 and 5.0) through step 1000. The gate biases are in a gradient local minimum.

**Analysis:** The gradient for bias b is ∂L/∂b = ∂L/∂gate · gate · (1-gate). At sigmoid(-3) ≈ 0.047, the factor gate·(1-gate) ≈ 0.045. The gradient for weight w is ∂L/∂gate · gate · (1-gate) · h[t], which includes an additional factor h[t] (std≈1). Both are similar magnitude. The issue may be that the optimal gate values (-3.0 for write, 5.0 for decay) are already near-optimal for the current learned weights, so the gradient doesn't push them away.

**Potential fix:** Add a small gradient noise or annealing schedule for biases. Or accept the frozen state as evidence that the initialization was well-calibrated.

### 7.3 Mirror log_scale frozen at 0

**Symptom:** log_scale ≈ 0 (exp=1) for all 24 layers at step 1000. The mirror paths are active (all 4 paths produce non-zero signals), but the per-dim output scaling is uniform (identity).

**Analysis:** The gradient for log_scale is ∂L/∂log_scale = ∂L/∂mirror · mirror. At step 1000, the model hasn't yet learned to trust or distrust the mirror correction. The mirror output (tanh of summed deviations) is non-zero but small. As training progresses, the gradient should accumulate and push log_scale away from 0.

**Timeline:** In the flat MLP run, log_scale started moving at step ~5000. Expected similar timeline for GroupedMLP.

### 7.4 Compute constraint

**Symptom:** ~167 tok/s on MX550 at B=2, L=128, fp32. At this rate, 500K steps × 256 tok/step = 128M tokens seen in a full run. The full dataset is 2.86B tokens, so the model will see ~4.5% of the data.

**Impact:** The model is heavily under-trained. Each token is seen ~once. Convergence to competitive perplexity would require either:
- More steps (longer training time)
- Larger batch size (more VRAM)
- AMP (fp16/mixed precision — risky with the bilinear bind whose product w_u·w_v·W_out can overflow)

---

## 8. Hypothesis: Layer Specialization

Based on the monotonic growth of lambda_k (0.5 → 1.5) and the eff_rank profiles across layers:

### Layer stack as frequency filter bank

```
L0 ─── L4 ─── L8 ─── L12 ─── L16 ─── L20 ─── L23
0.5    0.67   0.80   0.93    1.15    1.37    1.50
↓λ     →λ     →λ     →λ      →λ      →λ      →λ
Low freq ───→ Mid ───→ High ───→ Detail ───→ Compress
(Context)   (Syntax) (Lexical) (Pragmatic) (LM head)
```

### Evidence

1. **lambda_k monotonicity** — Confirmed in two separate runs (flat MLP and GroupedMLP). The growth is linear with layer index, suggesting a hardwired bias toward frequency amplification in deeper layers.

2. **eff_rank plateau (L0-L17)** — In GroupedMLP, eff_rank stays at ~55-62/112 for the first 18 layers. This suggests all these layers are operating in the same "processing regime" — they are learning features of similar complexity. The drop starts at L18, which corresponds to lambda_k > 1.2.

3. **Per-group uniformity** — All 8 groups in GroupedMLP have similar eff_rank at every layer. At L23, all 8 groups converge to eff_rank 10-13. This is not random collapse of a few groups; it is systematic compression.

### Predictions (to test at step 5000, 10000, 50000)

1. L0-L4 lambda_k will remain < 0.7 (low frequencies encode context, which is stable)
2. L5-L12 lambda_k will develop larger per-dim variance (syntax requires dimension-specific frequency sensitivity)
3. L13-L18 per-group eff_rank will diverge (groups specialize to different lexical categories)
4. L19-L22 lambda_k std will grow fastest (detail refinement requires most frequency differentiation)
5. L23 eff_rank will remain compressed regardless of training length (structural invariant)

---

## 10. AdaptiveController: Fully Self-Tuning Hyperparameters

The AdaptiveController eliminates all manual hyperparameter tuning for VSA memory
gates, making the architecture fully self-regulating based on cognitive mirror state.

### Two Fundamental Signals

```
exploration = min(1, |mirror| / 0.3)
    How much correction is the mirror applying.
    High → model is actively adjusting, needs aggressive learning.
    Low → model is stable, needs conservative parameters.

differentiation = min(1, var(log_scale) / 0.1)
    How specialized has the mirror become (per-dim scaling).
    High → mirror has learned which dims to trust/suppress.
    Low → mirror hasn't differentiated, still exploring.
```

### Adapted Parameters

| Parameter | Range | Signal | Math |
|-----------|-------|--------|------|
| `b_d` (decay bias) | [3.0, 6.0] → τ ≈ [20, 400] | exploration | `6.0 - expl × 3.0` |
| `b_i` (write bias) | [-5.0, -1.0] → i_gate ≈ [0.007, 0.269] | exploration | `-5.0 + expl × 4.0` |
| `w_mem2v` scale | [0.5, 1.0] | differentiation | `1.0 - diff × 0.5` |
| EMA α (global state) | [0.90, 0.99] | differentiation | `0.90 + diff × 0.09` |
| Noise scale (gates) | [0.001, 0.05] | differentiation | `0.05 - diff × 0.049` |

### Intuition

- **High exploration** (mirror making large corrections): memory should be short
  (low b_d/tau) and write-heavy (high b_i/i_gate) to capture the corrections.

- **High differentiation** (mirror has specialized per dimension): memory should
  yield to the mirror (low w_mem2v_scale) and update its self-model slowly
  (high EMA alpha).

### Gate Bias Parameterization

Gate biases are excluded from the optimizer (bypassed in `param_groups`) and set
directly via `bn.Parameter.fill_()` before each forward pass. This avoids gradient
conflict between adaptive control and learned updates.

### Integration Points

```
WideBindStack.forward():
    1. Compute expl, diff from all layers' mirrors
    2. Fill_ all layer.b_d, layer.b_i with adaptive values
    3. Pass mem2v_scale to each WideBindBlock
    4. Use adaptive ema_alpha for global state aggregation

WideBindBlock.forward():
    1. Accept mem2v_scale parameter
    2. Apply: enhanced = bind_out + mem_read × w_mem2v × mem2v_scale + mirror
```


## 9. Mathematical Appendix

### A. Prefix Scan Associativity Proof

The recurrence mem[t] = a[t] · mem[t-1] + b[t] can be written as:

```
(mem[t], 1) = (a[t], b[t]) ⊗ (mem[t-1], 1)
            = (a[t] · mem[t-1] + b[t], 1)
```

The operator ⊗ on pairs (a, b) is associative:

((a₁, b₁) ⊗ (a₂, b₂)) ⊗ (a₃, b₃) 
= (a₁ · a₂, a₂ · b₁ + b₂) ⊗ (a₃, b₃)
= (a₁ · a₂ · a₃, a₃ · (a₂ · b₁ + b₂) + b₃)
= (a₁ · a₂ · a₃, a₂ · a₃ · b₁ + a₃ · b₂ + b₃)

= (a₁, b₁) ⊗ ((a₂, b₂) ⊗ (a₃, b₃))
= (a₁, b₁) ⊗ (a₂ · a₃, a₃ · b₂ + b₃)
= (a₁ · a₂ · a₃, a₂ · a₃ · b₁ + a₃ · b₂ + b₃)

Same result. Associativity holds, enabling the parallel tree scan.

### B. Why K=16 bottleneck works

Consider a single layer: h ∈ ℝᴰ, W_proj ∈ ℝᴰˣᴷ, W_out ∈ ℝᴷˣᴰ.

```
forward: bind = (W_projᵀ h ⊙ w_u ⊙ w_v)ᵀ W_out
               = hᵀ W_proj · diag(w_u) · diag(w_v) · W_out
               = hᵀ M
```

Where M = W_proj · diag(w_u) · diag(w_v) · W_out ∈ ℝᴰˣᴰ.

The rank of M is at most K (since the intermediate representation is K-dimensional). During backpropagation:

```
∂bind/∂h = Mᵀ  (full D×D matrix, rank K)
∂bind/∂W_proj = h ⊗ (diag(w_u) · diag(w_v) · W_out)  (outer product)
```

The gradient flows through all D×D paths via the rank-K reconstruction. Each element of h influences all D elements of bind through the K-dimensional bottleneck. This is in contrast to element-wise VSA where ∂hᵢ/∂memⱼ = 0 for i ≠ j.

### C. GroupedMLP vs Flat MLP: Effective Capacity

**Flat MLP:** D → D → D

```
h_mlp = W_down · SiLU(W_up · h)
```

rank(h_mlp) ≤ D. The SiLU activation does not increase rank — it only provides element-wise nonlinearity. The effective capacity is bounded by the rank of W_down · W_up ≤ D.

**GroupedMLP:** D → G × (d → expand·d → d) → D

```
h_g = h_g  (per group g of size d)
h_g_up = SiLU(W_up_g · h_g)  ∈ ℝᵉˣᵈ
h_g_down = W_down_g · h_g_up  ∈ ℝᵈ
```

Per group: rank(h_g_down) ≤ d. Total rank ≤ G · d = D. Same bound.

**BUT:** The intermediate representation is G × expand·d dims per timestep, which is expand · D = 8 × 896 = 7,168 total intermediate dims (vs 896 in flat MLP). These 7,168 activations pass through 8 independent SiLU functions, each operating on its own 896-dim subspace. The nonlinearity is applied 8× more, providing richer feature interactions within each group.

### D. DCT-II Orthogonality

The DCT-II basis V satisfies:

```
V[n,k] = α_n · sqrt(2/D) · cos(π · n · (k + 0.5) / D)
```

Where α₀ = 1/√2 and αₙ = 1 for n > 0.

Proof of orthogonality:

```
Σₖ V[n₁,k] · V[n₂,k] 
  = (2/D) · Σₖ cos(π·n₁·(k+0.5)/D) · cos(π·n₂·(k+0.5)/D) · αₙ₁ · αₙ₂
  = δ[n₁, n₂]
```

For n₁ = n₂ = 0:
```
Σₖ V[0,k]² = (2/D) · Σₖ cos²(0) · (1/√2)² = (2/D) · D · (1/2) = 1
```

For n₁ = n₂ > 0:
```
Σₖ V[n,k]² = (2/D) · Σₖ cos²(π·n·(k+0.5)/D) · 1 = 1
```

For n₁ ≠ n₂: the cosine product integrates to zero over the DCT lattice.

### E. Effective Rank Computation

Effective rank (or stable rank) of a matrix A ∈ ℝᵐˣⁿ with singular values σ₁ ≥ σ₂ ≥ ... ≥ σ_min(m,n):

```
eff_rank(A) = Σ σᵢ² / σ₁² = ||A||_F² / ||A||₂²
```

Interpretation: the number of dimensions that carry significant signal. eff_rank = 112 means all dimensions are equally utilized. eff_rank = 4 means only 4 out of 896 dimensions matter — the rest are noise.

For GroupedMLP, per-group eff_rank is computed for each W_up[g] ∈ ℝᵈˣᵉˣᵈ:
- m = d = 112 (input dim)
- n = expand·d = 896 (hidden dim)
- min(m, n) = 112 singular values
- eff_rank ∈ [1, 112]
