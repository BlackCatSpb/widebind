# WideBind Architecture

**161M params, L=32, D=4096, G=32, d=128, bind_K=64, BottleneckBind (Fibonacci-twisted).**

A language model with no transformer layers: no softmax, no attention, no QKV projections, no KV-cache.

## Core Components

### Partitioned Embedding (Sparse Block Codes)

`vocab → 32 segments × 128 dims`. Each token gets a deterministic 6-out-of-32 sparse code via combinadic unranking. Code activates exactly 6 of 32 segments; the other 26 stay silent. This is not a projection — it's **addressing**.

### Pre-LN RMS Norm

`h = x / RMS(x) * w`. No mean subtraction. One learned scale vector per layer.

### Depthwise Conv1d (k=48)

Local context (n-gram patterns) on each of D=4096 channels independently. Stateful buffer carries kernel-1 steps across batch boundaries.

### BottleneckBind (D→K=64→D, Fibonacci-twisted)

The only cross-dimensional mixing mechanism in the model. Three modes:

- **off** — `(hp·w_u) ⊙ (hp·w_v) @ W_out`, classic bilinear bottleneck. Max diff vs legacy = 0.0.
- **shift** — sum of S shifted bilinear products. Each shift rolls the K-space by a golden-ratio offset. With tied ocular: `acc @ W_out`.
- **cascade** — Fibonacci-nested higher-order terms: `a[1]=hp·w_u[0]`, `a[2]=hp·w_v[0]`, `a[n]=normalize(cross(a[n-1], a[n-2]))`.

`tie_bind=True` (default): `W_out = W_proj^T` via forward pre-hook. Autoencoder constraint saves `K·D = 262K` params.

w_u/v init std=1.0 (critical: std³ product).

### VSA Memory (Chunked Prefix Scan)

Recurrent superposition: `mem[t] = decay[t]·mem[t-1] + i_gate[t]·h[t]`.

Parallelized via 2-level hierarchical scan (CHUNK=32): first level scans each chunk from zero, second level chains chunk states sequentially (O(L/CHUNK) steps).

- fp32 guard: entire scan runs in float32 regardless of model dtype
- Surprisal-gated write: `i_gate = softplus(h·w_i + b_i + γ·||ê||₂)`
- Per-channel decay up to τ≈163K (b_d=12.0)
- Dual readout: full memory + leaf (intra-chunk) + cross-chunk context
- First moment readout: mu = prefix_scan(decay, h·i_gate·w_k_mu)

### GroupedCognitiveMirror (32 experts × d=128, k_l∈{4,8,16})

Self-correction ensemble. Each expert computes 4 EMA-normalized signals in its K-space:

1. **temp** — deviation from memory centroid
2. **pred** — normalized prediction error (`alpha_g * hp_prev`)
3. **smooth** — local non-smoothness (causal conv1d)
4. **sym** — bilinear trajectory asymmetry (`(hp·u)·(hp_prev·v)`)

Signals are weighted by learnable softmax (4 weights per expert). φ-coordinate init: per-expert scalars initialized geometrically by depth.

`mirror = tanh(linear) + α·linear` — skip connection. `expert_gate = sigmoid(|pred_error| @ w_gate + b_gate)` blocks experts with good predictions.

Alpha (scalar per expert) replaces old W_pred (G×k×k matrix). 32 params vs 32K; 1024× stronger per-param gradient.

### DCT Spectral

`h_dct = h·V^T; h = h + (h_dct * λ_k)·V`. Learned frequency scaling per DCT component. λ_k initialized linearly from 0.5 (L0) to 1.5 (L31).

### GroupedMLP (32 groups, expand=4)

`h → split(32×128) → SiLU(·@W_up(128→512)) @ W_down(512→128) → merge`. 87.9% of all parameters.

### Partitioned Head

32 readout vectors r_k (d=128 each). `logit_v = Σ_k z_{vk} · ⟨h_k, r_k⟩`. No cross-talk: segment k only reads segment k of final h.

### AdaptiveController

5 VSA hyperparams controlled by two signals from Mirror:
- `exploration = min(1, |mirror| / 0.3)` — how actively the model is adjusting
- `differentiation = min(1, var(log_scale) / 0.1)` — how specialized the mirror is

| Param | Range | Depends on |
|---|---|---|
| b_d (decay bias) | [2.0, 12.0] → τ ≈ [8, 163K] (per-channel) | exploration + layer depth |
| b_i (write gate) | [-3.0, -1.5] → i_gate ≤ 0.27 | exploration |
| w_mem2v_scale | [0.5, 1.0] | differentiation |
| EMA α (state) | [0.90, 0.99] | differentiation |
| EMA α (δ_var) | [0.80, 0.99] | differentiation |

### MirrorLRScheduler

Bidirectional LR modulation: `LR = mult × base_lr`, where `mult = f(var(ls), |1-α|, gate_var, |mirror|/threshold)`. LR can grow (up to 3× base) or decay (down to 0.05×). No forced cosine decay.

## Parameter Distribution (tied, D=4096, L=32)

| Component | Params | % |
|---|---|---|
| Embed + LM Head | 8,192 | 0.01 |
| BottleneckBind (K=64) | 262,208 | 0.17 |
| GroupedCognitiveMirror | 713,632 | 0.47 |
| Conv1d (k=48) | 6,291,456 | 4.12 |
| DCT Spectral | 131,072 | 0.09 |
| VSA gates | 1,179,648 | 0.77 |
| GroupedMLP (expand=4) | 134,348,800 | 87.99 |
| **Total** | **~161M** | **100** |

## Scaling

`D = G·128`, `L = G`, `bind_K = 64`, `k = 32`. One number G determines all dimensions. VSA is O(L log L) — sequence length does not affect weights, only activations.
