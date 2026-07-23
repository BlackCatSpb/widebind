# WideBind: Architecture Analysis

**Два варианта:** Main (D=4096, G=32, ~161M) и Mini (D=896, G=8, 12.23M).
BottleneckBind shift mode (default). GroupedCognitiveMirror с private memory.

## Architecture Summary

| Component | Role |
|---|---|
| PartitionedEmbedding | Sparse 6/32 block codes, G×(D/G) segments. ~8K params. |
| BottleneckBind | D→K↔D bilinear mixing, shift mode (golden-ratio twisted, S=4). |
| VSA Memory | Vector superposition, chunked prefix scan, τ per-channel up to 163K. |
| GroupedCognitiveMirror | G experts, 3-layer (L0 signals/L1 private mem/L2 gate), K-space. |
| Private Memory | Cross-expert recall, contradiction gate, Knowledge Graph. |
| GroupedMLP | G groups × SiLU(expand=4). ~88% params. |
| DCT Spectral | Learned per-component frequency scaling. |
| PartitionedHead | G readout vectors, no cross-talk, weight-tying with Embed. |

## Key Design Decisions

- **No softmax/attention** — sigmoid (VSA gates), tanh (mirror), hardtanh (private mem clamp)
- **KV-cache = O(D)** — 16KB per layer, константа от длины последовательности
- **Meta-cognitive layers**: L0 (сигналы, опасные) → L1 (private memory EMA, безопасная) → L2 (gate, самонастройка)
- **Private memory**: soft-competition T=0.5, EMA decay [0.990, 0.999], writes delayed 5000 steps
- **Contradiction gate**: `disagreement = ||hp - help_k|| / ||hp||; gate = sigmoid(w_contra)`
- **Expert Knowledge Graph**: `concept_sim_ema`, `behavior_div_ema`, `trust_matrix` (все G×G)
- **Scalar α** — 32/G params vs 32K; 1024× сильнее градиент
- **MirrorLR**: `mult = min(var_mult, alpha_mult, gate_mult) × mag_factor × loss_damp × train_damp`, cap 1.0
- **log_scale L2**: штраф ls > log(10) для предотвращения взрыва exp(ls)
- **AdaptiveController**: все VSA гиперпараметры из mirror сигналов

## Training Dynamics

- Mini (MX550): loss 10.86→10.37 за 275 steps, g_var 0.0014→0.0048 (специализация растёт)
- Main (T4, pre-fix): training loss расходился (12→49) при eval loss 6.5→6.4 — echo chamber collapse из-за private memory writes в рандомные K-space. Фикс: `_pm_write_delay=5000` + `accum_steps=8`
- Никаких NaN. Gradient clipping 0.5. FP32 (без AMP).

## BottleneckBind Modes

| Mode | Description | Params |
|---|---|---|
| off | `(hp·u)⊙(hp·v) @ W_out` — legacy | w_u/v (G,K) |
| shift (default) | Sum S golden-ratio shifted bilinear products | w_u/v (S,K) + (S)W_out |
| cascade | Fibonacci-nested monomials with normalization | w_u/v (S,K) + mix_logit |

Shift + multi-ocular (default): rank ≥ K×S. 4×8=32 dims эффективного ранга.
