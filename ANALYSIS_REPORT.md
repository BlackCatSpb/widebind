# WideBind: Architecture Analysis

**161M params. D=4096, L=32, G=32, d=128, bind_K=64, BottleneckBind (3 modes).**

## Architecture Summary

| Component | Role |
|---|---|
| PartitionedEmbedding | Sparse 6/32 block codes, 32×128 segments. 8192 params. |
| BottleneckBind | D→K=64→D bilinear mixing. Fibonacci-twisted (off/shift/cascade). |
| VSA Memory | Vector superposition with chunked prefix scan. τ per-channel up to 163K. |
| GroupedCognitiveMirror | 32 experts with 4 EMA-normalized signals, scalar α, K-space gate. |
| GroupedMLP | 32 groups × (128→512→128, SiLU). 87.9% params. |
| DCT Spectral | Learned per-component frequency scaling (λ_k). |
| PartitionedHead | 32 readout vectors, no cross-talk. |

## Key Design Decisions

- **No softmax anywhere** — only sigmoid (VSA gates) and tanh (mirror correction)
- **KV-cache = O(D) per layer** — constant 16KB, independent of sequence length
- **bind_K=64** — D/K=64, eff_rank 38-50/64 (59-78%)
- **tie_bind=True** — autoencoder constraint saves 262K params
- **Chunked VSA scan (CHUNK=32)** — numerically stable at any sequence length
- **fp32 guard** — VSA scan forced to float32 regardless of model dtype
- **Scalar α** replaces old W_pred (32 params vs 32K); 1024× stronger gradient
- **Bidirectional MirrorLR** — no forced cosine decay; LR adapts to specialization
- **AdaptiveController** — all VSA hyperparams derived from mirror signals

## Training Dynamics

Loss drops from ~10.9 to ~1.6 over ~7000 steps on Colab T4. LR self-regulates (up to 3× base). No NaN, stable gradients, correct accumulation confirmed on MX550.

## BottleneckBind Modes

| Mode | Description | Params |
|---|---|---|
| off | `(hp·u)⊙(hp·v) @ W_out` — legacy regression | minimal |
| shift | Sum of S golden-ratio shifted bilinear products | w_u/v (S,K) |
| cascade | Fibonacci-nested monomials with normalization | w_u/v (S,K) + mix_logit |

Micro-benchmarked: shift mode produces finite output with std≈0.1 at init.
