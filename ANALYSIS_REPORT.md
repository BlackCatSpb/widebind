# WideBind: Architecture Analysis

**Два варианта:** Main (D=4096, L=32, G=32, ~255M default) и Mini (D=896, L=12, G=8, ~17.6M).
BottleneckBind shift mode (default, S=4). GroupedCognitiveMirror. Подробное описание — в `README.md`.

## Architecture Summary

| Component | Role |
|---|---|
| PartitionedEmbedding | Sparse 6/32 block codes → плотное смешивание (mix K×K) → RoPE. basis 4096 + mix 1024. |
| BottleneckBind | D→K↔D bilinear mixing, shift mode (golden-ratio twisted, S=4, multi-ocular). |
| VSA Memory | 4-масштабная векторная суперпозиция, learnable τ, chunked prefix scan, fp32 guard. |
| GroupedCognitiveMirror | G экспертов, 3-слойная мета-система (L0 signals / L1 private mem / L2 gate), K-space, staircase k=8/16/32. |
| Private Memory | Cross-expert recall, contradiction gate, Knowledge Graph (опционально, private_mem). |
| GroupedMLP | G групп × SwiGLU (expand=4). ~79% параметров. |
| DCT Spectral | Learnable per-frequency scaling. |
| PartitionedHead | K readout-векторов, no cross-talk, weight-tying с Embed. |

## Key Design Decisions

- **Без softmax/attention в теле** — sigmoid (гейты), tanh (mirror).
- **«KV-кэш» = O(D)** — вектор состояния на слой, константа от длины последовательности.
- **Мета-слои**: L0 (сигналы, опасные) → L1 (private memory EMA, безопасная) → L2 (gate, самонастройка).
- **Private memory** (опц.): soft-competition T=0.5, EMA decay [0.990, 0.999], запись после 5000 forward-шагов.
- **Contradiction gate** (опц.): disagreement = |hp − help_k| / |hp|.
- **Expert Knowledge Graph** (опц.): concept_sim, behavior_div, trust_matrix (все G×G).
- **MirrorLR**: counter-cyclical множители, объединяемые средним геометрическим
  `(var_mult · alpha_mult · gate_mult)^(1/3) · mag_factor`, clamp [0.05, 1.0], плюс loss_lr_factor (eval).
- **log_scale L2**: штраф exp(log_scale) > 10.
- **AdaptiveController**: VSA-гиперпараметры из сигналов exploration/differentiation.
- **λ_d LR-иерархия**: mirror и gate на λ¹, vsa на λ⁻², mlp λ⁻¹, embed λ⁻².

## Training Dynamics (историческое)

- Mini (MX550): на старых данных loss снижался (10.86→10.37 за 275 шагов), g_var росла (специализация).
- Main (T4, pre-fix): training loss расходился при стабильном eval — echo chamber collapse
  из-за записи private memory в рандомные K-space. Фикс: `_pm_write_delay=5000` + `accum_steps=8`.
- Никаких NaN. Gradient clipping 0.5. FP32 (без AMP).

### ⚠️ Важное замечание о данных (авг. 2026)

Ранее опубликованные «реальные» результаты (CE 11.27 → 5.3, val_ppl 13233 → 268, D=2560/L=24, 86.39M)
получены на **испорченном корпусе**: ~92% токенов были U+FFFD (замена повреждённой кодировки).
Эти цифры — артефакт, они не отражают усвоение языка и не должны использоваться как метрика.

После перегенерации данных (39 жанров, 2.46GB, 0% U+FFFD, валидная кириллица) обучение перезапущено.
Эталонный порог CE = ln(50000) ≈ 10.82 (пол случайного угадывания); ниже порога — реальное усвоение структуры.

## BottleneckBind Modes

| Mode | Description | Параметры (на слой) |
|---|---|---|
| off | Два билинейных произведения, без сдвигов | W_proj + w_u/v (+ W_out при tie=False) |
| shift (default) | Сумма S=4 golden-ratio shifted билинейных произведений, multi-ocular | W_proj + S·W_out + w_u/v (S,K) |
| cascade | Фибоначчи-вложенные моночлены с нормировкой и mix_logit | W_proj + W_out + mix_logit |

Shift + multi-ocular: ранг ≤ min(S·K, D) (Main: min(256, 4096) = 256).
