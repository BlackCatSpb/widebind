# WideBind Architecture

**Два варианта:** Main (D=4096, G=32, ~161M) и Mini (D=896, G=8, 12.23M).
Гибридная LM без transformer-слоёв: ни softmax, ни attention, ни QKV, ни KV-cache.

## Core Components

### Partitioned Embedding (Sparse Block Codes)

`vocab → G segments × (D/G) dims`. Deterministic 6-out-of-K sparse code via combinadic unranking. Код активирует ровно 6 сегментов; остальные молчат. Не проекция — **адресация**.

Mix-матрица K×K с ортогональной инициализацией: `codes → sigmoid(M·codes)` — каждый бит влияет на все сегменты (rank expansion).

### Pre-LN RMS Norm

`h = x / RMS(x) * w`. Без вычитания среднего. Один scale вектор на слой.

### Depthwise Conv1d (k=48)

Локальный контекст на каждом из D каналов независимо. Stateful buffer через границы батчей.

### BottleneckBind (D→K↔D, shift mode default)

Единственный cross-dimensional mixing механизм. Три режима:

- **off** — `(hp·w_u) ⊙ (hp·w_v) @ W_out`
- **shift** (default) — сумма S shifted bilinear произведений с golden-ratio roll. `bind_twist_S=4`, `bind_twist_ocular='multi'` (per-shift W_out). `shift + multi-ocular` даёт ранг = K×S.
- **cascade** — Фибоначчи-вложенные моночлены

`tie_bind=True`: `W_out = W_proj^T` (автоэнкодер).

### VSA Memory (Chunked Prefix Scan)

`mem[t] = decay[t]·mem[t-1] + i_gate[t]·h[t]`.

Параллелизация через hierarchical scan (CHUNK=32): первый уровень сканирует каждый чанк с нуля, второй сцепляет состояния чанков (O(L/CHUNK) шагов).

- fp32 guard: весь scan в float32 независимо от dtype модели
- Surprisal-gated write: `i_gate = softplus(h·w_i + b_i + γ·||ê||₂)`
- Per-channel decay до τ≈163K (b_d=12.0)
- Dual readout + first moment

### GroupedCognitiveMirror (G экспертов)

Трёхслойная метакогнитивная архитектура саморефлексии:

**Layer 0 — Cognitive Signals** (веса, опасные):
5 EMA-нормированных сигналов коррекции в K-space (k=32):
1. **temp** — отклонение от центроида памяти
2. **pred** — нормированная ошибка предсказания (`α_g · hp_{t-1}`)
3. **smooth** — локальная негладкость (causal conv1d)
4. **sym** — биллинейная асимметрия траектории
5. **help** — cross-expert recall из private memory (L1)

Сигналы взвешиваются learnable softmax (G+1 вес). φ-координата: per-expert инициализация по глубине.

`mirror = tanh(W·k + bias) + α·W·k` — skip connection (α=0.1: градиент для log_scale при насыщенном tanh). `expert_gate = sigmoid(|pred_error| @ w_gate + b_gate + delta_gate)`.

**Layer 1 — Private Memory** (EMA, безопасная):
- Банк G×k: EMA уверенных K-space состояний (`conf = sigmoid(-|pred_error|)`)
- Soft-competition write: `conf^T / sum(conf^T)`, T=0.5, социальное давление (contra_expert + isolation)
- Read: cross-expert attention `q=hp·uncert @ keys`, help_k = weighted recall
- Contradiction gate: `disagreement = ||hp - help_k|| / ||hp||; gate = sigmoid(w_contra)`
- Expert Knowledge Graph: `concept_sim_ema (G×G)`, `behavior_div_ema`, `trust_matrix`
- Writes delayed на 5000 forward steps (предотвращение echo chamber collapse)

**Layer 2 — Meta-Gate** (α + contradiction, самонастройка):
- gradient-adaptive gate: `delta_var` EMA определяет активность эксперта
- learnable usefulness predictor: softmax-конкуренция экспертов
- Self-organizing timescales: `alpha_diag` ← `sigmoid(2.2 - log(relative_var))`

### DCT Spectral

Learned frequency scaling per DCT component. λ_k от 0.5 (L0) до 1.5 (L31).

### GroupedMLP (G групп, expand=4)

`h → split → SiLU(@W_up(D/G → 4D/G)) @ W_down → merge`. ~88% параметров.

### Partitioned Head

G readout векторов. `logit_v = Σ_k z_{vk} · ⟨h_k, r_k⟩`. Без cross-talk между сегментами.

Embed + Head разделяют basis (weight tying encode/decode).

### AdaptiveController

5 VSA гиперпараметров из двух сигналов Mirror:
- `exploration = min(1, |mirror| / 0.3)`
- `differentiation = min(1, var(log_scale) / 0.1)`

### MirrorLRScheduler

`mult = min(var_mult, alpha_mult, gate_mult) × mag_factor × loss_lr_factor × train_loss_lr_factor`

- `var_mult`: var(log_scale) — counter-cyclical (LR ↓ когда var растёт)
- `alpha_mult`: |1-α| — counter-cyclical
- `gate_mult`: gate_var — counter-cyclical
- `mag_factor`: |mirror| / threshold — cap ≤ 1.0
- `loss_lr_factor`: eval loss > 1.02× best → ×0.5
- `train_loss_lr_factor`: training loss sustained increase over 100 steps → ×0.7
- log_scale L2: штраф ls > 2.3 (exp > 10) с весом 0.01
- Кап: mult ∈ [0.05, 1.0] (никогда выше base LR)

Во время warmup (λ_d-выведенный, fib(11)+50 blend): α_override=1.0→0.0, temperature annealing 2.0→0.5.

## Parameter Distribution (tied, D=4096, L=32)

| Component | Params | % |
|---|---|---|
| Embed + LM Head | 8,192 | 0.01 |
| BottleneckBind (K=64) | 262,208 | 0.17 |
| GroupedCognitiveMirror | 713,632 | 0.47 |
| Private Memory | 1,024 | <0.01 |
| Conv1d (k=48) | 6,291,456 | 4.12 |
| DCT Spectral | 131,072 | 0.09 |
| VSA gates | 1,179,648 | 0.77 |
| GroupedMLP (expand=4) | 134,348,800 | 87.99 |
| **Total** | **~161M** | **100** |

## Variants

| | Mini | Main |
|---|---|---|
| D | 896 | 4096 |
| G | 8 | 32 |
| bind_K | 32 | 64 |
| Params | 12,232,836 (12.23M) | ~161M |
| LR | 3e-4 | 3e-4 |
| Accum | 8 | 8 |
| VRAM | 2.1 GB (MX550) | 11-16 GB (T4) |

## Scaling

`D = G·128`, `L = G`, `bind_K = D/G`, `k = 32`. VSA O(L log L) — длина последовательности не влияет на веса.
