# Training Log — WideBind (2026-07)

> **⚠️ Эта история — эпоха W_pred (D=3584, L=32, G=32, mirror_k=8, 221M).**
> **Текущая архитектура (2026-07-16+): D=4096, L=32, G=32, mirror_k=32, alpha scalar per expert, 293M.**
> **W_pred заменён на alpha. Все чекпоинты этой эпохи несовместимы.**

**Architecture (old):** WideBind D=3584 L=32 K=32 G=32 mirror_k=8
**Params (old):** 221,740,064 (221.74M)
**Data:** 1,429,837,807 tokens (2 streams, clean .bin)
**GPU:** Colab T4, B=4, seq_len=64, fp32
**Init:** gate_pred_scale_init=0.0 (β=0.5), W_pred≈I (eye×0.99 + noise×0.01)
**LR:** 0.0003, warmup=2000, scheduler=mirror

---

## Run 1 — β=0.5 fresh start

| Step | Train Loss | β0 | βL | α_skip | tok/s | Notes |
|------|-----------|----|----|--------|-------|-------|
| 0 | 10.9585 | 0.003 | 0.003 | 0.100 | 201 | Init |
| 100 | 10.0215 | 0.003 | 0.003 | 0.100 | 244 | |
| 200 | 10.1421 | 0.003 | 0.004 | 0.100 | 241 | |
| 300 | 10.6762 | 0.003 | 0.005 | 0.100 | 240 | |
| 400 | 6.9265 | 0.003 | 0.008 | 0.100 | 239 | |
| 500 | 10.7270 | 0.003 | 0.008 | 0.100 | 239 | Spike |
| 600 | 6.3853 | 0.003 | 0.009 | 0.100 | 239 | |
| 700 | 6.8465 | 0.003 | 0.009 | 0.100 | 239 | |
| 800 | 6.4704 | 0.003 | 0.008 | 0.100 | 239 | |
| 900 | 6.3821 | 0.003 | 0.009 | 0.100 | 238 | |
| 1000 | 6.8329 | 0.003 | 0.008 | 0.100 | 238 | |
| 1100 | 6.3584 | 0.003 | 0.008 | 0.100 | 238 | |
| 1200 | 6.9656 | 0.003 | 0.009 | 0.100 | 238 | |
| 1300 | 6.8758 | 0.003 | 0.009 | 0.100 | 238 | |
| 1400 | 6.3485 | 0.003 | 0.006 | 0.100 | 238 | βL упал |
| 1500 | 6.5600 | 0.003 | 0.006 | 0.100 | 238 | |
| 1600 | 6.5589 | 0.003 | 0.006 | 0.100 | 238 | |
| 1700 | 6.4433 | 0.003 | 0.005 | 0.100 | 238 | |
| 1800 | 6.5218 | 0.003 | 0.006 | 0.100 | 238 | |
| 1900 | 6.5249 | 0.003 | 0.004 | 0.100 | 238 | |
| **2000** | **6.5339** | **0.003** | **0.003** | **0.100** | **238** | **Eval: val_loss=6.575** |

### Checkpoint: best.pt (step 2000) — Полный анализ

#### Архитектура
| Параметр | Значение |
|----------|----------|
| Params | 221,740,064 |
| D / K | 3584 / 32 |
| MLP | 32 groups × 8× expand |
| seq_len / batch | 64 / 4 |
| LR | 0.0003 (mirror scheduler) |

#### Веса
| Метрика | Значение |
|---------|----------|
| Mean | 0.0026 |
| Std | 0.1388 |
| Min | -4.746 |
| Max | 5.443 |
| Output std (fwd) | 0.9999 |
| Output mean (fwd) | 0.0155 |

#### Gate & Prediction (все слои — frozen at init)
| Метрика | Значение |
|---------|----------|
| β0 (L0) | 0.5008 |
| β31 (L31) | 0.5007 |
| β mean ± std | 0.5001 ± 0.0008 |
| gate_pred_scale range | [-0.0032, 0.0089] |
| W_pred \|I-diff\| mean (min) | 0.0086 (L29) |
| W_pred \|I-diff\| mean (max) | 0.0091 (L21) |
| W_pred diag (L0 / L31) | 0.987 / 0.987 |
| w_pred_scale μ / σ | 0.499 / 0.001 |
| skip_alpha μ | 0.1000 |
| var(log_scale) per-layer | 0.002477 |
| log_skip_alpha μ (L31) | -2.2955 |

#### Per-Layer Analysis (summary)

| Layer | β | \|W_pred-I\| | skip_α | log_scale σ | w_temp σ | w_global σ |
|-------|---|-------------|--------|-------------|---------|-----------|
| 0 | 0.5008 | 0.0087 | 0.1000 | 0.0490 | 0.9836 | 0.9640 |
| 5 | 0.5008 | 0.0088 | 0.1000 | 0.0496 | 0.9510 | 0.9529 |
| 10 | 0.5007 | 0.0087 | 0.1000 | 0.0498 | 0.9834 | 1.0031 |
| 15 | 0.5008 | 0.0087 | 0.1000 | 0.0483 | 1.0415 | 0.9902 |
| 20 | 0.4993 | 0.0089 | 0.1000 | 0.0499 | 1.0077 | 1.0454 |
| 25 | 0.5008 | 0.0086 | 0.1001 | 0.0506 | 0.8824 | 0.9740 |
| 30 | 0.5022 | 0.0089 | 0.1004 | 0.0486 | 0.9838 | 1.0455 |
| 31 | 0.5007 | 0.0089 | 0.1007 | 0.0505 | 0.9647 | 0.9651 |

**Bind proj rank:** 31.9 (все слои) — K=32 почти полностью используется.
**MLP eff rank:** 108.4/112 — 97% utilisation.
**i_gate:** 0.047 (frozen at init, b_i=-3.00).
**τ (decay):** 8 (L0) → 149 (L31) — растёт линейно с lambda_k.
**dvar_mod / grad_mod:** log=-2.303 (exp=0.1) — frozen at init.
**Conv ||W||:** 4.13 (L0) → 5.39 (L31) — нижние слои учатся сильнее.

#### Adaptive Controller
| Метрика | Значение |
|---------|----------|
| Exploration | 0.807 |
| Differentiation | 0.031 |
| b_d (τ bias) | 3.386 |
| b_i (i_gate) | -1.789 |
| w_mem2v_scale | 0.985 |
| EMA α | 0.903 |
| Noise scale | 0.048 |

#### Диагноз (step 2000)
- **Core MLP/Bind:** работают — loss с 10.96 → 6.58. eff_rank=108/112.
- **β:** gate_pred_scale заморожен на init (±0.003). Градиента нет.
- **W_pred:** всё ещё identity (\|I-diff\|=0.0088, init=~0.01). Не учится.
- **skip_alpha:** frozen at 0.100.
- **log_scale:** var=0.0025 — ноль per-dim специализации.
- **dvar_mod/grad_mod:** frozen at init (log=-2.303).
- **Вывод:** β=0.5 слишком высок для свежего W_pred. Gate режет 50% сигнала,
   W_pred не может выйти из identity. Требуется снижение gate_pred_scale_init
   или ждать ~5-10K шагов для накопления градиента к ранним слоям.


### Step 2000 — best.pt
| Metric | Value |
|--------|-------|
| Step | 2000 |
| best_val_loss | 6.5750534725189205 |
| beta_0 | 0.5008 |
| beta_31 | 0.5007 |
| beta mean / std | 0.5001 / 0.0008 |
| gate_pred_scale range | [-0.0032, 0.0089] |
| W_pred diff from I (max) | 0.0091 |
| W_pred diag L0/L31 | 0.987 / 0.987 |
| skip_alpha mean | 0.1000 |
| var(log_scale) mean | 0.002477 |
| log_scale sigma mean | 0.0498 |
| MLP eff_rank mean | 108.4 |
| Bind rank mean | 31.9 |
| tau range | [149, 149] |
| i_gate | 0.047 |
| w_pred_scale mu | 0.499 |
| log_dvar_mod_scale | -2.303 |
| log_grad_mod_scale | -2.303 |
| Exploration | 0.8060 |
| Differentiation | 0.030967 |
| Output std (fwd) | 0.9999 |
| Weights std | 0.1388 |

### Step 10000 — step_10000.pt
| Metric | Value |
|--------|-------|
| Step | 10000 |
| best_val_loss | 6.5692786645889285 |
| beta_0 | 0.5008 |
| beta_31 | 0.4982 |
| beta mean / std | 0.5000 / 0.0009 |
| gate_pred_scale range | [-0.0071, 0.0113] |
| W_pred diff from I (max) | 0.0107 |
| W_pred diag L0/L31 | 0.972 / 0.972 |
| skip_alpha mean | 0.1000 |
| var(log_scale) mean | 0.002406 |
| log_scale sigma mean | 0.0490 |
| MLP eff_rank mean | 108.4 |
| Bind rank mean | 31.9 |
| tau range | [149, 149] |
| i_gate | 0.047 |
| w_pred_scale mu | 0.491 |
| log_dvar_mod_scale | -2.303 |
| log_grad_mod_scale | -2.303 |
| Exploration | 0.7827 |
| Differentiation | 0.030074 |
| Output std (fwd) | 1.0000 |
| Weights std | 0.1383 |

---

## Session 2026-07-14 — Catch-22 Analysis & Fix

### Три причины catch-22 (W_pred frozen after 10K steps)

| Причина | Механизм | Доказательство |
|---|---|---|
| **Weight decay (3:1)** | lr·wd·\|W_pred\| = 1.96e-6; lr·\|grad\|/√N = 6.72e-7/step | wd в 2.9× сильнее градиента |
| **Gate saturation** (trained) | sigmoid_deriv: 0.042 (old) vs 0.198 (new) при 5× W_proj | Gate тупеет когда W_proj растёт |
| **Per-param grad 42× меньше** | einsum усредняет градиент W_pred по B×L; pred_scale — поэлементно | W_pred: 2048 params, grad 7e-5; pred_scale: 256 params, grad 1e-3 |

### Исправления в core/model.py

1. **Gate: hp + β·pred_error → |pred_error|** — gate открывается при плохом предсказании. 32× больше градиента к W_pred.
2. **W_pred weight_decay=0** — moved в gate_no_decay. WD убивал слабый градиент.
3. **Auxiliary loss** — MSE(pred_k, hp.detach()) с weight=0.01. ~50% градиента к W_pred.
4. **compute_loss** — обратно совместим: `(h, targets)` = CE; `(h, targets, pred_weight)` = CE+aux.

### Верификация (63 теста, все PASS)

| Тест | Результат |
|---|---|
| 46 unit tests (test_model.py) | 46/46 PASS |
| mini_test --full (16 checks) | 16/16 PASS |
| Gate comparison (old vs new) | 12/12 PASS — W_pred grad 32× больше, sigmoid_deriv 0.198 vs 0.042 |
| AR-1 synthetic: W_pred учится | \|I-diff\| 0.0084→0.0322 (4×), diag→0.801 (target 0.8) |
| Full model D=3584, L=32 fwd+bwd | 19.9s, 32/32 слоёв с W_pred grad |
| Memory leak (50 iter) | 16MB growth |
| Checkpoint step_10000 load | 0 missing, 0 unexpected. W_pred frozen (\|I-diff\|=0.010), pred_scale=0.491 |

### Анализ чекпоинта step_10000

| Параметр | Init | Step 10000 (old code) | Вывод |
|---|---|---|---|
| W_pred \|I-diff\| | ~0.010 | 0.0104 | Не двигался |
| W_pred diag | ~0.990 | 0.972 | Чуть уменьшился (wd толкает к 0) |
| w_pred_scale | 0.1 | 0.491 | Вырос в 5× (градиент сильнее wd) |
| W_proj std | 0.183 | 0.178 | Не вырос (1.0×) — gate не насыщен |
| beta (gate_pred_scale) | 0.5 | 0.500 | Не изменился |
| b_gate | 0 | ~0.000 | Не изменился |

### Gradient per-param comparison

| Параметр | Shape | Params | Grad norm | Per-param grad |
|---|---|---|---|---|
| pred_scale | (32, 8) | 256 | ~1e-3 | **6.3e-5** |
| W_pred | (32, 8, 8) | 2048 | ~7e-5 | **1.5e-6** (42× меньше) |

### Следующий шаг (неактуально — архитектура изменена)

**Смена курса (2026-07-16):**
- W_pred (G×k×k) → alpha (G, scalar per expert): 1024× сильнее градиент, учится за сотни шагов
- lo/hi k-space split удалён: pred_error течёт во все k=32 dims
- G=16→32: d/k с 8:1 до 4:1, вдвое сильнее временной сигнал
- D=3584→4096: 293M params
- Свежий старт (все старые чекпоинты несовместимы)

## Session 2026-07-17 — Expert Deadlock SOLVED (Bidirectional LR + Alpha Override)

### Архитектура
- D=4096, G=32, L=32, k=32, d=128, expand=4
- alpha (G,) — scalar per expert
- Bidirectional MirrorLR (нет forced cosine)
- Alpha Warmup Override (0.5→0.25 during warmup)
- tie_bind=True, tie_mirror_proj=True
- 150,865,744 params

### Resume с best.pt (step 2000, val_loss=5.92)

| Step | Loss | Val Loss | \|1-alpha\| | gate_var | var(ls) | LR mult |
|---|---|---|---|---|---|---|
| 2000 | 7.11 | 5.92 | 0.0077 | 0.0017 | 0.00771 | 0.88 |
| 2500 | 5.51 | — | 0.0101 | 0.0064 | 0.0078 | 2.34 |
| 3000 | 4.42 | — | 0.0117 | 0.0197 | 0.0078 | 3.00 |
| 3500 | 4.16 | — | 0.0127 | 0.0353 | 0.0079 | 3.00 |
| 4000 | 3.36 | 3.35 | 0.0138 | 0.0481 | 0.0079 | 3.00 |
| 4500 | 3.15 | — | 0.0135 | 0.0564 | 0.0079 | 3.00 |
| 5000 | 2.44 | — | 0.0143 | 0.0712 | 0.0079 | 3.00 |
| 5500 | 1.86 | — | 0.0142 | 0.0746 | 0.0080 | 3.00 |
| 6000 | 1.72 | **1.90** | 0.0143 | 0.0789 | 0.0080 | 3.00 |
| 6500 | 1.83 | — | 0.0142 | 0.0766 | 0.0080 | 3.00 |
| 6900 | 1.64 | — | 0.0147 | 0.0910 | 0.0080 | 3.00 |

### Диагноз (step 6900)

**Что работает ✅:**
- Expert deadlock полностью решён. Все 32 слоя с |1-alpha| > 0.01.
- gate_var 0.091 — gate активно дифференцирует экспертов.
- LR на 3× ceiling — модель ускоряет сама себя.
- val_loss 1.90 и продолжает падать — нет признаков насыщения.
- Checkpoint 159 MB (FCF-CPR uint8).

**Что отличается от ожиданий:**
- `var(ls)` растёт медленно (0.0077→0.0080, +4%) — специализация идёт через gate_var и alpha, не через log_scale.
- LR упирается в ceiling 3.0 — возможно, стоит поднять `lr_max_ratio`.

### Файлы
- `core/model.py` — `GroupedCognitiveMirror._alpha_override`, `MirrorLRScheduler` bidirectional logic
- `notebooks/cloud.ipynb` — упрощённый конструктор MirrorLRScheduler
- `checkpoints/best.pt` — step 6000, val_loss=1.9035, 159 MB
