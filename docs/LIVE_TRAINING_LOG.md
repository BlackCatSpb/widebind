# Живой журнал обучения WideBind

> Единый актуальный лог текущего запуска. Запуск с нуля (fresh start) — предыдущий прогон (val=8.734@16776) удалён.

## Текущий запуск (Colab T4, fp32, fresh start 2)

- Контур: `notebooks/colab.ipynb`, Colab T4, **fp32 (`use_amp=False`)**, ~33-43 tok/s, ~6.4 GB VRAM.
- Fresh start; `eval_interval=233`, `save_interval=233`, `max_steps=300000`.
- Watchdog `FailureDetector`: откат на `best.pt` при CE-спайке > `watchdog_ce=15.0`.
- In-core SemanticBridge active: `bridge_conn=0.1`, `bridge_dim=256`, params = 2,034,948.
- **Streaming Memory Bank** включён: `memory_bank=True`, `mem_min_write_mat=0.3`.
- **Исправления этого запуска:**
  - Убран `bridge_readiness` из `step_gate` (scalar уничтожал deep-first gradient)
  - Исправлен aux_dict logging (float vs tensor)
  - seq_len=128 (не влезает в T4 с 256)
  - **Per-layer maturation gating для memory bank** — L1/L2/L3 гейтируются mat_gate[i] >= 0.3
  - **Val ppl display** —科学记数法 вместо clamp

### Конфигурация

| Параметр | Значение |
|---|---|
| D / слои / G / bind_K | 2560 / 24 / 32 / 32 |
| vocab / seq_len (обуч.) | 65536 / 128 |
| Параметров | 191,370,592 (191.37M) |
| bridge_glu, intent_bridge, explicit_reasoning, reasoning_adaptive, collective_read_out, variable_precision, bind_twist_gate, maturation_enabled, triad_reason | True |
| bridge_conn | 0.1 (вес aux) |
| T0 / T_delay / delta | 8000 / 8000 / 4000 |
| use_amp | False (fp32) |
| memory_bank | True (L1=3, L2=16, L3=8) |
| mem_min_write_mat | 0.3 |
| mem_bridge_dim | 256 |
| mem_injection_scale | -2.0 |

### Текущий чекпоинт (`checkpoints/best.pt`)

| Метрика | Значение |
|---|---|
| step | 932 |
| val_loss | 10.8955 |
| val_ppl | 53,933.97 |
| active_depth | 8 |
| mat gate | [0.023, 0.145] mean=0.068 — deep-first gradient работает |
| mod_mlp | 0.331 |
| intent_w | 0.7457 |
| NaN / Inf | 0 / 0 |
| Memory Bank | L1=0, L2=0, L3=0 (0b) — mat=0.068 < 0.3, не активен |

### Траектория val

```
step   233: val=13.2381 val_ppl=561,322.75  (first eval)
step   466: val=11.0132 val_ppl=60,671.19
step   699: val=10.9503 val_ppl=56,973.13  ← previous best
step   932: val=10.8955 val_ppl=53,933.97  ← current best
```

### Траектория train CE

| Step  | CE    | mat deep | mat shallow | mod_mlp | intent_w | bridge_conn | val    |
|-------|-------|----------|-------------|---------|----------|-------------|--------|
| 0     | 11.77 | 0.119    | 0.018       | 0.330   | 0.050    | 1.078       | -      |
| 55    | 44.33 | 0.121    | 0.018       | 0.331   | 0.081    | 0.299       | -      |
| 110   | 44.33 | 0.122    | 0.018       | 0.331   | 0.084    | 0.136       | -      |
| 165   | 38.45 | 0.123    | 0.019       | 0.330   | 0.094    | 0.161       | -      |
| 220   | 13.29 | 0.124    | 0.019       | 0.329   | 0.132    | 0.104       | -      |
| 233   | -     | -        | -           | -       | -        | -           | 13.238 |
| 275   | 11.03 | 0.126    | 0.019       | 0.330   | 0.217    | 0.105       | -      |
| 330   | 11.03 | 0.127    | 0.020       | 0.330   | 0.261    | 0.145       | -      |
| 385   | 10.99 | 0.129    | 0.020       | 0.331   | 0.347    | 0.136       | -      |
| 440   | 10.99 | 0.130    | 0.020       | 0.333   | 0.391    | 0.117       | -      |
| 466   | -     | -        | -           | -       | -        | -           | 11.013 |
| 495   | 10.97 | 0.132    | 0.020       | 0.332   | 0.531    | 0.214       | -      |
| 550   | 10.96 | 0.133    | 0.021       | 0.332   | 0.640    | 0.123       | -      |
| 605   | 10.98 | 0.135    | 0.021       | 0.331   | 0.656    | 0.110       | -      |
| 660   | 10.92 | 0.136    | 0.021       | 0.329   | 0.656    | 0.110       | -      |
| 699   | -     | -        | -           | -       | -        | -           | 10.950 |
| 715   | 10.84 | 0.138    | 0.021       | 0.329   | 0.686    | 0.111       | -      |
| 770   | 10.82 | 0.140    | 0.022       | 0.330   | 0.693    | 0.111       | -      |
| 825   | 10.88 | 0.141    | 0.022       | 0.330   | 0.693    | 0.111       | -      |
| 880   | 10.87 | 0.143    | 0.022       | 0.331   | 0.711    | 0.110       | -      |
| 932   | -     | -        | -           | -       | -        | -           | 10.896 |
| 935   | 10.74 | 0.145    | 0.023       | 0.331   | 0.746    | 0.166       | -      |
| 990   | 10.80 | 0.146    | 0.023       | 0.332   | 0.746    | 0.167       | -      |
| 1045  | 10.74 | 0.148    | 0.023       | 0.332   | 0.899    | 0.184       | -      |
| 1100  | 10.80 | 0.150    | 0.024       | 0.331   | 0.911    | 0.210       | -      |
| 1155  | 10.76 | 0.151    | 0.024       | 0.331   | 0.911    | 0.210       | -      |

### Наблюдения

**Фаза 1 (step 0–220): Старт и spike**
- CE spike 11→44→44→38→13 — норма для fresh start (gradient accumulation)
- Maturation gate: deep=0.12, shallow=0.02 (correct spread)
- Intent weight быстро растёт: 0.05→0.13

**Фаза 2 (step 220–700): Plateau на ~11.0**
- CE стабилизировался на ~11.0 (≈ random baseline ln(65536)=11.09)
- Причина: maturation gate ~0.06 → concept/bridge/mirror еле активны
- Intent weight: 0.21→0.66 (насыщение)
- bridge_conn: 0.10–0.14 (норма)

**Фаза 3 (step 700–1155): Медленное снижение**
- CE: 10.84→10.74→10.76 — стабильный дрейф вниз
- val: 10.95→10.90→10.89
- Maturation deep: 0.138→0.151 — рамп продолжается, но медленно
- Intent weight: 0.69→0.91 — насыщается
- **Нет NaN/Inf, нет wake-up** — система здорова
- **Memory bank всё ещё неактивен** (mat=0.07 < 0.3)

### Прогноз

- **Step 3000–4000:** Maturation deep > 0.25, bridge начнёт включаться
- **Step 4000–8000:** Maturation deep > 0.3–0.4, concept layers начнут рождаться, CE < 10.0
- **Step 8000–16000:** Maturation deep > 0.5–0.7, все компоненты активны
- **Step 8000–10000:** Memory bank начнёт писать (mat > 0.3)
- **Цель:** val < 8.5 (лучше прошлого раунда 8.734)
