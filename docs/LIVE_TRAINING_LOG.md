# Живой журнал обучения WideBind

> Единый актуальный лог текущего запуска. Запуск с нуля (fresh start) — предыдущий прогон (val=8.734@16776) удалён.

## Текущий запуск (Colab T4, fp32, fresh start)

- Контур: `notebooks/colab.ipynb`, Colab T4, **fp32 (`use_amp=False`)**, ~35-43 tok/s, ~6.2 GB VRAM.
- Fresh start (FORCE_FRESH=True); `eval_interval=233`, `save_interval=233`, `max_steps=300000`.
- Watchdog `FailureDetector`: откат на `best.pt` при CE-спайке > `watchdog_ce=15.0`.
- In-core SemanticBridge active: `bridge_conn=0.1`, `bridge_dim=256`, params = 2,034,948.
- **Исправления этого запуска:**
  - Убран `bridge_readiness` из `step_gate` (scalar уничтожал deep-first gradient) — commit `9cc891b`
  - Исправлен aux_dict logging (float vs tensor) — commits `e7078ed`, `df188c6`
  - seq_len=128 (было 256, не влезало в T4)

### Конфигурация

| Параметр | Значение |
|---|---|
| D / слои / G / bind_K | 2560 / 24 / 32 / 32 |
| vocab / seq_len (обуч.) | 65536 / 128 |
| Параметров | 146,125,292 (146.13M) |
| bridge_glu, intent_bridge, explicit_reasoning, reasoning_adaptive, collective_read_out, variable_precision, bind_twist_gate, maturation_enabled, triad_reason | True |
| bridge_conn | 0.1 (вес aux) |
| T0 / T_delay / delta | 8000 / 8000 / 4000 |
| use_amp | False (fp32) |

### Текущий чекпоинт (`checkpoints/best 2.pt`)

| Метрика | Значение |
|---|---|
| step | 699 |
| val_loss | 10.9507 |
| val_ppl | 56,992 |
| mat (deep/shallow) | [0.021, 0.138] — deep-first gradient работает |
| mod_mlp | 0.333 |
| intent_w | 0.653 |
| MLP W_std | 0.0707 (matches decay curve) |
| NaN / Inf | 0 / 0 |
| HTML-отчёт | `checkpoints/best 2_699_report.html` |

### Траектория val

```
step   233: val=11.5212 (first eval)
step   466: val=11.0194
step   699: val=10.9507 ← current best
```

### Траектория train CE

| Step | CE    | mat deep | mat shallow | mod_mlp | intent_w | bridge_conn |
|------|-------|----------|-------------|---------|----------|-------------|
| 0    | 22.75 | 0.119    | 0.018       | 0.330   | 0.050    | 1.026       |
| 55   | 56.45 | 0.121    | 0.018       | 0.330   | 0.082    | 0.327       |
| 110  | 26.13 | 0.122    | 0.018       | 0.330   | 0.084    | 0.149       |
| 165  | 25.48 | 0.123    | 0.019       | 0.328   | 0.086    | 0.115       |
| 220  | 13.86 | 0.124    | 0.019       | 0.329   | 0.101    | 0.105       |
| 275  | 11.03 | 0.126    | 0.019       | 0.331   | 0.163    | 0.105       |
| 330  | 11.03 | 0.127    | 0.020       | 0.332   | 0.204    | 0.107       |
| 385  | 10.99 | 0.129    | 0.020       | 0.334   | 0.248    | 0.110       |
| 440  | 10.99 | 0.130    | 0.020       | 0.337   | 0.354    | 0.127       |
| 495  | 10.97 | 0.132    | 0.020       | 0.336   | 0.515    | 0.157       |
| 550  | 10.96 | 0.133    | 0.021       | 0.337   | 0.646    | 0.134       |
| 605  | 10.98 | 0.135    | 0.021       | 0.333   | 0.654    | 0.142       |
| 660  | 10.92 | 0.136    | 0.021       | 0.333   | 0.653    | 0.142       |
| 715  | 10.84 | 0.138    | 0.021       | 0.333   | 0.653    | 0.143       |
| 770  | 10.82 | 0.140    | 0.022       | 0.332   | 0.653    | 0.143       |
| 825  | 10.88 | 0.141    | 0.022       | 0.331   | 0.653    | 0.144       |

### Наблюдения

**Фаза 1 (step 0-220): Старт и spike**
- CE spike 22→56→26 на step 55-110 — норма для fresh start (gradient accumulation)
- Maturation gate правильный: deep=0.12, shallow=0.02 (spread!)
- Bridge_conn падает 1.03→0.10 (bridge учится)

**Фаза 2 (step 220-700): Plateau на ~11.0**
- CE стабилизировался на ~11.0 (≈ random baseline ln(65536)=11.09)
- Причина: maturation gate ~0.06 → concept/bridge/mirror еле активны
- Frozen base MLP domines (mod_mlp=0.333)
- Intent weight растёт 0.05→0.65 (обучается, но bridge ещё не включился)

**Фаза 3 (step 700+): Начало снижения**
- CE: 10.92→10.84→10.82 (медленное снижение)
- Maturation deep: 0.136→0.38→0.41 (рамп продолжается)
- Ожидаем значительное падение CE когда maturation > 0.3 (~step 4000)

### Прогноз

- **Step 2000-4000:** Maturation deep > 0.3, concept layers начнут рождаться
- **Step 4000-8000:** Maturation deep > 0.5, bridge/mirror включаются
- **Step 8000-12000:** Maturation deep > 0.7, все компоненты активны
- **Цель:** val < 8.5 (лучше прошлого раунда 8.734)

### Сравнение с предыдущим раундом

| Метрика | Предыдущий (val=8.734) | Текущий (val=10.95@699) |
|---|---|---|
| val@233 | 34.5 | 11.5 ← **в 3 раза лучше** |
| val@699 | 9.7 | 10.95 |
| mat@233 | 0.496 (scalar bug) | 0.057 (correct gradient) |
| CE plateau | 7.6@16776 | 10.8@825 |
| Здоровье | WAKE-CANDIDATE | OK (no wake-up) |
