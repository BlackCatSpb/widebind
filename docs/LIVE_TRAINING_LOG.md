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

### Траектория val

```
step   233: val=13.2381 val_ppl=561,322.75   (first eval)
step   466: val=11.0132 val_ppl=60,671.19
step   699: val=10.9503 val_ppl=56,973.13   ← best.pt
step   932: val=10.8955 val_ppl=53,933.97   ← best 2.pt
step  1165: val=10.8568 val_ppl=51,884.04   ← best 3.pt
step  1398: val=10.7936 val_ppl=48,707.87   ← best 4.pt
step  1631: val=10.7343 val_ppl=45,905.87   ← best 5
step  1864: val=10.6869 val_ppl=43,776.44   ← best 5.pt
step  2097: val=10.6463 val_ppl=42,037.69   ← best 6.pt
step  2330: val=10.5937 val_ppl=39,883.06
step  2563: val=10.5760 val_ppl=39,187.00   ← best 7
step  2796: val=10.4896 val_ppl=35,946.84
step  3029: val=10.4282 val_ppl=33,814.25
step  3262: val=10.3800 val_ppl=32,209.44
step  3495: val=10.3291 val_ppl=30,595.19
step  3728: val=10.2952 val_ppl=29,584.25
step  3961: val=10.2442 val_ppl=28,111.53
step  4194: val=10.1844 val_ppl=26,488.01
step  4427: val=10.1253 val_ppl=24,984.76
step  4660: val=10.0892 val_ppl=24,108.01
step  4893: val=10.0407 val_ppl=22,963.42
step  5126: val=9.9943 val_ppl=21,901.78
step  5359: val=9.9477 val_ppl=20,883.80
step  5592: val=9.8994 val_ppl=19,918.40
step  5825: val=9.8525 val_ppl=19,005.14
step  6058: val=9.8032 val_ppl=18,102.19
step  6291: val=9.7579 val_ppl=17,298.03
step  6524: val=9.7183 val_ppl=16,644.29
step  6757: val=9.6800 val_ppl=16,016.56
step  6990: val=9.6297 val_ppl=15,222.45
step  7223: val=9.5887 val_ppl=14,618.55
step  7456: val=9.5476 val_ppl=14,040.76
step  7689: val=9.5072 val_ppl=13,489.69
step  7922: val=9.4665 val_ppl=12,954.57
step  8155: val=9.4222 val_ppl=12,388.29
step  8388: val=9.3914 val_ppl=12,015.59
step  8621: val=9.3506 val_ppl=11,533.19
step  8854: val=9.3074 val_ppl=11,045.47
step  9553: val=9.1808 val_ppl=9,714.69   ← best 8 (novelty gate active)
step 10019: val=9.1642 val_ppl=9,552.23   ← best 9 (current)
```

### Траектория train CE (выборочно)

| Step  | CE    | mat deep | mat shallow | mod_mlp | intent_w | bridge_conn | val    | Memory Bank | Depth | Notes |
|-------|-------|----------|-------------|---------|----------|-------------|--------|-------------|-------|-------|
| 0     | 11.77 | 0.119    | 0.018       | 0.330   | 0.050    | 1.078       | -      | L1=0 L2=0 L3=0 | 8/24 | Start |
| 220   | 13.29 | 0.124    | 0.019       | 0.329   | 0.132    | 0.104       | -      | L1=0 L2=0 L3=0 | 8/24 | CE spike |
| 699   | 10.92 | 0.136    | 0.021       | 0.329   | 0.656    | 0.110       | 10.950 | L1=0 L2=0 L3=0 | 8/24 | |
| 1398  | 10.63 | 0.160    | 0.026       | 0.330   | 0.910    | 0.208       | 10.794 | L1=0 L2=0 L3=0 | 8/24 | intent_w saturated |
| 2330  | 10.45 | 0.197    | 0.033       | 0.326   | 0.907    | 0.208       | 10.594 | L1=0 L2=0 L3=0 | 8/24 | |
| 2796  | 10.01 | 0.215    | 0.036       | 0.323   | 0.906    | 0.209       | 10.490 | L1=0 L2=0 L3=0 | 8/24 | CE<10 imminent |
| 3495  | 9.99  | 0.242    | 0.040       | 0.320   | 0.906    | 0.209       | 10.329 | L1=0 L2=0 L3=0 | 8/24 | |
| 3728  | 9.93  | 0.250    | 0.041       | 0.318   | 0.906    | 0.209       | 10.295 | L1=0 L2=0 L3=0 | **12/24** | **DepthController plateaus at 12** |
| 4194  | 9.78  | 0.268    | 0.045       | 0.316   | 0.907    | 0.209       | 10.184 | L1=0 L2=0 L3=0 | **16/24** | **DepthController: 16** |
| 4427  | 9.71  | 0.277    | 0.046       | 0.315   | 0.907    | 0.209       | 10.125 | L1=0 L2=0 L3=0 | **20/24** | **DepthController: 20** |
| 4660  | 9.59  | 0.287    | 0.048       | 0.314   | 0.906    | 0.209       | 10.089 | L1=0 L2=0 L3=0 | **24/24** | **DepthController: 24 (MAX)** |
| 4675  | 9.55  | 0.288    | 0.048       | 0.314   | 0.907    | 0.210       | -      | **L1=25 L2=17 L3=1** | 24/24 | **MEMORY BANK WOKE UP** |
| 4785  | 9.31  | 0.292    | 0.049       | 0.313   | 0.907    | 0.210       | -      | L1=1103 L2=30 L3=1 | 24/24 | L1 growing fast |
| 5126  | 8.79  | 0.305    | 0.051       | 0.312   | 0.907    | 0.210       | 9.994 | L1=12046 L2=305 L3=5 | 24/24 | **val < 10.0!** |
| 5390  | 8.17  | 0.314    | 0.053       | 0.312   | 0.907    | 0.210       | -      | L1=20354 L2=869 L3=7 | 24/24 | L3 concepts: 7 |
| 5610  | 7.50  | 0.322    | 0.055       | 0.312   | 0.907    | 0.210       | -      | L1=29849 L2=881 L3=8 | 24/24 | L3 birthed all 8 |
| 6105  | 6.98  | 0.340    | 0.059       | 0.312   | 1.230    | 0.210       | -      | L1=56537 L2=1554 L3=8 | 24/24 | L3 spiking |
| 6215  | 6.54  | 0.344    | 0.059       | 0.312   | 1.330    | 0.210       | -      | L1=60487 L2=1949 L3=8(39) | 24/24 | L3=39 births |
| 6325  | 6.00  | 0.348    | 0.060       | 0.312   | 1.350    | 0.210       | -      | L1=74757 L2=3555 L3=8(139) | 24/24 | L3=139 births! |
| 6545  | 5.19  | 0.356    | 0.062       | 0.312   | 1.380    | 0.210       | -      | L1=96307 L2=6870 L3=8(247) | 24/24 | |
| 6765  | 4.41  | 0.365    | 0.064       | 0.312   | 1.400    | 0.210       | -      | L1=120244 L2=10898 L3=8(299) | 24/24 | |
| 7040  | 3.65  | 0.375    | 0.066       | 0.312   | 1.404    | 0.210       | -      | L1=148986 L2=18448 L3=8(522) | 24/24 | L2 doubling |
| 7260  | 3.33  | 0.383    | 0.068       | 0.312   | 1.404    | 0.210       | -      | L1=177235 L2=22069 L3=8(664) | 24/24 | **lbg_global=1.0!** |
| 7535  | 3.15  | 0.393    | 0.070       | 0.312   | 1.404    | 0.210       | -      | L1=211263 L2=27748 L3=8(1009) | 24/24 | L3=1009 births |
| 8085  | 3.01  | 0.412    | 0.075       | 0.312   | 1.404    | 0.210       | -      | L1=286243 L2=32464 L3=8(2965) | 24/24 | |
| 8580  | 2.85  | 0.430    | 0.080       | 0.312   | 1.404    | 0.210       | -      | L1=363553 L2=50023 L3=8(4867) | 24/24 | **mat=0.311** |
| 9020  | 2.70  | 0.448    | 0.084       | 0.312   | 1.404    | 0.210       | -      | L1=462624 L2=58443 L3=8(6779) | 24/24 | **mat=0.333** |
| 9553  | 2.45  | 0.594    | 0.167       | 0.351   | 1.424    | 0.109       | 9.181 | L1=40920 L2=40920(8782c) L3=8(4331b) | 24/24 | **Novelty gate active** |
| 10019 | 2.30  | 0.620    | 0.183       | 0.349   | 1.445    | 0.117       | 9.164 | L1=95087 L2=95087(25444c) L3=8(6650b) | 24/24 | **L2 consumed tripled** |

### Ключевые вехи

| Шаг | Событие |
|-----|---------|
| 233 | Первый eval: val=13.238 |
| 3728 | DepthController поднял active_depth до 12/24 (plateau) |
| 4194 | DepthController: 16/24 |
| 4427 | DepthController: 20/24 |
| 4660 | DepthController: 24/24 (максимум) |
| 4675 | **Memory Bank проснулся** — L1=25, L2=17, L3=1 |
| 4840 | L3 concepts: 3 births |
| 5126 | **val < 10.0** (val=9.994) |
| 5390 | L3 concepts: 7 births |
| 5610 | L3 concepts: 8 (все 8 родились) |
| 6105 | L3=8, concept count растёт |
| 6215 | L3=8(39 births), L2=1949 |
| 6325 | L3=8(139 births), L2=3555 — L3 spiking |
| 6545 | L3=8(247 births), L2=6870 |
| 7260 | **lbg_global_ready = 1.0** — layer bridge gate fully open |
| 7535 | L3=8(1009 births), L2=27748 |
| 8085 | L3=8(2965 births), L2=32464 |
| 8580 | **mat_gate = 0.311** (threshold breach!) |
| 8854 | **val = 9.307** |
| 9553 | **Novelty gate active** — L2 consumed=8782, L3 births=4331 |
| 10019 | **val = 9.164** (current best), L2 consumed=25444 (3x) |

### Наблюдения

**Фаза 1 (step 0–220): Старт и spike**
- CE spike 11→44→44→38→13 — норма для fresh start
- Maturation gate: deep=0.12, shallow=0.02

**Фаза 2 (step 220–700): Plateau на ~11.0**
- CE ~11.0 (≈ random baseline ln(65536)=11.09)
- Maturation gate ~0.06-0.14, memory bank dormant

**Фаза 3 (step 700–3728): Медленное снижение, maturation ramp**
- CE: 10.84→9.93, val: 10.95→10.30
- Maturation deep: 0.138→0.250, readiness: 0.444→0.853
- Intent weight saturated ~0.91

**Фаза 4 (step 3728–4675): DepthController expansion**
- Active depth: 8→12→16→20→24 (full expansion over ~940 steps)
- Depth plateaus at 12, then jumps in bursts
- Memory bank still dormant (mat=0.25-0.29)

**Фаза 5 (step 4675–5600): Memory Bank activation**
- Memory bank wakes up at step 4675 (L1=25, L2=17, L3=1)
- L1 grows exponentially: 25→29849 in ~935 steps
- L3 concepts: 1→8 births by step 5610
- CE drops: 9.55→7.50, val: 10.09→9.90

**Фаза 6 (step 5600–7260): L3 explosion and layer bridge gate**
- L3 births explode: 8→522 in ~1430 steps
- L2 grows: 881→22069
- intent_w: 0.907→1.404 (increases from memory bank usage)
- **lbg_global_ready = 1.0** at step 7260

**Фаза 7 (step 7260–9020): Mature phase**
- Memory bank: L1=462k, L2=58k, L3=8(6779 births)
- mat_gate reaches 0.333 (threshold = 0.3)
- CE: 3.33→2.70, val: 9.59→9.31
- mod_mlp stable at 0.312

**Фаза 8 (step 9020–10019): Novelty gate and L2 consumed**
- **Novelty gate active** — concept birth decoupled from maturation
- L2 consumed: 8782→25444 (3x growth)
- L3 births: 4331→6650
- val: 9.31→9.16 (improving)
- maturation: 0.333→0.386

### Масштаб memory bank

```
Step  | L1 entries | L2 entries | L2 consumed | L3 births | L3 capacity
4675  | 25         | 17         | 0           | 1         | 1/8
5610  | 29,849     | 881        | -           | 8         | 8/8 (full)
6215  | 60,487     | 1,949      | -           | 39        | 8/8
6325  | 74,757     | 3,555      | -           | 139       | 8/8
7040  | 148,986    | 18,448     | -           | 522       | 8/8
7535  | 211,263    | 27,748     | -           | 1,009     | 8/8
8085  | 286,243    | 32,464     | -           | 2,965     | 8/8
8580  | 363,553    | 50,023     | -           | 4,867     | 8/8
9020  | 462,624    | 58,443     | -           | 6,779     | 8/8
9553  | 40,920     | 40,920     | 8,782       | 4,331     | 8/8
10019 | 95,087     | 95,087     | 25,444      | 6,650     | 8/8
```

> **Наблюдение:** L2 consumed вырос с 8,782 до 25,444 за 466 шагов (3x). L3 активно кластеризует L2, освобождая слоты. Это подтверждает эффективность consumed tracking.

### Прогноз

- **Step 10000–15000:** val < 9.0, mat_gate > 0.5, L2 consumed > 50,000
- **Step 15000–20000:** val < 8.5 (цель), mat_gate > 0.6
- **Step 20000–30000:** val < 8.0, memory bank fully saturated
- **Цель:** val < 8.5 (лучше прошлого раунда 8.734)

**Текущий темп:** val снижается на ~0.001 за 100 шагов. При текущей скорости 23 tok/s до step 15000 осталось ~21 час.

---

## Новый запуск (Colab T4, hybrid attention, fresh start 3)

> **2026-09-02:** Новый запуск с нуля после унификации hybrid attention.

### Изменения相对于 предыдущего запуска:
- **Hybrid attention** везде: `_memory_attention()`, cascade mixing, manifold beam read
- `log_temp` → `log_tau` (единое имя параметров)
- L2 keys: `F.normalize() + sigmoid(key_log_scale)`
- L2/L3 vals: `sigmoid(val_log_scale)`
- `analyze.py` исправлен (tokens param, quantile crash, unexpected keys)

### Траектория val

```
step     0: loss=9172.12  ce=23.67  mat=0.055  scale=-0.964  tok/s=22
step    55: loss=11169.33 ce=36.32  mat=0.056  scale=-0.964  tok/s=40 (spike, warmup)
step   110: loss=10847.88 ce=31.56  mat=0.056  scale=-0.964  tok/s=40
step   165: loss=10527.15 ce=28.85  mat=0.057  scale=-0.964  tok/s=40
step   220: loss=10583.49 ce=11.05  mat=0.058  scale=-0.964  tok/s=40
step   233: val=11.0681 val_ppl=64,100
step   275: loss=11532.91 ce=11.03  mat=0.058  scale=-0.964  tok/s=32
step   330: loss=11291.37 ce=11.03  mat=0.059  scale=-0.964  tok/s=33
step   385: loss=11739.30 ce=10.99  mat=0.060  scale=-0.964  tok/s=34
step   440: loss=10691.61 ce=10.99  mat=0.061  scale=-0.964  tok/s=34
step   466: val=11.0115 val_ppl=60,600
step   495: loss=10897.84 ce=10.97  mat=0.061  scale=-0.964  tok/s=31
step   550: loss=10551.71 ce=10.96  mat=0.062  scale=-0.964  tok/s=32
step   605: loss=10450.48 ce=10.97  mat=0.063  scale=-0.964  tok/s=32
step   660: loss=10899.25 ce=10.92  mat=0.064  scale=-0.964  tok/s=33
step   699: val=10.9498 val_ppl=56,900   ← best.pt
step   715: loss=12180.94 ce=10.84  mat=0.064  scale=-0.964  tok/s=31
step   770: loss=13038.94 ce=10.82  mat=0.065  scale=-0.964  tok/s=31
step   825: loss=13530.45 ce=10.88  mat=0.066  scale=-0.964  tok/s=32
step   880: loss=12099.60 ce=10.87  mat=0.067  scale=-0.964  tok/s=32
step   932: val=10.8937 val_ppl=53,800   ← best.pt
step   935: loss=13294.66 ce=10.73  mat=0.068  scale=-0.964  tok/s=31
step   990: loss=13995.67 ce=10.79  mat=0.069  scale=-0.964  tok/s=31
step  1045: loss=14211.06 ce=10.73  mat=0.070  scale=-0.964  tok/s=32
step  1100: loss=15153.06 ce=10.80  mat=0.070  scale=-0.964  tok/s=32
step  1155: loss=14678.31 ce=10.76  mat=0.071  scale=-0.964  tok/s=32
step  1165: val=10.8391 val_ppl=51,000   ← best.pt
step  1210: loss=14479.18 ce=10.78  mat=0.072  scale=-0.964  tok/s=31
step  1265: loss=13859.47 ce=10.78  mat=0.073  scale=-0.964  tok/s=32
```

### Наблюдения

1. **CE spike на step 55** (23→36) — аномалия при lr warmup, восстановился к step 220
2. **Maturation растёт медленно:** 0.055→0.073 за 1265 шагов. До порога 0.3 нужно ~20k шагов
3. **scale=-0.964 не двигается** — memory bank пуст, hybrid параметры не задействованы, градиент не течёт
4. **L1/L2/L3 пусты** — maturation ещё не достиг порогов (ожидаемо)
5. **VRAM 6.4GB** — T4 с запасом
6. **tok/s 31-40** — стабильно
7. **Total loss растёт** (12k→15k) — aux losses увеличиваются, CE снижается. Тревожно: aux может начать доминировать над CE
8. **intent_w saturating** ~0.78 — максимальный вклад intent
9. **val_loss** 11.07→10.84 — стабильное снижение

### Прогноз

- **Step 2000-3000:** maturation начнёт расти быстрее
- **Step 5000-8000:** maturation > 0.1, L1/L2 начнут заполняться
- **Step 10000-15000:** maturation > 0.3, memory bank активируется, hybrid scale начнёт обучаться
- **Цель:** val < 10.0 к step 5000, val < 9.0 к step 15000
