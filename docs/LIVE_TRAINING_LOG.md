# Живой журнал обучения WideBind

> Единственный актуальный лог текущего запуска. Исторические прогоны (прогон №2 / AMP / fp32-расхождение) удалены — при необходимости см. git-историю.

## Текущий запуск (Colab T4, fp32, архитектура зрелости по компетентности моста)

- Контур: `notebooks/colab.ipynb`, Colab T4, **fp32 (`use_amp=False`)**, ~28 tok/s, ~9.8 GB VRAM.
- Резюм `checkpoints/best.pt` (`FORCE_FRESH=False`); `eval_interval=2000`, `save_interval=987`, `max_steps=300000`.
- Watchdog `FailureDetector`: откат на `best.pt` при CE-спайке > `watchdog_ce=15.0`.
- При резюме cognitive gate принудительно «открывается» (`mod_scale_mlp → cfg.mlp_mod_scale_reopen ≈ 1.099`), поэтому зрелость `mat` после резюма стартует снизу (~0.07) и снова набирает рамп.
- In-core SemanticBridge active: `bridge_conn=0.1`, `bridge_dim=256`, params = 2,034,948 (внутри модели).

### Конфигурация (Большая, обучаемая)

| Параметр | Значение |
|---|---|
| D / слои / G / bind_K | 2560 / 24 / 32 / 32 |
| vocab / seq_len (обуч.) | 65536 / 128 |
| Параметров | 146,125,268 (146.13M) |
| bridge_glu, intent_bridge, explicit_reasoning, reasoning_adaptive, collective_read_out, variable_precision, bind_twist_gate, maturation_enabled, triad_reason | True |
| bridge_conn | 0.1 (вес aux) |
| pm_write_delay | 0 (игнорируется при maturation_enabled) |
| mask_eos | False |
| use_amp | False (fp32) |

### Текущий чекпоинт (`checkpoints/best.pt`)

| Метрика | Значение |
|---|---|
| Шаг сохранения best | **8854** (обучение продолжается, последний лог step 8965). ⚠️ Локальный `checkpoints/best.pt` — пока snapshot шага 7223; новый best (8854) лежит в ФС Colab и не скопирован (свежий analyze не снят). |
| val_loss / ppl | 9.4425 / ~12613 |
| Зрелость `mat` | 0.716–0.755 (после прорыва осел ниже пика 0.755 — новый устойчивый режим) |
| `bridge_conn` (raw cosine-loss моста) | ~0.09 в новом бассейне (во время перестройки падал до ~0.015 — мост был гипер-компетентен и дестабилизировал ствол; после прорыва баланс восстановлен) |
| `mod_mlp` (live) | ~0.327–0.331 (здоров, не схлопнут) |
| Скорость / VRAM | ~28 tok/s на T4 / ~9.8 GB |
| NaN / Inf | 0 / 0 |

### Траектория val (монотонно улучшается)

10.0216 (5126) → 9.9749 (5359) → 9.9283 (5592) → 9.8816 (5825) → 9.8366 (6058) → 9.7956 (6291) → 9.7516 (6524) → 9.7113 (6757) → 9.6820 (6990) → 9.6390 (7223) → *перестройка (7689=11.09, 7922=15.74)* → **9.5201 (8388) → 9.4821 (8621) → 9.4425 (8854)** — новый best после прорыва.

### ⚠️ Событие перестройки (step 7689 → 7922)

После best **9.6390@7223** валидация пошла вверх: **EVAL 7689 = 11.09** (≈ порог случайного ln(65536)=11.09), затем **EVAL 7922 = 15.74 / ppl 6.86M** — катастрофический скачок. Одновременно `mat` вырос **0.735 → 0.748** (time-рамп `M_l = max(time_ramp, bridge_readiness)` продолжает расти с шагом, T0=20000 / T_delay=20000, и тянет гейт к 1.0), `bridge_conn` (raw) упал до **~0.02** (мост предельно компетентен, его injection в ствол открыт на полную), LR принудительно снижен до **1.5e-4**.

Интерпретация: живые ветви (BridgeGLU, injection семантического моста, intent-шина, запись приватной памяти) раскрылись **полностью** и ломают прежнее устойчивое внутреннее пространство — отсюда «перестройка». **Не обязательно расходимость до dead**: CE в трейне держится ~9.0–9.4 (ниже порога `watchdog_ce=15.0`), значит ствол жив; но val-траектория временно вне устойчивости.

**`best.pt` НЕ перезаписан** (эвалы хуже 9.6390, save не сработал) — чекпоинт 9.6390@7223 сохранён. Действия: наблюдать; если val не оседает обратно < ~10 за следующие ~1–2k шагов или CE пробьёт 15 — рассмотреть откат на `best.pt`@7223 (watchdog или ручной rollback, `FORCE_FRESH` остаётся False).

После EVAL 7922 адаптивный контроллер LR (`lr_adapt`) среагировал на нестабильность: `mult` упал до **0.25**, LR снижен до **7.5e-5** (step 7975+). `mat` продолжает расти (0.748 → 0.750), `bridge_conn` raw ~0.015–0.031 (мост предельно компетентен). CE трейна вернулся к **~8.7–9.3** (уровень до blowup) — ствол стабилизируется под пониженным LR. Новый eval после 7922 ещё не было; перестройка продолжается под наблюдением.

### ✅ Прорыв: перестройка разрешилась новым best (step 8155 → 8854)

После blowup 7922 (и временного снижения LR до 7.5e-5) валидация пошла **вниз** и пробила прежний плато 9.6390:

- **EVAL 8155 = 12.14 / ppl 186704** (промежуточный, хуже best — не сохранён).
- **EVAL 8388 = 9.5201 / ppl 13631** → **Saved best** (первый новый best после прорыва).
- **EVAL 8621 = 9.4821 / ppl 13122** → **Saved best**.
- **EVAL 8854 = 9.4425 / ppl 12613** → **Saved best** (текущий best). CE трейна ~8.5–9.6; `mat` осел к **0.716–0.726** (новый устойчивый режим, ниже пика 0.755 во время перестройки); `bridge_conn` raw вернулся к **~0.09** (сбалансированный, не гипер-компетентный как ~0.015 в момент дестабилизации); LR восстановлен до **~3e-4** (`lr_adapt` вернул mult к ~1.0 после прорыва).

Интерпретация: перестройка — **успешный прорыв** в лучший бассейн. Модель «переломала» старое устойчивое внутреннее пространство и нашла более обобщающее (val 9.44 вместо 9.64). Ствол оставался жив (CE < 15), watchdog не срабатывал. Обучение продолжается к цели `val ≈ 8.5`. Локальный `checkpoints/best.pt` — пока snapshot 7223; новый best 8854 лежит в ФС Colab (нужно скопировать для свежего analyze).

### Фазовый переход (созревание и каскад, step 4950 → 6105)

Единый гейт зрелости `M_l = max(time_ramp, bridge_readiness)`:

- **Рампа зрелости (4950–5940):** после резюма `mat` стартует с ~0.07 и поднимается по time/τ-рампе до плато **0.624** (step ~5225–5940). `bridge_conn` (raw) в этой фазе **высокий (0.23–0.24)** — мост ещё не компетентен, живые ветви прикрыты.
- **Каскад созревания (~5995–6105):** `bridge_readiness` догоняет — `bridge_conn` (raw) падает **0.237 → 0.074 → 0.038** (мост научился предсказывать эталонный next-token), `mat` прыгает **0.626 → 0.684 → 0.734**, `ce` **9.29 → 8.80**. Здоровый каскад (не расходимость): maturation-гейт держит `ρ(J_l) ≈ 1`.
- **Плато (6105 → 7205+):** `mat` стабилизируется **0.730–0.735**, `bridge_conn` (raw) ~0.03–0.07, живые ветви (live BridgeGLU, injection моста, intent-шина, запись приватной памяти) раскрыты полностью; `mod_mlp`~0.327, CE плавает 8.5–10.1.

### Вердикт analyze (`checkpoints/best.pt`@7223) — WAKE-CANDIDATE

- MLP проснулся: `W_std` = 0.0704 ≈ decay-базис (0.0698, dev +0.0006); gate max 0.751 (порог WAKE 0.75). `sigmoid(mod_scale_mlp)` = 0.749, `sigmoid(mod_scale_mem)` = 0.667.
- Maturation gate 0.7355 (полностью открыт, max 0.736); `bridge_readiness` max = 0.780, `tau_norm` max = 0.997 — мост реально включается.
- Intent-шина: слои несут **дополняющие** сигналы (cross-layer cosine offdiag = 0.0524, далеко от 1); `bus_head_proj` растёт с zero-init (стенсил учится: norm 0.489), `bus` norm вырос до 4269.
- Концепты: slots 76/192, births в пустых слоях **[14]** (штатное рождение нового концепта, WATCH).
- Triad (`triad_reason`): при генерации ствол ре-циркулируется, если `_conf < 0.5` (до `triad_max_passes=3`), бленд `h = 0.5·h + 0.5·h2`. Только inference.
- Минорные WATCH: `pred` aux DEAD (`requires_grad=False`, `cos_sim(diversity, CE)=-0.72`, но `||gCE||≫||gDIV||` ⇒ вклад capped в 0); **72 unexpected tensors** = буферы `collective._resvar_ema/_resvar_var/_mature_count` (пересоздаются на резюме, EMA зрелости коллектива сбрасывается — кандидат на рефактор: зарегистрировать в `__init__`).
-   HTML-отчёт: `checkpoints/best_7223_report.html`.

> ⚠️ Локальный `checkpoints/best.pt` на момент analyze — snapshot шага **7223** (val 9.6390). Новый best **8854** (val 9.4425) существует в ФС Colab, но не скопирован сюда — свежий вердикт по 8854 ещё не снят.

**Цель:** выйти к историческому рубежу `val ≈ 8.5` при сохранении устойчивого CE.

### Сырой лог (текущий запуск, step 4934 → 8965)

```
In-core SemanticBridge active (bridge_conn=0.1, bridge_dim=256, params=2,034,948)
Training: step 4934 -> 300000
  (295066 steps remaining)
  batch=1 seq=128 -> tokens/step=128
  lr_adapt: var(ls)=13.826442 |1-a|=0.096238 gate_var=0.027234 |mirror|=229.4501 tau_var=13.826442 mult=1.0000 lr=3.00e-04 ls_mult[min=1.000 max=1.000]
step=  4950  loss=17244.4960  ce=10.1561  mod_mlp=0.370 mod_std=0.017 lr=2.71e-04  tok/s=33  mem=9.8GB  intent_w=1.0857  mlp_out=680.6 usef=0.500  mat=0.071[0.071,0.075]
  aux: alpha_novelty=-0.0005 balance=0.0037 branch=62.7507 bridge_conn=0.1166 decorr=0.0435 div=-0.1164 diversity=88.7674 gate_l1=0.6044 gate_repulse=-0.0385 gradalign=22.9064 intent_tau=0.6592 ls_reg=10.5243 nuc=161.4851 pred=2.7078 ranking=16882.2461 reinforce=0.0978 signal_ent=1.4241 w_m2v=0.1583
step=  5005  loss=16589.2152  ce=9.4930  mod_mlp=0.355 mod_std=0.024 lr=2.63e-04  tok/s=34  mem=9.8GB  intent_w=1.1559  mlp_out=691.7 usef=0.499  mat=0.301[0.301,0.301]
  aux: alpha_novelty=-0.0005 balance=0.0061 branch=59.8303 bridge_conn=0.2346 decorr=0.0492 div=-0.1164 diversity=90.7959 gate_l1=0.5793 gate_repulse=-0.0557 gradalign=22.4302 intent_tau=0.6592 ls_reg=10.5211 nuc=163.5605 pred=2.7327 ranking=16226.8066 reinforce=0.1065 signal_ent=1.4241 w_m2v=0.1583
step=  5060  loss=18776.6878  ce=10.1048  mod_mlp=0.346 mod_std=0.032 lr=2.78e-04  tok/s=34  mem=9.8GB  intent_w=1.1567  mlp_out=700.4 usef=0.499  mat=0.446[0.446,0.446]
  aux: alpha_novelty=-0.0005 balance=0.0070 branch=56.8113 bridge_conn=0.2367 decorr=0.0488 div=-0.1164 diversity=88.3108 gate_l1=0.5777 gate_repulse=-0.0585 gradalign=21.9366 intent_tau=0.6592 ls_reg=10.5180 nuc=158.1258 pred=2.6891 ranking=18425.1426 reinforce=0.1127 signal_ent=1.4241 w_m2v=0.1583
step=  5115  loss=21175.7445  ce=9.8732  mod_mlp=0.341 mod_std=0.041 lr=2.83e-04  tok/s=34  mem=9.8GB  intent_w=1.1565  mlp_out=706.1 usef=0.499  mat=0.526[0.526,0.526]
  aux: alpha_novelty=-0.0005 balance=0.0074 branch=61.0523 bridge_conn=0.2372 decorr=0.0539 div=-0.1164 diversity=84.9034 gate_l1=0.5948 gate_repulse=-0.0641 gradalign=21.3297 intent_tau=0.6592 ls_reg=10.5147 nuc=163.6150 pred=2.6937 ranking=20818.6875 reinforce=0.1211 signal_ent=1.4241 w_m2v=0.1583
  EVAL step=5126: val_loss=10.0216 val_ppl=22507.09
  Saved best to best.pt
step=  5170  loss=19855.1412  ce=9.6998  mod_mlp=0.335 mod_std=0.049 lr=2.88e-04  tok/s=28  mem=9.8GB  intent_w=1.1563  mlp_out=706.1 usef=0.499  mat=0.619[0.619,0.619]
  aux: alpha_novelty=-0.0005 balance=0.0081 branch=62.9412 bridge_conn=0.2367 decorr=0.0550 div=-0.1164 diversity=86.9810 gate_l1=0.5825 gate_repulse=-0.0673 gradalign=20.5239 intent_tau=0.6592 ls_reg=10.5113 nuc=161.5979 pred=2.7100 ranking=19497.1094 reinforce=0.1270 signal_ent=1.4241 w_m2v=0.1583
step=  5225  loss=18873.1661  ce=9.8335  mod_mlp=0.335 mod_std=0.056 lr=2.92e-04  tok/s=29  mem=9.8GB  intent_w=1.1561  mlp_out=702.2 usef=0.499  mat=0.621[0.621,0.621]
  aux: alpha_novelty=-0.0005 balance=0.0073 branch=62.9849 bridge_conn=0.2363 decorr=0.0539 div=-0.1164 diversity=95.8580 gate_l1=0.5923 gate_repulse=-0.0671 gradalign=20.2378 intent_tau=0.6592 ls_reg=10.5079 nuc=163.2755 pred=2.6554 ranking=18504.7402 reinforce=0.1254 signal_ent=1.4241 w_m2v=0.1583
step=  5280  loss=20022.3371  ce=9.3670  mod_mlp=0.336 mod_std=0.063 lr=2.97e-04  tok/s=29  mem=9.8GB  intent_w=1.1558  mlp_out=698.4 usef=0.500  mat=0.622[0.622,0.622]
  aux: alpha_novelty=-0.0005 balance=0.0068 branch=63.9334 bridge_conn=0.2351 decorr=0.0524 div=-0.1164 diversity=91.8136 gate_l1=0.6061 gate_repulse=-0.0651 gradalign=19.9691 intent_tau=0.6592 ls_reg=10.5044 nuc=166.3732 pred=2.6110 ranking=19654.6777 reinforce=0.1277 signal_ent=1.4241 w_m2v=0.1583
step=  5335  loss=19809.8223  ce=9.4990  mod_mlp=0.336 mod_std=0.067 lr=2.98e-04  tok/s=30  mem=9.8GB  intent_w=1.1556  mlp_out=696.9 usef=0.500  mat=0.622[0.622,0.622]
  aux: alpha_novelty=-0.0005 balance=0.0067 branch=68.2098 bridge_conn=0.2344 decorr=0.0514 div=-0.1164 diversity=118.4021 gate_l1=0.6052 gate_repulse=-0.0647 gradalign=19.8936 intent_tau=0.6592 ls_reg=10.5009 nuc=163.1034 pred=2.5865 ranking=19414.5410 reinforce=0.1282 signal_ent=1.4241 w_m2v=0.1583
  EVAL step=5359: val_loss=9.9749 val_ppl=21480.85
  Saved best to best.pt
step=  5390  loss=21579.3361  ce=9.3163  mod_mlp=0.335 mod_std=0.070 lr=3.02e-04  tok/s=27  mem=9.8GB  intent_w=1.1554  mlp_out=691.4 usef=0.499  mat=0.624[0.624,0.624]
  aux: alpha_novelty=-0.0005 balance=0.0063 branch=66.6551 bridge_conn=0.2335 decorr=0.0529 div=-0.1164 diversity=115.7410 gate_l1=0.6178 gate_repulse=-0.0615 gradalign=19.8435 intent_tau=0.6592 ls_reg=10.4974 nuc=163.3369 pred=2.5811 ranking=21188.2617 reinforce=0.1295 signal_ent=1.4241 w_m2v=0.1583
  lr_adapt: var(ls)=13.797299 |1-a|=0.096083 gate_var=0.064116 |mirror|=219.7716 tau_var=13.802939 mult=0.9915 lr=2.97e-04 ls_mult[min=1.000 max=1.003]
step=  5445  loss=18699.6473  ce=9.7415  mod_mlp=0.335 mod_std=0.073 lr=3.01e-04  tok/s=28  mem=9.8GB  intent_w=1.1552  mlp_out=696.2 usef=0.499  mat=0.624[0.624,0.624]
  aux: alpha_novelty=-0.0005 balance=0.0067 branch=66.7463 bridge_conn=0.2338 decorr=0.0518 div=-0.1164 diversity=116.4736 gate_l1=0.5911 gate_repulse=-0.0619 gradalign=19.4496 intent_tau=0.6592 ls_reg=10.4939 nuc=162.3531 pred=2.5551 ranking=18308.7637 reinforce=0.1243 signal_ent=1.4241 w_m2v=0.1583
step=  5500  loss=18804.9149  ce=9.4932  mod_mlp=0.335 mod_std=0.075 lr=3.00e-04  tok/s=29  mem=9.8GB  intent_w=1.1550  mlp_out=703.7 usef=0.499  mat=0.624[0.624,0.624]
  aux: alpha_novelty=-0.0005 balance=0.0069 branch=65.7132 bridge_conn=0.2342 decorr=0.0523 div=-0.1164 diversity=101.0818 gate_l1=0.5851 gate_repulse=-0.0624 gradalign=19.2372 intent_tau=0.6592 ls_reg=10.4903 nuc=162.5631 pred=2.5455 ranking=18430.7266 reinforce=0.1232 signal_ent=1.4241 w_m2v=0.1583
step=  5555  loss=18910.3498  ce=10.0384  mod_mlp=0.335 mod_std=0.076 lr=2.99e-04  tok/s=29  mem=9.8GB  intent_w=1.1547  mlp_out=701.7 usef=0.498  mat=0.624[0.624,0.624]
  aux: alpha_novelty=-0.0005 balance=0.0072 branch=64.3634 bridge_conn=0.2343 decorr=0.0520 div=-0.1163 diversity=100.3827 gate_l1=0.5795 gate_repulse=-0.0636 gradalign=19.0990 intent_tau=0.6592 ls_reg=10.4868 nuc=161.5101 pred=2.5411 ranking=18538.8711 reinforce=0.1230 signal_ent=1.4241 w_m2v=0.1583
  EVAL step=5592: val_loss=9.9283 val_ppl=20502.73
  Saved best to best.pt
step=  5610  loss=19157.2335  ce=9.3778  mod_mlp=0.334 mod_std=0.077 lr=2.98e-04  tok/s=28  mem=9.8GB  intent_w=1.1545  mlp_out=706.7 usef=0.497  mat=0.624[0.624,0.624]
  aux: alpha_novelty=-0.0005 balance=0.0080 branch=63.7023 bridge_conn=0.2340 decorr=0.0499 div=-0.1163 diversity=83.6108 gate_l1=0.5715 gate_repulse=-0.0650 gradalign=19.0605 intent_tau=0.6592 ls_reg=10.4833 nuc=163.5666 pred=2.5419 ranking=18801.8398 reinforce=0.1271 signal_ent=1.4241 w_m2v=0.1583
step=  5665  loss=17934.9361  ce=9.4183  mod_mlp=0.334 mod_std=0.078 lr=2.94e-04  tok/s=28  mem=9.8GB  intent_w=1.1543  mlp_out=712.4 usef=0.497  mat=0.624[0.624,0.624]
  aux: alpha_novelty=-0.0005 balance=0.0084 branch=62.2630 bridge_conn=0.2354 decorr=0.0511 div=-0.1163 diversity=77.1884 gate_l1=0.5576 gate_repulse=-0.0686 gradalign=18.8985 intent_tau=0.6592 ls_reg=10.4799 nuc=163.8314 pred=2.5500 ranking=17587.2734 reinforce=0.1245 signal_ent=1.4241 w_m2v=0.1583
step=  5720  loss=18132.1395  ce=9.1653  mod_mlp=0.334 mod_std=0.079 lr=2.96e-04  tok/s=28  mem=9.8GB  intent_w=1.1541  mlp_out=715.9 usef=0.497  mat=0.623[0.623,0.623]
  aux: alpha_novelty=-0.0005 balance=0.0084 branch=60.4343 bridge_conn=0.2362 decorr=0.0510 div=-0.1163 diversity=70.8315 gate_l1=0.5583 gate_repulse=-0.0692 gradalign=18.9417 intent_tau=0.6592 ls_reg=10.4764 nuc=164.9314 pred=2.5469 ranking=17791.7773 reinforce=0.1252 signal_ent=1.4241 w_m2v=0.1583
step=  5775  loss=17723.0460  ce=9.2842  mod_mlp=0.334 mod_std=0.079 lr=3.01e-04  tok/s=29  mem=9.8GB  intent_w=1.1539  mlp_out=725.2 usef=0.496  mat=0.623[0.623,0.623]
  aux: alpha_novelty=-0.0005 balance=0.0084 branch=61.6536 bridge_conn=0.2371 decorr=0.0519 div=-0.1163 diversity=60.9052 gate_l1=0.5551 gate_repulse=-0.0663 gradalign=19.2160 intent_tau=0.6592 ls_reg=10.4729 nuc=163.5091 pred=2.5415 ranking=17392.4297 reinforce=0.1228 signal_ent=1.4241 w_m2v=0.1583
  EVAL step=5825: val_loss=9.8816 val_ppl=19567.84
  Saved best to best.pt
step=  5830  loss=17756.0448  ce=9.0678  mod_mlp=0.334 mod_std=0.079 lr=3.04e-04  tok/s=28  mem=9.8GB  intent_w=1.1537  mlp_out=718.2 usef=0.496  mat=0.622[0.622,0.622]
  aux: alpha_novelty=-0.0005 balance=0.0083 branch=63.3093 bridge_conn=0.2357 decorr=0.0504 div=-0.1163 diversity=82.9631 gate_l1=0.5515 gate_repulse=-0.0647 gradalign=19.2616 intent_tau=0.6592 ls_reg=10.4694 nuc=163.2122 pred=2.5289 ranking=17402.2051 reinforce=0.1215 signal_ent=1.4241 w_m2v=0.1583
step=  5885  loss=17262.2982  ce=9.3803  mod_mlp=0.335 mod_std=0.077 lr=3.01e-04  tok/s=28  mem=9.8GB  intent_w=1.1534  mlp_out=716.6 usef=0.497  mat=0.622[0.622,0.622]
  aux: alpha_novelty=-0.0005 balance=0.0083 branch=62.4052 bridge_conn=0.2377 decorr=0.0499 div=-0.1163 diversity=74.6484 gate_l1=0.5523 gate_repulse=-0.0652 gradalign=19.7435 intent_tau=0.6592 ls_reg=10.4659 nuc=161.8000 pred=2.5315 ranking=16918.2949 reinforce=0.1205 signal_ent=1.4241 w_m2v=0.1583
  lr_adapt: var(ls)=13.768724 |1-a|=0.095611 gate_var=0.066043 |mirror|=218.4194 tau_var=13.774396 mult=1.0000 lr=3.00e-04 ls_mult[min=1.001 max=1.006]
step=  5940  loss=17648.0902  ce=9.2284  mod_mlp=0.336 mod_std=0.076 lr=3.00e-04  tok/s=28  mem=9.8GB  intent_w=1.1532  mlp_out=707.1 usef=0.499  mat=0.622[0.622,0.622]
  aux: alpha_novelty=-0.0005 balance=0.0088 branch=63.4966 bridge_conn=0.2371 decorr=0.0526 div=-0.1163 diversity=79.6487 gate_l1=0.5490 gate_repulse=-0.0666 gradalign=19.8044 intent_tau=0.6592 ls_reg=10.4624 nuc=166.6372 pred=2.5356 ranking=17293.2500 reinforce=0.1212 signal_ent=1.4241 w_m2v=0.1583
step=  5995  loss=17977.1111  ce=9.2865  mod_mlp=0.336 mod_std=0.076 lr=2.66e-04  tok/s=28  mem=9.8GB  intent_w=1.1715  mlp_out=718.2 usef=0.500  mat=0.626[0.626,0.626]
  aux: alpha_novelty=-0.0005 balance=0.0123 branch=58.9524 bridge_conn=0.0741 decorr=0.0538 div=-0.1163 diversity=31.8635 gate_l1=0.5435 gate_repulse=-0.0856 gradalign=19.5268 intent_tau=0.6592 ls_reg=10.4585 nuc=163.0489 pred=2.5218 ranking=17678.5879 reinforce=0.1420 signal_ent=1.4241 w_m2v=0.1583
step=  6050  loss=14516.9445  ce=8.8047  mod_mlp=0.330 mod_std=0.074 lr=2.40e-04  tok/s=29  mem=9.8GB  intent_w=1.2942  mlp_out=705.9 usef=0.502  mat=0.684[0.684,0.684]
  aux: alpha_novelty=-0.0005 balance=0.0242 branch=42.1315 bridge_conn=0.0377 decorr=0.0518 div=-0.1162 diversity=31.8160 gate_l1=0.4931 gate_repulse=-0.1362 gradalign=19.9710 intent_tau=0.6592 ls_reg=10.4541 nuc=165.5912 pred=2.5810 ranking=14232.8096 reinforce=0.1899 signal_ent=1.4241 w_m2v=0.1582
  EVAL step=6058: val_loss=9.8366 val_ppl=18705.62
  Saved best to best.pt
step=  6105  loss=11549.9395  ce=9.0469  mod_mlp=0.327 mod_std=0.075 lr=2.63e-04  tok/s=28  mem=9.8GB  intent_w=1.3451  mlp_out=730.7 usef=0.501  mat=0.734[0.734,0.734]
  aux: alpha_novelty=-0.0005 balance=0.0280 branch=39.5124 bridge_conn=0.0698 decorr=0.0563 div=-0.1162 diversity=34.2751 gate_l1=0.4550 gate_repulse=-0.1576 gradalign=19.4945 intent_tau=0.6591 ls_reg=10.4509 nuc=163.9374 pred=2.5915 ranking=11267.8594 reinforce=0.1951 signal_ent=1.4242 w_m2v=0.1582
step=  6160  loss=11216.2164  ce=9.3949  mod_mlp=0.327 mod_std=0.075 lr=2.78e-04  tok/s=28  mem=9.8GB  intent_w=1.3621  mlp_out=748.1 usef=0.502  mat=0.732[0.732,0.732]
  aux: alpha_novelty=-0.0005 balance=0.0296 branch=39.9740 bridge_conn=0.0711 decorr=0.0552 div=-0.1162 diversity=37.1533 gate_l1=0.4302 gate_repulse=-0.1606 gradalign=18.7122 intent_tau=0.6591 ls_reg=10.4477 nuc=161.6926 pred=2.5511 ranking=10933.5430 reinforce=0.1974 signal_ent=1.4242 w_m2v=0.1582
step=  6215  loss=9639.9325  ce=9.3267  mod_mlp=0.326 mod_std=0.073 lr=2.77e-04  tok/s=28  mem=9.8GB  intent_w=1.4545  mlp_out=725.0 usef=0.501  mat=0.735[0.735,0.735]
  aux: alpha_novelty=-0.0005 balance=0.0380 branch=31.2010 bridge_conn=0.0623 decorr=0.0564 div=-0.1162 diversity=51.3389 gate_l1=0.3968 gate_repulse=-0.1829 gradalign=19.6438 intent_tau=0.6592 ls_reg=10.4438 nuc=165.5657 pred=2.5944 ranking=9347.1074 reinforce=0.2154 signal_ent=1.4242 w_m2v=0.1582
step=  6270  loss=9165.0054  ce=9.3284  mod_mlp=0.328 mod_std=0.074 lr=2.91e-04  tok/s=29  mem=9.8GB  intent_w=1.4578  mlp_out=738.4 usef=0.502  mat=0.733[0.733,0.733]
  aux: alpha_novelty=-0.0005 balance=0.0360 branch=33.2024 bridge_conn=0.0681 decorr=0.0572 div=-0.1162 diversity=34.6335 gate_l1=0.3975 gate_repulse=-0.1762 gradalign=19.7579 intent_tau=0.6592 ls_reg=10.4405 nuc=162.3660 pred=2.5786 ranking=8889.9805 reinforce=0.2101 signal_ent=1.4242 w_m2v=0.1582
  EVAL step=6291: val_loss=9.7956 val_ppl=17954.66
  Saved best to best.pt
step=  6325  loss=9111.0518  ce=9.2120  mod_mlp=0.327 mod_std=0.074 lr=2.99e-04  tok/s=28  mem=9.8GB  intent_w=1.4575  mlp_out=741.3 usef=0.500  mat=0.730[0.730,0.730]
  aux: alpha_novelty=-0.0005 balance=0.0312 branch=34.9204 bridge_conn=0.0694 decorr=0.0565 div=-0.1162 diversity=22.6763 gate_l1=0.4095 gate_repulse=-0.1639 gradalign=19.5623 intent_tau=0.6592 ls_reg=10.4370 nuc=166.1653 pred=2.5451 ranking=8842.8105 reinforce=0.1952 signal_ent=1.4242 w_m2v=0.1582
step=  6380  loss=10322.1458  ce=9.6259  mod_mlp=0.327 mod_std=0.077 lr=3.02e-04  tok/s=28  mem=9.8GB  intent_w=1.4999  mlp_out=754.1 usef=0.499  mat=0.730[0.730,0.730]
  aux: alpha_novelty=-0.0005 balance=0.0308 branch=38.0919 bridge_conn=0.0683 decorr=0.0580 div=-0.1161 diversity=18.7145 gate_l1=0.4085 gate_repulse=-0.1596 gradalign=18.9672 intent_tau=0.6591 ls_reg=10.4335 nuc=164.7466 pred=2.6532 ranking=10056.1914 reinforce=0.1908 signal_ent=1.4242 w_m2v=0.1582
  lr_adapt: var(ls)=13.732357 |1-a|=0.095741 gate_var=0.155633 |mirror|=227.6137 tau_var=13.738281 mult=1.0110 lr=3.03e-04 ls_mult[min=1.001 max=1.008]
step=  6435  loss=10555.8609  ce=8.9819  mod_mlp=0.328 mod_std=0.076 lr=3.03e-04  tok/s=28  mem=9.8GB  intent_w=1.5167  mlp_out=752.5 usef=0.499  mat=0.730[0.730,0.730]
  aux: alpha_novelty=-0.0005 balance=0.0297 branch=39.1991 bridge_conn=0.0685 decorr=0.0596 div=-0.1161 diversity=20.4201 gate_l1=0.4131 gate_repulse=-0.1558 gradalign=18.9245 intent_tau=0.6591 ls_reg=10.4300 nuc=164.0620 pred=2.7407 ranking=10288.3750 reinforce=0.1876 signal_ent=1.4242 w_m2v=0.1582
step=  6490  loss=10355.7069  ce=9.8227  mod_mlp=0.327 mod_std=0.078 lr=3.07e-04  tok/s=28  mem=9.8GB  intent_w=1.5167  mlp_out=747.1 usef=0.497  mat=0.730[0.730,0.730]
  aux: alpha_novelty=-0.0005 balance=0.0261 branch=40.6108 bridge_conn=0.0686 decorr=0.0584 div=-0.1161 diversity=21.0399 gate_l1=0.4287 gate_repulse=-0.1458 gradalign=18.6426 intent_tau=0.6591 ls_reg=10.4265 nuc=162.6291 pred=2.7426 ranking=10087.0537 reinforce=0.1781 signal_ent=1.4242 w_m2v=0.1583
  EVAL step=6524: val_loss=9.7516 val_ppl=17181.54
  Saved best to best.pt
step=  6545  loss=10596.1228  ce=9.8317  mod_mlp=0.328 mod_std=0.079 lr=3.15e-04  tok/s=28  mem=9.8GB  intent_w=1.5164  mlp_out=761.5 usef=0.498  mat=0.730[0.730,0.730]
  aux: alpha_novelty=-0.0005 balance=0.0212 branch=48.3606 bridge_conn=0.0687 decorr=0.0546 div=-0.1161 diversity=29.1893 gate_l1=0.4471 gate_repulse=-0.1280 gradalign=18.4454 intent_tau=0.6591 ls_reg=10.4228 nuc=164.3101 pred=2.6901 ranking=10310.1211 reinforce=0.1634 signal_ent=1.4242 w_m2v=0.1583
step=  6600  loss=10743.3384  ce=9.9444  mod_mlp=0.328 mod_std=0.081 lr=3.12e-04  tok/s=28  mem=9.8GB  intent_w=1.5161  mlp_out=753.7 usef=0.498  mat=0.730[0.730,0.730]
  aux: alpha_novelty=-0.0005 balance=0.0194 branch=48.4665 bridge_conn=0.0675 decorr=0.0533 div=-0.1161 diversity=36.6611 gate_l1=0.4631 gate_repulse=-0.1218 gradalign=18.7255 intent_tau=0.6591 ls_reg=10.4192 nuc=163.7982 pred=2.6826 ranking=10449.8760 reinforce=0.1587 signal_ent=1.4242 w_m2v=0.1583
step=  6655  loss=11003.3808  ce=10.0861  mod_mlp=0.328 mod_std=0.081 lr=3.12e-04  tok/s=28  mem=9.8GB  intent_w=1.5159  mlp_out=748.3 usef=0.499  mat=0.730[0.730,0.730]
  aux: alpha_novelty=-0.0005 balance=0.0172 branch=49.5062 bridge_conn=0.0659 decorr=0.0536 div=-0.1161 diversity=31.8268 gate_l1=0.4853 gate_repulse=-0.1118 gradalign=18.7288 intent_tau=0.6591 ls_reg=10.4156 nuc=160.4449 pred=2.5947 ranking=10716.9922 reinforce=0.1505 signal_ent=1.4242 w_m2v=0.1583
step=  6710  loss=10669.8212  ce=9.6265  mod_mlp=0.328 mod_std=0.082 lr=3.08e-04  tok/s=28  mem=9.8GB  intent_w=1.5156  mlp_out=733.1 usef=0.499  mat=0.731[0.731,0.731]
  aux: alpha_novelty=-0.0005 balance=0.0159 branch=52.5265 bridge_conn=0.0652 decorr=0.0523 div=-0.1161 diversity=25.3604 gate_l1=0.4949 gate_repulse=-0.1097 gradalign=18.8497 intent_tau=0.6591 ls_reg=10.4120 nuc=164.2590 pred=2.5875 ranking=10383.4072 reinforce=0.1488 signal_ent=1.4242 w_m2v=0.1583
  EVAL step=6757: val_loss=9.7113 val_ppl=16503.49
  Saved best to best.pt
step=  6765  loss=10493.4582  ce=9.5128  mod_mlp=0.328 mod_std=0.081 lr=2.99e-04  tok/s=28  mem=9.8GB  intent_w=1.5153  mlp_out=708.0 usef=0.499  mat=0.733[0.733,0.733]
  aux: alpha_novelty=-0.0005 balance=0.0166 branch=50.9702 bridge_conn=0.0618 decorr=0.0515 div=-0.1161 diversity=17.6157 gate_l1=0.4870 gate_repulse=-0.1133 gradalign=18.7729 intent_tau=0.6591 ls_reg=10.4084 nuc=167.0909 pred=2.5863 ranking=10213.7188 reinforce=0.1538 signal_ent=1.4242 w_m2v=0.1583
step=  6820  loss=10920.2238  ce=9.8700  mod_mlp=0.328 mod_std=0.080 lr=3.04e-04  tok/s=28  mem=9.8GB  intent_w=1.5151  mlp_out=701.5 usef=0.499  mat=0.733[0.733,0.733]
  aux: alpha_novelty=-0.0005 balance=0.0159 branch=55.5065 bridge_conn=0.0608 decorr=0.0538 div=-0.1161 diversity=16.6013 gate_l1=0.4958 gate_repulse=-0.1084 gradalign=18.7795 intent_tau=0.6591 ls_reg=10.4049 nuc=163.7702 pred=2.6465 ranking=10639.8525 reinforce=0.1497 signal_ent=1.4242 w_m2v=0.1583
step=  6875  loss=11424.3584  ce=9.4552  mod_mlp=0.328 mod_std=0.080 lr=3.15e-04  tok/s=28  mem=9.8GB  intent_w=1.5148  mlp_out=680.7 usef=0.498  mat=0.733[0.733,0.733]
  aux: alpha_novelty=-0.0005 balance=0.0126 branch=62.4556 bridge_conn=0.0614 decorr=0.0534 div=-0.1161 diversity=51.1574 gate_l1=0.5207 gate_repulse=-0.0936 gradalign=19.4586 intent_tau=0.6591 ls_reg=10.4014 nuc=160.6016 pred=2.6862 ranking=11105.3242 reinforce=0.1389 signal_ent=1.4242 w_m2v=0.1583
step=  6930  loss=12887.7856  ce=9.5039  mod_mlp=0.327 mod_std=0.080 lr=3.19e-04  tok/s=28  mem=9.8GB  intent_w=1.5145  mlp_out=689.9 usef=0.497  mat=0.733[0.733,0.733]
  aux: alpha_novelty=-0.0005 balance=0.0105 branch=61.0194 bridge_conn=0.0619 decorr=0.0543 div=-0.1161 diversity=29.8183 gate_l1=0.5382 gate_repulse=-0.0830 gradalign=19.3088 intent_tau=0.6591 ls_reg=10.3977 nuc=163.3574 pred=2.6897 ranking=12588.8506 reinforce=0.1332 signal_ent=1.4242 w_m2v=0.1583
  lr_adapt: var(ls)=13.702960 |1-a|=0.096055 gate_var=0.082228 |mirror|=230.5272 tau_var=13.708824 mult=1.0633 lr=3.19e-04 ls_mult[min=1.001 max=1.009]
step=  6985  loss=13284.0741  ce=9.1737  mod_mlp=0.327 mod_std=0.082 lr=3.12e-04  tok/s=28  mem=9.8GB  intent_w=1.5142  mlp_out=700.3 usef=0.497  mat=0.733[0.733,0.733]
  aux: alpha_novelty=-0.0005 balance=0.0103 branch=60.3771 bridge_conn=0.0618 decorr=0.0544 div=-0.1161 diversity=33.8055 gate_l1=0.5440 gate_repulse=-0.0820 gradalign=18.9101 intent_tau=0.6591 ls_reg=10.3940 nuc=162.8321 pred=2.6682 ranking=12983.0654 reinforce=0.1345 signal_ent=1.4242 w_m2v=0.1583
  EVAL step=6990: val_loss=9.6820 val_ppl=16026.00
  Saved best to best.pt
step=  7040  loss=12626.4141  ce=8.4676  mod_mlp=0.326 mod_std=0.082 lr=3.02e-04  tok/s=28  mem=9.8GB  intent_w=1.5140  mlp_out=720.9 usef=0.497  mat=0.733[0.733,0.733]
  aux: alpha_novelty=-0.0005 balance=0.0117 branch=60.1104 bridge_conn=0.0615 decorr=0.0529 div=-0.1161 diversity=67.5895 gate_l1=0.5327 gate_repulse=-0.0877 gradalign=18.4942 intent_tau=0.6591 ls_reg=10.3905 nuc=163.0561 pred=2.6662 ranking=12292.8057 reinforce=0.1379 signal_ent=1.4242 w_m2v=0.1583
step=  7095  loss=12500.9960  ce=8.8784  mod_mlp=0.328 mod_std=0.084 lr=2.95e-04  tok/s=28  mem=9.8GB  intent_w=1.5137  mlp_out=733.3 usef=0.499  mat=0.733[0.733,0.733]
  aux: alpha_novelty=-0.0005 balance=0.0124 branch=58.6687 bridge_conn=0.0634 decorr=0.0549 div=-0.1161 diversity=61.0174 gate_l1=0.5359 gate_repulse=-0.0938 gradalign=18.3102 intent_tau=0.6591 ls_reg=10.3870 nuc=160.3240 pred=2.6859 ranking=12177.8838 reinforce=0.1430 signal_ent=1.4242 w_m2v=0.1583
step=  7150  loss=11188.0758  ce=8.7941  mod_mlp=0.327 mod_std=0.083 lr=2.93e-04  tok/s=28  mem=9.8GB  intent_w=1.5135  mlp_out=723.9 usef=0.499  mat=0.733[0.733,0.733]
  aux: alpha_novelty=-0.0005 balance=0.0135 branch=57.6694 bridge_conn=0.0640 decorr=0.0551 div=-0.1160 diversity=60.1811 gate_l1=0.5184 gate_repulse=-0.1008 gradalign=18.4919 intent_tau=0.6591 ls_reg=10.3836 nuc=163.6138 pred=2.6593 ranking=10863.4619 reinforce=0.1455 signal_ent=1.4242 w_m2v=0.1583
step=  7205  loss=10737.4645  ce=9.5438  mod_mlp=0.326 mod_std=0.084 lr=2.56e-04  tok/s=28  mem=9.8GB  intent_w=1.6004  mlp_out=767.4 usef=0.498  mat=0.735[0.735,0.735]
  aux: alpha_novelty=-0.0005 balance=0.0261 branch=38.0365 bridge_conn=0.0335 decorr=0.0553 div=-0.1160 diversity=89.1218 gate_l1=0.4419 gate_repulse=-0.1516 gradalign=18.3614 intent_tau=0.6591 ls_reg=10.3799 nuc=162.8055 pred=2.6676 ranking=10403.8291 reinforce=0.1887 signal_ent=1.4242 w_m2v=0.1583
  EVAL step=7223: val_loss=9.6390 val_ppl=15351.94
  Saved best to best.pt
step=  7260  loss=9759.5801  ce=8.7531  mod_mlp=0.326 mod_std=0.083 lr=2.68e-04  tok/s=28  mem=9.8GB  intent_w=1.6259  mlp_out=760.0 usef=0.498  mat=0.734[0.734,0.734]
  aux: alpha_novelty=-0.0005 balance=0.0321 branch=33.4588 bridge_conn=0.0668 decorr=0.0537 div=-0.1160 diversity=46.1346 gate_l1=0.4172 gate_repulse=-0.1701 gradalign=18.6101 intent_tau=0.6591 ls_reg=10.3767 nuc=163.8017 pred=2.6684 ranking=9473.0469 reinforce=0.2051 signal_ent=1.4242 w_m2v=0.1582
step=  7315  loss=9725.1104  ce=9.2970  mod_mlp=0.326 mod_std=0.084 lr=2.83e-04  tok/s=28  mem=9.8GB  intent_w=1.6259  mlp_out=758.5 usef=0.499  mat=0.733[0.733,0.733]
  aux: alpha_novelty=-0.0005 balance=0.0314 branch=35.2925 bridge_conn=0.0672 decorr=0.0533 div=-0.1160 diversity=41.6581 gate_l1=0.4178 gate_repulse=-0.1673 gradalign=18.5441 intent_tau=0.6591 ls_reg=10.3735 nuc=161.1418 pred=2.6674 ranking=9443.4072 reinforce=0.2016 signal_ent=1.4242 w_m2v=0.1583
step=  7370  loss=9448.3372  ce=8.7470  mod_mlp=0.327 mod_std=0.082 lr=2.90e-04  tok/s=28  mem=9.8GB  intent_w=1.6257  mlp_out=752.0 usef=0.499  mat=0.732[0.732,0.732]
  aux: alpha_novelty=-0.0005 balance=0.0309 branch=36.1515 bridge_conn=0.0662 decorr=0.0533 div=-0.1160 diversity=17.1866 gate_l1=0.4199 gate_repulse=-0.1674 gradalign=18.8001 intent_tau=0.6591 ls_reg=10.3702 nuc=163.4018 pred=2.6993 ranking=9188.2520 reinforce=0.2009 signal_ent=1.4242 w_m2v=0.1583
step=  7425  loss=9465.9513  ce=9.2295  mod_mlp=0.328 mod_std=0.082 lr=2.95e-04  tok/s=28  mem=9.8GB  intent_w=1.6254  mlp_out=747.5 usef=0.501  mat=0.732[0.732,0.732]
  aux: alpha_novelty=-0.0005 balance=0.0303 branch=36.2459 bridge_conn=0.0650 decorr=0.0546 div=-0.1160 diversity=14.7597 gate_l1=0.4232 gate_repulse=-0.1668 gradalign=19.0145 intent_tau=0.6591 ls_reg=10.3668 nuc=162.3045 pred=2.7179 ranking=9208.5811 reinforce=0.2002 signal_ent=1.4242 w_m2v=0.1583
  lr_adapt: var(ls)=13.673193 |1-a|=0.095776 gate_var=0.167113 |mirror|=241.7639 tau_var=13.678796 mult=0.9863 lr=2.96e-04 ls_mult[min=1.001 max=1.010]
  EVAL step=7456: val_loss=9.7607 val_ppl=17338.47
step=  7480  loss=9497.0666  ce=8.8263  mod_mlp=0.329 mod_std=0.081 lr=2.98e-04  tok/s=28  mem=9.8GB  intent_w=1.6251  mlp_out=739.9 usef=0.502  mat=0.732[0.732,0.732]
  aux: alpha_novelty=-0.0005 balance=0.0292 branch=36.4771 bridge_conn=0.0633 decorr=0.0575 div=-0.1160 diversity=24.9303 gate_l1=0.4297 gate_repulse=-0.1654 gradalign=19.0552 intent_tau=0.6591 ls_reg=10.3634 nuc=162.1724 pred=2.7119 ranking=9229.7920 reinforce=0.1990 signal_ent=1.4242 w_m2v=0.1583
step=  7535  loss=9203.3830  ce=8.8863  mod_mlp=0.330 mod_std=0.079 lr=3.01e-04  tok/s=28  mem=9.8GB  intent_w=1.6249  mlp_out=734.0 usef=0.502  mat=0.732[0.732,0.732]
  aux: alpha_novelty=-0.0005 balance=0.0277 branch=37.5599 bridge_conn=0.0626 decorr=0.0551 div=-0.1160 diversity=11.4359 gate_l1=0.4357 gate_repulse=-0.1607 gradalign=18.9723 intent_tau=0.6591 ls_reg=10.3599 nuc=161.5496 pred=2.7160 ranking=8949.1641 reinforce=0.1937 signal_ent=1.4242 w_m2v=0.1583
  EVAL step=7689: val_loss=11.0899 val_ppl=65506.07
step=  7700  loss=9557.5917  ce=9.3681  mod_mlp=0.329 mod_std=0.077 lr=1.50e-04  tok/s=28  mem=9.8GB  intent_w=1.7163  mlp_out=712.2 usef=0.501  mat=0.722[0.722,0.722]
  aux: alpha_novelty=-0.0005 balance=0.0314 branch=35.5200 bridge_conn=0.0843 decorr=0.0525 div=-0.1159 diversity=19.3035 gate_l1=0.4179 gate_repulse=-0.1700 gradalign=19.0770 intent_tau=0.6591 ls_reg=10.3493 nuc=162.9566 pred=2.6534 ranking=9295.6230 reinforce=0.1999 signal_ent=1.4242 w_m2v=0.1581
step=  7755  loss=8298.2740  ce=9.0166  mod_mlp=0.328 mod_std=0.076 lr=1.50e-04  tok/s=28  mem=9.8GB  intent_w=1.7880  mlp_out=727.9 usef=0.501  mat=0.733[0.733,0.733]
  aux: alpha_novelty=-0.0005 balance=0.0335 branch=31.1364 bridge_conn=0.0216 decorr=0.0548 div=-0.1159 diversity=6.4541 gate_l1=0.3999 gate_repulse=-0.1687 gradalign=18.8842 intent_tau=0.6591 ls_reg=10.3468 nuc=163.5444 pred=2.6675 ranking=8053.5566 reinforce=0.2015 signal_ent=1.4242 w_m2v=0.1580
step=  7810  loss=8214.2540  ce=9.1472  mod_mlp=0.326 mod_std=0.075 lr=1.50e-04  tok/s=28  mem=9.8GB  intent_w=1.8374  mlp_out=712.4 usef=0.500  mat=0.742[0.742,0.742]
  aux: alpha_novelty=-0.0005 balance=0.0355 branch=28.9960 bridge_conn=0.0160 decorr=0.0524 div=-0.1159 diversity=8.0508 gate_l1=0.3961 gate_repulse=-0.1696 gradalign=19.0560 intent_tau=0.6591 ls_reg=10.3445 nuc=162.2924 pred=2.6918 ranking=7971.0186 reinforce=0.2015 signal_ent=1.4242 w_m2v=0.1580
step=  7865  loss=8074.0453  ce=9.0356  mod_mlp=0.326 mod_std=0.075 lr=1.49e-04  tok/s=28  mem=9.8GB  intent_w=1.8566  mlp_out=715.6 usef=0.500  mat=0.746[0.746,0.746]
  aux: alpha_novelty=-0.0005 balance=0.0361 branch=30.1148 bridge_conn=0.0242 decorr=0.0549 div=-0.1159 diversity=3.9566 gate_l1=0.3991 gate_repulse=-0.1722 gradalign=19.3132 intent_tau=0.6591 ls_reg=10.3426 nuc=165.5135 pred=2.6948 ranking=7830.4033 reinforce=0.2038 signal_ent=1.4242 w_m2v=0.1580
step=  7920  loss=8668.2774  ce=9.1071  mod_mlp=0.326 mod_std=0.075 lr=1.50e-04  tok/s=29  mem=9.8GB  intent_w=1.8917  mlp_out=716.3 usef=0.500  mat=0.748[0.748,0.748]
  aux: alpha_novelty=-0.0005 balance=0.0352 branch=32.6400 bridge_conn=0.0303 decorr=0.0571 div=-0.1159 diversity=6.0092 gate_l1=0.4003 gate_repulse=-0.1683 gradalign=19.4430 intent_tau=0.6591 ls_reg=10.3406 nuc=166.1099 pred=2.6734 ranking=8419.2754 reinforce=0.1995 signal_ent=1.4241 w_m2v=0.1580
  EVAL step=7922: val_loss=15.7416 val_ppl=6862653.58 - идет перестройка внутреннего пространства
  lr_adapt: var(ls)=13.645029 |1-a|=0.095813 gate_var=0.156830 |mirror|=242.0786 tau_var=13.649048 mult=0.2500 lr=7.50e-05 ls_mult[min=1.001 max=1.010]
step=  7975  loss=9323.3761  ce=9.2668  mod_mlp=0.326 mod_std=0.073 lr=7.50e-05  tok/s=28  mem=9.8GB  intent_w=1.8919  mlp_out=733.8 usef=0.501  mat=0.748[0.748,0.748]
  aux: alpha_novelty=-0.0005 balance=0.0302 branch=33.8738 bridge_conn=0.0308 decorr=0.0601 div=-0.1159 diversity=5.1596 gate_l1=0.4200 gate_repulse=-0.1549 gradalign=19.9317 intent_tau=0.6591 ls_reg=10.3397 nuc=163.7485 pred=2.6929 ranking=9075.6670 reinforce=0.1851 signal_ent=1.4241 w_m2v=0.1580
step=  8030  loss=8629.0476  ce=8.6672  mod_mlp=0.326 mod_std=0.072 lr=7.50e-05  tok/s=28  mem=9.8GB  intent_w=1.9124  mlp_out=730.7 usef=0.500  mat=0.750[0.750,0.750]
  aux: alpha_novelty=-0.0005 balance=0.0332 branch=35.1595 bridge_conn=0.0146 decorr=0.0606 div=-0.1159 diversity=5.9304 gate_l1=0.4062 gate_repulse=-0.1606 gradalign=20.2864 intent_tau=0.6591 ls_reg=10.3386 nuc=161.3159 pred=2.6854 ranking=8381.9941 reinforce=0.1912 signal_ent=1.4241 w_m2v=0.1580
step=  8085  loss=8615.3287  ce=10.7629  mod_mlp=0.326 mod_std=0.072 lr=7.50e-05  tok/s=29  mem=9.8GB  intent_w=1.9204  mlp_out=736.4 usef=0.500  mat=0.752[0.752,0.752]
  aux: alpha_novelty=-0.0005 balance=0.0339 branch=33.5756 bridge_conn=0.0140 decorr=0.0609 div=-0.1159 diversity=5.8840 gate_l1=0.3997 gate_repulse=-0.1584 gradalign=20.2574 intent_tau=0.6590 ls_reg=10.3377 nuc=162.9605 pred=2.6460 ranking=8366.2393 reinforce=0.1904 signal_ent=1.4241 w_m2v=0.1580
step=  8140  loss=8515.1822  ce=9.0990  mod_mlp=0.327 mod_std=0.072 lr=7.50e-05  tok/s=29  mem=9.8GB  intent_w=1.9277  mlp_out=735.2 usef=0.501  mat=0.753[0.753,0.753]
  aux: alpha_novelty=-0.0005 balance=0.0343 branch=33.6943 bridge_conn=0.0168 decorr=0.0594 div=-0.1159 diversity=7.5885 gate_l1=0.3952 gate_repulse=-0.1575 gradalign=20.2153 intent_tau=0.6590 ls_reg=10.3367 nuc=165.8490 pred=2.6190 ranking=8263.1172 reinforce=0.1902 signal_ent=1.4241 w_m2v=0.1580
  EVAL step=8155: val_loss=12.1373 val_ppl=186704.33
step=  8195  loss=10128.4274  ce=9.0670  mod_mlp=0.326 mod_std=0.072 lr=3.75e-05  tok/s=28  mem=9.8GB  intent_w=1.9380  mlp_out=748.7 usef=0.500  mat=0.754[0.754,0.754]
  aux: alpha_novelty=-0.0005 balance=0.0325 branch=37.5423 bridge_conn=0.0140 decorr=0.0600 div=-0.1159 diversity=8.7202 gate_l1=0.3998 gate_repulse=-0.1547 gradalign=20.2599 intent_tau=0.6590 ls_reg=10.3361 nuc=166.6111 pred=2.5978 ranking=9870.6289 reinforce=0.1877 signal_ent=1.4241 w_m2v=0.1580
step=  8250  loss=10389.3304  ce=9.4172  mod_mlp=0.327 mod_std=0.070 lr=3.75e-05  tok/s=28  mem=9.8GB  intent_w=1.9384  mlp_out=761.9 usef=0.500  mat=0.755[0.755,0.755]
  aux: alpha_novelty=-0.0005 balance=0.0322 branch=40.0051 bridge_conn=0.0152 decorr=0.0613 div=-0.1159 diversity=9.9292 gate_l1=0.3993 gate_repulse=-0.1522 gradalign=20.4167 intent_tau=0.6590 ls_reg=10.3356 nuc=163.6655 pred=2.5986 ranking=10130.2969 reinforce=0.1850 signal_ent=1.4241 w_m2v=0.1580
step=  8305  loss=9890.5964  ce=9.2073  mod_mlp=0.328 mod_std=0.072 lr=3.75e-05  tok/s=29  mem=9.8GB  intent_w=1.9384  mlp_out=755.3 usef=0.500  mat=0.755[0.755,0.755]
  aux: alpha_novelty=-0.0005 balance=0.0326 branch=41.8740 bridge_conn=0.0159 decorr=0.0610 div=-0.1159 diversity=9.6233 gate_l1=0.3938 gate_repulse=-0.1519 gradalign=20.2653 intent_tau=0.6590 ls_reg=10.3352 nuc=162.9141 pred=2.5908 ranking=9631.1260 reinforce=0.1842 signal_ent=1.4241 w_m2v=0.1580
step=  8360  loss=9075.5963  ce=8.7528  mod_mlp=0.328 mod_std=0.073 lr=3.74e-05  tok/s=29  mem=9.8GB  intent_w=1.9383  mlp_out=753.1 usef=0.500  mat=0.755[0.755,0.755]
  aux: alpha_novelty=-0.0005 balance=0.0339 branch=43.6568 bridge_conn=0.0165 decorr=0.0609 div=-0.1159 diversity=8.2976 gate_l1=0.3898 gate_repulse=-0.1558 gradalign=19.9980 intent_tau=0.6590 ls_reg=10.3347 nuc=164.3120 pred=2.6210 ranking=8814.9658 reinforce=0.1875 signal_ent=1.4241 w_m2v=0.1580
  EVAL step=8388: val_loss=9.5201 val_ppl=13631.05
  Saved best to best.pt
step=  8415  loss=8887.5812  ce=8.8437  mod_mlp=0.328 mod_std=0.072 lr=3.02e-04  tok/s=28  mem=9.8GB  intent_w=1.9382  mlp_out=753.1 usef=0.500  mat=0.754[0.754,0.754]
  aux: alpha_novelty=-0.0005 balance=0.0314 branch=40.2896 bridge_conn=0.0170 decorr=0.0590 div=-0.1159 diversity=5.5814 gate_l1=0.3978 gate_repulse=-0.1507 gradalign=19.9247 intent_tau=0.6590 ls_reg=10.3329 nuc=161.4272 pred=2.6356 ranking=8635.8838 reinforce=0.1830 signal_ent=1.4241 w_m2v=0.1580
  lr_adapt: var(ls)=13.635810 |1-a|=0.095820 gate_var=0.158342 |mirror|=237.2534 tau_var=13.638478 mult=0.9583 lr=2.87e-04 ls_mult[min=1.001 max=1.007]
step=  8470  loss=11769.2945  ce=9.1840  mod_mlp=0.327 mod_std=0.069 lr=2.91e-04  tok/s=28  mem=9.8GB  intent_w=1.9936  mlp_out=753.2 usef=0.499  mat=0.754[0.754,0.754]
  aux: alpha_novelty=-0.0005 balance=0.0303 branch=33.3418 bridge_conn=0.0150 decorr=0.0598 div=-0.1158 diversity=7.7635 gate_l1=0.4308 gate_repulse=-0.1667 gradalign=20.1193 intent_tau=0.6590 ls_reg=10.3292 nuc=166.3524 pred=2.6014 ranking=11516.9180 reinforce=0.1911 signal_ent=1.4241 w_m2v=0.1580
step=  8525  loss=8094.8478  ce=9.0717  mod_mlp=0.326 mod_std=0.070 lr=2.84e-04  tok/s=29  mem=9.8GB  intent_w=2.0559  mlp_out=740.2 usef=0.499  mat=0.744[0.744,0.744]
  aux: alpha_novelty=-0.0005 balance=0.0461 branch=30.1975 bridge_conn=0.0889 decorr=0.0625 div=-0.1158 diversity=4.5506 gate_l1=0.3784 gate_repulse=-0.1918 gradalign=19.5917 intent_tau=0.6589 ls_reg=10.3250 nuc=159.8787 pred=2.6335 ranking=7855.8701 reinforce=0.2201 signal_ent=1.4240 w_m2v=0.1580
step=  8580  loss=7924.3089  ce=8.8712  mod_mlp=0.327 mod_std=0.074 lr=2.93e-04  tok/s=29  mem=9.8GB  intent_w=2.0564  mlp_out=741.7 usef=0.498  mat=0.733[0.733,0.733]
  aux: alpha_novelty=-0.0005 balance=0.0432 branch=30.5030 bridge_conn=0.0914 decorr=0.0622 div=-0.1158 diversity=5.8594 gate_l1=0.3824 gate_repulse=-0.1835 gradalign=19.0955 intent_tau=0.6589 ls_reg=10.3217 nuc=162.0956 pred=2.6084 ranking=7682.2207 reinforce=0.2130 signal_ent=1.4240 w_m2v=0.1580
  EVAL step=8621: val_loss=9.4821 val_ppl=13122.12
  Saved best to best.pt
step=  8635  loss=7409.9347  ce=9.6091  mod_mlp=0.329 mod_std=0.079 lr=2.96e-04  tok/s=28  mem=9.8GB  intent_w=2.0561  mlp_out=736.7 usef=0.499  mat=0.718[0.718,0.718]
  aux: alpha_novelty=-0.0005 balance=0.0394 branch=31.1629 bridge_conn=0.0917 decorr=0.0614 div=-0.1158 diversity=6.2171 gate_l1=0.3865 gate_repulse=-0.1750 gradalign=18.8081 intent_tau=0.6589 ls_reg=10.3183 nuc=163.4566 pred=2.5995 ranking=7165.0293 reinforce=0.2050 signal_ent=1.4240 w_m2v=0.1580
step=  8690  loss=7423.5867  ce=9.1326  mod_mlp=0.329 mod_std=0.080 lr=2.98e-04  tok/s=28  mem=9.8GB  intent_w=2.0557  mlp_out=739.4 usef=0.499  mat=0.718[0.718,0.718]
  aux: alpha_novelty=-0.0005 balance=0.0378 branch=32.3246 bridge_conn=0.0918 decorr=0.0599 div=-0.1158 diversity=6.3897 gate_l1=0.3906 gate_repulse=-0.1732 gradalign=18.8970 intent_tau=0.6589 ls_reg=10.3149 nuc=161.6258 pred=2.5919 ranking=7179.5762 reinforce=0.2026 signal_ent=1.4240 w_m2v=0.1580
step=  8745  loss=7614.7562  ce=9.3414  mod_mlp=0.330 mod_std=0.080 lr=3.03e-04  tok/s=29  mem=9.8GB  intent_w=2.0554  mlp_out=741.8 usef=0.501  mat=0.717[0.717,0.717]
  aux: alpha_novelty=-0.0005 balance=0.0350 branch=37.3865 bridge_conn=0.0925 decorr=0.0597 div=-0.1158 diversity=33.5822 gate_l1=0.3971 gate_repulse=-0.1666 gradalign=19.3794 intent_tau=0.6589 ls_reg=10.3115 nuc=163.6557 pred=2.5777 ranking=7335.7812 reinforce=0.1983 signal_ent=1.4240 w_m2v=0.1580
step=  8800  loss=8159.4972  ce=8.5140  mod_mlp=0.331 mod_std=0.080 lr=3.07e-04  tok/s=29  mem=9.8GB  intent_w=2.0551  mlp_out=751.6 usef=0.501  mat=0.717[0.717,0.717]
  aux: alpha_novelty=-0.0005 balance=0.0323 branch=39.2229 bridge_conn=0.0936 decorr=0.0600 div=-0.1158 diversity=34.1128 gate_l1=0.4026 gate_repulse=-0.1578 gradalign=19.4478 intent_tau=0.6589 ls_reg=10.3080 nuc=165.0656 pred=2.5491 ranking=7877.5298 reinforce=0.1919 signal_ent=1.4240 w_m2v=0.1580
  EVAL step=8854: val_loss=9.4425 val_ppl=12612.93
  Saved best to best.pt
step=  8855  loss=8864.8481  ce=8.8638  mod_mlp=0.331 mod_std=0.081 lr=3.06e-04  tok/s=28  mem=9.8GB  intent_w=2.0548  mlp_out=757.9 usef=0.500  mat=0.716[0.716,0.716]
  aux: alpha_novelty=-0.0005 balance=0.0297 branch=37.5443 bridge_conn=0.0940 decorr=0.0601 div=-0.1158 diversity=18.2233 gate_l1=0.4147 gate_repulse=-0.1535 gradalign=19.7206 intent_tau=0.6589 ls_reg=10.3044 nuc=165.1575 pred=2.5706 ranking=8599.7070 reinforce=0.1870 signal_ent=1.4240 w_m2v=0.1580
step=  8910  loss=9333.7558  ce=9.2232  mod_mlp=0.330 mod_std=0.082 lr=3.09e-04  tok/s=28  mem=9.8GB  intent_w=2.0544  mlp_out=746.9 usef=0.499  mat=0.716[0.716,0.716]
  aux: alpha_novelty=-0.0005 balance=0.0263 branch=42.0214 bridge_conn=0.0932 decorr=0.0599 div=-0.1158 diversity=8.9997 gate_l1=0.4268 gate_repulse=-0.1433 gradalign=19.7509 intent_tau=0.6589 ls_reg=10.3009 nuc=161.3177 pred=2.5947 ranking=9076.7832 reinforce=0.1765 signal_ent=1.4240 w_m2v=0.1580
  lr_adapt: var(ls)=13.603713 |1-a|=0.095776 gate_var=0.145079 |mirror|=215.9910 tau_var=13.609920 mult=1.0188 lr=3.06e-04 ls_mult[min=1.001 max=1.009]
step=  8965  loss=11197.4635  ce=8.5357  mod_mlp=0.328 mod_std=0.083 lr=2.94e-04  tok/s=28  mem=9.8GB  intent_w=2.1406  mlp_out=762.8 usef=0.499  mat=0.726[0.726,0.726]
  aux: alpha_novelty=-0.0005 balance=0.0337 branch=41.7329 bridge_conn=0.0275 decorr=0.0574 div=-0.1157 diversity=23.2902 gate_l1=0.4163 gate_repulse=-0.1643 gradalign=19.6337 intent_tau=0.6588 ls_reg=10.2970 nuc=162.2160 pred=2.5452 ranking=10926.5195 reinforce=0.1982 signal_ent=1.4240 w_m2v=0.1580
```
