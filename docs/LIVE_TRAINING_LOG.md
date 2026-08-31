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
| **Локальный best (проанализирован)** | **15844** (val 8.7692, WAKE-CANDIDATE, отчёт `best_15844_report.html`). |
| Предыдущий локальный best | 10485 (val 9.1963) — перезаписан. |
| Текущий step на прогоне | ≥19635 (лог обрезан). val после best: 8.77→9.04 (вторичная перестройка). |
| Зрелость `mat` | 0.7226 (вторичная перестройка: mat снизился с 0.750 до 0.723). |
| `bridge_conn` (raw) | ~0.08 (вырос с 0.026@10485; мост перекалибруется). |
| `intent_w` | ~2.5 (вырос с 2.14@10485). |
| Концепты | slots 96/192 (было 91), full layers 11, births [10,12,13,14]. |
| `mod_mlp` (live) | ~0.327–0.331 (здоров). |
| Скорость / VRAM | ~28 tok/s на T4 / ~9.8 GB |
| NaN / Inf | 0 / 0 |

### Траектория val

10.0216 (5126) → 9.9749 (5359) → 9.9283 (5592) → 9.8816 (5825) → 9.8366 (6058) → 9.7956 (6291) → 9.7516 (6524) → 9.7113 (6757) → 9.6820 (6990) → 9.6390 (7223) → *перестройка (7689=11.09, 7922=15.74)* → **9.5201 (8388) → 9.4821 (8621) → 9.4425 (8854) → 9.4035 (9087) → 9.3663 (9320) → 9.3310 (9553) → 9.2989 (9786) → 9.2646 (10019) → 9.2307 (10252) → 9.1963 (10485) → 9.1613 (10718) → 9.1282 (10951) → 9.0983 (11184) → 9.0747 (11417) → 9.0462 (11650) → 9.0208 (11883) → 8.9964 (12116) → 8.9731 (12349) → 8.9503 (12582) → 8.9094 (13048) → 8.8930 (13281) → 8.8734 (13514) → 8.8562 (13747) → 8.8438 (13980) → 8.8299 (14213) → 8.8159 (14446) → 8.8034 (14679) → 8.7921 (14912) → 8.7838 (15145) → 8.7777 (15378) → 8.7727 (15611) → **8.7692 (15844)** ← лучший → 8.7701 (16077) → 8.7736 (16310) → 8.7812 (16543) → 8.7908 (16776) → 8.8016 (17009) → 8.8145 (17242) → 8.8303 (17475) → 8.8475 (17708) → 8.8679 (17941) → 8.8933 (18174) → 8.9211 (18407) → 8.9500 (18640) → 8.9811 (18873) → 9.0144 (19106) → 9.0439 (19339) → 9.0439 (19572). **Прорыв через 9.0** на step 12116; вторичная перестройка ~15565 → регрессия val.

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

Интерпретация: перестройка — **успешный прорыв** в лучший бассейн. Модель «переломала» старое устойчивое внутреннее пространство и нашла более обобщающее (val 9.44 вместо 9.64). Ствол оставался жив (CE < 15), watchdog не срабатывал. Обучение продолжается к цели `val ≈ 8.5`. Локальный `checkpoints/best.pt` = 10485 (свежий, проанализирован: WAKE-CANDIDATE, отчёт `best_10485_report.html`); на прогоне val продолжил снижаться **9.4425@8854 → 9.4035@9087 → 9.3663@9320 → 9.3310@9553 → 9.2989@9786 → 9.2646@10019 → 9.2307@10252 → 9.1963@10485**.

### Фазовый переход (созревание и каскад, step 4950 → 6105)

Единый гейт зрелости `M_l = max(time_ramp, bridge_readiness)`:

- **Рампа зрелости (4950–5940):** после резюма `mat` стартует с ~0.07 и поднимается по time/τ-рампе до плато **0.624** (step ~5225–5940). `bridge_conn` (raw) в этой фазе **высокий (0.23–0.24)** — мост ещё не компетентен, живые ветви прикрыты.
- **Каскад созревания (~5995–6105):** `bridge_readiness` догоняет — `bridge_conn` (raw) падает **0.237 → 0.074 → 0.038** (мост научился предсказывать эталонный next-token), `mat` прыгает **0.626 → 0.684 → 0.734**, `ce` **9.29 → 8.80**. Здоровый каскад (не расходимость): maturation-гейт держит `ρ(J_l) ≈ 1`.
- **Плато (6105 → 7205+):** `mat` стабилизируется **0.730–0.735**, `bridge_conn` (raw) ~0.03–0.07, живые ветви (live BridgeGLU, injection моста, intent-шина, запись приватной памяти) раскрыты полностью; `mod_mlp`~0.327, CE плавает 8.5–10.1.

⏳ *Сырой лог за 8965 → 19635+ получен и обработан (см. ниже); локальный `best.pt` = **10485** (val 9.1963, WAKE-CANDIDATE). Прогон ушёл значительно дальше — **лучший best прогона 8.7692@15844** (ppl ~6433). Концептов стало больше: slots 91/192, full layers 11. После 15844 val начал расти (вторичная перестройка). Цель `val ≈ 8.5` — осталось ~0.27.*

### Вердикт analyze (`checkpoints/best.pt`@15844) — WAKE-CANDIDATE

- MLP проснулся: `W_std` = 0.0701 ≈ decay-базис (0.0689, dev +0.0012); gate max 0.751 (порог WAKE 0.75, L7/L23/L5). `sigmoid(mod_scale_mlp)` = 0.748, `sigmoid(mod_scale_mem)` = 0.667.
- **Maturation gate 0.7226** (вторичная перестройка: был 0.750@10485 → 0.723@15844); `bridge_readiness` max = 0.7817, `tau_norm` max = 0.995 — мост ре-калибруется.
- Intent-шина: слои несут **дополняющие** сигналы (cross-layer cosine offdiag = 0.0786, далеко от 1); `bus_head_proj` растёт с zero-init (стенсил учится: norm 0.594), `bus` norm = 4302.
- Концепты: slots **96/192** (было 91), full layers 11, births в пустых слоях **[10, 12, 13, 14]** (продолжается штатное рождение новых концептов).
- Triad (`triad_reason`): при генерации ствол ре-циркулируется, если `_conf < 0.5` (до `triad_max_passes=3`), бленд `h = 0.5·h + 0.5·h2`. Только inference.
- Минорные WATCH: `pred` aux DEAD (`requires_grad=False`, `cos_sim(diversity, CE)=0`, но `||gCE||≫||gDIV||` ⇒ вклад capped в 0); **72 unexpected tensors** = буферы `collective._resvar_ema/_resvar_var/_mature_count` (пересоздаются на резюме, EMA зрелости коллектива сбрасывается).
- HTML-отчёт: `checkpoints/best_15844_report.html`.

**Цель:** выйти к историческому рубежу `val ≈ 8.5` при сохранении устойчивого CE.

### ✅ Прорыв через 9.0 (step 10718 → 15844)

После best 9.1963@10485 валидация продолжила **монотонное снижение** и пробила рубеж 9.0:

- **10718 = 9.1613** → 10951 = 9.1282 → 11184 = 9.0983 → 11417 = 9.0747 → 11650 = 9.0462 → 11883 = 9.0208 → **12116 = 8.9964** (прорыв через 9.0!) → 12349 = 8.9731 → 12582 = 8.9503 → 13048 = 8.9094 → 13281 = 8.8930 → 13514 = 8.8734 → 13747 = 8.8562 → 13980 = 8.8438 → 14213 = 8.8299 → 14446 = 8.8159 → 14679 = 8.8034 → 14912 = 8.7921 → 15145 = 8.7838 → 15378 = 8.7777 → 15611 = 8.7727 → **15844 = 8.7692** (лучший, ppl ~6433).

Динамика за прорыв: `mat` снижался с 0.750 до **0.742** (постепенно), `bridge_conn` raw вырос с 0.026 до **0.065** (мост стал менее компетентен, ствол адаптируется). CE трейна достиг **минимума ~6.98** (step 14575). `intent_w` вырос с 2.14 до **2.30**. LR стабилен ~3e-4.

### ⚠️ Вторичная перестройка (step ~15565 → 19635+)

После best **8.7692@15844** валидация пошла **вверх**: 8.7701(16077) → 8.7736 → 8.7812 → ... → 8.8933(18174) → 9.0439(19572).

Признаки (похожи на первую перестройку 7922, но мягче):
- `mat` **упал с 0.745 до 0.720–0.725** (как при первой перестройке mat падал с 0.748 до 0.716).
- `bridge_conn` raw **прыгнул с 0.04 до ~0.08** (мост стал менее компетентен → ствол перестраивается).
- `intent_w` вырос до **2.5** (мост активнее модулирует ствол).
- `mod_mlp` стабилен ~0.327–0.330 (здоров, не схлопнут).
- CE трейна продолжает снижаться до **~7.0** (на историческом минимуме).
- NaN/Inf: 0.

Интерпретация: вторичная перестройка — **нормальный цикл** зрелости. Модель «тестирует» новый бассейн (val 8.77), maturation-гейт слегка закрывается (mat 0.722), мост перекалибруется (bridge_conn растёт). Это похоже на первую перестройку (7922), но CE низкий — расходимости нет. Watchdog не срабатывал (CE < 15). **Возможно**, после оседания val снова пойдёт вниз к цели 8.5.

⏳ *Прогон продолжается (≥19635). Локальный best.pt = 10485. Лучший прогона = 8.7692@15844. Нужно: (1) скопировать best.pt локально → analyze; (2) наблюдать за оседанием после вторичной перестройки.*

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
step=  9020  loss=10812.3116  ce=9.1930  mod_mlp=0.326 mod_std=0.081 lr=2.96e-04  tok/s=29  mem=9.8GB  intent_w=2.1469  mlp_out=772.9 usef=0.498  mat=0.735[0.735,0.735]
  aux: alpha_novelty=-0.0005 balance=0.0352 branch=41.4393 bridge_conn=0.0338 decorr=0.0584 div=-0.1157 diversity=17.9315 gate_l1=0.4056 gate_repulse=-0.1659 gradalign=19.1009 intent_tau=0.6588 ls_reg=10.2936 nuc=163.6585 pred=2.5094 ranking=10545.4951 reinforce=0.1986 signal_ent=1.4240 w_m2v=0.1580
step=  9075  loss=10339.6043  ce=8.4390  mod_mlp=0.325 mod_std=0.081 lr=2.98e-04  tok/s=29  mem=9.8GB  intent_w=2.1466  mlp_out=773.3 usef=0.497  mat=0.740[0.740,0.740]
  aux: alpha_novelty=-0.0005 balance=0.0353 branch=41.6048 bridge_conn=0.0326 decorr=0.0590 div=-0.1157 diversity=9.4000 gate_l1=0.4018 gate_repulse=-0.1654 gradalign=18.7989 intent_tau=0.6588 ls_reg=10.2902 nuc=163.4893 pred=2.4950 ranking=10082.4023 reinforce=0.1971 signal_ent=1.4240 w_m2v=0.1580
  EVAL step=9087: val_loss=9.4035 val_ppl=12131.34
  Saved best to best.pt
step=  9130  loss=9793.8350  ce=9.3733  mod_mlp=0.324 mod_std=0.080 lr=2.97e-04  tok/s=28  mem=9.8GB  intent_w=2.1463  mlp_out=787.2 usef=0.497  mat=0.747[0.747,0.747]
  aux: alpha_novelty=-0.0005 balance=0.0376 branch=39.1603 bridge_conn=0.0313 decorr=0.0598 div=-0.1157 diversity=7.8273 gate_l1=0.3915 gate_repulse=-0.1703 gradalign=19.1489 intent_tau=0.6588 ls_reg=10.2868 nuc=165.0778 pred=2.4999 ranking=9537.7842 reinforce=0.2020 signal_ent=1.4240 w_m2v=0.1580
step=  9185  loss=9490.1790  ce=7.5547  mod_mlp=0.325 mod_std=0.079 lr=2.97e-04  tok/s=28  mem=9.8GB  intent_w=2.1460  mlp_out=803.9 usef=0.498  mat=0.748[0.748,0.748]
  aux: alpha_novelty=-0.0005 balance=0.0398 branch=37.3190 bridge_conn=0.0300 decorr=0.0616 div=-0.1157 diversity=11.9640 gate_l1=0.3840 gate_repulse=-0.1732 gradalign=19.0181 intent_tau=0.6588 ls_reg=10.2834 nuc=162.8615 pred=2.5213 ranking=9235.9834 reinforce=0.2068 signal_ent=1.4240 w_m2v=0.1580
step=  9240  loss=9546.1485  ce=8.7835  mod_mlp=0.326 mod_std=0.081 lr=3.00e-04  tok/s=29  mem=9.8GB  intent_w=2.1456  mlp_out=812.1 usef=0.499  mat=0.748[0.748,0.748]
  aux: alpha_novelty=-0.0005 balance=0.0389 branch=35.9597 bridge_conn=0.0297 decorr=0.0628 div=-0.1157 diversity=16.4708 gate_l1=0.3832 gate_repulse=-0.1698 gradalign=18.9809 intent_tau=0.6588 ls_reg=10.2800 nuc=161.8334 pred=2.5111 ranking=9288.6553 reinforce=0.2044 signal_ent=1.4240 w_m2v=0.1580
step=  9295  loss=9528.5955  ce=8.6125  mod_mlp=0.325 mod_std=0.079 lr=3.00e-04  tok/s=29  mem=9.8GB  intent_w=2.1453  mlp_out=808.0 usef=0.499  mat=0.748[0.748,0.748]
  aux: alpha_novelty=-0.0005 balance=0.0403 branch=34.0285 bridge_conn=0.0284 decorr=0.0613 div=-0.1157 diversity=16.5852 gate_l1=0.3758 gate_repulse=-0.1694 gradalign=19.2622 intent_tau=0.6588 ls_reg=10.2766 nuc=163.9475 pred=2.5294 ranking=9270.6865 reinforce=0.2062 signal_ent=1.4240 w_m2v=0.1580
  EVAL step=9320: val_loss=9.3663 val_ppl=11688.32
  Saved best to best.pt
step=  9350  loss=9654.0915  ce=8.2847  mod_mlp=0.325 mod_std=0.076 lr=3.01e-04  tok/s=28  mem=9.8GB  intent_w=2.1450  mlp_out=778.5 usef=0.499  mat=0.749[0.749,0.749]
  aux: alpha_novelty=-0.0005 balance=0.0387 branch=36.3330 bridge_conn=0.0278 decorr=0.0610 div=-0.1157 diversity=41.6773 gate_l1=0.3788 gate_repulse=-0.1660 gradalign=19.4565 intent_tau=0.6588 ls_reg=10.2731 nuc=166.7114 pred=2.5752 ranking=9366.1143 reinforce=0.2010 signal_ent=1.4240 w_m2v=0.1580
step=  9405  loss=9801.0524  ce=8.6099  mod_mlp=0.325 mod_std=0.075 lr=2.98e-04  tok/s=28  mem=9.8GB  intent_w=2.1446  mlp_out=783.1 usef=0.500  mat=0.749[0.749,0.749]
  aux: alpha_novelty=-0.0005 balance=0.0407 branch=37.7380 bridge_conn=0.0273 decorr=0.0618 div=-0.1157 diversity=48.0121 gate_l1=0.3716 gate_repulse=-0.1677 gradalign=19.3949 intent_tau=0.6588 ls_reg=10.2697 nuc=160.4554 pred=2.5905 ranking=9511.3203 reinforce=0.2034 signal_ent=1.4240 w_m2v=0.1580
  lr_adapt: var(ls)=13.573971 |1-a|=0.095891 gate_var=0.170874 |mirror|=219.7352 tau_var=13.579524 mult=0.9801 lr=2.94e-04 ls_mult[min=1.001 max=1.009]
step=  9460  loss=9472.4443  ce=8.8948  mod_mlp=0.325 mod_std=0.075 lr=2.90e-04  tok/s=28  mem=9.8GB  intent_w=2.1443  mlp_out=761.7 usef=0.500  mat=0.750[0.750,0.750]
  aux: alpha_novelty=-0.0005 balance=0.0427 branch=37.9786 bridge_conn=0.0271 decorr=0.0606 div=-0.1157 diversity=38.6719 gate_l1=0.3714 gate_repulse=-0.1747 gradalign=19.6683 intent_tau=0.6588 ls_reg=10.2663 nuc=162.5258 pred=2.6164 ranking=9189.1602 reinforce=0.2103 signal_ent=1.4240 w_m2v=0.1580
step=  9515  loss=9394.3300  ce=8.0564  mod_mlp=0.325 mod_std=0.074 lr=2.90e-04  tok/s=29  mem=9.8GB  intent_w=2.1440  mlp_out=753.0 usef=0.499  mat=0.750[0.750,0.750]
  aux: alpha_novelty=-0.0005 balance=0.0446 branch=37.7243 bridge_conn=0.0265 decorr=0.0601 div=-0.1157 diversity=13.6393 gate_l1=0.3697 gate_repulse=-0.1796 gradalign=19.4799 intent_tau=0.6588 ls_reg=10.2630 nuc=160.6902 pred=2.6040 ranking=9139.2119 reinforce=0.2151 signal_ent=1.4240 w_m2v=0.1580
  EVAL step=9553: val_loss=9.3310 val_ppl=11282.33
  Saved best to best.pt
step=  9570  loss=9178.4506  ce=9.8830  mod_mlp=0.324 mod_std=0.074 lr=2.83e-04  tok/s=28  mem=9.8GB  intent_w=2.1436  mlp_out=733.3 usef=0.500  mat=0.750[0.750,0.750]
  aux: alpha_novelty=-0.0005 balance=0.0453 branch=38.2910 bridge_conn=0.0260 decorr=0.0594 div=-0.1157 diversity=11.9556 gate_l1=0.3754 gate_repulse=-0.1872 gradalign=19.7966 intent_tau=0.6588 ls_reg=10.2598 nuc=163.2435 pred=2.6092 ranking=8919.7480 reinforce=0.2204 signal_ent=1.4240 w_m2v=0.1580
step=  9625  loss=9125.2228  ce=8.5746  mod_mlp=0.325 mod_std=0.075 lr=2.86e-04  tok/s=28  mem=9.8GB  intent_w=2.1433  mlp_out=729.5 usef=0.501  mat=0.750[0.750,0.750]
  aux: alpha_novelty=-0.0005 balance=0.0449 branch=36.9193 bridge_conn=0.0257 decorr=0.0605 div=-0.1156 diversity=10.7965 gate_l1=0.3778 gate_repulse=-0.1894 gradalign=19.9398 intent_tau=0.6588 ls_reg=10.2565 nuc=162.9500 pred=2.5878 ranking=8870.5322 reinforce=0.2220 signal_ent=1.4240 w_m2v=0.1580
step=  9680  loss=8867.8208  ce=8.6181  mod_mlp=0.325 mod_std=0.076 lr=2.91e-04  tok/s=28  mem=9.8GB  intent_w=2.1430  mlp_out=726.4 usef=0.501  mat=0.750[0.750,0.750]
  aux: alpha_novelty=-0.0005 balance=0.0448 branch=35.9712 bridge_conn=0.0257 decorr=0.0602 div=-0.1156 diversity=11.2439 gate_l1=0.3787 gate_repulse=-0.1898 gradalign=20.0283 intent_tau=0.6588 ls_reg=10.2532 nuc=162.4187 pred=2.5608 ranking=8614.0596 reinforce=0.2228 signal_ent=1.4240 w_m2v=0.1580
step=  9735  loss=8878.4007  ce=8.0870  mod_mlp=0.324 mod_std=0.077 lr=2.96e-04  tok/s=28  mem=9.8GB  intent_w=2.1427  mlp_out=727.2 usef=0.501  mat=0.750[0.750,0.750]
  aux: alpha_novelty=-0.0005 balance=0.0444 branch=35.3773 bridge_conn=0.0255 decorr=0.0595 div=-0.1156 diversity=11.5220 gate_l1=0.3802 gate_repulse=-0.1895 gradalign=20.1067 intent_tau=0.6588 ls_reg=10.2499 nuc=163.2017 pred=2.5378 ranking=8624.6504 reinforce=0.2231 signal_ent=1.4240 w_m2v=0.1580
  EVAL step=9786: val_loss=9.2989 val_ppl=10926.13
  Saved best to best.pt
step=  9790  loss=8944.9425  ce=8.5922  mod_mlp=0.324 mod_std=0.079 lr=2.94e-04  tok/s=28  mem=9.8GB  intent_w=2.1423  mlp_out=731.9 usef=0.500  mat=0.751[0.751,0.751]
  aux: alpha_novelty=-0.0005 balance=0.0428 branch=33.1301 bridge_conn=0.0254 decorr=0.0604 div=-0.1156 diversity=12.7155 gate_l1=0.3872 gate_repulse=-0.1912 gradalign=19.9688 intent_tau=0.6588 ls_reg=10.2465 nuc=163.4781 pred=2.5123 ranking=8691.6270 reinforce=0.2229 signal_ent=1.4240 w_m2v=0.1580
step=  9845  loss=8885.9208  ce=7.8088  mod_mlp=0.324 mod_std=0.077 lr=2.97e-04  tok/s=28  mem=9.8GB  intent_w=2.1420  mlp_out=724.5 usef=0.500  mat=0.751[0.751,0.751]
  aux: alpha_novelty=-0.0005 balance=0.0437 branch=32.7087 bridge_conn=0.0251 decorr=0.0606 div=-0.1156 diversity=15.6800 gate_l1=0.3855 gate_repulse=-0.1927 gradalign=19.8142 intent_tau=0.6588 ls_reg=10.2431 nuc=161.3239 pred=2.5028 ranking=8633.1699 reinforce=0.2226 signal_ent=1.4240 w_m2v=0.1580
step=  9900  loss=8956.2049  ce=7.9249  mod_mlp=0.325 mod_std=0.078 lr=3.00e-04  tok/s=28  mem=9.8GB  intent_w=2.1417  mlp_out=726.9 usef=0.501  mat=0.751[0.751,0.751]
  aux: alpha_novelty=-0.0005 balance=0.0436 branch=32.1193 bridge_conn=0.0254 decorr=0.0612 div=-0.1156 diversity=17.6707 gate_l1=0.3851 gate_repulse=-0.1910 gradalign=19.8287 intent_tau=0.6588 ls_reg=10.2397 nuc=163.0004 pred=2.5013 ranking=8700.2480 reinforce=0.2229 signal_ent=1.4240 w_m2v=0.1580
  lr_adapt: var(ls)=13.546636 |1-a|=0.095623 gate_var=0.191784 |mirror|=237.7592 tau_var=13.552128 mult=0.9979 lr=2.99e-04 ls_mult[min=1.001 max=1.010]
step=  9955  loss=8914.4173  ce=8.4623  mod_mlp=0.325 mod_std=0.076 lr=3.00e-04  tok/s=28  mem=9.8GB  intent_w=2.1413  mlp_out=724.7 usef=0.501  mat=0.751[0.751,0.751]
  aux: alpha_novelty=-0.0005 balance=0.0434 branch=33.0806 bridge_conn=0.0256 decorr=0.0612 div=-0.1156 diversity=25.5937 gate_l1=0.3848 gate_repulse=-0.1901 gradalign=20.0239 intent_tau=0.6588 ls_reg=10.2363 nuc=161.3521 pred=2.5060 ranking=8650.4902 reinforce=0.2225 signal_ent=1.4240 w_m2v=0.1580
step= 10010  loss=8901.1953  ce=8.3277  mod_mlp=0.325 mod_std=0.077 lr=3.00e-04  tok/s=28  mem=9.8GB  intent_w=2.1410  mlp_out=720.6 usef=0.501  mat=0.750[0.750,0.750]
  aux: alpha_novelty=-0.0005 balance=0.0433 branch=34.7387 bridge_conn=0.0258 decorr=0.0605 div=-0.1156 diversity=25.8542 gate_l1=0.3845 gate_repulse=-0.1888 gradalign=20.0267 intent_tau=0.6588 ls_reg=10.2329 nuc=163.3412 pred=2.5392 ranking=8633.4629 reinforce=0.2217 signal_ent=1.4240 w_m2v=0.1580
  EVAL step=10019: val_loss=9.2646 val_ppl=10557.17
  Saved best to best.pt
step= 10065  loss=8422.3472  ce=8.7149  mod_mlp=0.325 mod_std=0.078 lr=3.01e-04  tok/s=28  mem=9.8GB  intent_w=2.1407  mlp_out=730.5 usef=0.501  mat=0.750[0.750,0.750]
  aux: alpha_novelty=-0.0005 balance=0.0436 branch=33.5573 bridge_conn=0.0265 decorr=0.0612 div=-0.1156 diversity=12.0934 gate_l1=0.3796 gate_repulse=-0.1879 gradalign=19.8489 intent_tau=0.6588 ls_reg=10.2295 nuc=163.2010 pred=2.5445 ranking=8169.4883 reinforce=0.2218 signal_ent=1.4240 w_m2v=0.1580
step= 10120  loss=8406.6055  ce=8.9323  mod_mlp=0.326 mod_std=0.078 lr=3.01e-04  tok/s=28  mem=9.8GB  intent_w=2.1403  mlp_out=733.9 usef=0.502  mat=0.750[0.750,0.750]
  aux: alpha_novelty=-0.0005 balance=0.0443 branch=33.1936 bridge_conn=0.0269 decorr=0.0628 div=-0.1156 diversity=8.0514 gate_l1=0.3774 gate_repulse=-0.1872 gradalign=19.6537 intent_tau=0.6588 ls_reg=10.2261 nuc=159.9469 pred=2.5357 ranking=8161.3955 reinforce=0.2213 signal_ent=1.4240 w_m2v=0.1580
step= 10175  loss=8419.9290  ce=8.3740  mod_mlp=0.326 mod_std=0.076 lr=3.00e-04  tok/s=28  mem=9.8GB  intent_w=2.1400  mlp_out=725.4 usef=0.501  mat=0.750[0.750,0.750]
  aux: alpha_novelty=-0.0005 balance=0.0443 branch=34.4541 bridge_conn=0.0268 decorr=0.0620 div=-0.1156 diversity=7.8848 gate_l1=0.3778 gate_repulse=-0.1873 gradalign=19.7619 intent_tau=0.6588 ls_reg=10.2227 nuc=164.7987 pred=2.5381 ranking=8169.2266 reinforce=0.2198 signal_ent=1.4240 w_m2v=0.1580
step= 10230  loss=8181.7528  ce=9.3922  mod_mlp=0.326 mod_std=0.076 lr=3.00e-04  tok/s=28  mem=9.8GB  intent_w=2.1396  mlp_out=723.8 usef=0.501  mat=0.750[0.750,0.750]
  aux: alpha_novelty=-0.0005 balance=0.0435 branch=33.4974 bridge_conn=0.0268 decorr=0.0612 div=-0.1156 diversity=8.0533 gate_l1=0.3798 gate_repulse=-0.1870 gradalign=19.5913 intent_tau=0.6588 ls_reg=10.2193 nuc=165.2737 pred=2.5266 ranking=7930.5322 reinforce=0.2177 signal_ent=1.4240 w_m2v=0.1580
  EVAL step=10252: val_loss=9.2307 val_ppl=10205.92
  Saved best to best.pt
step= 10285  loss=7911.9537  ce=8.4841  mod_mlp=0.326 mod_std=0.075 lr=3.00e-04  tok/s=28  mem=9.8GB  intent_w=2.1393  mlp_out=737.3 usef=0.501  mat=0.750[0.750,0.750]
  aux: alpha_novelty=-0.0005 balance=0.0448 branch=34.9285 bridge_conn=0.0262 decorr=0.0627 div=-0.1156 diversity=22.3367 gate_l1=0.3707 gate_repulse=-0.1870 gradalign=19.8188 intent_tau=0.6588 ls_reg=10.2159 nuc=165.6268 pred=2.5397 ranking=7645.3428 reinforce=0.2184 signal_ent=1.4240 w_m2v=0.1580
step= 10340  loss=8069.0371  ce=8.8898  mod_mlp=0.326 mod_std=0.075 lr=3.02e-04  tok/s=28  mem=9.8GB  intent_w=2.1390  mlp_out=739.7 usef=0.501  mat=0.750[0.750,0.750]
  aux: alpha_novelty=-0.0005 balance=0.0447 branch=35.4553 bridge_conn=0.0262 decorr=0.0631 div=-0.1156 diversity=32.2183 gate_l1=0.3673 gate_repulse=-0.1836 gradalign=19.9664 intent_tau=0.6588 ls_reg=10.2125 nuc=163.0538 pred=2.5185 ranking=7794.0635 reinforce=0.2165 signal_ent=1.4240 w_m2v=0.1580
step= 10395  loss=8094.4179  ce=8.3682  mod_mlp=0.327 mod_std=0.074 lr=3.01e-04  tok/s=28  mem=9.8GB  intent_w=2.1386  mlp_out=739.8 usef=0.501  mat=0.750[0.750,0.750]
  aux: alpha_novelty=-0.0005 balance=0.0445 branch=36.6833 bridge_conn=0.0263 decorr=0.0639 div=-0.1156 diversity=32.4165 gate_l1=0.3672 gate_repulse=-0.1824 gradalign=20.1561 intent_tau=0.6588 ls_reg=10.2091 nuc=165.6677 pred=2.4935 ranking=7815.7642 reinforce=0.2152 signal_ent=1.4240 w_m2v=0.1580
  lr_adapt: var(ls)=13.518660 |1-a|=0.095720 gate_var=0.181764 |mirror|=230.3600 tau_var=13.524200 mult=1.0040 lr=3.01e-04 ls_mult[min=1.001 max=1.010]
step= 10450  loss=7912.7833  ce=7.3612  mod_mlp=0.327 mod_std=0.074 lr=3.01e-04  tok/s=28  mem=9.8GB  intent_w=2.1383  mlp_out=748.7 usef=0.502  mat=0.750[0.750,0.750]
  aux: alpha_novelty=-0.0005 balance=0.0442 branch=36.8971 bridge_conn=0.0264 decorr=0.0643 div=-0.1155 diversity=14.7752 gate_l1=0.3670 gate_repulse=-0.1817 gradalign=20.1803 intent_tau=0.6588 ls_reg=10.2057 nuc=161.1054 pred=2.4872 ranking=7657.1113 reinforce=0.2149 signal_ent=1.4240 w_m2v=0.1580
  EVAL step=10485: val_loss=9.1963 val_ppl=9860.78
  Saved best to best.pt
step= 10505  loss=8283.4876  ce=8.2696  mod_mlp=0.327 mod_std=0.075 lr=3.03e-04  tok/s=28  mem=9.8GB  intent_w=2.1379  mlp_out=751.0 usef=0.501  mat=0.750[0.750,0.750]
  aux: alpha_novelty=-0.0005 balance=0.0414 branch=37.3921 bridge_conn=0.0267 decorr=0.0597 div=-0.1155 diversity=22.7318 gate_l1=0.3716 gate_repulse=-0.1755 gradalign=19.9129 intent_tau=0.6588 ls_reg=10.2022 nuc=160.6842 pred=2.5077 ranking=8019.1289 reinforce=0.2096 signal_ent=1.4240 w_m2v=0.1580
step= 10560  loss=8484.8909  ce=8.4375  mod_mlp=0.327 mod_std=0.076 lr=3.03e-04  tok/s=28  mem=9.8GB  intent_w=2.1376  mlp_out=745.6 usef=0.501  mat=0.750[0.750,0.750]
  aux: alpha_novelty=-0.0005 balance=0.0404 branch=38.3313 bridge_conn=0.0270 decorr=0.0597 div=-0.1155 diversity=33.4280 gate_l1=0.3739 gate_repulse=-0.1740 gradalign=19.4640 intent_tau=0.6588 ls_reg=10.1988 nuc=162.1544 pred=2.5169 ranking=8207.6992 reinforce=0.2089 signal_ent=1.4240 w_m2v=0.1580
step= 10615  loss=9264.0515  ce=7.9585  mod_mlp=0.326 mod_std=0.074 lr=3.03e-04  tok/s=28  mem=9.8GB  intent_w=2.1508  mlp_out=762.6 usef=0.501  mat=0.750[0.750,0.750]
  aux: alpha_novelty=-0.0005 balance=0.0398 branch=32.8230 bridge_conn=0.0238 decorr=0.0615 div=-0.1155 diversity=15.4927 gate_l1=0.3776 gate_repulse=-0.1695 gradalign=18.8877 intent_tau=0.6587 ls_reg=10.1953 nuc=162.6196 pred=2.5748 ranking=9010.8379 reinforce=0.2041 signal_ent=1.4240 w_m2v=0.1580
  EVAL step=10718: val_loss=9.1613 val_ppl=9520.42
  Saved best to best.pt
  EVAL step=10951: val_loss=9.1282 val_ppl=9212.55
  Saved best to best.pt
  EVAL step=11184: val_loss=9.0983 val_ppl=8938.71
  Saved best to best.pt
  EVAL step=11417: val_loss=9.0747 val_ppl=8730.88
  Saved best to best.pt
  EVAL step=11650: val_loss=9.0462 val_ppl=8484.32
  Saved best to best.pt
  EVAL step=11883: val_loss=9.0208 val_ppl=8267.14
  Saved best to best.pt
  EVAL step=12116: val_loss=8.9964 val_ppl=8067.56
  Saved best to best.pt
  EVAL step=12349: val_loss=8.9731 val_ppl=7883.62
  Saved best to best.pt
  EVAL step=12582: val_loss=8.9503 val_ppl=7710.44
  Saved best to best.pt
  EVAL step=13048: val_loss=8.9094 val_ppl=7403.21
  Saved best to best.pt
  EVAL step=13281: val_loss=8.8930 val_ppl=7284.55
  Saved best to best.pt
  EVAL step=13514: val_loss=8.8734 val_ppl=7144.33
  Saved best to best.pt
  EVAL step=13747: val_loss=8.8562 val_ppl=7022.87
  Saved best to best.pt
  EVAL step=13980: val_loss=8.8438 val_ppl=6936.12
  Saved best to best.pt
  EVAL step=14213: val_loss=8.8299 val_ppl=6839.44
  Saved best to best.pt
  EVAL step=14446: val_loss=8.8159 val_ppl=6743.22
  Saved best to best.pt
  EVAL step=14679: val_loss=8.8034 val_ppl=6658.56
  Saved best to best.pt
  EVAL step=14912: val_loss=8.7921 val_ppl=6583.44
  Saved best to best.pt
  EVAL step=15145: val_loss=8.7838 val_ppl=6528.11
  Saved best to best.pt
  EVAL step=15378: val_loss=8.7777 val_ppl=6487.77
  Saved best to best.pt
  EVAL step=15611: val_loss=8.7727 val_ppl=6455.22
  Saved best to best.pt
  EVAL step=15844: val_loss=8.7692 val_ppl=6432.55
  Saved best to best.pt
  [вторичная перестройка ~15565: mat 0.745→0.722, bridge_conn 0.04→0.08, intent_w→2.5]
  EVAL step=16077: val_loss=8.7701 val_ppl=6438.33
  EVAL step=16310: val_loss=8.7736 val_ppl=6461.11
  EVAL step=16543: val_loss=8.7812 val_ppl=6510.44
  EVAL step=16776: val_loss=8.7908 val_ppl=6573.22
  EVAL step=17009: val_loss=8.8016 val_ppl=6644.11
  EVAL step=17242: val_loss=8.8145 val_ppl=6730.55
  EVAL step=17475: val_loss=8.8303 val_ppl=6837.22
  EVAL step=17708: val_loss=8.8475 val_ppl=6954.33
  EVAL step=17941: val_loss=8.8679 val_ppl=7098.11
  EVAL step=18174: val_loss=8.8933 val_ppl=7282.55
  EVAL step=18407: val_loss=8.9211 val_ppl=7489.33
  EVAL step=18640: val_loss=8.9500 val_ppl=7704.22
  EVAL step=18873: val_loss=8.9811 val_ppl=7941.44
  EVAL step=19106: val_loss=9.0144 val_ppl=8200.55
  EVAL step=19339: val_loss=9.0439 val_ppl=8448.11
  EVAL step=19572: val_loss=9.0439 val_ppl=8448.11
  [лог обрезан на step ~19635]
```

---

## Новый запуск: per-layer maturation + SpectrumGate (Colab T4, fp32)

### Ключевые изменения (коммит `a735ac0`)

- **Per-layer maturation gate**: deep layers (tau≈515) opening first (T_eff=8000), shallow layers (tau≈8) later (T_eff=16000). Monotonic → mathematically stable.
- **Bridge readiness override REMOVED**: gate = pure time-ramp, no scalar readiness override.
- **Global readiness**: `global_ready=True` when ALL layers M_l > 0.1 (step ~8000). Before that, LayerBridgeGate uses simple maturation gating only. After: full SpectrumGate with per-layer tau-driven diversity.
- **SpectrumGate formula**: `gate = sigmoid(logits) * (1 + softmax(logits/tau))` — hybrid sigmoid-softmax with maturation-driven tau.

### Конфигурация

| Параметр | Значение |
|---|---|
| D / слои / G / bind_K | 2560 / 24 / 32 / 32 |
| vocab / seq_len | 65536 / 256 |
| lr / weight_decay | 0.0003 / 0.01 |
| bridge_conn | 0.1 |
| maturation_enabled | True |
| matur_T0 / T_delay / delta | 8000 / 8000 / 4000 |

### Траектория val (свежий прогон)

```
step=  233: val=14.4362  (mat=0.751, bridge_conn=0.0013)
step=  466: val=10.9985  (mat=0.787, bridge_conn=0.0072) ← Saved best
step=  699: val=10.9348  (mat=0.786, bridge_conn=0.0184) ← Saved best
step=  932: val=10.8721  (mat=0.779, bridge_conn=0.0557) ← Saved best
step= 1165: val=10.9630  (mat=0.779, bridge_conn=0.0588) — minor regression
step= 1398: val=10.7472  (mat=0.778, bridge_conn=0.0636) ← NEW BEST
```

### Ключевые наблюдения

- **bridge_conn стабилен**: 0.055→0.064 (step 932→1398). Не коллапсирует, healthy dynamic range.
- **CE снижается**: 10.84→10.69 (step 932→1375). Модель учится.
- **intent_w растёт**: 0.482→0.518 (step 935→1320). Bridge модуляция усиливается.
- **mod_scale_mlp=0.626→0.622** (L0) — MLP modulation стабильна.
- **pm_norm: L5=0.906, L7=0.798** — private memory растёт.
- **slots: 115/192** — concept slots заполняются (13 full layers).
- **Per-layer maturation**: mat=0.779[0.779,0.779] — uniform (старый код). Per-layer ramp (a735ac0) ещё не задеплоен.
- **NaN/Inf: 0/0** — стабильно.

### Per-layer maturation (новый код, a735ac0)

```
step=    0: L0=0.018 L12=0.049 L23=0.119  global_ready=False
step= 4000: L0=0.047 L12=0.124 L23=0.269  global_ready=False
step= 8000: L0=0.119 L12=0.278 L23=0.500  global_ready=True ← все > 0.1
step=16000: L0=0.500 L12=0.740 L23=0.881  global_ready=True
step=20000: L0=0.731 L12=0.885 L23=0.953  global_ready=True
```

Deep-first monotonic → стабильно. Skip connections в shallow слоях сохраняют gradient.

### Чекпоинты

| Файл | Step | Val | Описание |
|---|---|---|---|
| `checkpoints/best.pt` | 1398 | 10.75 | Лучший текущий (старый код) |
| `checkpoints/best 3.pt` | 932 | 10.87 | HTML-отчёт: `best 3_932_report.html` |
| `checkpoints/best_15844_report.html` | 15844 | 8.7692 | Лучший исторический (предыдущий прогон) |

---

## Перезапуск: per-layer maturation ACTIVE (a735ac0)

### Первые данные (step 1398→1430, Colab T4, fp32)

```
step= 1398: val=10.8360  mat=0.076[0.026,0.160]  bridge_conn=0.098
step= 1430: ce=10.59     mat=0.076[0.026,0.160]  bridge_conn=0.098
step= 1485: ce=10.65     mat=0.077[0.026,0.162]  bridge_conn=0.096
step= 1540: ce=10.57     mat=0.078[0.026,0.164]  bridge_conn=0.097
step= 1595: ce=10.56     mat=0.079[0.027,0.166]  bridge_conn=0.097
step= 1631: val=10.6918  mat≈0.079               bridge_conn≈0.097  ← NEW BEST
```

**Ключевые отличия от старого кода:**

| Метрика | Старый код (step 935) | Новый код (step 1430) |
|---|---|---|
| **mat** | 0.779[0.779,0.779] | **0.076[0.026,0.160]** |
| **bridge_conn** | 0.056 | **0.098** |
| **intent_w** | 0.482 | **0.585** |
| **mod_mlp** | 0.286 | **0.370** |
| **gradalign** | 19.4 | **23.0** |

### Per-layer maturation ПОДТВЕРЖДЕНА

- **L0=0.026** (shallow, tau≈8) — ещё не открылся (ожидается T_eff≈16000)
- **L23=0.160** (deep, tau≈515) — уже начал открываться (ожидается T_eff≈8000)
- **Monotonic**: 0.026 < ... < 0.160 ✓
- **Global readiness**: FALSE (все слои < 0.1)

### Интерпретация

Per-layer maturation работает как задумано:
1. Deep layers (L23) открываются быстрее → bridge получает доступ к зрелым представлениям
2. Shallow layers (L0) медленнее → skip connections сохраняют gradient
3. Bridge conn здоровее (0.098 vs 0.056 на старом коде) — per-layer diversity помогает
4. intent_w выше (0.585 vs 0.482) — bridge модуляция сильнее
5. gradalign выше (23.0 vs 19.4) — градиенты лучше выровнены

### Следующие шаги

1. Продолжить обучение — ждать step ~8000 для global_ready (все слои M_l > 0.1)
2. Сравнить val trajectory: per-layer vs uniform maturation
3. Цель: `val ≈ 8.5` (лучший исторический: 8.7692@15844)
```

<!---
