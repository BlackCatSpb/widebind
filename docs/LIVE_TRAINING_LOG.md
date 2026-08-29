# Живой журнал обучения WideBind

> Единственный актуальный лог текущего запуска. Исторические прогоны (прогон №2 / AMP / fp32-расхождение) удалены — при необходимости см. git-историю.

## Текущий запуск (Colab T4, fp32, архитектура зрелости по компетентности моста)

- Контур: `notebooks/colab.ipynb`, Colab T4, **fp32 (`use_amp=False`)**, ~28 tok/s, ~9.8 GB VRAM.
- Резюм `checkpoints/best.pt` (`FORCE_FRESH=False`); `eval_interval=2000`, `save_interval=987`, `max_steps=300000`.
- Watchdog `FailureDetector`: откат на `best.pt` при CE-спайке > `watchdog_ce=15.0`.
- При резюме cognitive gate принудительно «открывается» (`mod_scale_mlp → cfg.mlp_mod_scale_reopen ≈ 1.099`), чтобы ядро не резюмилось «спящим».

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
| Шаг сохранения best | 6757 (обучение продолжается) |
| val_loss / ppl | 9.7113 / ~16503 |
| Зрелость `mat` | 0.731 (все 24 слоя равномерно) |
| `bridge_conn` (raw cosine-loss моста) | ~0.07 — мост компетентен |
| `mod_mlp` (live) | ~0.328 (здоров, не схлопнут) |
| Скорость / VRAM | ~28 tok/s на T4 / ~9.8 GB |
| NaN / Inf | 0 / 0 |

### Траектория val (монотонно улучшается)

9.9749 (5359) → 9.9283 (5592) → 9.8816 (5825) → 9.8366 (6058) → 9.7956 (6291) → 9.7516 (6524) → **9.7113 (6757)**

### Фазовый переход (каскад созревания, ~5995–6215)

Единый гейт зрелости `M_l = max(time_ramp, bridge_readiness)` открыл живые ветви **по компетентности моста**, а не по слепым часам:

- `bridge_conn` (raw) упал **0.237 → 0.037** — in-core SemanticBridge научился предсказывать эталонный next-token embedding;
- `mat` вырос **0.624 → 0.735**; `ce` **9.3 → 8.80**.

Здоровый каскад (не расходимость): maturation-гейт держит `ρ(J_l) ≈ 1`, ствол и метакогнитивные ветви учатся как единое целое. После каскада `mat` продолжает медленно расти (0.730 → 0.733), живые ветви (live BridgeGLU, injection семантического моста, intent-шина, запись приватной памяти) раскрыты полностью.

### Вердикт analyze (`checkpoints/best.pt`@6757) — WAKE-CANDIDATE

- MLP проснулся: `W_std` = 0.0705 ≈ decay-базис (0.0699, dev +0.0006); gate max 0.751 (порог WAKE 0.75). `sigmoid(mod_scale_mlp)` = 0.750, `sigmoid(mod_scale_mem)` = 0.667.
- Maturation gate 0.731 (открыт); `bridge_readiness` max = 0.778, `tau_norm` max = 0.997 — мост реально включается.
- Intent-шина: слои несут **дополняющие** сигналы (cross-layer cosine offdiag = 0.0195, близко к 0); `bus_head_proj` растёт с zero-init (стенсил учится: norm 0.48).
- Концепты: slots 76/192, births в пустых слоях **[14]** (штатное рождение нового концепта, WATCH).
- Triad (`triad_reason`): при генерации ствол ре-циркулируется, если `_conf < 0.5` (до `triad_max_passes=3`), бленд `h = 0.5·h + 0.5·h2`. Только inference.
- Минорные WATCH: `pred` aux DEAD (`requires_grad=False`, `cos_sim(diversity, CE)=0`) — diversity-выравнивание не даёт градиента; **72 unexpected tensors** = буферы `collective._resvar_ema/_resvar_var/_mature_count` (пересоздаются на резюме, EMA зрелости коллектива сбрасывается — кандидат на рефактор: зарегистрировать в `__init__`).

**Цель:** выйти к историческому рубежу `val ≈ 8.5` при сохранении устойчивого CE.

### Сырой лог (фрагмент мониторинга, step 6325 → 6820)

```
Saved best to best.pt
step=  6325  loss=9111.0518  ce=9.2120  mod_mlp=0.327 mod_std=0.074 lr=2.99e-04  tok/s=28  mem=9.8GB  intent_w=1.4575  mlp_out=741.3 usef=0.500  mat=0.730[0.730,0.730]
  aux: alpha_novelty=-0.0005 balance=0.0312 branch=34.9204 bridge_conn=0.0694 decorr=0.0565 div=-0.1162 diversity=22.6763 gate_l1=0.4095 gate_repulse=-0.1639 gradalign=19.5623 intent_tau=0.6592 ls_reg=10.4370 nuc=166.1653 pred=2.5451 ranking=8842.8105 reinforce=0.1952 signal_ent=1.4242 w_m2v=0.1582
step=  6380  loss=10322.1458  ce=9.6259  mod_mlp=0.327 mod_std=0.077 lr=3.02e-04  tok/s=28  mem=9.8GB  intent_w=1.4999  mlp_out=754.1 usef=0.499  mat=0.730[0.730,0.730]
  aux: alpha_novelty=-0.0005 balance=0.0308 branch=38.0919 bridge_conn=0.0683 decorr=0.0580 div=-0.1161 diversity=18.7145 gate_l1=0.4085 gate_repulse=-0.1596 gradalign=18.9672 intent_tau=0.6591 ls_reg=10.4335 nuc=164.7466 pred=2.6532 ranking=10056.1914 reinforce=0.1908 signal_ent=1.4242 w_m2v=0.1582
  lr_adapt: var(ls)=13.732357 |1-a|=0.095741 gate_var=0.155633 |mirror|=227.6137 tau_var=13.738281 mult=1.0110 lr=3.03e-04 ls_mult[min=1.001 max=1.008]
step=  6435  loss=10555.8609  ce=8.9819  mod_mlp=0.328 mod_std=0.076 lr=3.03e-04  tok/s=28  mem=9.8GB  intent_w=1.5167  mlp_out=752.5 usef=0.499  mat=0.730[0.730,0.730]
  aux: alpha_novelty=-0.0005 balance=0.0297 branch=39.1991 bridge_conn=0.0685 decorr=0.0596 div=-0.1161 diversity=20.4201 gate_l1=0.4131 gate_repulse=-0.1558 gradalign=18.9245 intent_tau=0.6591 ls_reg=10.4300 nuc=164.0620 pred=2.7407 ranking=10288.3750 reinforce=0.1876 signal_ent=1.4242 w_m2v=0.1582
step=  6490  loss=10355.7069  ce=9.8227  mod_mlp=0.327 mod_std=0.078 lr=3.07e-04  tok/s=28  mem=9.8GB  intent_w=1.5167  mlp_out=747.1 usef=0.497  mat=0.730[0.730,0.730]
  aux: alpha_novelty=-0.0005 balance=0.0261 branch=40.6108 bridge_conn=0.0686 decorr=0.0584 div=-0.1161 diversity=21.0399 gate_l1=0.4287 gate_repulse=-0.1458 gradalign=18.6426 intent_tau=0.6591 ls_reg=10.4265 nuc=162.6291 pred=2.7426 ranking=10087.0537 reinforce=0.1781 signal_ent=1.4242 w_m2v=0.1582
  EVAL step=6524: val_loss=9.7516 val_ppl=17181.54
  Saved best to best.pt
step=  6545  loss=10596.1228  ce=9.8317  mod_mlp=0.328 mod_std=0.079 lr=3.15e-04  tok/s=28  mem=9.8GB  intent_w=1.5164  mlp_out=761.5 usef=0.498  mat=0.730[0.730,0.730]
  aux: alpha_novelty=-0.0005 balance=0.0212 branch=48.3606 bridge_conn=0.0687 decorr=0.0546 div=-0.1161 diversity=29.1893 gate_l1=0.4471 gate_repulse=-0.1280 gradalign=18.4454 intent_tau=0.6591 ls_reg=10.4228 nuc=164.3101 pred=2.6901 ranking=10310.1211 reinforce=0.1634 signal_ent=1.4242 w_m2v=0.1582
step=  6600  loss=10743.3384  ce=9.9444  mod_mlp=0.328 mod_std=0.081 lr=3.12e-04  tok/s=28  mem=9.8GB  intent_w=1.5161  mlp_out=753.7 usef=0.498  mat=0.730[0.730,0.730]
  aux: alpha_novelty=-0.0005 balance=0.0194 branch=48.4665 bridge_conn=0.0675 decorr=0.0533 div=-0.1161 diversity=36.6611 gate_l1=0.4631 gate_repulse=-0.1218 gradalign=18.7255 intent_tau=0.6591 ls_reg=10.4192 nuc=163.7982 pred=2.6826 ranking=10449.8760 reinforce=0.1587 signal_ent=1.4242 w_m2v=0.1582
step=  6655  loss=11003.3808  ce=10.0861  mod_mlp=0.328 mod_std=0.081 lr=3.12e-04  tok/s=28  mem=9.8GB  intent_w=1.5159  mlp_out=748.3 usef=0.499  mat=0.730[0.730,0.730]
  aux: alpha_novelty=-0.0005 balance=0.0172 branch=49.5062 bridge_conn=0.0659 decorr=0.0536 div=-0.1161 diversity=31.8268 gate_l1=0.4853 gate_repulse=-0.1118 gradalign=18.7288 intent_tau=0.6591 ls_reg=10.4156 nuc=160.4449 pred=2.5947 ranking=10716.9922 reinforce=0.1505 signal_ent=1.4242 w_m2v=0.1582
step=  6710  loss=10669.8212  ce=9.6265  mod_mlp=0.328 mod_std=0.082 lr=3.08e-04  tok/s=28  mem=9.8GB  intent_w=1.5156  mlp_out=733.1 usef=0.499  mat=0.731[0.731,0.731]
  aux: alpha_novelty=-0.0005 balance=0.0159 branch=52.5265 bridge_conn=0.0652 decorr=0.0523 div=-0.1161 diversity=25.3604 gate_l1=0.4949 gate_repulse=-0.1097 gradalign=18.8497 intent_tau=0.6591 ls_reg=10.4120 nuc=164.2590 pred=2.5875 ranking=10383.4072 reinforce=0.1488 signal_ent=1.4242 w_m2v=0.1582
  EVAL step=6757: val_loss=9.7113 val_ppl=16503.49
  Saved best to best.pt
step=  6765  loss=10493.4582  ce=9.5128  mod_mlp=0.328 mod_std=0.081 lr=2.99e-04  tok/s=28  mem=9.8GB  intent_w=1.5153  mlp_out=708.0 usef=0.499  mat=0.733[0.733,0.733]
  aux: alpha_novelty=-0.0005 balance=0.0166 branch=50.9702 bridge_conn=0.0618 decorr=0.0515 div=-0.1161 diversity=17.6157 gate_l1=0.4870 gate_repulse=-0.1133 gradalign=18.7729 intent_tau=0.6591 ls_reg=10.4084 nuc=167.0909 pred=2.5863 ranking=10213.7188 reinforce=0.1538 signal_ent=1.4242 w_m2v=0.1582
step=  6820  loss=10920.2238  ce=9.8700  mod_mlp=0.328 mod_std=0.080 lr=3.04e-04  tok/s=28  mem=9.8GB  intent_w=1.5151  mlp_out=701.5 usef=0.499  mat=0.733[0.733,0.733]
  aux: alpha_novelty=-0.0005 balance=0.0159 branch=55.5065 bridge_conn=0.0608 decorr=0.0538 div=-0.1161 diversity=16.6013 gate_l1=0.4958 gate_repulse=-0.1084 gradalign=18.7795 intent_tau=0.6591 ls_reg=10.4049 nuc=163.7702 pred=2.6465 ranking=10639.8525 reinforce=0.1497 signal_ent=1.4242 w_m2v=0.1582
```
