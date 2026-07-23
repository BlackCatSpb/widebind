# WideBind Main: Архитектурный обзор

**~161M параметров (default) / ~187M (multi-ocular S=4), 32 слоя, D=4096, 32 эксперта.**
Трёхслойный дифференцируемый аппарат саморефлексии:
Знание (L0) → Мета-знание (L1) → Арбитр (L2).

Гибридная LM без transformer-слоёв: ни softmax attention, ни QKV, ни KV-cache, ни ReLU.
Вся архитектура — композиция дифференцируемых резервуаров:

```
token → PartitionedEmbedding (sparse 6/32 код)
      → [WideBindBlock × 32]
         → RMSNorm → Conv1d (k=48)
         → BottleneckBind (D→K↔D, билинейное скрещивание)
         → VSA Memory (4-масштабная суперпозиция, τ=8..512)
         → GroupedCognitiveMirror (32 эксперта×32 слоя, k=4/8/16 staircase)
         → DCT Spectral (частотная фильтрация)
         → GroupedMLP (32 группы × expand=4)
      → Final RMSNorm → PartitionedHead → logits
```

---

## 1. Философия и мотивация

### 1.1 Проблема современных LLM

Все современные языковые модели — Transformer, MoE, RWKV, Mamba — имеют фундаментальное ограничение: **у них нет внутренней модели собственного знания**.

- **Transformer**: attention между токенами. Модель не знает, что она знает.
- **MoE**: gate выбирает экспертов, но эксперты не общаются. Нет коллективной памяти.
- **Дифференцируемость**: все symbolic-надстройки (EVA, Reflexion) работают через LLM-вызовы, не дифференцируемы.

### 1.2 Трёхслойная архитектура саморефлексии

| Слой | Название | Сущность | Пластичность |
|------|----------|----------|-------------|
| L0 | Knowledge | W_proj, alpha_diag, MLP | **Опасна** — правка весов разрушает знания |
| L1 | Meta-Knowledge | `_private_mem[g]` — EMA уверенных K-space состояний | **Безопасна** — отношение к фактам, не факты |
| L2 | Arbiter | Gate с 5-6 компонентами, contradiction | Самонастраивается |

**Принцип**: модель знает (L0), знает что знает (L1), и решает кому верить (L2).
Все три слоя — одна дифференцируемая функция. Градиент течёт через всё одновременно,
но с разной скоростью и разным уровнем риска.

### 1.3 Что это даёт

1. **Безопасная пластичность** — L1 можно менять без риска для L0.
2. **Коллективный разум** — cross-expert attention между 32 экспертами.
3. **Детекция противоречий** — concept_sim высок, behavior_div высок = противоречие.
4. **Социальная динамика** — trust_matrix, dominance, isolation, social_pressure.

---

## 2. Partitioned Embedding — разреженные блочные коды

### 2.1 Механизм

Входные токены (V=50000) кодируются не проекцией, а **адресацией** через разреженный
блочный код (6-out-of-32). Каждый токен получает детерминированный бинарный код
с ровно 6 единицами из 32 через комбинаторную систему счисления (combinadic).

```python
basis = nn.Parameter(K=32, d=128)  # 32 сегмента × 128 = 4096
code = sparse_block_codes(vocab)[token]  # (K,) — 6 единиц, 26 нулей
h = code @ basis  # (d,) → (D=4096)
```

### 2.2 Mix-матрица

```python
mix = sigmoid(codes @ M)  # K×K ортогональная mix-матрица
h_mix = code @ mix @ basis  # каждый бит влияет на все сегменты (rank expansion)
```

### 2.3 Преимущества

- Каждый из 32 сегментов выровнен с mirror-группой (1:1).
- Sparse-вычисления: только 6 из 32 сегментов активны.
- Детерминированность: одинаковые токены → одинаковые коды.
- V=50000, K=32: каждому сегменту ≈ 1563 токена, d=128.

### 2.4 Параметры

basis: K × d = 32 × 128 = 4,096. Инициализация: нормальное распределение std=0.02.

---

## 3. Partitioned Head — сегментированный выход

Обратная операция к Partitioned Embedding:

```python
h_g = h.reshape(B, L, K, d)  # разделение на 32 сегмента
scores = einsum('blkd,kd->blk', h_g, readout)  # каждый сегмент против своего readout
logits = scores @ codes.T + token_bias  # агрегация через коды
```

- **Weight tying**: readout = embed.basis (один и тот же параметр encode/decode).
- **Per-token bias**: `torch.zeros(V)` — частотный prior.
- Без cross-talk между сегментами: каждый сегмент отвечает за свою часть словаря.

### 3.1 Zeckendorf Readout (альтернатива)

При `zeckendorf_readout=True` вместо PartitionedHead используется дерево
Фибоначчи-Цекендорфа (F=23 бита для V=50000): logits по разрядам, бинарная
кросс-энтропия. Экспериментальный режим.

---

## 4. Pre-LN RMS Norm

```python
h = x / RMS(x) * w  # без вычитания среднего
RMS(x) = sqrt(mean(x²) + eps)
```

Один learnable scale-вектор `w ∈ ℝᴰ` на слой. Без bias.

---

## 5. Depthwise Conv1d (k=48)

Локальный контекст на каждом из D=4096 каналов независимо.

```python
conv = Conv1d(D, D, kernel_size=48, groups=D, bias=False)
h_conv = conv(h_perm)  # (B, D, L) depthwise
```

- **Groups=D**: каждый канал — свой фильтр (4096 фильтров × 48 = 196,608 весов).
- **Causal padding**: stateful buffer через границы батчей (conv_state).
- Инициализация: нормальное std=0.01 (λ_d-производное).
- Skip-connection: `h = h + h_conv`.

---

## 6. VSA Memory — 4-масштабная векторная суперпозиция

### 6.1 Механизм

VSA (Vector-Symbolic Architecture) Memory хранит суперпозицию векторов
с экспоненциальным затуханием на 4 фиксированных временных масштабах:

```python
τ_s = [8, 32, 128, 512]  # 4 масштаба
d_s = exp(-1 / τ_s)       # decay per scale
```

### 6.2 Write gate

```python
i_gate = softplus(h · w_i + b_i + γ · ||pred_error||₂)
mem[t] = d_s · d_mod · mem[t-1] + i_gate · h[t]
```

- **Surprisal-gated**: gamma_surprisal масштабирует ошибку предсказания mirror
  (`_cached_pred_error_norm`) — удивление открывает write gate.
- **Content-dependent decay**: `d_mod = sigmoid(h · w_d + b_d)` — per-channel
  модуляция скорости забывания.
- **Noise injection**: в training режиме `i_gate *= 1 + noise_scale · randn`.

### 6.3 Dual readout + first moment

```python
mem_read = w_mem2v · mem + (1 - w_mem2v) · h  # взвешенное чтение
mu = exp(w_k_mu · mem + w_q_mu · h + w_mu_mem)  # первый момент
```

### 6.4 Per-channel временные константы

```python
b_d_init = 2.0 + 3.0 · layer_frac  # L0: τ≈7, L31: τ≈150
b_d ∈ [2.0, vsa_b_d_max]  # vsa_b_d_max=12.0 → τ до ≈163K
```

- AdaptiveController регулирует b_d динамически из exploration сигнала.
- `vsa_b_d_smooth=0.999`: плавный lerp к target (≈1000 шагов).

### 6.5 fp32 guard

Весь сканирование в float32 независимо от dtype модели — критично для
долгой памяти под AMP.

### 6.6 Параметры (per layer)

w_i (D), b_i (D), w_d (D), b_d (D), w_q (D), w_q_leaf (D), w_q_ctx (D),
w_mem2v (D), gamma_surprisal (1), scale_w (S × D), w_k_mu (D), w_q_mu (D),
w_mu_mem (D) = ~11 × D + S×D + 1 ≈ 61,441 на слой.

---

## 7. BottleneckBind — межканальное скрещивание

### 7.1 Концепция

BottleneckBind проецирует D=4096 → K=64 через билинейное скрещивание
и проецирует обратно: `D → K ↔ K → D`. Единственный cross-dimensional
mixing механизм (без attention).

### 7.2 Режимы

#### off: диагональное билинейное произведение

```python
prod = (hp · w_u) ⊙ (hp · w_v)  # поэлементное произведение двух проекций
return prod @ W_out
```

Ранг ≤ K=64. Без сдвигов.

#### shift (по умолчанию для S=4)

```python
for s in range(S):
    prod = (hp · w_u[s]) ⊙ roll(hp · w_v[s], golden_shift[s])
    acc += prod @ W_out[s]  # multi-ocular: per-shift W_out
```

- **Golden-ratio сдвиги**: `shift[s] = floor(s · K / φ) mod K`.
- **Multi-ocular** (авто при tie_bind + S>1): отдельный W_out[s] на сдвиг.
  Ранг = min(S·K, D) = min(256, 4096).

#### cascade: Фибоначчи-вложенные моночлены

```python
a[1] = hp · w_u[0]
a[2] = hp · w_v[0]
a[n] = (a[n-1] · w_u[n-1]) ⊙ roll(a[n-2] · w_v[n-1], shift[n-1])
a[n] = normalize(a[n]) · ||a[1]||  # ремасштабирование
return Σ softmax(mix_logit)[n] · a[n] @ W_out[n]
```

Степени моночленов: [1, 1, 2, 3, 5, 8, ...] (F_n).

#### gated aperture (bind_twist_gate)

Per-token adaptive gate: `g = sigmoid(W_gate_proj(hp))` — каждый сдвиг
получает свой вес `g[s]`, модулирующий `prod[s]`.

### 7.3 Инициализация

- w_u, w_v: std=1.0 (критичен — градиент ∝ std³).
- W_proj: Linear(D→K), инициализация по умолчанию (kaiming).
- W_out: std=0.02.
- tie_bind: W_out = W_proj^T через forward pre-hook (автоэнкодер).

### 7.4 Параметры

Default (S=1, off):
- W_proj: D·K = 262,144; W_out: K·D = 262,144; w_u, w_v: 2·K = 128.
- Total per layer: 524,416. Total 32 layers: 16,781,312 (10.4%).

Multi-ocular (S=4, shift):
- W_proj: D·K = 262,144; W_out: S·K·D = 1,048,576; w_u, w_v: S·K·2 = 512.
- Total per layer: 1,311,232. Total 32 layers: 41,959,424 (22.5%).

---

## 8. GroupedCognitiveMirror — ансамбль из 32 экспертов

### 8.1 Структура

32 эксперта, каждый в своём d=128 подпространстве (D/G = 128).
Каждый эксперт:

1. Проецирует h[g] (d=128) → hp[g] (k) через W_proj (d×k)
2. Вычисляет 4-5 сигналов коррекции в K-space
3. EMA-нормирует сигналы
4. Смешивает через learnable softmax
5. delta = Σ w_i · signal_i
6. delta @ W_out (k→d) → tanh + skip_alpha · linear → gate → exp(log_scale)

```
h_g (B, L, d) → W_proj (d→k) → hp (B, L, k)
  → [signals: temp, pred_error, smooth, sym, (help_k)]
  → EMA norm → softmax mix → delta (B, L, k)
  → W_out (k→d) → linear → gate(sigmoid)
  → mirror = (tanh(linear) + skip_alpha · linear) · gate · exp(log_scale)
```

### 8.2 Staircase k (D=4096)

При `mirror_k_staircase=True` (по умолчанию) k варьирует по глубине:

| Треть слоёв | k | d/k |
|:-----------:|:-:|:---:|
| L0-L10      | 4 | 32  |
| L11-L21     | 8 | 16  |
| L22-L31     | 16| 8   |

Ранние слои (сенсорные): маленькое k, большое d/k — широкий K-space анализ.
Глубокие слои (концептуальные): большое k, малое d/k — высокая точность K-space.

### 8.3 Пять сигналов K-space

| Сигнал | Формула | Семантика |
|--------|---------|-----------|
| temp_k | hp - mc_k | Отклонение от центроида памяти (что изменилось) |
| pred_error | hp - α · hp_prev | Ошибка предсказания K-space траектории |
| smooth_k | hp - conv1d(hp) | Локальная негладкость |
| sym_k | (hp·w_u) ⊙ (hp_prev·w_v) | Билинейное временное cross-term |
| help_k (L1) | attn @ private_mem | Коллективная память уверенных экспертов |

При `private_mem=False` (режим без L1): 4 сигнала (без help_k).

### 8.4 EMA-нормировка сигналов

```python
signal_norm_ema[n] = 0.999 · ema + 0.001 · RMS(signal[n])
signal_normed = signal / (ema[n] + 1e-8)
```

Текущий RMS каждого сигнала сглаживается (τ ≈ 1000 шагов).
Инициализация ema = 3.0 (консервативная — подавление на старте).

### 8.5 Learnable signal weights

```python
w = softmax(signal_log_weights)  # n_signals весов, sum=1
delta = Σ w[i] · signals_normed[i]
```

Энтропийная регуляризация: `-signal_entropy_weight · H(w)` в loss.
Вес 0.001 — мягкое поощрение равномерного использования.

### 8.6 Predictive mirror (alpha)

Каждый K-space канал имеет свою временную константу:

```python
τ_k ∈ [2.0, 200.0]  # логарифмическая шкала
α_k = exp(-1 / τ_k)
pred_k = α_kg · hp_prev  # per-expert, per-dim предсказание
pred_error = hp - pred_k  # ошибка предсказания
```

- alpha_diag: init из τ-иерархии, (G, k).
- w_pred_scale: init=3.0, learnable per-expert масштаб ошибки.

### 8.7 Skip connection (L0)

```python
linear = delta @ W_out  # (B, L, G, d)
mirror_raw = tanh(linear) + skip_alpha · linear
```

skip_alpha: геометрическая init по глубине (ρ^layer_idx):
L0 ≈ 17, L31 ≈ 0.10. Обеспечивает per-dim градиент для log_scale
даже при насыщении tanh.

### 8.8 K-Space gate (L2)

5 компонент gate_logits:

1. **|pred_error| @ w_gate** — «я не знаю этот паттерн» (L0)
2. **|delta| @ w_delta_gate** — «я применяю коррекцию» (Mirror)
3. **grad_mod** — «меня учит loss» (backprop)
4. **dvar_mod** — «я стабилен/нестабилен» (internal)
5. **disagreement · w_contra** — «я противоречу коллективу» (Arbiter)
6. **contra_expert** — «эксперт систематически противоречив» (коллектив)

```python
gate = sigmoid(gate_logits)  # (B, L, G)
mirror = mirror_raw · gate.unsqueeze(-1)  # gated
```

### 8.9 Self-organizing usefulness

Каждый эксперт предсказывает свою полезность:

```python
usefulness_logits = predictor(delta)  # (B, L, G)
threshold = median(usefulness_logits)  # per-token
usefulness = sigmoid((logits - threshold) / temperature)
```

Temperature самоорганизуется: homeostatic control к target_entropy=0.75·G·log(2).
Usefulness модулирует MLP выход.

### 8.10 tie_mirror_proj

При `tie_mirror_proj=True`: W_out = W_proj^T (автоэнкодер).
Синхронизация через forward pre-hook.

### 8.11 Параметры (per layer, k=4/8/16)

| Параметр | k=4 | k=8 | k=16 |
|----------|:---:|:---:|:----:|
| W_proj (d×k) | 32·4·128=16,384 | 32·8·128=32,768 | 32·16·128=65,536 |
| W_out (k×d) | 16,384 | 32,768 | 65,536 |
| alpha_diag | 32·4=128 | 32·8=256 | 32·16=512 |
| w_gate/b_gate | 32·4+32=160 | 32·8+32=288 | 32·16+32=544 |
| w_delta_gate | 128 | 256 | 512 |
| log_scale | 32·128=4,096 | 4,096 | 4,096 |
| прочие | ~1,500 | ~1,500 | ~1,500 |
| **Total** | **~38K** | **~72K** | **~138K** |

32 слоя: ≈ 11 × 38K + 11 × 72K + 10 × 138K = 418K + 792K + 1.38M ≈ 2.59M
(с tie_mirror_proj ≈ 1.87M, т.к. W_out не хранится).

---

## 9. Private Memory Bank — мета-познание (L1)

### 9.1 Концепция

Коллективная память уверенных K-space состояний экспертов.
Каждый эксперт накапливает EMA своих состояний, когда уверен,
не противоречит коллективу и не под социальным давлением.

```python
_private_mem[g]  # (G, k) — persistent buffer
```

### 9.2 Механизм записи

Три условия:

**Уверенность**: `conf = sigmoid(-|pred_error|)`
**Непротиворечие**: `contra = sigmoid(||hp-help_k||/||hp|| - 1)`
**Социальное давление**: `social_pressure = 1 - 0.5 · sigmoid(relu(contra_expert) + isolation)`

```python
conf_plastic = conf · (1 - contra) · social_pressure  # [0, 1]
conf_bc = conf_plastic^T · G / sum(conf_plastic^T)  # soft-competition T=0.5
weighted_hp = mean(conf_bc · hp.detach(), dim=(0,1))  # (G, k)
pm_decay = 0.999 - 0.009 · sigmoid(3.0 - ||pm||)  # [0.990, 0.999]
_private_mem = pm_decay · mem + (1 - pm_decay) · weighted_hp
```

### 9.3 Delayed write (_pm_write_delay)

```python
_pm_step += 1
if _pm_step < 5000:
    return  # пишем только после 5000 forward шагов
```

5000 шагов ≈ 640K токенов (accum=8, batch=2, seq_len=128).
Предотвращает echo chamber collapse на старте, когда все эксперты
одинаковы и пишут друг в друга одно и то же.

G=32 требует большей задержки, чем G=8 (Mini), из-за более высокого
риска эхо-камеры.

### 9.4 Механизм чтения (cross-expert attention)

```python
uncert = sigmoid(|pred_error|)     # неуверенность
q = hp · uncert                     # запрос от неуверенных экспертов
keys = private_mem.detach().clone() # заморожен (нет градиентного цикла)
attn = softmax(q @ keys.T / √k)     # (B, L, G, G) — кто у кого спрашивает
help_k = attn @ keys                # взвешенная сумма
trust = 1 - contra                  # доверие к коллективу
help_k = help_k · sigmoid(w_help) · trust
```

- w_help init = log(3) ≈ 1.1 → sigmoid ≈ 0.75 (сильное начальное присутствие).
- trust = 1 - contra: высокое противоречие → низкое доверие.
- keys заморожен `.detach().clone()` — эксперт не может изменить память,
  из которой читает.

### 9.5 help_k как 5-й сигнал

help_k добавляется к 4 базовым сигналам (temp, pred_error, smooth, sym):
все 5 смешиваются через learnable softmax.

---

## 10. Expert Knowledge Graph (L1.5)

Граф G×G (32×32), отражающий отношения между экспертами.

### 10.1 Concept Similarity

```python
pm_n = _private_mem / ||_private_mem||.clamp(min=1e-10)
concept_sim = pm_n @ pm_n.T  # (G, G) — cosine similarity
_concept_sim_ema = 0.999 · ema + 0.001 · concept_sim
```

### 10.2 Behavior Divergence

```python
hp_avg = mean(hp, dim=(0,1))  # (G, k) — усреднённое за шаг
behavior_sim = F.normalize(hp_avg) @ F.normalize(hp_avg).T
behavior_div = 1 - behavior_sim
```

### 10.3 Contradiction Graph

```python
contra_graph = concept_sim · behavior_div  # думают похоже, действуют по-разному
contra_expert[g] = mean_j(contra_graph[g, j])
```

### 10.4 Trust Matrix

```python
_trust_matrix = 0.999 · trust + 0.001 · mean(attn, dim=(0,1))
dominance[g] = sum_j trust_matrix[j, g]
isolation[g] = 1 - sum_j trust_matrix[g, j] / G
```

### 10.5 EVA-inspired mapping

| EVA (symbolic) | WideBind (K-space) |
|---------------|--------------------|
| \|v1-v2\|/\|v1\| | disagreement = \|hp - help_k\| / \|hp\| |
| 1-cos(e1,e2) | concept_sim = cos(pm[g1], pm[g2]) |
| 1-Jaccard(w1,w2) | behavior_div = 1 - cos(hp_avg[g1], hp_avg[g2]) |
| is_a chain | contra_graph = concept_sim · behavior_div |
| A→B→C→A | broken trust: t[i,j]·t[j,k]·(1-t[k,i]) |
| Ambiguity | contra_expert[g] = mean_j(contra_graph[g,j]) |

---

## 11. Arbiter: K-Space Gate (L2, детали)

### 11.1 grad_mod — gradient signal

```python
grad_mod = exp(log_grad_mod_scale) · tanh(prev_grad_norm + grad_mod_bias)
prev_grad_norm  # устанавливается hook'ом после backward
```

Если loss «толкает» подпространство эксперта → gate открывается.

### 11.2 dvar_mod — variance signal

```python
dvar = mean(var(delta, dim=(0,1)), dim=-1)  # (G,)
dvar_mod = exp(log_dvar_mod_scale) · tanh(dvar + dvar_mod_bias)
```

Стабильная коррекция → gate открыт. Нестабильная → закрыт.

### 11.3 Contradiction signal

```python
gate_logits += disagreement · w_contra
gate_logits += contra_expert
```

w_contra init +0.01: disagreement открывает gate.
contra_expert: систематически противоречивый эксперт всегда открыт.

---

## 12. GroupedMLP — Feed-Forward

### 12.1 Архитектура

32 группы × (128 → 512 → 128, SiLU). ~83% параметров.

```python
h = F.rms_norm(h, (D,), norm_w)          # pre-LN
h = h.reshape(B, L, G, d)                 # split на G групп
h = SiLU(einsum('blgd,gdf->blgf', h, W_up))  # d → expand·d
h = einsum('blgf,gfd->blgd', h, W_down)      # expand·d → d
```

Per-group: d · 4d + 4d · d = 2 · d² · expand = 2 · 128² · 4 = 131,072.
Per layer: 32 × 131,072 = 4,194,304.
Total 32 layers: 134,217,728 (83.2% default).

### 12.2 Usefulness modulation

```python
modulation = usefulness · sigmoid(mod_scale_mlp)  # от mirror
out = mlp(h_g) · modulation.unsqueeze(-1)
```

Эксперты с высокой predicted usefulness получают больший вес в MLP.

---

## 13. DCT Spectral — частотная фильтрация

```python
V_dct = dct_basis(D)  # (D, D) — DCT-II ортонормированный базис
h_dct = V_dct.T @ h            # преобразование в частотную область
h_dct = h_dct · lambda_k        # per-frequency масштабирование
h = V_dct @ h_dct               # обратное преобразование
```

- lambda_k init: `base + 0.1 · (1.0 → 0.5 линейно по частотам)`
  base = 0.5 + layer_idx / 31 ([0.5, 1.5]).
- Per-dim вариация: низкие частоты boost, высокие cut (1/f эффект).
- lambda_k learnable: модель учится, какие частоты важны для данного слоя.

---

## 14. AdaptiveController — самоорганизующиеся гиперпараметры

Два фундаментальных сигнала управляют каждым параметром:

```python
exploration = min(1, |mirror| / λ⁻²)        # λ⁻² ≈ 0.296
differentiation = min(1, var(log_scale) / λ⁻⁴)  # λ⁻⁴ ≈ 0.087
```

λ_d иерархия (d=3): λ₃ ≈ 1.839. Все границы выведены из λ_d.

### Что регулируется:

| Параметр | Диапазон | От чего |
|----------|----------|---------|
| b_d (decay bias) | [2.0, vsa_b_d_max] | exploration (высокая → короткая память) |
| b_i (write gate) | [-3.0, -1.5] | exploration |
| w_mem2v_scale | [λ⁻¹, 1.0] ≈ [0.544, 1.0] | differentiation |
| ema_alpha | [1-λ⁻⁶, 1-λ⁻⁸] ≈ [0.974, 0.992] | differentiation |
| noise_scale | [λ⁻⁸, λ⁻⁶] ≈ [0.0076, 0.026] | differentiation |
| tanh_bias_mod | [1.0, 1.5] | exploration |
| spectral_mod | [1±λ⁻⁴] ≈ [0.913, 1.087] | differentiation |
| pred_weight | [λ⁻⁶, λ⁻²] ≈ [0.026, 0.296] | differentiation |

### Layer-wise

```python
for layer in layers:
    l_expl, l_diff = AdaptiveController.layer_stats(layer)
    layer.b_d = lerp(target_b_d(l_expl), smooth=0.999)
    layer.b_i = lerp(target_b_i(l_expl), smooth=0.999)
```

---

## 15. MirrorLRScheduler — самоорганизующийся LR

### 15.1 Механизм

```python
mult = min(var_mult, alpha_mult, gate_mult) * mag_factor
       * loss_lr_factor * train_loss_lr_factor
mult = clamp(mult, 0.05, 1.0)
lr = base_lr * mult
```

### 15.2 Компоненты

| Компонент | Что измеряет | Поведение |
|-----------|-------------|-----------|
| var_mult | var(log_scale) / target_var | Counter-cyclical: специализация растёт → LR падает |
| alpha_mult | \|1-α\| / target_var | Counter-cyclical: ошибка предсказания падает → LR падает |
| gate_mult | gate_var / target_var | Gate стабилизируется → LR падает |
| mag_factor | \|mirror\| / mag_threshold | cap ≤ 1.0: сильная коррекция → LR не растёт |
| loss_lr_factor | val_loss > 1.02·best → ×0.5 | ReduceLROnPlateau, сброс на new best |
| train_loss_lr_factor | train_loss ↑ 5% за 100 шагов → ×0.7 | Демпфирование тренировочного loss |

### 15.3 Warmup

λ_d-выведенный warmup (fib(11)+50 = 199 шагов):
- alpha_override: 1.0 → 0.0 (плавный переход identity → learned)
- temperature: 2.0 → 0.5 (размытая конкуренция → острая)

---

## 16. λ_d Hierarchy — иерархия параметров

При `lambda_lr_hierarchy=True` (по умолчанию) группы параметров
получают LR, масштабированный по степеням λ_d:

| Группа | Примеры параметров | LR mult (λ=1.839) |
|--------|-------------------|:-----------------:|
| p=-2: embed | embed, readout | 0.296× |
| p=-1: mlp | MLP W_up/W_down | 0.544× |
| p= 0: default | conv, norm, head | 1.000× |
| p=+1: mirror | W_proj, alpha, log_scale | 1.839× |
| p=+2: gate | w_gate, b_gate, gates | 3.384× |
| vsa (λ⁻⁴) | b_d, b_i | 0.087× |

Более глубокие слои (мета-познание, gates) учатся быстрее.
Базовые слои (embed, MLP) — медленнее и стабильнее.

---

## 17. Loss

| Компонент | Вес | Описание |
|-----------|-----|----------|
| CE (PAD/EOS masked) | 1.0 | Стандартный cross-entropy |
| pred_loss (K-space) | adaptive 0.05–1.0 | MSE предсказания K-space |
| gate_l1 | 0.001 | Разреженность gate (L1) |
| reinforce | 0.01 | Gate должен совпадать с usefulness |
| balance | 0.01 | Load balancing (энтропия использования → logG) |
| diversity | 0.001 | ||cov(MLP_groups) - I||² |
| signal_entropy | 0.001 | -H(omega) — равномерность сигналов |
| nuclear | 1e-5 | Ядерная норма W_proj (стохастическая) |
| orth | 1e-4 | Ортогональность W_proj ||Ŵ^TŴ - I||² |
| w_m2v_hierarchy | 0.001 | Push w_m2v к target ∝ σ(ln τ) |
| branch_balance | 0.0 | Выравнивание log-var conv/bind/mirror |
| log_scale_l2 | 0.01 | Штраф exp(ls) > 10 |

### 17.1 Surprisal-weighted CE

При `surprisal_weight > 0`:
```python
w = (CE / mean(CE))^γ  # информативные токены получают больший вес
ce_loss = mean(w · CE)
```

### 17.2 PAD/EOS masking

Токены 0 (PAD) и 2 (EOS) исключены из CE loss.

---

## 18. Параметры

### 18.1 Default (S=1, bind_twist_mode=off)

| Компонент | Параметров | % |
|-----------|-----------|----|
| Embed + LM Head | 4,096 | <0.01 |
| Embed + LM Head (basis+head) | 55,120 | 0.03 |
| BottleneckBind (K=64, S=1) | 16,781,312 | 10.40 |
| GroupedCognitiveMirror | 1,867,712 | 1.16 |
| Conv1d (k=48, groups=D) | 6,291,456 | 3.90 |
| DCT Spectral (lambda_k) | 131,072 | 0.08 |
| VSA Gates (4-масштабная) | 1,835,040 | 1.14 |
| Private Memory (w_help, w_contra) | 1,056 | <0.01 |
| GroupedMLP (expand=4) | 134,348,800 | 83.22 |
| **Total** | **~161,443,632** | **100** |

### 18.2 Multi-Ocular (S=4, bind_twist_mode=shift)

| Компонент | Параметров | % |
|-----------|-----------|----|
| Embed + LM Head | 55,120 | 0.03 |
| BottleneckBind (S=4, multi) | 41,959,424 | 22.48 |
| GroupedCognitiveMirror | 1,867,712 | 1.00 |
| Conv1d (k=48, groups=D) | 6,291,456 | 3.37 |
| DCT Spectral | 131,072 | 0.07 |
| VSA Gates | 1,835,040 | 0.98 |
| Private Memory | 1,056 | <0.01 |
| GroupedMLP (expand=4) | 134,348,800 | 71.99 |
| **Total** | **~186,621,744** | **100** |

### 18.3 K-space mirror staircase (32 слоя)

| Диапазон слоёв | k | Параметров (total) |
|:-------------:|:---:|:----------------:|
| L0-L10 (11 слоёв) | 4 | ~418K |
| L11-L21 (11 слоёв) | 8 | ~792K |
| L22-L31 (10 слоёв) | 16 | ~1.38M |

---

## 19. Ключевые отличия от Mini (D=896, G=8)

| Аспект | Mini | Main |
|--------|------|------|
| D | 896 | 4096 |
| G (экспертов) | 8 | 32 |
| L (слоёв) | 12 | 32 |
| bind_K | 32 | 64 |
| d per expert | 112 | 128 |
| k (staircase) | 32 (flat) | 4/8/16 (по глубине) |
| Embedding | Partitioned (8×112) | Sparse Block Codes (6/32) |
| VSA | single-scale (τ per channel) | 4-scale (τ=8,32,128,512) |
| AdaptiveController | нет | λ_d-выведенный |
| λ_d LR hierarchy | нет | да (p=-2..+2) |
| Vocab | 11,000 | 50,000 |
| Параметров (default) | 11.20M | 161.44M |
| VRAM | 2.1 GB (MX550) | 11-16 GB (T4) |

---

## 20. Масштабирование

Формулы зависимостей:
```
D = G · 128    (d = 128 фиксировано)
L = G          (число слоёв = числу экспертов)
bind_K = D/G = 128  (для Main: переопределено в cfg.bind_K=64)
k (mirror) = staircase 4/8/16 (λ_d-оптимизировано)
VSA O(L · log(CHUNK)) — длина последовательности не влияет на веса
```

Все конфигурации, кроме bind_K, следуют λ_d-иерархии:
- warmup = fib(λ_d²+1) + 50
- boundaries: target_var = λ^{-2}, mag_threshold = λ^{-2}, ...
- LR иерархия: степени λ_d^p для разных типов параметров

---

*WideBind Main — C. BlackCatSpb, July 2026*
