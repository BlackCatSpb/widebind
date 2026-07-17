# WideBind: Полный анализ архитектуры и предложения по улучшению

**Дата анализа:** 2026-07-19
**Чекпоинт:** step_25000 (150.9M params, D=4096, L=32)

---

## 1. Обзор проекта

WideBind — это экспериментальная языковая модель с нетривиальной архитектурой, вдохновлённая VSA (Vector Symbolic Architectures) и заменяющая стандартный transformer attention на систему **32 когнитивных экспертов-зеркал** с предиктивным самомоделированием. Проект нацелен на экстремальную параметрическую эффективность: эмбеддинг + head занимают всего 58K параметров (0.04% модели).

---

## 2. Архитектура: математические обоснования

### 2.1 PartitionedEmbedding (Разрежённое блочное кодирование)

**Математика:**
Каждый токен `v` кодируется бинарным кодом `z_v ∈ {0,1}^32` с ровно S=6 единицами, генерируемым через **combinadic unranking** (комбинаторная система счисления):

```
C(32,6) = 906,192 ≥ 50,000  (ёмкость кодов > словарь)
```

D разбивается на K=32 сегмента по d=128 dims. Эмбеддинг:

```
h_v = concat_k[ z_{v,k} · basis[k] ] = (z_v ⊗ I_d) · vec(basis)
```

**Ключевые свойства:**
- **Параметры:** K·d = D = **4,096** (vs V·D = 204.8M у стандартного эмбеддинга — **50,000× компрессия**)
- **Ранг:** ≤ 32 (все d столбцов сегмента k параллельны) — документированное rank=3584 неточно
- **Разрежённость:** только 18.75% размерностей активны (6/32 бита)
- **Выравнивание 1:1:** сегмент = бит кода = эксперт зеркала — вся модель следует блочно-диагональной структуре

### 2.2 Bind Layer (VSA-биндинг)

**Математика:**
```
hp = h @ W_proj ∈ ℝ^K
bind_out = ((hp ⊙ w_u) ⊙ (hp ⊙ w_v)) @ W_out ∈ ℝ^D
```

Это **квадратичная форма ранга K**:
```
bind_out_j = Σ_k W_out[k,j]·(w_u[k]·w_v[k])·(Σ_i W_proj[i,k]·h_i)²
```

**Параметры:** D·K + 2K = **262,272** (tied) или 524,416 (untied). При tie_bind=True: W_out = W_projᵀ — tied-weight autoencoder с bottleneck K=64.

### 2.3 GroupedCognitiveMirror (Ключевая инновация)

**32 эксперта**, каждый владеет d=128 сегментом в D=4096 пространстве.

**Предиктивное зеркало (AR(1)):**
```
pred_k = α_g · hp_prev        # α_g — скалярный коэффициент AR(1)
pred_error = (hp − pred_k) ⊙ w_pred_scale ⊙ pred_scale_mod
```
- α init=0.98, заменяет полную матрицу перехода (G,k,k)=32,768 → 32 параметра
- **Вспомогательный лосс:** `L_pred = MSE(pred_k, hp.detach())` — градиент течёт в α и W_proj

**4 корректирующих сигнала:**
| Сигнал | Формула | Смысл |
|--------|---------|-------|
| temp | (hp − mc_k) ⊙ w_temp | Отклонение от центроида памяти |
| pred | (hp − α·hp_prev) ⊙ w_pred_scale | Темпоральная ошибка предсказания |
| smooth | hp − conv_smooth(hp) | Локальная некогерентность |
| sym | (hp ⊙ w_sym_u) ⊙ (hp_prev ⊙ w_sym_v) | Билинейное темпоральное связывание |

**Гейт-адаптация:**
```
gate_logits = einsum(|pred_error|, w_gate) + b_gate
            + exp(log_grad_mod)·tanh(prev_grad_norm + grad_bias)
            + exp(log_dvar_mod)·tanh(delta_var + dvar_bias)
expert_gate = σ(gate_logits)
```

**SelfOrganizingMirror:**
```
usefulness = σ(MLP(delta))           # 1,089 параметров
mlp_mod = usefulness ⊙ σ(mod_scale_mlp)
mem_mod = usefulness ⊙ σ(mod_scale_mem)
```

Каждый эксперт **самооценивает свою полезность** и через mlp_mod/mem_mod регулирует вклад MLP и VSA-памяти в соседних ветвях блока.

**Параметры зеркала/слой:** **146,785** (tied).

### 2.4 WideBindBlock (Полный поток)

```
h ← RMSNorm(h)
h ← h + Conv1d_causal(h)              # 48-tap depthwise
bind_out = Bind(h)                     # §2.2
i_gate = softplus(h ⊙ w_i + b_i)      # (0,∞), init≈0.078
decay = σ(h ⊙ w_d + b_d)              # (0,1), τ=e^{b_d}
mem = VSA_scan(decay, h ⊙ i_gate)     # O(L log L) scan
mirror, mlp_mod, mem_mod = Mirror(h, mem, global_state)
h ← h + bind_out + (mem_read ⊙ mem_mod ⊙ w_mem2v) + mirror
h ← h + Spectral(h)                    # DCT-based coloring
h ← h + (MLP(h) ⊙ mlp_mod)            # GroupedMLP, 8D²/G params
```

**VSA память:**
```
mem_t = decay_t ⊙ mem_{t−1} + i_gate_t ⊙ h_t
```
τ = 1/(1−σ(b_d)): L0 τ≈8 → L31 τ≈150. При b_d_max=12: τ≈163K.

**Параметры слоя:** **4,845,025** — MLP 86.6%, bind 5.4%, conv 4.1%, mirror 3.0%, VSA 0.76%, spectral 0.08%.

### 2.5 PartitionedHead (Структурно-связанный head)

**Математика:**
```
score_k = ⟨h_k, r_k⟩                 # K=32 скалярных скора
logit_v = Σ_k z_{v,k}·score_k + b_v
```

**Параметры:** K·d + V = **54,096** (vs 204.8M у стандартного head — 3,785× сжатие).

**Отношение к эмбеддингу:** транспонированная структура по той же кодовой матрице Z. Если readout ≡ basis, то logit_v = ⟨h, E_v⟩ — классический weight-tied LM head. Но здесь они **развязаны параметрически**, но **связаны структурно** через Z.

### 2.6 AdaptiveController + MirrorLRScheduler

**Два наблюдаемых:**
```
expl = min(1, |mirror|/λ⁻²)      # λ⁻²=0.296
diff = min(1, var(log_scale)/λ⁻⁴)  # λ⁻⁴=0.087
```

**8 модуляций** (все функции λ₃ ≈ 1.8393):

| Механизм | Формула | Диапазон |
|----------|---------|----------|
| layer_b_i | −3 + 1.5·expl | [−3, −1.5] → softplus [0.048, 0.201] |
| layer_b_d | b_d_max − expl·(b_d_max − b_d_min) | b_d_max=12 |
| w_mem2v_scale | 1 − diff·(1−λ⁻¹) | [0.544, 1.0] |
| ema_alpha | (1−λ⁻⁶) + diff·(λ⁻⁶−λ⁻⁸) | [0.974, 0.992] |
| noise_scale | λ⁻⁶ − diff·(λ⁻⁶−λ⁻⁸) | [0.0076, 0.026] |
| tanh_bias_mod | 1 + λ⁻²·expl | [1, 1.296] |
| spectral_mod | 1 + λ⁻⁴·(2diff−1) | [0.913, 1.087] |
| pred_scale_mod | clamp(δvar_g/mean, 0.1, 3.0) | [0.1, 3.0] |

**MirrorLRScheduler** (упрощённо):
```
mult = clamp(var_mult · alpha_mult · gate_mult · mag_factor, 0.05, 3.0)
lr = base_lr · mult
```

### 2.7 λ₃-иерархия (Математическая основа)

λ₃ ≈ 1.83929 — **трибоначчи-константа**, корень x³ = x² + x + 1.

**Ключевое свойство:** F_{n+1}/F_n → λ₃, поэтому:
- **Пороги:** λ⁻¹=0.544, λ⁻²=0.296, λ⁻³=0.161, λ⁻⁴=0.087, λ⁻⁶=0.026, λ⁻⁸=0.008
- **Временные константы:** α_EMA = 1−λ⁻ᵏ → τ = λᵏ (11.4, 38.7, 131 шагов)
- **Шаги:** Fibonacci numbers (55, 233, 987, 2584, 6765)

---

## 3. Идентифицированные проблемы

### 3.1 Критические

| Проблема | Описание | Файл |
|----------|----------|------|
| **MirrorLRScheduler не использует λ-параметры** | `target_var`, `lr_min_ratio`, `max_decay_steps`, `var_min_for_lr_decay` — мёртвые поля. Живая логика — только warmup, base_lr, mag_threshold=λ⁻² | `core/model.py:1057` |
| **Compression silently drops b_i/b_d variance** | Tier S хранит только b[0], но b_i/b_d — обучаемые параметры с `vsa_b_lr_mult=0.1`. После градиентных шагов они могут стать не-равномерными, и компрессия **бесшумно теряет per-channel вариацию** | `compression/fcf_cpr.py:114` |
| **Mode collapse при генерации** | После 1-2 токенов модель коллапсирует в повторение (`Его!!!!!!!!!`) — признание недоученности | — |
| **Embedding rank** | Документация утверждает rank=3584, реальный rank ≤ 32 (все d столбцов сегмента параллельны) | `core/model.py:143` |
| **PartitionedHead still materializes (B,L,V) logits** | K·V FLOPs ≈ 1.6M/token, но logits memory = (B,L,V) = 50000 элементов на токен — экономия только на параметрах, не на памяти | `core/model.py:190` |

### 3.2 Архитектурные

| Проблема | Описание |
|----------|----------|
| **state detach каждый forward** | Все (mem, mu, conv) состояния `.detach()` → усечённый BPTT между чанками. Градиент не течёт через границу батча |
| **α_override во время warmup** | Принудительная установка α=0.5→0.25 отключает обучение AR(1) динамике в самом начале |
| **pred_error без нормализации** | pred_error растёт с ростом hp, что может дестабилизировать гейт |
| **Gate variance signal** | `gate_var` измеряет дисперсию σ(gate_logits), но не саму специализацию — высокая дисперсия может означать шум, а не разнообразие |

### 3.3 Инженерные

| Проблема | Описание |
|----------|----------|
| **PartitionedEmbedding: 32 kernel launches** | Python loop по K=32 с `.item()` вызовами — неэффективно |
| **evaluate() тратит forward passes** | `evaluate` в train.py делает 100 батчей × forward без state reuse |
| **Checkpoint optimizer state** | Adam moments (2× model size) передаются некомпрессированными |

---

## 4. Предложения по улучшению

### 4.1 Исправление MirrorLRScheduler (HIGH)

**Проблема:** Создавался для управления LR через λ-иерархию, но реальная логика — просто clamp(warmup, 0.05, 3.0).

**Решение:** Реализовать задокументированную формулу:
```python
lr_mult = max(λ⁻⁶, (1 − v/target_var) · min(1, m/mag_threshold))
```
или добавить стабилизатор:
```python
# Текущая growth-ratio схема нестабильна (val_loss diverged: 1.72 → 4.23)
# Добавить damping при росте loss:
if val_loss_ema > prev_val_loss_ema * 1.05:
    lr_mult *= 0.9  # reduce LR when loss grows
```

### 4.2 Исправление compression b_i/b_d (HIGH)

**Решение:** Добавить проверку однородности:
```python
def is_scalar_gate(key, value):
    if any(s in key for s in ('.b_i', '.b_d')):
        # Проверить: все ли элементы равны?
        if value.std() < 1e-8:
            return True  # uniform — safe to compress
    return False  # non-uniform — use Tier W
```

### 4.3 Улучшение генерации (HIGH)

**Проблема:** Mode collapse после 1-2 токенов.

**Решения:**
1. **Repetition penalty:** Отслеживать последние N токенов и снижать их логиты:
   ```python
   logits[:, recent_ids] *= repetition_penalty  # ~0.9
   ```
2. **Minimum probability sampling:** Отбрасывать токены с p < threshold
3. **Nucleus sampling (top-p):** Вместо top_k использовать cumulative probability
4. **Longer training:** val_loss должен быть < 1.5 для качественной генерации

### 4.4 Оптимизация PartitionedHead (MEDIUM)

**Проблема:** (B,L,V) logits memory — 50000 floats на токен.

**Решение:** Использовать ZeckendorfReadout для inference (O(B·K) memory), или применить partitioned softmax:
```python
# Вместо полных logits, считать только top-K кандидатов
scores = ...  # (B, L, K=32)
topk_scores, topk_indices = torch.topk(scores, k=100, dim=-1)
# Вычислить logits только для этих 100 кандидатов
```

### 4.5 Векторизация PartitionedEmbedding (LOW)

**Решение:** Заменить Python loop на gather:
```python
def forward(self, x):
    codes_batch = self.codes[x]  # (B, L, K)
    h = torch.zeros(B, L, D)
    for k in range(self.K):
        o = int(self._offsets[k].item())  # ← bottleneck
        h[:, :, o:o+d] += codes_batch[:, :, k:k+1] * self.basis[k]
    return h

# Оптимизация:
segment_ids = torch.arange(self.K).repeat_interleave(self.d)
h = torch.zeros(B, L, D, device=x.device)
h.scatter_add_(2, segment_ids.expand(B, L, D), codes_batch.repeat_interleave(self.d, dim=2))
```

### 4.6 Улучшение SelfOrganizingMirror (MEDIUM)

**Проблема:** `usefulness_predictor` обучается на одном батче, может быть шумным.

**Решение:** Добавить EMA для usefulness:
```python
self.register_buffer('_usefulness_ema', torch.ones(G))
# В forward:
usefulness = 0.9 * self._usefulness_ema + 0.1 * usefulness_new
self._usefulness_ema = usefulness.detach()
```

### 4.7 Защита от дивергенции при single-genre training (HIGH)

**Проблема:** val_loss вырос с 1.72 до 4.23 при обучении на одном жанре.

**Решение:**
1. **Gradient clipping** более агрессивный: `grad_clip=0.5` вместо 1.0
2. **Early stopping** на val_loss с patience=3 eval'а
3. **LR warmup restart** при смене данных:
   ```python
   # При резюме с новыми данными:
   scheduler.restart_warmup(steps=100)
   ```

---

## 5. Математическая оценка эффективности

### 5.1 Параметрическая эффективность

| Компонент | Параметры | % модели |
|-----------|-----------|----------|
| Embedding | 4,096 | 0.003% |
| Head | 54,096 | 0.036% |
| Mirror (32 слоя) | 4,697,120 | 3.1% |
| MLP (32 слоя) | 134,348,800 | 86.6% |
| Bind (32 слоя) | 8,392,704 | 5.4% |
| Conv (32 слоя) | 6,291,456 | 4.1% |
| VSA + Spectral + Norm | ~500K | 0.3% |
| **Итого** | **~155M** | **100%** |

**Ключевой инсайт:** MLP занимает 86.6% параметров — это узкое место. Bind+Mirror+Conv+VSA вместе дают уникальную функциональность за ~12% параметров.

### 5.2 Вычислительная эффективность

| Операция | FLOPs/token |
|----------|-------------|
| Standard Transformer | ~2·D²·L (attention) + ~8·D²·L (MLP) |
| WideBind | ~2·D·K·L (bind) + ~4·D·d·G·L (MLP) + ~3·D·L (conv) + ~10·D·L (mirror) |
| С D=4096, L=32, K=64, G=32, d=128 | ~2× меньше FLOPs, но ~10× меньше параметров |

### 5.3 Компрессия

FCF_CPR: **>4× сжатие** (160MB vs ~700MB):
- Tier R: V_dct (D² per layer) + codes → 0 байт
- Tier S: b_i/b_d → 4 байта
- Tier W: uint8 quantization → 4:1

---

## 6. Приоритетные действия

| Приоритет | Действие | Эффект |
|-----------|----------|--------|
| **P0** | Исправить compression b_i/b_d (проверка однородности) | Предотвращение потери обученной динамики |
| **P0** | Добавить защиту от дивергенции (early stopping + LR damping) | Стабильность обучения |
| **P1** | Исправить MirrorLRScheduler (реализовать задокументированную формулу) | Корректная адаптация LR |
| **P1** | Добавить repetition penalty при генерации | Качество генерации |
| **P2** | Векторизация PartitionedEmbedding | Ускорение forward pass |
| **P2** | EMA для usefulness в SelfOrganizingMirror | Стабильность самоорганизации |
| **P3** | ZeckendorfReadout для inference | Экономия памяти |
| **P3** | Оптимизация evaluate() | Ускорение eval |

---

## 7. Заключение

WideBind представляет собой **архитектурно инновационную** модель с несколькими нестандартными идеями:

1. **Разрежённое блочное кодирование** — 50,000× компрессия эмбеддинга с сохранением структурной выразительности
2. **32 эксперта-зеркала с AR(1) предсказанием** — внутренняя динамическая модель на уровне слоя
3. **λ₃-иерархия** — единая математическая основа для всех гиперпараметров
4. **SelfOrganizingMirror** — самооценка полезности экспертов как дифференцируемый мета-контроллер

Основные проблемы — **нестабильность обучения** (дивергенция val_loss при смене домена) и **неиспользуемый потенциал λ-иерархии** в scheduler. После исправления этих проблем модель имеет потенциал для эффективного обучения на ограниченных ресурсах.

---

# РАУНД 2: Углублённый анализ и исправления (2026-07-19)

## 8. Новые находки (все исправлены)

### 8.1 Критические баги

| # | Баг | Файл | Статус |
|---|-----|------|--------|
| 1 | **ZeckendorfReadout.predict() декодировал LSB-first вместо MSB-first** — 9/10 токенов неверны. `zeckendorf_code()` идёт `reversed(fibs)` (MSB-first), бит уровня k = коэффициент при `fibs[K-1-k]`, а predict накапливал `fibs[k]`. Скрыт клампом `min(token_id, vocab-1)` | `core/zeckendorf_readout.py:187` | ✅ Исправлен + отсечение фантомов |
| 2 | **Loss damping в scheduler — one-step no-op**: `mult *= 0.5` применялся ровно к одному шагу оптимизатора, потом disarm навсегда. Переписан на персистентный `_loss_lr_factor` (ReduceLROnPlateau-семантика: halving при регрессе >2%, reset при новом best, floor 0.1) | `core/model.py:1057` | ✅ Исправлен |
| 3 | **state_dict scheduler не сохранял `_init_*`, `_best_val_loss`** → до 6× скачок LR при resume (ростовые якоря пере-захватывались). Теперь сохраняются все поля; старые чекпоинты грузятся через `.get()` с дефолтами | `core/model.py:1174` | ✅ Исправлен |
| 4 | **2× LR-клифф после warmup**: `var_mult` имел нейтраль 0.5 при growth=1. Теперь все growth-мультипликаторы нейтральны при growth=1: `clamp(growth, 0.5, 2.0)` | `core/model.py:1137` | ✅ Исправлен |
| 5 | **prepare_data.py: нет разделителя документов** — книги склеивались. Добавлен `<|eos|>` (id=2) после каждого документа; pass-2 failure теперь заполняет слот EOS вместо нулей (pad id 0); добавлена защита uint16 (vocab < 65536) | `scripts/prepare_data.py` | ✅ Исправлен |
| 6 | **analyze_checkpoint.py: KeyError gate_beta** — мёртвый ключ от старой архитектуры, заменён на `m.b_gate`. Также: краш `--log` при отсутствии best_val_loss, перепутанные заголовки Memory/Spectr., мутация модели при анализе (теперь `adaptive=False`) | `scripts/analyze_checkpoint.py` | ✅ Исправлен |
| 7 | **Repetition penalty sign-blind**: `logits[rid] *= 0.85` УВЕЛИЧИВАЛ вероятность для отрицательных логитов. Заменён на вычитание `-= 2.0` | `scripts/generate.py` | ✅ Исправлен |
| 8 | **gen_demo.py: первый токен из hidden state как embedding** (OOD-вход) + `adaptive=True` при генерации. Теперь первый токен берётся из логитов prompt-forward, вся генерация с `adaptive=False` | `scripts/gen_demo.py` | ✅ Исправлен |

### 8.2 Архитектурные находки

| Находка | Детали |
|---------|--------|
| **ZeckendorfReadout: дефицит словаря** | Дерево нормировано над листьями (доказано индукцией), но листьев F_{K+2} > V — ~59% листьев «фантомные» (≥V). Распределение над словарём не нормировано; обучение вытесняет фантомную массу → 0. Для точной нормировки нужен словарь размера F_{K+2} (Фибоначчи) или ре-нормировка |
| **temporal_zeckendorf — мёртвый код** | Не импортируется в model.py, флаг конфига нигде не читается. Плюс баг масштаба `trace` (делит на summand-count=6 вместо max-fib-idx=30) и немонотонность theta (zlen осциллирует) |
| **wpred_ablation — исторический артефакт** | Подходы A (no lo/hi split) и B (scalar alpha) уже влиты в нативное зеркало; C (InfoNCE) не принят. ApproachAMirror переписан как alias (был несовместим с интерфейсом: 3-tuple return + новые kwargs) |
| **benchmark_readout: методологическая проблема** | «Real hidden states» берутся из необученной модели с несдвинутыми таргетами (предсказание последнего входного токена) — секция бессмысленна |
| **Тесты** | 6 тестов зеркала ожидали одиночный тензор вместо 3-tuple SelfOrganizingMirror — обновлены. **61/61 PASS** |

### 8.3 Верификация scheduler (математика)

Диапазоны мультипликаторов: `var/alpha/gate_mult ∈ [0.5, 2.0]` (нейтраль при growth=1), `mag_factor ∈ [0.2, 1.0]` (чистый cap). Произведение ∈ [0.025, 8.0] → после clamp [0.05, 3.0]. Damping персистентен: `mult *= _loss_lr_factor` каждый шаг, `_loss_lr_factor ∈ [0.1, 1.0]`.

**Проверено численно:**
- Round-trip state_dict (старый и новый формат) ✅
- Damping: halving при регрессе, повторный halving, floor 0.1, гистерезис 2%, reset при новом best ✅
- Zeckendorf MSB-first decode round-trip (500 samples) ✅
- predict() диапазон [0, vocab) для greedy и sampling ✅

## 9. Обновлённые приоритеты

| Приоритет | Действие | Статус |
|-----------|----------|--------|
| ~~P0~~ | Персистентный loss damping в scheduler | ✅ Сделано |
| ~~P0~~ | Полный state_dict scheduler | ✅ Сделано |
| ~~P0~~ | Zeckendorf predict decode bug | ✅ Сделано |
| ~~P0~~ | EOS-разделители в prepare_data | ✅ Сделано |
| ~~P0~~ | Compression b_i/b_d однородность | ✅ Сделано (ранее) |
| ~~P1~~ | Repetition penalty sign-safe | ✅ Сделано |
| ~~P1~~ | analyze_checkpoint gate_beta + заголовки | ✅ Сделано |
| ~~P1~~ | gen_demo/generate adaptive=False | ✅ Сделано |
| ~~P1~~ | Тесты под SelfOrganizingMirror | ✅ 61/61 |
| **P2** | Решение по дефициту словаря Zeckendorf (Fibonacci-vocab или ре-нормировка) | Открыто |
| **P2** | Удалить или подключить temporal_zeckendorf (мёртвый код) | Открыто |
| **P2** | EMA для usefulness в SelfOrganizingMirror | Открыто |
| **P3** | Векторизация PartitionedEmbedding (32 kernel launches) | Открыто |
| **P3** | benchmark_readout: обученная модель + сдвинутые таргеты | Открыто |
| **P3** | Компрессия optimizer state (Adam moments 2× модели) | Открыто |

## 10. Итог раунда 2

Кодовая база приведена в консистентное состояние: **все найденные баги исправлены, 61/61 тестов проходят**. Scheduler теперь контр-цикличен (mag cap) и имеет настоящий ReduceLROnPlateau-демпфинг с полным состоянием при resume. Генерация получила корректный первый токен, sign-safe penalty и adaptive=False. Данные получили границы документов.

Ключевой оставшийся архитектурный вопрос: **дефицит словаря ZeckendorfReadout** (59% фантомных листьев) — требует дизайн-решения (Fibonacci-sized vocab vs ре-нормировка). Для немедленного обучения это не блокер: PartitionedHead — рабочий head по умолчанию.
