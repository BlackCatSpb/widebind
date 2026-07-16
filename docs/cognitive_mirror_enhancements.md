# Cognitive Mirror: Предлагаемые улучшения

> **⚠️ Исторический документ (эпоха W_pred, D=3584, mirror_k=8).**
> **Текущая архитектура (2026-07-16): alpha (G,) вместо W_pred, G=32, mirror_k=32, D=4096, 293M.**
> **W_pred не учился 10000 шагов — заменён на scalar alpha. Skip connection (п.4) реализован.**
> **См. `core/model.py` `GroupedCognitiveMirror`.**

## Текущее состояние

```
hp = h @ W_proj                          # D → K
temp_k  = (hp - mem_centroid) * w_temp   # временное отклонение
smooth_k = hp - conv1x3(hp)              # локальная когерентность
sym_k   = (hp * w_u) * (hp_prev * w_v)  # билинейная траектория

delta = temp_k + smooth_k + sym_k
delta = rms_norm(delta, K)

mirror = tanh(delta @ W_out + tanh_bias)         # K → D
mirror = mirror * exp(log_scale)                  # per-dim масштаб

h = h + mirror
```

**Проблема:** `var(log_scale) ≈ 0` за 22000 шагов — per-dim специализации нет.

---

## 1. Lacuna Correction (амплитудные провалы)

**Идея из EVA-Ai (KCA):**
```
E_lacuna = (1 - attention) @ value_knowledge
```
Заполняет пробелы: размерности, которые модель игнорирует, получают принудительный push.

**Математика:**

Per-dim RMS энергии `h`:
```
rms_d = sqrt(mean_{b,l}(h[b,l,d]^2))
```

Размерности с `rms_d < threshold` — «мёртвые» — получают коррекцию:
```
lacuna_mask_d = relu(thresh - rms_d)        ∊ [0, thresh]  (≈0 для активных, >0 для мёртвых)
lacuna_k = lacuna_mask @ W_proj * hp        # push в K-space
```

**Градиентный эффект:** каждая D-размерность получает градиент пропорционально её lacuna_mask. Мёртвые dims получают большой gradient push. Это разрывает петлю «мёртвая dim → нет градиента → мёртвая dim».

**Проблема:** `W_proj` — D×K. lacuna_mask (D) → lacuna_k (K) теряет размерностную информацию (K=16 ≪ D=3584). Все мёртвые dims коллапсируют в один K-сигнал.

**Решение:** lacuna_mask должен напрямую влиять на mirror output, не проходя через узкое K-горло. Это означает: добавить лакунарный сигнал в D-space, а не K-space.

---

## 2. Variance Contradiction

**Идея из EVA-Ai (KCA):**
```
E_contra[i] = H[u] - H[v]   при cosine_sim(u,v) < threshold
```
Размерности с конфликтующими паттернами раздвигаются.

**В терминах WideBind — temporal variance:**
```
hp_var = (hp - hp_prev)^2     # квадратичное отклонение от предыдущего шага
contra_k = -hp_var * w_contra  # подавление нестабильных dims
```

**Математика:**
- Если dimension d резко меняется между t-1 и t, variance велика
- Отрицательный сигнал подавляет эту размерность
- Эффект: сглаживание нестабильных dims, усиление стабильных

**Проблема:** уже есть smooth_k = hp - conv1x3(hp), который делает то же самое (локальная когерентность). `contra_k` — это просто квадратичная версия smooth_k. Возможно, избыточно.

---

## 3. Mirror Gate

**Идея из EVA-Ai (KnowledgeConsciousAttention):**
```
gamma = sigmoid(concat(X_prev, E_corr) @ W_g + b_g)
X_new = X_prev + gamma * E_corr
```

Научиться доверять mirror'у когда он прав, игнорировать когда шумит:
```
mirror_gate = sigmoid(h @ w_mirror_gate + b_mirror_gate)   # (B, L, D)
h = h + mirror_gate * mirror
```

**Математика:**
- gate ≈ 1: mirror коррекция применяется полностью
- gate ≈ 0: mirror игнорируется
- Градиент по w_mirror_gate: если mirror был полезен (loss ↓), gate открывается; если вреден (loss ↑), gate закрывается

**Проблема:** на ранних этапах gate закроется (0), и mirror перестанет учиться — та же проблема, что и с i_gate. Нужен индуктивный сдвиг (b_mirror_gate > 0) или обязательная коррекция.

---

## 4. Skip Connection через tanh (решение root cause)

**Проблема:** градиент к log_scale[d] = `dL/d(mirror[d]) * mirror[d]`. В mirror = `tanh(linear) * exp(log_scale)`, tanh насыщается (±1) и производная tanh² ≈ 0 для больших значений. Градиент пропадает.

**Решение:** добавить линейный skip вокруг tanh:
```
mirror = tanh(linear + tanh_bias) + alpha * linear
mirror = mirror * exp(log_scale)
```
где `alpha = 0.1` (гиперпараметр) или `nn.Parameter` (learnable).

**Математика — градиент к log_scale:**
```
d(mirror)/d(log_scale) = mirror * exp(log_scale) / exp(log_scale)  ... нет, проще:
   
mirror = A * exp(log_scale), где A = tanh(linear) + alpha * linear

dL/d(log_scale[d]) = dL/d(mirror[d]) * A[d] * exp(log_scale[d])
```

Ключевой момент: **A[d]** имеет per-dim variance ЛЮБОЙ величины, в отличие от tanh(linear)[d] который ограничен [-1, 1] и симметричен. Компонента `alpha * linear` может быть любой величины и несимметричной, давая каждой D-размерности уникальный градиент к log_scale[dim].

**При alpha=0.1:** для типичного linear ~ N(0, 1), A ~ tanh(±1-2) + 0.1*(±1-2) ≈ ±0.76 + ±0.15 = [±0.61, ±0.91]. Вариация A между dims: σ(A) ≈ 0.15 (против ≈0.01 без skip). Достаточно для дифференциации log_scale.

**Доказательство (chain rule полный):**
```
dL/d(log_scale[d]) = dL/d(mirror[d]) * A[d] * exp(log_scale[d])
dL/d(linear[d])    = dL/d(mirror[d]) * ((1-tanh²(linear[d])) + alpha) * exp(log_scale[d])
```

Градиент к tanh_bias и W_out теперь содержит и tanh-компоненту, и линейную. Линейная компонента не насыщается — градиент течёт всегда.

---

## Сравнительный анализ

| Метод | Сложность | Решает root cause? | Побочные эффекты |
|---|---|---|---|
| **Skip connection (α=0.1)** | ~1 строка | **ДА** — даёт per-dim градиент для log_scale | Нет |
| Lacuna correction | ~10 строк, новый сигнал | Частично — помогает мёртвым dims | Избыточен при работающем skip |
| Variance contradiction | ~5 строк | Нет — дублирует smooth_k | Избыточен |
| Mirror gate | ~10 строк, новые параметры | Нет — gate закроется, mirror умрёт | **Опасен** — повторяет проблему i_gate |
| Convergence detection | ~5 строк | Нет — оптимизация, не обучение | Не нужен на ранней стадии |

**Вывод:** только **Skip Connection** решает фундаментальную проблему. Всё остальное — оптимизация после того, как mirror начнёт работать.

---

## Практическая рекомендация

1. **Skip connection** — делаем сейчас, alpha=0.1 (learnable делать не будем — лишний гиперпараметр)
2. **Всё остальное** — откладываем, пока не убедимся, что var(ls) растёт
3. **После 5000 шагов** с новым mirror: если var(ls) всё ещё ≈ 0 — переходим к лакунарной коррекции

---

## Код изменений

```python
# В CognitiveMirror.__init__:
self.skip_alpha = 0.1  # fixed scalar (можно nn.Parameter)

# В CognitiveMirror.forward:
linear = delta @ self.W_out
mirror = torch.tanh(linear + self.tanh_bias) + self.skip_alpha * linear
mirror = mirror * torch.exp(self.log_scale)
```
