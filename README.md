# WideBind

Гибридная не-трансформерная языковая модель на основе VSA-памяти, bottleneck bind и когнитивного зеркала. Без softmax, без attention, без QKV.

**Параметры:** 41.22M | **Слои:** 24 | **D:** 896 | **Bind K:** 16 | **bottleneck:** 896

---

## Архитектура

### Блок WideBind (per-layer)

```
h → Pre-LN → Conv1d(depthwise, kernel=48) → h + conv
  → Bind (D→K=16→bilinear→D)
  → VSA Memory (prefix scan, O(L log L))
  → Cognitive Mirror (3 bounded paths)
  → enhanced = bind + mem_read * w_mem2v + mirror → h += enhanced
  → Spectral (DCT basis, learned lambda_k)
  → MLP (D→bottleneck→D, RMS norm, SiLU)
  → h_out
```

### VSA Memory
```
mem[t] = decay[t] · mem[t-1] + i_gate[t] · h[t]
decay[t] = sigmoid(h[t] · w_d + b_d)     ∈ (0, 1)
i_gate[t] = sigmoid(h[t] · w_i + b_i)    ∈ (0, 1)
```

- **τ ≈ 150** (b_d=5.0) — полный контекст 128 токенов с переносом между батчами
- Ассоциативный параллельный prefix scan (O(L log L))
- Первый момент (mu) для нормализации смещения

### Cognitive Mirror

Три пути самосогласованности в K-пространстве (16-dim):

| Путь | Формула | Смысл |
|---|---|---|
| Temporal | (h[t] - mem_centroid) · w_temp | отклонение от VSА-памяти слоя |
| Global | (h[t] - global_state) · w_global | отклонение от self-model всех слоёв |
| Smooth | h[t] - conv1x3(h[t]) | локальная когерентность |
| Symmetry | (h[t] · w_u) · (h[t-1] · w_v) | билинейная самосогласованность |

Все пути → K-space → rms_norm → tanh(W_out) → **exp(log_scale)** per-dim.
tanh гарантирует bounded correction. log_scale (init=0, exp=1) с полноценным градиентом.

### Spectral (DCT)

DCT-II базис с per-dim learned lambda_k. lambda_k растёт от 0.5 (L0) до 1.5 (L23) — модель учится усиливать высокие частоты на глубоких слоях.

### Embedding / LM Head

Zeckendorf коды Фибоначчи (K=23) + learned linear projection (D→23→50000).
50K словарный запас, BOS=1, EOS=2, PAD=0.

---

## Гипотеза специализации слоёв

На основе lambda_k (step 2000):

| Слой | lambda_k | Предполагаемая роль |
|---|---|---|
| L0 | 0.67 | Входной буфер — минимальная обработка, пропускает сигнал |
| L1-L4 | 0.67-0.76 | **Низкие частоты** — контекстная интеграция, выделение темы |
| L5-L12 | 0.76-1.02 | **Средние частоты** — синтаксис, локальные зависимости |
| L13-L18 | 1.02-1.28 | **Высокие частоты** — лексика, точное словоупотребление |
| L19-L22 | 1.33-1.46 | **Детализация** — уточнение, прагматика |
| L23 | 1.50 | **Сжатие в LM head** — коллапс в низкоранговое пространство (eff_rank≈4) |

lambda_k растёт монотонно по слоям, что согласуется с иерархической обработкой языка: первые слои строят контекст, последние — точный выбор токена.

---

## Статус обучения (step 2000)

| Метрика | Значение |
|---|---|
| Train loss | 1.34 (step 1500) |
| Val loss | 5.33 (ppl 205) |
| LR | 3e-4 (cosine decay, осталось 498K шагов) |
| Data | 2.86B токенов (ACTION 1.1B + DETECT 1.8B) |
| VRAM | ~1.9 GB (MX550, B=2, L=128) |
| Tok/s | ~250 |

---

## Известные проблемы

1. **MLP expansion = 1×** — bottleneck=896 не даёт MLP расширять признаки. eff_rank=228/896 (26% утилизации). Для 2× нужна другая конфигурация (n_layers=12, bottleneck=1792).
2. **Last-layer collapse** — L23 eff_rank≈4. Симптом 1× expansion, не исправляется архитектурой.
3. **b_d / b_i frozen at init** — decay (5.0) и gate (-3.0) не двигаются. init оказался в локальном минимуме по градиенту.

---

Тренировка: `python train.py --data-dir <path> [--resume auto]`
Отчёты: `<checkpoint>_report.html` генерируется автоматически.
