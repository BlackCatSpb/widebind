# WideBind

Гибридная не-трансформерная языковая модель. Без softmax, без attention, без QKV.

**163M параметров | 24 слоя | D=3584 | K=16 bottleneck bind | Grouped MLP (32×112→896→112) | GroupedCognitiveMirror (32 эксперта) | Partitioned Embed/Head (K=32, S=6)**

---

## Мотивация

Трансформеры доминируют в NLP, но O(L²) attention и неограниченный KV-cache — архитектурные ограничения, а не фичи. WideBind исследует альтернативу:

- **VSA-память** — состояние O(D) на слой (336 KB на все 24 слоя), O(L log L) параллельный prefix scan
- **Bottleneck bind (D→K→D)** — билинейная проекция через K=16. Решает проблему диагонального якобиана чистых element-wise VSA (модели умирали после 4 слоёв)
- **Grouped MLP** — D=896 разбивается на 8 независимых групп, каждая с 8× внутренним expansion (112→896→112). Параметров столько же, сколько в плоском 896→896→896, но каждая группа учится в своём подпространстве
- **Cognitive Mirror** — bounded self-consistency: temp/pred/smooth/sym пути, frequency-adaptive K (lo/hi), predictive self-model (W_pred), gradient-adaptive gate (delta_var EMA). **Grouped** — 32 эксперта, каждый в K=8 подпространстве.
- **DCT Spectral** — learned per-dim частотная маска (lambda_k растёт 0.5→1.5 по слоям)
- **Partitioned Embedding** — sparse block codes (K=32, S=6), 32 сегмента × 112 dims, 1:1 с mirror группами

---

## Архитектура (блок)

```
h → Pre-LN (RMS)
  → Depthwise Conv1d (k=48, groups=D)
  → Bottleneck Bind (D→K=16→bilinear→D)
  → VSA Memory (prefix scan, τ≈150)
  → Cognitive Mirror (3 local + 1 global path)
  → h += bind + mem_read + mirror
  → DCT Spectral (h_dct * lambda_k)
  → Grouped MLP (8 групп × 112→896→112, SiLU)
  → h_out
```

### VSA Memory

```
mem[t] = sigmoid(h·w_d + b_d) · mem[t-1] + sigmoid(h·w_i + b_i) · h[t]
```
τ ≈ 150 (b_d=5.0) — первый токен сохраняется на ~42% после 128 шагов. Ассоциативный параллельный prefix scan.

### Grouped MLP

| G | d | expand | Параметров | Эффект |
|---|---|---|---|---|
| 8 | 112 | 8× | 1,606,528 (93.6% слоя) | 8× expansion per group |

Каждая группа: SiLU(W_down_g · SiLU(W_up_g · h_g)). Mixing между группами — через residual + conv + bind + mirror соседних блоков.

### Cognitive Mirror

Четыре пути в K=16 → rms_norm → tanh(W_out) → exp(log_scale):

1. **Temporal:** h − mem_centroid (ошибка предсказания VSA-памяти)
2. **Global:** h − cross-layer EMA (отклонение от self-model всех слоёв)
3. **Smooth:** h − conv1×3(h) (локальная когерентность)
4. **Symmetry:** (h·w_u) · (h[t-1]·w_v) (билинейная самосогласованность)

tanh гарантирует bounded correction; exp(log_scale) — per-dim амплитуда.

### Partitioned Embedding / LM Head

Sparse block codes (K=32, S=6) — комбинаторная система счисления, ровно 6 из 32 бит на токен. D=3584 разбит на 32 сегмента × 112. Каждый бит — свой basis vector w_k ∈ ℝ¹¹². Gradient grouping: ∂L/∂w_k = 0 при z_k=0. 3584 параметра против 179.2M learned embedding.

---

## Быстрый старт

### Требования
- Python 3.10+, PyTorch 2.0+
- 2 GB VRAM (MX550) или больше
- Токен стримы в `.bin` формате (uint16 numpy array)

### Тренировка
```bash
python train.py --data-dir /path/to/token_streams --save-dir checkpoints
```

Ключевые флаги: `--batch-size 2 --seq-len 128 --n-layers 24 --lr 3e-4 --warmup 1000`

Resume с последнего: `--resume auto`

### Отчёты по чекпоинтам
При каждом сохранении генерируется HTML-отчёт:
```
checkpoints/step_5000_report.html
```

---

## Структура проекта

| Файл | Назначение |
|---|---|
| `core.py` | WideBindBlock, GroupedMLP, CognitiveMirror, VSA prefix scan, эмбеддинги |
| `config.py` | WideBindConfig dataclass (все гиперпараметры) |
| `train.py` | Streaming trainer — AdamW + cosine LR + checkpointing |
| `analyze_checkpoint.py` | Генерация HTML-отчёта из `.pt` чекпоинта |
| `TRAINING_LOG.md` | Живой лог тренировки, сравнения, изменения архитектуры |

---

## Статус обучения

Текущие результаты в [TRAINING_LOG.md](TRAINING_LOG.md). Последнее: step 1000, val_loss=1.99, ppl=7.32.

---

## Известные проблемы

1. **Last-layer collapse** — L23 eff_rank ≈ 12/112 на группу (было 4/896 в плоском MLP — улучшение в 22×, но коллапс остаётся). Структурно: LM head (896→23→50000) агрессивно сжимает на последнем слое.
2. **b_d / b_i frozen at init** — decay (5.0) и write gate bias (-3.0) не двигаются. init — локальный минимум градиента.
3. **Mirror frozen** — log_scale ≈ 0 (exp=1) первые 1000 шагов. Пути зеркала активны, но амплитуда не учится.
4. **Плоский MLP с bottleneck=D мёртв** — rank ≤ D. GroupedMLP решает это внутренним 8× expansion в каждой группе.

---

## Гипотеза специализации слоёв (по lambda_k)

| Слой | lambda_k | Предполагаемая роль |
|---|---|---|
| L0-L4 | 0.50-0.67 | Низкие частоты — интеграция контекста, тема |
| L5-L12 | 0.72-0.98 | Средние частоты — синтаксис, локальные зависимости |
| L13-L18 | 1.02-1.28 | Высокие частоты — лексика, точный выбор слов |
| L19-L22 | 1.33-1.46 | Детализация — прагматика, уточнение |
| L23 | 1.50 | Сжатие в LM head |

lambda_k растёт монотонно с глубиной, что согласуется с иерархической обработкой языка.
