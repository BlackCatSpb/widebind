# WideBind

**Не-трансформерная языковая модель** на когнитивных зеркалах, VSA-памяти и bottleneck bind. Никакого softmax, attention или QKV.

Архитектура вдохновлена тем, как мозг обрабатывает последовательности: **векторная суперпозиция** вместо attention, **биллинейный баттлнек** вместо multi-head projection, **когнитивное зеркало** для самокоррекции.

---

## Быстрый старт

### Инференс (локально, MX550 2GB)

```bash
python scripts/gen_demo.py
```

### Тренировка (Colab T4 16GB)

1. Загрузить `core/`, `compression/`, `scripts/`, `notebooks/` на Google Drive в папку `widebind_data/src/`
2. Открыть `notebooks/colab.ipynb` → Run All

### Анализ чекпоинта

```bash
python scripts/analyze_checkpoint.py checkpoints/step_N.pt
# → step_N_report.html
```

---

## Чем WideBind не является

| Это НЕ | Потому что |
|---|---|
| Transformer | Нет attention, нет QKV, нет softmax |
| Pure VSA / hyperdimensional computing | Есть bottleneck bind (D→K→D) — даёт скрещивание размерностей |
| SSM / State Space Model | VSA memory — это не линейная рекуррентность, а prefix scan с сигмоидными гейтами |
| RNN | Prefix scan параллелизуется за O(L log L), не O(L) последовательно |
| MoE | Все 32 группы MLP всегда активны |

---

## Архитектура за 30 секунд

```
tokens → Zeckendorf code (K=23) → Linear(K→D=3584) → [Block × 24] → LM Head
```

**Один блок:**

```
h → RMSNorm → Conv1d depthwise → Bind (D→K=16→D) → VSA Memory → Cognitive Mirror → DCT Spectral → GroupedMLP → h'
```

Компоненты:

- **Bind** — проекция в K=16, биллинейная склейка, проекция обратно. Единственный механизм скрещивания размерностей.
- **VSA Memory** — prefix scan: `mem[t] = sigmoid(decay) · mem[t-1] + sigmoid(i_gate) · h[t]`. Векторная суперпозиция, не матрица ковариации.
- **Cognitive Mirror** — 4 пути самоконтроля в K-space: temporal (отклонение от памяти), global (отклонение от предыдущих слоёв), smoothness (предсказуемость соседями), symmetry (биллинейная согласованность).
- **DCT Spectral** — весовое масштабирование частот DCT-II. L0: все частоты ×0.5, L23: все частоты ×1.5.
- **GroupedMLP** — D=3584 разбит на 32 группы по 112, каждая с 8× внутренним расширением. Имитация большого MLP без роста параметров.

Параметры: **165M** (D=3584, 24 слоя).

---

## Ключевые идеи (для специалиста)

### Почему не attention?

Attention — O(L²). Prefix scan — O(L log L). VSA memory — O(D) состояния на слой, независимо от длины последовательности.

### Почему K=16 работает?

Bind — это `h·W_proj → u·v → W_out`. Матрица преобразования M = W_proj · diag(u) · diag(v) · W_out имеет **ранг K**, но размер D×D. Градиент течёт через все D×K + K×D путей — никакого диагонального затухания, как в pure VSA.

При K=16 и D=3584: grad/param > 0.4 на ините. При K=1: grad/param ≈ 0. При K=896 (полная проекция): grad/param ≈ 0.9, но 1.6M параметров на слой вместо 28K.

### Cognitive Mirror — что это?

Четыре сигнала рассогласования в K-space:

1. **Temporal:** `h[t] - mean(mem)` — текущий вход не похож на накопленную память
2. **Global:** `h[t] - global_state` — текущий вход не похож на агрегат предыдущих слоёв
3. **Smoothness:** `h[t] - conv1x3(h[t])` — не-гладкий переход
4. **Symmetry:** `(h[t]·u) · (h[t-1]·v)` — биллинейное рассогласование соседних шагов

Сумма → rms_norm → tanh(W_out) → **exp(log_scale)**. tanh гарантирует коррекцию в [-1, 1]; log_scale — пер-дим амплитуда.

### AdaptiveController — автоподстройка

Никаких ручных гиперпараметров для гейтов памяти. Два сигнала:

- **exploration** = `mean(|mirror|) / 0.3` — зеркало активно → короткая память, агрессивная запись
- **differentiation** = `var(log_scale) / 0.1` — зеркало специализировалось → меньше памяти, стабильнее EMA

Всё считается в `.forward()` одной строкой: `layer.b_i.fill_(b_i_val)`.

### FCF-CPR сжатие

Uniform 8-bit per tensor с per-tensor min/max. Удаляет детерминированные буферы (V_dct, Zeckendorf codes). 3.22 GB → 1.48 GB (с оптимизатором), 165 MB (model-only). MSE 1.7e-5.

---

## Проект

```
WideBind/
├── core/                    # Модель
│   ├── config.py            # WideBindConfig — все гиперпараметры
│   └── model.py             # WideBindStack, блоки, зеркало, MLP, адаптивный контроллер
├── compression/             # FCF-CPR сжатие чекпоинтов
│   └── fcf_cpr.py           # save_compressed / load_compressed
├── scripts/                 # Всё, что можно запустить
│   ├── train.py             # Локальная тренировка (CPU/MX550)
│   ├── colab_train.py       # Тренировка на Colab (T4, fp16, auto-batch)
│   ├── analyze_checkpoint.py # HTML-отчёт по чекпоинту
│   ├── generate.py          # Генерация текста (с токенизатором)
│   ├── gen_demo.py          # Демо генерации из сжатого чекпоинта
│   └── run_generate.py      # Быстрый тест генерации
├── tests/
│   └── test_infer.py        # Бенчмарк инференса (VRAM, tok/s)
├── notebooks/
│   └── colab.ipynb          # Colab для D=3584 на T4
├── docs/
│   ├── ARCHITECTURE.md      # Полное описание архитектуры
│   └── TRAINING_LOG.md      # Логи тренировок
├── config.py / wbconfig.py  # Шимы для старых чекпоинтов
├── checkpoints/             # Веса (gitignored)
└── requirements.txt
```

---

## Статус

- **Архитектура:** финальная (CognitiveMirror + GroupedMLP + AdaptiveController)
- **Сжатие:** FCF-CPR (8-bit uniform, 11.5× без потерь качества)
- **Тренировка:** D=896 (41M) — **val_loss 2.27, ppl 9.67** | D=3584 (165M) — **step 15000**, train_loss 2.7-3.0
- **Инференс:** fp16, 0.98 GB VRAM, 15 tok/s (MX550)
- **Токенизатор:** BPE, vocab=50000, русский

---

## Производительность

| Конфиг | Параметры | VRAM (fp16) | tok/s (MX550) | tok/s (T4) |
|--------|-----------|-------------|---------------|------------|
| D=896, 24L | 41M | 0.5 GB | 25 | 420 |
| D=3584, 24L | 165M | 0.98 GB | 15 | 214 |

---

## Структура параметров на слой (D=3584, G=32, expand=8×)

| Компонент | Параметров | Доля |
|---|---|---|
| GroupedMLP | 6,422,528 | 93.5% |
| Bottleneck Bind | 172,048 | 2.5% |
| Cognitive Mirror | 172,049 | 2.5% |
| VSA Memory (гейты + момент) | 50,176 | 0.7% |
| Depthwise Conv1d (k=48) | 172,032 | 2.5% |
| Spectral lambda_k | 3,584 | 0.05% |
| **Итого на слой** | **6,992,417** | 100% |
| Embedding + LM Head | 165,616 | 0.1% |
| **Total** | **165,313,024** | 165M |
