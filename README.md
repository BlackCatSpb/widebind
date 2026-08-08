# EVA — Единая Вычислительная Архитектура (WideBind)

**Экспериментальная LLM архитектура на основе самоорганизации и коллективной динамики.**

## Основные принципы

- **Без attention** — вместо него VSA (Vector Symbolic Architecture) + Mirror
- **Без softmax** — factorized Bernoulli/Gaussian head
- **Multi-scale память** — VSA с τ ∈ {8, 32, 128, 512}
- **Коллективный интеллект** — эксперты с shared memory и meta-trust
- **Lambda_d hierarchy** — все гиперпараметры из одного числа (λ₃ ≈ 1.839)

## Архитектура

```
Input tokens
    ↓
[Embedding: sparse codes → sigmoid mixing → basis @ + RoPE]
    ↓
For each of 24 layers:
    ├─ [Pre-LN: RMSNorm]
    ├─ [Conv1d: depthwise 48-tap]
    ├─ [TrajectorySpiralBind: D→K complex rotation + HRR hybrid]
    ├─ [VSA Memory: 4-scale τ={8,32,128,512}]
    ├─ [Mirror: 32 experts, 5 signals, adaptive alpha]
    ├─ [Variable Precision: exact attention when needed]
    ├─ [Output: bind + memory + mirror + collective concept]
    ├─ [Spectral: DCT adaptive]
    ├─ [MLP: 32 groups × (80→320→80) SwiGLU]
    └─ residual add
    ↓
[Final RMSNorm]
    ↓
[SigmoidCodedHead: factorized Bernoulli + normalize]
    ↓
logits
```

## Структура проекта

```
WideBind/
├── core/                    # Основной код архитектуры
│   ├── config.py            # WideBindConfig — все параметры
│   ├── block.py             # WideBindBlock — основной блок
│   ├── stack.py             # WideBindStack — полная модель
│   ├── bind.py              # TrajectorySpiralBind, SpiralBind
│   ├── mirror.py            # GroupedCognitiveMirror
│   ├── embedding.py         # PartitionedEmbedding, SigmoidCodedHead
│   ├── concept_layer.py     # CollectiveConceptLayer (adaptive maturity)
│   ├── reasoning.py         # Variable Precision + Explicit Reasoning
│   ├── mlp.py               # GroupedMLP (SwiGLU)
│   ├── vsa_utils.py         # DCT, sparse codes, prefix scan
│   ├── lambda_utils.py      # Lambda_d hierarchy
│   ├── curriculum.py        # CurriculumTracker
│   ├── live_inference.py    # LiveInference
│   ├── amp_optim.py         # AmpAdam
│   └── archive/             # Старые модули (amp_codec, zeckendorf)
│
├── scripts/                 # Скрипты
│   ├── train.py             # Основной цикл обучения
│   ├── generate.py          # Генерация текста
│   ├── generate_russian.py  # Мониторинг генерации на русском
│   ├── checkpoint_inspector.py  # Анализ чекпоинтов
│   └── archive/             # Старые/тестовые скрипты
│
├── notebooks/               # Jupyter ноутбуки
│   └── colab.ipynb           # Основной ноутбук для Colab
│
├── tests/                   # Тесты
│   ├── test_model.py        # Тесты модели
│   └── test_infer.py        # Тесты генерации
│
├── docs/                    # Документация
│   ├── ARCHITECTURE.md      # Описание архитектуры
│   ├── AMPLITUDE_CODEC.md   # Старый кодек (справка)
│   └── archive/             # Старые документы
│
├── wb/                      # Данные
│   └── russian_tokenizer/   # BPE токенайзер (65k vocab)
│
├── checkpoints/             # Чекпоинты обучения
├── archive/                 # Архивные файлы
└── README.md                # Этот файл
```

## Текущая модель

| Параметр | Значение |
|----------|----------|
| D | 2560 |
| Layers | 24 |
| Experts (G) | 32 |
| bind_K | 32 |
| vocab | 65536 |
| seq_len | 128 |
| Parameters | ~89.7M |
| VRAM (T4) | ~12GB |

## Обучение

```bash
python scripts/train.py --data-dir ./data
```

Или в Colab: запустите `notebooks/colab.ipynb`.

## Генерация

```bash
python scripts/generate.py checkpoints/best.pt --prompt "Привет" --tokens 50
```

## Лицензия

Experimental — используйте на свой страх и риск.
