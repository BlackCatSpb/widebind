# WideBind

Гибридная LM без transformer-слоёв: ни softmax, ни attention, ни QKV-проекций.
Два варианта на одном коде:

| | Mini | Main |
|---|---|---|
| D | 896 | 4096 |
| G (групп) | 8 | 32 |
| Параметров | 12.23M | ~161M |
| VRAM/токен | 2.1 GB | 11-16 GB |
| Устройство | MX550 | T4/A100 |
| Accum | 8 | 8 |

## Архитектура

```
token_ids → PartitionedEmbedding → [WideBindBlock × G] → RMS Norm → PartitionedHead → logits
```

Каждый блок:

```
h → RMSNorm → Conv1d → BottleneckBind(D→K↔D) → VSA Memory → GroupedCognitiveMirror → DCT Spectral → GroupedMLP
```

**BottleneckBind** — скрещивание размерностей через K=32/64. Три режима:
- `off` — `(h·w_u) ⊙ (h·w_v) @ W_out`
- `shift` — сумма S shifted bilinear произведений (golden-ratio roll, multi-ocular)
- `cascade` — Фибоначчи-вложенные моночлены

**VSA Memory** — векторная суперпозиция с chunked prefix scan, fp32 guard, τ per-channel до ~160K.

**GroupedCognitiveMirror** — G экспертов, 3-слойная метакогнитивная архитектура:
- **L0** (weights): 5 сигналов коррекции (temp/pred/smooth/sym/help), learnable softmax-веса
- **L1** (private memory): EMA уверенных K-space состояний, cross-expert recall через attention, contradiction gate, expert Knowledge Graph
- **L2** (meta-gate): α-самонастройка, gradient-adaptive gate, MirrorLR scheduler

Подробно: `docs/ARCHITECTURE.md`

## Параметры (main, D=4096, G=32, tied)

| Компонент | Параметров | % |
|---|---|---|
| Embed + LM Head | 8,192 | 0.01 |
| Bind (K=64) | 262K | 0.17 |
| GroupedCognitiveMirror | 714K | 0.47 |
| Private Memory | 1K | <0.01 |
| Conv1d | 6.3M | 4.12 |
| VSA gates | 1.2M | 0.77 |
| GroupedMLP | 134.3M | 87.99 |
| **Total** | **~161M** | **100** |

## Тренировка

- Данные: 3 потока (ADVENTUR/DRAMA/FANTASY), ~6.3B токенов
- AdamW (0.9, 0.95), LR=3e-4, accum=8 (effective batch = B×L×accum)
- Gradient clipping 0.5, FP32 (без AMP)
- MirrorLRScheduler: var(log_scale) + gate_var + |1-α| + training-loss trend damping
- log_scale L2 (ls > 2.3) + signal entropy + gate L1 + diversity + nuclear + orth
- Private memory writes: delayed 5000 forward steps, EMA decay, soft-competition T=0.5
- HTML-отчёты: `scripts/analyze_checkpoint.py`

## Варианты

```
# Mini (MX550, 2.1 GB)
D=896, G=8,  bind_K=32, k=32, accum=8, private_mem=True
11.20M params (default) / 12.23M (multi-ocular S=4), 0.10 GB VRAM

# Main (T4/Colab, 16 GB)
D=4096, G=32, bind_K=64, k=32, accum=8, private_mem=True
~161M params, ~11.5 GB VRAM
```

## Структура

```
core/            — config, model
compression/     — FCF-CPR (8-bit, ~3.8×)
scripts/         — train, colab_train, analyze, generate
notebooks/       — colab.ipynb
tests/           — unit tests
docs/            — архитектура, обзоры, логи
```

Генерация с метакогнитивным readout: `--show-mind`, `--context-mem`, `--continuous-learn`
