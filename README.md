# WideBind

**161M параметров, 32 слоя, D=4096.** Без единого transformer-слоя. Ни softmax, ни attention, ни QKV-проекций.

## Архитектура

```
token IDs → PartitionedEmbedding (32×128, sparse 6/32) → [WideBindBlock × 32] → Final RMS Norm → PartitionedHead → logits
```

Каждый блок:

```
h → RMSNorm → Conv1d(groups=D, k=48) → BottleneckBind(D→K=64→D) → VSA Memory (chunked scan) → GroupedCognitiveMirror (32 эксперта) → DCT Spectral → GroupedMLP (32 группы, ×4)
```

**BottleneckBind** — скрещивание размерностей через K=64 с Фибоначчи-твистом. Три режима:
- `off` — `(h·w_u) ⊙ (h·w_v) @ W_out`, классическая регрессия
- `shift` — сумма S штук shifted bilinear произведений (golden-ratio roll)
- `cascade` — Фибоначчи-вложенные моночлены: `a[n] = cross(a[n-1], a[n-2])`

tie_bind: W_out = W_proj^T (автоэнкодер).

**VSA Memory** — векторная суперпозиция с chunked prefix scan (CHUNK=32), surprisal-gated i_gate, dual readout + first moment. fp32 guard для численной стабильности. τ per-channel до ~163K (b_d=12.0).

**GroupedCognitiveMirror** — 32 эксперта, каждый в своём d=128, с 4 EMA-нормированными сигналами (temp/pred/smooth/sym), learnable softmax-весами, K-space gate. α — скаляр per expert, не W_pred.

**GroupedMLP** — 32 группы × (128→512→128, SiLU). 87.9% параметров.

**PartitionedEmbedding/Head** — 32 сегмента × 128, sparse 6-out-of-32 коды (combinadic). Никакого cross-talk между сегментами. 8192 параметра на весь эмбеддинг + голову.

**AdaptiveController** — 5 гиперпараметров VSA-памяти (b_d, b_i, mem2v_scale, EMA α, noise_scale) управляются двумя сигналами из Mirror: exploration и differentiation.

**MirrorLRScheduler** — LR растёт с var(log_scale), |1-α|, gate_var. Без forced cosine decay.

## Параметры (tied)

| Компонент | Параметров |
|---|---|
| Embed + LM Head | 8,192 |
| Bind (K=64) | 262K |
| GroupedCognitiveMirror | 714K |
| Conv1d | 6.3M |
| DCT Spectral | 131K |
| VSA gates | 1.2M |
| GroupedMLP | 134.3M |
| **Total** | **~161M** |

## Тренировка

- Данные: 3 потока ADVENTUR/DRAMA/FANTASY, ~6.3B токенов, uint16 memmap
- AdamW (0.9, 0.95), weight_decay=0.01, LR=3e-4
- Gradient accumulation (удвоение effective batch)
- Soft EOS reset: state *= 0.3 при EOS
- Chunked VSA scan (CHUNK=32) — никаких NaN
- Чекпоинты: step_{N}.pt + eval_{N}.pt на каждом eval
- CTRL+C → сохраняет step_{N}.pt для resume
- HTML-отчёты: `scripts/analyze_checkpoint.py`

## Mасштабирование

`D = G·128`, `L = G`, `bind_K = 64`, `k = 32`. Все размеры выведены из числа групп G.

## Структура

```
core/            — config, model, live_inference
compression/     — FCF-CPR (8-bit, ~3.8× сжатие)
scripts/         — train, colab_train, analyze_checkpoint, generate
tests/           — 59 тестов
docs/            — документация
```
