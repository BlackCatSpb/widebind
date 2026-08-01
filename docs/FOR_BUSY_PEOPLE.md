# WideBind для занятых

Языковая модель без transformer-слоёв. Нет attention, нет softmax, нет KV-cache.

## Три идеи

**1. Память — вектор, не матрица.** Один D-мерный вектор на слой, не матрица K×V всех прошлых токенов. ~16 KB на слой — генерация хоть миллион токенов без роста памяти.

**2. Скрещивание размерностей — биллинейность, не weighted sum.** Проекция D→K, покомпонентное произведение u⊙v (с golden-ratio сдвигами), проекция K→D.

**3. Мета-познание.** Трёхслойный cognitive mirror: веса (опасные) → private memory (безопасная EMA) → meta-gate (самонастройка). Cross-expert recall через contradiction gate (при private_mem).

## Два варианта

| | Mini | Main |
|---|---|---|
| Параметров | ~17.6M | ~255M |
| Групп (G) | 8 | 32 |
| D | 896 | 4096 |
| VRAM | ~2 GB (MX550) | 11–16 GB (T4) |
| Bottleneck K | 32 | 64 |

- **~79% параметров** — GroupedMLP (SwiGLU, expand=4)
- **K bottleneck**, shift mode (golden-ratio twisted, S=4), multi-ocular
- **Эмбеддинг + голова**: basis 4096 + mix 1024 + token_bias 50000 ≈ 55K (<0.05%)
- **VSA scan** — 4 масштаба, learnable τ, chunked prefix scan, fp32 guard
- **RoPE** — позиционное кодирование в эмбеддинге (θ=1e6), 0 параметров
- **Private memory** (опц.) — soft-competition write, Knowledge Graph, 3-слойная meta-reflection
- **MirrorLR** — без cosine decay, counter-cyclical (среднее геометрическое), loss-damped по eval
- **Инференс (fp16)**: ~0.5–0.6 GB VRAM

Подробное описание — в корневом [README.md](../README.md).
