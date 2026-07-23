# WideBind для занятых

Языковая модель без transformer-слоёв. Нет attention, нет softmax, нет KV-cache.

## Три идеи

**1. Память — вектор, не матрица.** Один D-мерный вектор на слой, не матрица K×V всех прошлых токенов. ~16 KB на слой — генерация хоть миллион токенов без роста памяти.

**2. Скрещивание размерностей — биллинейность, не weighted sum.** Проекция D→K, покомпонентное произведение u⊙v, проекция K→D.

**3. Мета-познание.** Трёхслойный cognitive mirror: веса (опасные) → private memory (безопасная EMA) → meta-gate (самонастройка). Cross-expert recall через contradiction gate.

## Два варианта

| | Mini | Main |
|---|---|---|
| Параметров | 12.23M | ~161M |
| Групп (G) | 8 | 32 |
| D | 896 | 4096 |
| VRAM | 2.1 GB (MX550) | 11-16 GB (T4) |
| Bottleneck K | 32 | 64 |

- **87.9% параметров** — GroupedMLP (SiLU, expand=4)
- **K=64 bottleneck**, shift mode (golden-ratio twisted)
- **Эмбеддинг + голова**: 8192 параметра (0.01%)
- **VSA scan** — O(L log L), chunked, fp32 guard
- **Private memory** — soft-competition write, Knowledge Graph, 3-слойная meta-reflection
- **MirrorLR** — без cosine decay, loss-damped при росте train/eval loss
- **Чекпоинт**: 159 MB (FCF-CPR 8-bit). **Инференс**: ~0.55 GB VRAM (fp16)
