# WideBind — Дневник архитектуры

> Контекст между сессиями. Ключевые решения, вехи, наблюдения.

## Текущее состояние (последнее обновление)

- **Step:** 11275 (Colab запущен)
- **Val:** 9.194@10019 (лучший)
- **Голова:** HybridSigmoidSoftmaxHead (sigmoid * (1 + softmax), tau=exp(log_temp))
- **Per-layer maturation:** L0=0.235, L23=0.691
- **bridge_conn:** 0.016 (стабильный)
- **intent_w:** 1.07
- **CE:** 9.076 (снижается)

## Ключевые решения

### 1. Per-layer maturation (коммит a735ac0)
- **Проблема:** bridge_conn collapse на uniform maturation
- **Решение:** deep layers (tau≈515) открываются быстрее (T_eff=8000), shallow (tau≈8) медленнее (T_eff=16000)
- **Формула:** `gate_l = sigmoid((t - (T0 + alpha*(1-tau_norm_l)*T_delay)) / delta_t)`
- **Результат:** bridge_conn стабилен 0.12→0.049 (нет коллапса)

### 2. SpectrumGate в bridge routing (коммит 9ad91f9)
- **Формула:** `gate = sigmoid(logits) * (1 + softmax(logits/tau))`
- **Tau:** из maturation — `effective_tau = tau_max*(1-mat) + tau_min*mat`
- **Свойство:** sigmoid сохраняет базу, softmax добавляет emphasis

### 3. HybridSigmoidSoftmaxHead (коммит 5826661)
- **Проблема:** LM head чисто sigmoid → entropy 10.5, модель "гадает"
- **Решение:** перенести SpectrumGate в `_su()` метод головы
- **Формула:** `gate = sigmoid(zt) * (1 + softmax(zt/tau))`, tau = exp(log_temp)
- **Результат:** entropy 10.5→0.58, модель фокусируется
- **Стоимость:** 0 новых параметров, hot-swap, совместим с чекпоинтами

### 4. Tau без магических чисел
- Tau всегда берётся из модели:
  - bridge: `effective_tau = tau_max*(1-mat) + tau_min*mat`
  - LM head: `tau = exp(log_temp)`
  - SpectrumGate: `tau = exp(log_tau)` (learnable)
- Принцип: **никаких внешних гиперпараметров для tau**

## Философия архитектуры

- **Сигмоид** = анархия (все равны, нет фокуса)
- **Софтмакс** = диктатура (один победил, остальные мертвы)
- **Гибрид** = демократия с лидером (все участвуют, лучший ведёт)
- **Per-layer maturation** = эволюция (глубокие成熟 быстрее, shallow сохраняют gradient)
- **Bridge** = кросс-слойная коммуникация (не residual, а модуляция)

## Цели

| Цель | Статус |
|---|---|
| val < 9.0 | ✅ Достигнуто (8.77@15844, old run) |
| val < 8.5 | ⏳ Текущая цель |
| val < 8.0 | 🎯 Долгосрочная |
| Bridge collapse solved | ✅ Per-layer maturation |
| Hybrid head | ✅ Коммит 5826661 |
| Генерация связного текста | ⏳ Ожидается при val < 8.5 |

## Чекпоинты

| Файл | Step | Val | Описание |
|---|---|---|---|
| best.pt | 10019 | 9.194 | Per-layer maturation + hybrid head |
| best 5.pt | 10019 | 9.194 | Тот же |
| best_15844_report.html | 15844 | 8.7692 | Old run (uniform maturation) |

## Наблюдения

- При val ~9.2 модель ещё не генерирует связный текст (нормально)
- Гибридная голова снижает entropy на порядок → должна ускорить обучение
- Per-layer maturation + SpectrumGate = стабильная кросс-слойная коммуникация
- DepthController выбирает глубину динамически (12-24 слоя)
- Два disruption (step 8388, 9087) — нормальный цикл перестройки

## TODO

- [ ] Мониторить val после перезапуска с hybrid head
- [ ] Сравнить скорость схождения: hybrid vs original head
- [ ] Обновить README §18 после достижения val < 8.5
- [ ] Fisher-Rao integration (deferred)
