# Обновление на Google (Colab / Сферум / Cloud GPU)

## Что нужно обновить

### 1. Репозиторий

```bash
cd WideBind
git pull
```

### 2. Данные (если не обновлялись)

Проверить наличие файлов:
- `token_stream_DETECT_clean.bin`
- `token_stream_ACTION_clean.bin`

Если отсутствуют — скопировать из backup.

### 3. Зависимости

```bash
pip install -r requirements.txt  # если есть
# Или минимально:
pip install torch numpy tokenizers
```

PyTorch должен быть версии с CUDA, совместимой с GPU (для Сферум — `pip install torch --index-url https://download.pytorch.org/whl/cu118`).

### 4. Запуск

**Ноутбук:**
- Открыть `notebooks/cloud.ipynb` в Jupyter
- Запустить все ячейки последовательно
- При первом запуске ноутбук создаст модель (D=4096, ~293M params), подберёт batch_size и начнёт обучение

**Скрипт:**
```bash
python scripts/cloud_train.py --data-dir . --save-dir checkpoints
```

### 5. Что проверять при обучении

| Метрика | Ожидание | Что если нет |
|---|---|---|
| `|1-alpha|` | >0.01 (растёт от 0.016) | Alpha не учится → нет временной структуры |
| `gate_var` | >0.001 (эксперты дифференцируются) | Gate всегда открыт/закрыт |
| `val_loss` | Монотонное снижение | Проблема с LR или данными |
| `tok/s` | >1000 (T4) / >5000 (A100) | Проверить batch_size, dtype |

### 6. Если NCCL ошибка (Сферум)

`ImportError: libtorch_cuda.so: undefined symbol: ncclCommResume` — несоответствие версий PyTorch и NCCL.

**Workaround:** переустановить torch без NCCL:
```bash
pip install --force-reinstall --no-deps torch==2.1.0
```
или использовать `CUDA_VISIBLE_DEVICES=-1` для CPU (медленно, но работает).

### 7. Изменения в конфиге (defaults)

| Параметр | Старое | Новое | Причина |
|---|---|---|---|
| `tie_bind` | False | True | Автоэнкодер bind, −262K params |
| `tie_mirror_proj` | False | True | Автоэнкодер mirror, −295K params |
| `lambda_d_enabled` | False | True | λ₃-иерархия для всех констант |
| `alpha init` | 0.99 | 0.98 | +60% temporal gradient |
| `bind_K` | 32 | 64 | Ближе к теорет. оптимуму K≈100 |

Новые изменения:

| Параметр | Описание |
|---|---|
| `Bidirectional MirrorLR` | LR растёт при var(ls)/\|1-alpha\|/gate_var↑, падает при сходимости. Нет forced cosine. |
| `Alpha warmup override` | На время warmup alpha=0.5 (prинудительное обучение экспертов), после warmup отпускается |
| `scheduler:` | Убраны `lr_min_ratio`, `max_decay_steps`, `var_min_for_lr_decay` — больше не нужны |

Эти изменения активны при создании `WideBindConfig()` без явного override.

### 8. Resume

Ноутбук автоматически подхватывает **последний** чекпоинт (по номеру шага в имени файла): `step_N.pt` с максимальным N. Если такого нет — `best.pt` (содержит шаг сохранения как `step`).

При добавлении новых buffer'ов (`_alpha_override`) в model.py, resume работает через `strict=False`.

### 9. Ожидания после перезапуска

| Метрика | Что будет |
|---|---|
| `|1-alpha|` | За время warmup вырастет до 0.5 (forced). После отпускания — равновесие ~0.05-0.08 |
| `var(ls)` | Начнёт расти (специализация экспертов) |
| `gate_var` | Пик 0.03-0.07 (дифференциация), затем спад |
| `LR` | Плавающий 0.1× — 3.0× от base_lr |
| `loss` | Может быть выше первое время (alpha далеко от init), но быстро догонит |
