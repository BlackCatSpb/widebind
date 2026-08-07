import torch
import sys
import io
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from core import WideBindConfig, WideBindStack
from scripts.generate import generate

ckpt = torch.load('checkpoints/best.pt', map_location='cpu', weights_only=False)
cfg = ckpt['cfg']
model = WideBindStack(cfg)
model.load_state_dict(ckpt['model'], strict=False)
model.eval()

print(f'Чекпоинт: step={ckpt["step"]}, val_loss={ckpt.get("best_val_loss", 0):.4f}')
print(f'Параметры: {sum(p.numel() for p in model.parameters()):,}')
print()

prompts = [
    'Привет, как дела?',
    'Москва — столица России',
    'Искусственный интеллект — это',
    'В начале было Слово',
]

for prompt in prompts:
    try:
        text = generate(model, prompt, max_new_tokens=30, temperature=0.7, top_k=40)
        print(f'Промпт: {prompt}')
        print(f'Генерация: {text}')
        print()
    except Exception as e:
        print(f'Промпт: {prompt}')
        print(f'Ошибка: {e}')
        print()

print('=== Итоги генерации ===')
print(f'Mодель: WideBind-89M')
print(f'Шаг обучения: {ckpt["step"]}')
print(f'Validation loss: {ckpt.get("best_val_loss", 0):.4f}')
print(f'Статус: обучение продолжается')
print()
print('Примечание: val_loss=10.34 означает что модель ещё не обучена')
print('Для осмысленной генерации нужно val_loss < 9.0')
print('Прогноз: достижение val_loss=9.0 на шаге ~12000-15000')
