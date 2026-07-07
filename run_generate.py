"""Quick generation test."""
import sys, torch
sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)
from torch.serialization import add_safe_globals
from config import WideBindConfig
from core import WideBindStack
from generate import load_russian_tokenizer, generate

add_safe_globals([WideBindConfig])

ckpt = torch.load('checkpoints/step_5000.pt', map_location='cpu', weights_only=True)
cfg = ckpt.get('cfg', WideBindConfig())
model = WideBindStack(cfg)
model.load_state_dict(ckpt['model'])
print(f'Loaded step={ckpt.get("step","?")} params={model.param_count():,}')

prompts = [
    'Привет, как дела?',
    'Москва — столица',
]

for p in prompts:
    text = generate(model, p, max_new_tokens=50, temperature=1.2, top_k=50)
    print(f'> {p}')
    print(repr(text))
    print()

# Check raw token IDs for first prompt
tok = load_russian_tokenizer()
encoded = tok.encode('Привет, как дела?')
print(f'Prompt token IDs: {encoded.ids}')
print(f'Prompt tokens: {[tok.id_to_token(i) for i in encoded.ids]}')
