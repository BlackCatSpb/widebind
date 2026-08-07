import torch
import sys
sys.path.insert(0, '.')
from dataclasses import fields

ckpt = torch.load('checkpoints/best.pt', map_location='cpu', weights_only=False)
cfg = ckpt['cfg']

print('=== Checkpoint Config ===')
print(f"step={ckpt.get('step', '?')}")
print(f"best_val_loss={ckpt.get('best_val_loss', '?')}")
print()

for f in fields(cfg):
    v = getattr(cfg, f.name)
    if not f.name.startswith('_'):
        print(f'{f.name}: {v}')
