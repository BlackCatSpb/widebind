import torch
import sys
sys.path.insert(0, '.')
from core import WideBindConfig, WideBindStack

ckpt = torch.load('checkpoints/best.pt', map_location='cpu', weights_only=False)
cfg = ckpt['cfg']

model = WideBindStack(cfg)
missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
print(f"Loaded: {len(ckpt['model']) - len(unexpected)}/{len(ckpt['model'])} keys")
print(f"Skipped: {len(unexpected)} old buffers")
print(f"Missing: {len(missing)} new keys")
print(f"Step: {ckpt['step']}, val_loss: {ckpt['best_val_loss']:.4f}")

# Quick forward (no backward for speed)
x = torch.randint(0, 65536, (1, 32))
h = model.embed_tokens(x)
with torch.no_grad():
    out, state, gs = model(h, step=1631)
print(f"Forward OK: out.shape={out.shape}, NaN={torch.isnan(out).any().item()}")
print("RESUME READY")
