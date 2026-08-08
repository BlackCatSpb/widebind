import torch
import sys
sys.path.insert(0, '.')
from core import WideBindConfig, WideBindStack

ckpt = torch.load('checkpoints/best.pt', map_location='cpu', weights_only=False)
cfg = ckpt['cfg']

print(f"step={ckpt.get('step')}, val_loss={ckpt.get('best_val_loss'):.4f}")
print(f"Config: D={cfg.D}, L={cfg.n_layers}, bind_K={cfg.bind_K}, vocab={cfg.vocab}")
print(f"head_mode={cfg.head_mode}, bind_twist_mode={cfg.bind_twist_mode}")
print(f"collective_layer={cfg.collective_layer}, private_mem={cfg.private_mem}")

# Try loading
model = WideBindStack(cfg)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel params: {n_params:,} ({n_params/1e6:.2f}M)")

# Check state dict compatibility
old_sd = ckpt['model']
new_sd = model.state_dict()

old_keys = set(old_sd.keys())
new_keys = set(new_sd.keys())

only_old = old_keys - new_keys
only_new = new_keys - old_keys
common = old_keys & new_keys

print(f"\nState dict:")
print(f"  Old keys: {len(old_keys)}")
print(f"  New keys: {len(new_keys)}")
print(f"  Common: {len(common)}")
print(f"  Only in old: {len(only_old)}")
print(f"  Only in new: {len(only_new)}")

if only_old:
    print(f"\n  Only in old (sample): {sorted(list(only_old))[:5]}")
if only_new:
    print(f"  Only in new (sample): {sorted(list(only_new))[:5]}")

# Check shape mismatches
mismatch = 0
for k in common:
    if old_sd[k].shape != new_sd[k].shape:
        mismatch += 1
print(f"  Shape mismatches: {mismatch}")

# Try loading
if mismatch == 0:
    missing, unexpected = model.load_state_dict(old_sd, strict=False)
    print(f"\nCheckpoint COMPATIBLE!")
    print(f"  Loaded: {len(old_sd) - len(unexpected)}/{len(old_sd)} keys")
    print(f"  Skipped (old buffers): {len(unexpected)}")
    print(f"  Missing (new): {len(missing)}")
else:
    print(f"\nCheckpoint has {mismatch} shape mismatches, {len(only_new)} new keys")
