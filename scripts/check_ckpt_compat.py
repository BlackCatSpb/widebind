import torch
import sys
sys.path.insert(0, '.')
from core import WideBindConfig, WideBindStack

ckpt = torch.load('checkpoints/best.pt', map_location='cpu', weights_only=False)
old_cfg = ckpt['cfg']

print('=== Old checkpoint config ===')
print(f'D={old_cfg.D}, L={old_cfg.n_layers}, bind_K={old_cfg.bind_K}')
print(f'vocab={old_cfg.vocab}, mlp_groups={old_cfg.mlp_groups}')
print(f'amp_codec={old_cfg.amp_codec}')
print(f'head_mode: {getattr(old_cfg, "head_mode", "N/A")}')
print(f'bind_twist_mode: {getattr(old_cfg, "bind_twist_mode", "N/A")}')
print(f'private_mem={old_cfg.private_mem}')
print(f'collective_layer={old_cfg.collective_layer}')

cfg = WideBindConfig(
    D=2560, n_layers=24, bind_K=32, vocab=65536, mlp_groups=32,
    mlp_expand=4, seq_len=128, lr=3e-4, max_steps=300000,
    warmup_steps=101, scheduler='mirror', private_mem=True,
    expert_asymmetry=True, meta_trust=True, grad_clip=0.5,
    head_mode='sigmoid_coded', head_normalize=True,
    bind_twist_mode='trajectory_spiral', bind_traj_dims=3,
    hybrid_alpha_max=0.7, hybrid_alpha_min=0.3, w_pred_scale_init=3.0,
    collective_layer=True, collective_uncert_theta=0.5,
    collective_uncert_kappa=3.0, collective_contra_thresh=-0.1,
    collective_contra_gain=6.0, collective_maturity_thresh=0.12,
)
cfg.lambda_d_enabled = False

new_model = WideBindStack(cfg)

old_sd = ckpt['model']
new_sd = new_model.state_dict()

old_keys = set(old_sd.keys())
new_keys = set(new_sd.keys())

only_old = old_keys - new_keys
only_new = new_keys - old_keys
common = old_keys & new_keys

print(f'\n=== State dict comparison ===')
print(f'Old keys: {len(old_keys)}')
print(f'New keys: {len(new_keys)}')
print(f'Common: {len(common)}')
print(f'Only in old: {len(only_old)}')
print(f'Only in new: {len(only_new)}')

print(f'\n=== Only in old checkpoint (will be skipped) ===')
for k in sorted(only_old)[:15]:
    print(f'  {k}: {old_sd[k].shape}')
if len(only_old) > 15:
    print(f'  ... and {len(only_old)-15} more')

print(f'\n=== Only in new model (will be randomly initialized) ===')
for k in sorted(only_new)[:15]:
    print(f'  {k}: {new_sd[k].shape}')
if len(only_new) > 15:
    print(f'  ... and {len(only_new)-15} more')

shape_mismatch = []
for k in common:
    if old_sd[k].shape != new_sd[k].shape:
        shape_mismatch.append((k, old_sd[k].shape, new_sd[k].shape))

print(f'\n=== Shape mismatches in common keys: {len(shape_mismatch)} ===')
for k, old_shape, new_shape in shape_mismatch[:10]:
    print(f'  {k}: old={old_shape} new={new_shape}')

missing, unexpected = new_model.load_state_dict(old_sd, strict=False)
print(f'\n=== Load result ===')
print(f'Missing (not in ckpt): {len(missing)}')
print(f'Unexpected (in ckpt but not model): {len(unexpected)}')
