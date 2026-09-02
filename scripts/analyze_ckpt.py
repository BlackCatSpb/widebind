import sys; sys.path.insert(0, '.')
import torch, math
from core.config import WideBindConfig

ckpt = torch.load(r'C:\Users\black\OneDrive\Desktop\best.pt', map_location='cpu', weights_only=False)
sd = ckpt['model']

cfg = ckpt.get('cfg')
print('Config from checkpoint:')
print(f'  D={cfg.D}, n_layers={cfg.n_layers}, vocab={cfg.vocab}')
print(f'  memory_bank={cfg.memory_bank}')
print(f'  bridge_conn={cfg.bridge_conn}, bridge_dim={cfg.bridge_dim}')
print(f'  maturation_enabled={cfg.maturation_enabled}')

# Check all params for NaN/Inf
nan_count = 0
inf_count = 0
for k, v in sd.items():
    if v.isnan().any():
        nan_count += 1
        print(f'  NaN: {k}')
    if v.isinf().any():
        inf_count += 1
        print(f'  Inf: {k}')
print(f'\nTotal NaN: {nan_count}, Inf: {inf_count}')

# Check expert gate norms per layer
print('\nExpert gate norms per layer:')
for i in range(24):
    k = f'layers.{i}.mirror.W_gate'
    if k in sd:
        w = sd[k]
        print(f'  layer {i}: norm={w.norm():.4f} std={w.std():.6f}')

# Check maturation
log_tau = sd.get('maturation._log_tau')
if log_tau is not None:
    print(f'\nmaturation._log_tau: {log_tau}')
tau_l = sd.get('tau_l_dev')
if tau_l is not None:
    print(f'tau_l_dev: {tau_l}')

print(f'active_depth: {ckpt.get("active_depth")}')

# Check that all non-memory-bank params match a fresh model
print('\n--- Comparing with fresh model ---')
fresh_cfg = WideBindConfig(D=cfg.D, n_layers=cfg.n_layers, vocab=cfg.vocab,
    mlp_groups=cfg.mlp_groups, bind_K=cfg.bind_K, bridge_conn=cfg.bridge_conn,
    bridge_dim=cfg.bridge_dim, maturation_enabled=cfg.maturation_enabled,
    memory_bank=False)
from core.stack import WideBindStack
fresh = WideBindStack(fresh_cfg)
fresh_sd = fresh.state_dict()

# Compare shapes
mismatch = 0
for k in fresh_sd:
    if k in sd:
        if fresh_sd[k].shape != sd[k].shape:
            print(f'  SHAPE MISMATCH: {k}: fresh={fresh_sd[k].shape} ckpt={sd[k].shape}')
            mismatch += 1
    else:
        print(f'  MISSING in ckpt: {k}')
        mismatch += 1
for k in sd:
    if k not in fresh_sd and 'memory_bank' not in k:
        print(f'  EXTRA in ckpt (non-MB): {k}')
        mismatch += 1
print(f'Total mismatches: {mismatch}')
