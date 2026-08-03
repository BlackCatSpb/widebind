"""Comprehensive checkpoint analyzer for WideBind.
Usage:
  python scripts/analyze_checkpoint.py <path/to/checkpoint.pt>
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from torch.serialization import add_safe_globals
from core import WideBindConfig, WideBindStack
from core.checkpoints import find_latest_ckpt
add_safe_globals([WideBindConfig])

default_ckpt = find_latest_ckpt(os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))
if default_ckpt is None:
    print('No checkpoint found in checkpoints/; pass an explicit path.')
    sys.exit(1)
ckpt_path = sys.argv[1] if len(sys.argv) > 1 else default_ckpt
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
cfg = ckpt['cfg']
sd = ckpt['model']
model = WideBindStack(cfg)
missing, unexpected = model.load_state_dict(sd, strict=False)

print('=' * 72)
print('CHECKPOINT ANALYSIS')
print('=' * 72)
print(f'File:       {ckpt_path}')
print(f'Step:       {ckpt["step"]}')
print(f'Params:     {model.param_count() / 1e6:.2f}M')
print(f'Tensors:    {len(sd)}')
print(f'Missing:    {len(missing)} keys')
print(f'Unexpected: {len(unexpected)} keys')
if missing:
    for k in missing[:5]:
        print(f'  MISS: {k}')
if unexpected:
    for k in unexpected[:5]:
        print(f'  UNEXP: {k}')

print()
print('CONFIG')
print('-' * 72)
important = ['D', 'n_layers', 'mirror_k', 'G', 'k', 'lr', 'weight_decay',
             'conv_kernel', 'vocab', 'seq_len', 'mlp_groups', 'mlp_expand',
             'bind_K', 'ranking_weight', 'balance_weight', 'diversity_weight',
             'div_weight', 'orth_weight', 'nuclear_weight', 'gate_l1_weight',
             'log_scale_l2_weight', 'grad_clip', 'warmup_steps', 'max_steps',
             'max_decay_steps', 'lambda_lr_hierarchy', 'lambda_d',
             'diff_threshold', 'exploration_threshold', 'ema_alpha_min',
             'dtype', 'save_interval', 'log_interval']
for attr in important:
    val = getattr(cfg, attr, 'N/A')
    print(f'  {attr:30s} = {val}')

print()
print('OPTIMIZER PARAM GROUPS (LR multipliers)')
print('-' * 72)
groups = model.param_groups()
for g in groups:
    wd = g.get('weight_decay', 0)
    lr = g.get('lr', cfg.lr)
    n = len(g['params'])
    print(f'  lr={lr:.6f} wd={wd:.4f} n_params={n:4d}')

print()
print('LAYER STATS PER LAYER')
print('-' * 72)
header = f'{"L":>3s} {"alpha":>7s} {"|1-a|":>7s} {"log_scale":>10s} {"ls_std":>7s} {"w_help":>7s} {"skip":>7s} {"scale_w":>7s} {"conv_std":>7s}'
print(header)
print('-' * len(header))
for i, layer in enumerate(model.layers):
    a = layer.mirror.alpha_diag.data
    ls = layer.mirror.log_scale.data
    wh = torch.sigmoid(layer.mirror.w_help.data)
    lsa = torch.sigmoid(layer.mirror.log_skip_alpha.data)
    sw = torch.sigmoid(layer.scale_w.data)
    cw = layer.conv.weight.data
    print(f'{i:3d} {a.mean().item():7.4f} {1-a.mean().item():7.4f} {ls.mean().item():10.4f} {ls.std().item():7.4f} {wh.mean().item():7.4f} {lsa.mean().item():7.4f} {sw.mean().item():7.4f} {cw.std().item():7.4f}')

print()
print('VSA TIMESCALES')
print('-' * 72)
vsa_log = model._vsa_log_param
vsa_tau = torch.exp(torch.cumsum(torch.nn.functional.softplus(vsa_log), dim=0)) + 1.0
print(f'  tau[0]    = {vsa_tau[0].item():.2f}')
print(f'  tau[-1]   = {vsa_tau[-1].item():.2f}')
print(f'  tau ratio = {vsa_tau[-1].item() / vsa_tau[0].item():.1f}x')
print(f'  vsa_log   = softplus mean={torch.nn.functional.softplus(vsa_log).mean().item():.4f}')
td = model._tau_l_dev.data
print(f'  tau_l_dev mean={td.mean().item():.4f} std={td.std().item():.4f}')
b_d = model.b_d.data if hasattr(model, 'b_d') else None
b_i = model.b_i.data if hasattr(model, 'b_i') else None
if b_d is not None:
    b_d_min, b_d_max = b_d.min().item(), b_d.max().item()
    print(f'  b_d[{b_d.numel()}]: mean={b_d.mean().item():.4f} range=[{b_d_min:.4f}, {b_d_max:.4f}]')
if b_i is not None:
    b_i_min, b_i_max = b_i.min().item(), b_i.max().item()
    print(f'  b_i[{b_i.numel()}]: mean={b_i.mean().item():.4f} range=[{b_i_min:.4f}, {b_i_max:.4f}]')

print()
print('W_Q/W_I DYN (per layer)')
print('-' * 72)
for i in range(min(24, len(model.layers))):
    wq = model.layers[i].w_q_dyn.data
    wi = model.layers[i].w_i_dyn.data
    print(f'  L{i:2d} w_q_dyn shape={str(list(wq.shape)):>15s} mean={wq.mean().item():.6f} std={wq.std().item():.6f}')
    print(f'       w_i_dyn shape={str(list(wi.shape)):>15s} mean={wi.mean().item():.6f} std={wi.std().item():.6f}')

print()
print('MIRROR INTERNALS (first 3 layers)')
print('-' * 72)
for i in range(min(3, len(model.layers))):
    mir = model.layers[i].mirror
    print(f'  L{i}:')
    for attr in ['W_proj', 'W_out', 'w_gate', 'b_gate', 'w_b_read', 'b_b_read',
                 'log_scale', 'log_skip_alpha', 'w_help',
                 'alpha_diag', 'tanh_bias']:
        if hasattr(mir, attr):
            t = getattr(mir, attr).data
            info = f'shape={list(t.shape)} mean={t.mean().item():.4f} std={t.std().item():.4f}'
            if t.numel() <= 10:
                info += f' vals={t.flatten().tolist()}'
            elif attr == 'tanh_bias':
                info += f' min={t.min().item():.4f} max={t.max().item():.4f}'
            print(f'    {attr:25s}: {info}')

print()
print('MLP WEIGHTS (first 2 layers)')
print('-' * 72)
for i in range(min(2, len(model.layers))):
    mlp = model.layers[i].mlp
    for attr in ['W_gate', 'W_up', 'W_down', 'norm_w']:
        if hasattr(mlp, attr):
            t = getattr(mlp, attr).data
            print(f'  L{i} mlp.{attr:10s}: shape={str(list(t.shape)):>25s} mean={t.mean().item():.6f} std={t.std().item():.6f}')

print()
print('BIND PARAMS')
print('-' * 72)
bind = model.bind if hasattr(model, 'bind') else None
if bind is not None:
    for name, p in bind.named_parameters():
        print(f'  {name:30s}: shape={list(p.data.shape)} mean={p.data.mean().item():.6f} std={p.data.std().item():.6f}')

print()
print('LM_HEAD')
print('-' * 72)
lm = model.lm_head if hasattr(model, 'lm_head') else None
if lm is not None:
    for name, p in lm.named_parameters():
        print(f'  {name:30s}: shape={list(p.data.shape)} mean={p.data.mean().item():.6f} std={p.data.std().item():.6f}')

print()
print('TENSOR STATS (histogram of parameter values)')
print('-' * 72)
all_vals = torch.cat([p.data.flatten() for p in model.parameters()])
print(f'  Total scalars: {all_vals.numel()}')
print(f'  Global mean:   {all_vals.mean().item():.6f}')
print(f'  Global std:    {all_vals.std().item():.6f}')
print(f'  Global min:    {all_vals.min().item():.6f}')
print(f'  Global max:    {all_vals.max().item():.6f}')
q = torch.tensor([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
try:
    quants = torch.quantile(all_vals, q.to(all_vals.device))
    for qi, qv in zip(q.tolist(), quants.tolist()):
        print(f'  Q{qi*100:3.0f}: {qv:.6f}')
except:
    pass

print()
print(f'NaN in params: {torch.isnan(all_vals).any().item()}')
print(f'Inf in params: {torch.isinf(all_vals).any().item()}')

print()
print('=' * 72)
print('DONE')
