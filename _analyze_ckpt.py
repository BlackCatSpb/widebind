import torch, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from wbconfig import WideBindConfig

ckpt = torch.load('checkpoints/step_10000.pt', map_location='cpu', weights_only=False)
print('=== CHECKPOINT ANALYSIS ===')
print(f'Keys: {list(ckpt.keys())}')
print(f'Step: {ckpt.get("step")}')
print(f'Best val loss: {ckpt.get("best_val_loss")}')

if 'cfg' in ckpt:
    c = ckpt['cfg']
    print(f'\nConfig: D={c.D} layers={c.n_layers} groups={c.mlp_groups} expand={c.mlp_expand}')
    print(f'  seq_len={c.seq_len} lr={c.lr} max_steps={c.max_steps}')
    print(f'  bottleneck={c.bottleneck} bind_K={c.bind_K}')
    print(f'  conv_kernel={c.conv_kernel} grad_clip={c.grad_clip}')
    print(f'  scheduler={c.scheduler} warmup={c.warmup_steps}')

if 'model' in ckpt:
    sd = ckpt['model']
    total = sum(p.numel() for p in sd.values())
    print(f'\nModel: {len(sd)} keys, {total:,} total params')
    
    layer_keys = {}
    for k, v in sd.items():
        layer = k.split('.')[0] if '.' in k else k
        layer_keys.setdefault(layer, []).append(k)
    print(f'Layers found: {sorted(layer_keys.keys())}')
    
    print('\nAll weight shapes:')
    for k, v in sd.items():
        print(f'  {k}: {list(v.shape)}')
    
    # Check for NaN/Inf
    n_nan = sum(torch.isnan(v).sum().item() for v in sd.values() if v.is_floating_point())
    n_inf = sum(torch.isinf(v).sum().item() for v in sd.values() if v.is_floating_point())
    print(f'\nNaN values: {n_nan}, Inf values: {n_inf}')
    
    # Stats per layer
    print('\nWeight stats (mean/std/min/max):')
    for k, v in sd.items():
        if v.is_floating_point():
            print(f'  {k}: mean={v.mean():.6f} std={v.std():.6f} min={v.min():.6f} max={v.max():.6f}')

if 'optimizer' in ckpt:
    opt = ckpt['optimizer']
    print(f'\nOptimizer: state_count={len(opt.get("state",{}))} groups={len(opt.get("param_groups",[]))}')
    if opt.get('param_groups'):
        pg = opt['param_groups'][0]
        print(f'  lr={pg.get("lr")} betas={pg.get("betas")} eps={pg.get("eps")} weight_decay={pg.get("weight_decay")}')
    if opt.get('state'):
        first_key = list(opt['state'].keys())[0]
        first_state = opt['state'][first_key]
        print(f'  First state keys: {list(first_state.keys())}')
        if 'exp_avg' in first_state:
            total_exp = sum(v['exp_avg'].numel() for v in opt['state'].values())
            total_var = sum(v['exp_avg_sq'].numel() for v in opt['state'].values())
            print(f'  Total momentum params: {total_exp:,}')
            print(f'  Total variance params: {total_var:,}')
            # Check if optimizer states are all zeros (fresh)
            first_exp = list(opt['state'].values())[0]['exp_avg']
            print(f'  First momentum has {torch.count_nonzero(first_exp)} / {first_exp.numel()} non-zero')

if 'scheduler' in ckpt:
    sched = ckpt['scheduler']
    print(f'\nScheduler keys: {list(sched.keys())}')
    print(f'  _step={sched.get("_step")}')
    print(f'  _min_lr={sched.get("_min_lr")}')
    print(f'  forced_mult_factor={sched.get("forced_mult_factor")}')
    print(f'  mirror_mult_factor={sched.get("mirror_mult_factor")}')
    if 'log_scale' in sched:
        ls = sched['log_scale']
        print(f'  log_scale: shape={list(ls.shape)} mean={ls.mean():.6f} std={ls.std():.6f} min={ls.min():.6f} max={ls.max():.6f}')
        print(f'  var(log_scale)={ls.var():.8f}')
        print(f'  |mirror|(sum abs)={ls.abs().sum():.4f}')
    if 'bias_d' in sched:
        print(f'  bias_d: mean={sched["bias_d"].mean():.6f} std={sched["bias_d"].std():.6f}')
    if 'bias_i' in sched:
        print(f'  bias_i: mean={sched["bias_i"].mean():.6f} std={sched["bias_i"].std():.6f}')
