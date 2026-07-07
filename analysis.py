"""Deep analysis of WideBind architecture."""

import torch, os, glob, math
import torch.nn as nn
import torch.nn.functional as F
from torch.serialization import add_safe_globals
from config import WideBindConfig
from core import WideBindStack
add_safe_globals([WideBindConfig])

# --- Load ---
save_dir = 'checkpoints'
ckpts = sorted(glob.glob(os.path.join(save_dir, 'step_*.pt')))
if not ckpts:
    ckpts = sorted(glob.glob(os.path.join(save_dir, 'best.pt')))
latest = ckpts[-1] if ckpts else None

if latest:
    ckpt = torch.load(latest, map_location='cpu', weights_only=True)
    cfg = ckpt['cfg']
    print(f'Loaded: {latest}')
    print(f'Step: {ckpt["step"]}')
else:
    cfg = WideBindConfig()
    ckpt = None
    print('No checkpoint found')

# --- Model ---
model = WideBindStack(cfg)
model.eval()
if ckpt:
    model.load_state_dict(ckpt['model'], strict=False)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'\n=== Parameter Count ===')
print(f'Total: {total:,} ({total/1e6:.2f}M)')
print(f'Trainable: {trainable:,} ({trainable/1e6:.2f}M)')

# Per-module breakdown: only leaf modules
for name, mod in model.named_modules():
    if not name:
        continue
    if any(isinstance(m, nn.Module) for m in mod.children()):
        continue
    n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    if n > 0:
        print(f'  {name:50s} {n:>8,} params')

# --- Gradient stats from optimizer ---
if ckpt and 'optimizer' in ckpt:
    opt = ckpt['optimizer']
    if 'state' in opt:
        g_means, g_vars = [], []
        for pid, st in opt['state'].items():
            if 'exp_avg' in st:
                g = st['exp_avg']
                g_means.append(g.abs().mean().item())
                g_vars.append((g**2).mean().item())
        print(f'\n=== Gradient Stats (from AdamW exp_avg) ===')
        print(f'  Mean |grad|: {sum(g_means)/len(g_means):.6f}')
        print(f'  Mean grad^2: {sum(g_vars)/len(g_vars):.6f}')
        print(f'  RMS grad:    {math.sqrt(sum(g_vars)/len(g_vars)):.6f}')

# --- Weight distribution ---
all_w = torch.cat([p.data.flatten() for p in model.parameters()])
print(f'\n=== Global Weight Distribution ===')
print(f'  Mean: {all_w.mean():.4f}')
print(f'  Std:  {all_w.std():.4f}')
print(f'  Min:  {all_w.min():.4f}')
print(f'  Max:  {all_w.max():.4f}')

# Per-parameter norm
norms = [(name, p.data.norm().item(), p.data.std().item() if p.numel() > 1 else 0.0, p.numel())
         for name, p in model.named_parameters()]
norms.sort(key=lambda x: -x[1])
print(f'\n=== Top-20 params by ||W|| ===')
for name, nrm, std, sz in norms[:20]:
    print(f'  {name:55s} ||W||={nrm:8.1f}  std={std:6.4f}  size={sz:>6}')

print(f'\n=== Bottom-20 params by ||W|| (non-zero) ===')
for name, nrm, std, sz in [x for x in norms if x[1] > 0][-20:]:
    print(f'  {name:55s} ||W||={nrm:8.1f}  std={std:6.4f}  size={sz:>6}')

# --- Embedding analysis ---
print(f'\n=== Token Embedding (wte) ===')
wte = model.embed.proj.weight.data
print(f'  Shape: {list(wte.shape)}')
print(f'  Mean={wte.mean():.4f}  Std={wte.std():.4f}  Min={wte.min():.4f}  Max={wte.max():.4f}')
print(f'  Row norms: mean={wte.norm(dim=1).mean():.4f}  std={wte.norm(dim=1).std():.4f}')

print(f'\n=== Zeckendorf Codes ===')
codes = model.embed.codes
print(f'  Shape: {list(codes.shape)}')
print(f'  Non-zero per row: mean={codes.sum(dim=1).float().mean():.2f}')
print(f'  K (code length): {codes.shape[1]}')

print(f'\n=== LM Head ===')
lm = model.lm_head.proj.weight.data
print(f'  Shape: {list(lm.shape)}')
print(f'  Mean={lm.mean():.4f}  Std={lm.std():.4f}  Min={lm.min():.4f}  Max={lm.max():.4f}')
logit_W = codes.float() @ lm.float()  # (vocab, D)
print(f'  Combined proj (codes @ lm.T): shape={list(logit_W.shape)}, '
      f'std={logit_W.std():.4f}, mean={logit_W.mean():.4f}')

# --- Layer analysis ---
print(f'\n=== Per-Layer Summary (first 3 + last 2) ===')
for i in [0, 1, 2, 22, 23]:
    layer = model.layers[i]
    Wp = layer.W_proj.data
    wi = layer.w_i.data
    wd = layer.w_d.data
    wq = layer.w_q.data
    wmv = layer.w_mem2v.data
    up = layer.mlp_up.weight.data
    down = layer.mlp_down.weight.data
    
    _, s_Wp, _ = torch.svd(Wp)
    eff_rank_p = (s_Wp**2).sum() / s_Wp.max()**2
    up_s = torch.linalg.svdvals(up.float())
    eff_rank_up = (up_s**2).sum() / up_s.max()**2
    
    print(f'  L{i:2d}: ||Wp||={Wp.norm():5.1f} eff_r(Wp)={eff_rank_p:.2f} '
          f'std(wi)={wi.std():.4f} std(wd)={wd.std():.4f} '
          f'std(wq)={wq.std():.4f} std(wmv)={wmv.std():.4f} '
          f'||up||={up.norm():5.1f} eff_r(up)={eff_rank_up:.2f}')

# --- Capacity analysis ---
print(f'\n=== Capacity Analysis ===')
print(f'D={cfg.D}, K={cfg.bind_K}, bottleneck={cfg.bottleneck}')
print(f'Bind bilinear: D->K ({cfg.D}*{cfg.bind_K}={cfg.D*cfg.bind_K}) '
      f'+ K->D ({cfg.bind_K}*{cfg.D}) = {2*cfg.D*cfg.bind_K:,} params')
print(f'  Effective rank bound: K={cfg.bind_K}')
print(f'  Off-diagonal mixing: K*K = {cfg.bind_K**2} entries in bilinear step')
print(f'MLP: D->bottleneck ({cfg.D}*{cfg.bottleneck}={cfg.D*cfg.bottleneck:,}) '
      f'+ bottleneck->D = {2*cfg.D*cfg.bottleneck:,} params')
print(f'  Expansion ratio: bottleneck/D = {cfg.bottleneck/cfg.D:.1f}x')
print(f'VSA memory: O(D) state per layer ({cfg.D} dims)')
print(f'  Total state: {cfg.n_layers*cfg.D:,} scalars ('
      f'{cfg.n_layers*cfg.D*4/1024:.1f} KB)')

# --- Initialization sanity ---
print(f'\n=== Init Sanity (vs. Xavier Glorot) ===')
xavier_std = math.sqrt(2.0 / (cfg.D + cfg.bottleneck))
actual_mlp_std = model.layers[0].mlp_up.weight.data.std().item()
print(f'  MLP up target std (xavier): {xavier_std:.4f}')
print(f'  MLP up actual std:          {actual_mlp_std:.4f}')
print(f'  Ratio: {actual_mlp_std/xavier_std:.2f}x')

# Conv kernel
conv = model.layers[0].conv
print(f'\n=== Conv1d ===')
print(f'  kernel_size={conv.kernel_size[0]}  groups={conv.groups}  in_ch={conv.in_channels}')
print(f'  Weight shape: {list(conv.weight.shape)}')
print(f'  Weight std: {conv.weight.data.std():.4f}')

# DCT basis
print(f'\n=== DCT Basis ===')
V = model.layers[0].V_dct
print(f'  Shape: {list(V.shape)}')
print(f'  Orthogonality: ||V V^T - I|| = {(V @ V.T - torch.eye(cfg.D)).norm():.4f}')

# Lambda_k
lam = model.layers[0].lambda_k.data
print(f'  lambda_k: std={lam.std():.4f}  mean={lam.mean():.4f}')
print(f'  range: [{lam.min():.4f}, {lam.max():.4f}]')

# Mirror
print(f'\n=== Mirror Bind ===')
ms = model.layers[0].mirror_scale.data
print(f'  mirror_scale: {ms.item():.4f}')

# --- Memory gate statistics ---
print(f'\n=== Memory Gates (avg across layers) ===')
wi_means, wd_means, wq_means = [], [], []
for layer in model.layers:
    wi_means.append(layer.w_i.data.mean().item())
    wd_means.append(layer.w_d.data.mean().item())
    wq_means.append(layer.w_q.data.mean().item())
print(f'  w_i mean: {sum(wi_means)/len(wi_means):.4f}')
print(f'  w_d mean: {sum(wd_means)/len(wd_means):.4f}')
print(f'  w_q mean: {sum(wq_means)/len(wq_means):.4f}')

print(f'\n=== Forward Simulation (B=1, L=16) ===')
torch.manual_seed(42)
x = torch.randint(0, cfg.vocab, (1, 16))
h = model.embed_tokens(x)
out, state = model(h)
print(f'  Input:  {h.shape}  std={h.std():.4f}  mean={h.mean():.4f}')
print(f'  Output: {out.shape}  std={out.std():.4f}  mean={out.mean():.4f}')

# State shapes
for i, s in enumerate(state):
    mem, mu, conv = s
    print(f'  Layer {i} state: mem={list(mem.shape)} mu={list(mu.shape)} conv={list(conv.shape)}')

print('\n=== DONE ===')
