import torch
from torch.serialization import add_safe_globals
from config import WideBindConfig
from core import WideBindStack
add_safe_globals([WideBindConfig])

ckpt = torch.load('checkpoints/best.pt', map_location='cpu', weights_only=True)

print('=== CHECKPOINT ===')
print(f'Step: {ckpt["step"]}')
print(f'Best val_loss: {ckpt["best_val_loss"]:.4f}')
ep = ckpt['scheduler']['last_epoch']
print(f'Scheduler epoch: {ep}')

model = WideBindStack(ckpt['cfg'])
model.eval()
model.load_state_dict(ckpt['model'], strict=False)

print()
print('=== WEIGHT STATS ===')
for name, p in model.named_parameters():
    d = p.data
    print(f'  {name:55s}  mean={d.mean():+.4f}  std={d.std():.4f}  norm={d.norm():.1f}')

print()
print('=== MIRROR log_scale ===')
for i in range(ckpt['cfg'].n_layers):
    ls = model.layers[i].mirror.log_scale.data
    print(f'  L{i:2d}  mean={ls.mean():+.4f}  std={ls.std():.4f}  exp range=[{ls.exp().min():.4f}, {ls.exp().max():.4f}]')

print()
print('=== MEMORY GATES (layer 0 mean) ===')
l0 = model.layers[0]
for n in ['w_i', 'w_d', 'w_q', 'w_mem2v', 'w_k_mu', 'w_q_mu', 'w_mu_mem', 'b_i', 'b_d']:
    d = getattr(l0, n).data
    print(f'  {n:10s}  mean={d.mean():+.4f}  std={d.std():.4f}')

print()
print('=== MLP eff_rank (top 5 + bottom 2) ===')
for i in [0,1,2,3,4,22,23]:
    up_s = torch.linalg.svdvals(model.layers[i].mlp_up.weight.float())
    eff = (up_s**2).sum() / up_s.max()**2
    nrm = model.layers[i].mlp_up.weight.norm().item()
    print(f'  L{i:2d}  eff_rank={eff:.1f}  ||W||={nrm:.1f}')

print()
print('=== BIND eff_rank ===')
for i in [0,11,23]:
    s = torch.linalg.svdvals(model.layers[i].W_proj.float())
    eff = (s**2).sum() / s.max()**2
    nrm = model.layers[i].W_proj.norm().item()
    print(f'  L{i:2d}  eff_rank={eff:.1f}  ||W||={nrm:.1f}')

print()
x = torch.randint(0, 1000, (1, 16))
h = model.embed_tokens(x)
out, state = model(h)
print(f'Forward: output std={out.std():.4f}')
