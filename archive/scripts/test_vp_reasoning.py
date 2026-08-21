import torch
import sys
sys.path.insert(0, '.')
from core import WideBindConfig, WideBindStack

cfg = WideBindConfig()
cfg.D = 128
cfg.n_layers = 4
cfg.bind_K = 32
cfg.vocab = 100
cfg.mlp_groups = 4
cfg.lambda_d_enabled = False
cfg.head_mode = 'sigmoid_coded'
cfg.variable_precision = True
cfg.explicit_reasoning = True

model = WideBindStack(cfg)
n = sum(p.numel() for p in model.parameters())
print(f'Params: {n:,} ({n/1e6:.2f}M)')

x = torch.randint(0, 100, (1, 10))
h = model.embed_tokens(x)
out, state, gs = model(h, step=0)
print(f'Forward OK: out.shape={out.shape}, NaN={torch.isnan(out).any().item()}')

targets = torch.randint(0, 100, (1, 10))
loss, aux = model.compute_losses(out, targets)
print(f'Loss: {loss.item():.4f}')
print(f'Aux keys: {list(aux.keys())}')
print('ALL OK')
