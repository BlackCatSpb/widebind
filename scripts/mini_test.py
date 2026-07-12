"""Mini-test: 100 steps on MX550 — verify all optimizations work before Colab."""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from core import WideBindConfig, WideBindStack, AdaptiveController

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device} ({torch.cuda.get_device_name(0) if device=="cuda" else "N/A"})')

cfg = WideBindConfig(D=896, n_layers=4, mlp_groups=8, seq_len=64, batch_size=1,
                     w_pred_scale_init=0.5, lr=2e-4)

model = WideBindStack(cfg).to(device)
print(f'Params: {model.param_count()/1e6:.1f}M')

def snapshot(model, tag):
    with torch.no_grad():
        ac = AdaptiveController
        expl, diff = ac.stats(model.layers)
        ns = ac.noise_scale(model.layers)
        m0 = model.layers[0].mirror
        lm = model.layers[-1].mirror
        print(f'\n=== {tag} ===')
        print(f'  expl={expl:.4f}  diff={diff:.6f}  noise={ns:.6f}')
        print(f'  gate_pred_scale:  {m0.gate_pred_scale.item():+.4f} (L0)  {lm.gate_pred_scale.item():+.4f} (L31)')
        print(f'  W_pred.std:       {m0.W_pred.data.std().item():.4f}   pred_scale.std={m0.w_pred_scale.data.std().item():.4f}')
        print(f'  log_scale[L2].var:{diff:.6f}')
        print(f'  log_skip_alpha:   mean={m0.log_skip_alpha.data.mean().item():.4f}')
        print(f'  dvar_mod_bias:    mean={m0.dvar_mod_bias.data.mean().item():.4f}')
        print(f'  w_q[L0].mean:     {model.layers[0].w_q.data.mean().item():.4f}')

snapshot(model, 'BEFORE')

optimizer = torch.optim.AdamW(model.param_groups(cfg.lr), betas=(0.9, 0.95))

B, L = cfg.batch_size, cfg.seq_len

with torch.no_grad():
    model.eval()
    x0 = torch.randint(0, cfg.vocab, (B, L), device=device)
    h = model.embed_tokens(x0)
    out, _, _ = model(h)
    loss_before = model.compute_loss(out, x0)
    print(f'\nLoss before: {loss_before.item():.4f}')

model.train()
state = None
gs = None
t0 = time.time()

for step in range(100):
    x = torch.randint(0, cfg.vocab, (B, L), device=device)
    y = torch.randint(0, cfg.vocab, (B, L), device=device)
    
    optimizer.zero_grad()
    h = model.embed_tokens(x)
    out, state, gs = model(h, state=state, global_state=gs)
    loss = model.compute_loss(out, y)
    loss.backward()
    
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
    optimizer.step()
    
    state = [(s[0].detach(), s[1].detach(), s[2].detach()) if s is not None else None for s in state]
    gs = gs.detach()
    
    if (step + 1) % 20 == 0 or step < 3:
        vr = torch.cuda.memory_allocated() / 1e6 if device == 'cuda' else 0
        print(f'  step {step+1:3d}  loss={loss.item():.4f}  |g|={grad_norm:.4f}  VRAM={vr:.0f}MB')

t1 = time.time()
print(f'\nTime: {t1-t0:.1f}s ({100/(t1-t0):.1f} steps/s)')

with torch.no_grad():
    model.eval()
    x0 = torch.randint(0, cfg.vocab, (B, L), device=device)
    h = model.embed_tokens(x0)
    out, _, _ = model(h)
    loss_after = model.compute_loss(out, x0)
    print(f'Loss: {loss_before.item():.4f} -> {loss_after.item():.4f}  (delta: {loss_after.item()-loss_before.item():.4f})')

snapshot(model, 'AFTER')
print('Done.')
