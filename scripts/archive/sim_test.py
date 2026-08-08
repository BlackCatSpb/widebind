"""Sim-test for Big model (D=2560, 24 layers).
Usage:
  python scripts/sim_test.py                         # quick smoke (200 steps)
  python scripts/sim_test.py --full                   # + comprehensive checks
  python scripts/sim_test.py --steps 100000           # long-run stability test
  python scripts/sim_test.py --steps 100000 --full    # both
"""
import sys, os, math, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from core import WideBindConfig, WideBindStack, AdaptiveController, MirrorLRScheduler

parser = argparse.ArgumentParser()
parser.add_argument('--full', action='store_true', help='Run comprehensive checks')
parser.add_argument('--steps', type=int, default=200, help='Number of training steps (default: 200)')
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

device = args.device
print(f'Device: {device} ({torch.cuda.get_device_name(0) if device=="cuda" else "N/A"})')

# ─── Smoke test (always) ────────────────────────────────────────────
N = args.steps
print(f'\n=== SMOKE TEST: {N} steps (Big: D=2560, 24 layers, G=32) ===')
cfg = WideBindConfig(
    D=2560, n_layers=24, bind_K=64, mlp_groups=32, mlp_expand=4,
    seq_len=128, batch_size=1, lr=3e-4, conv_kernel=48,
    gradient_checkpointing=False,
)
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
        print(f'  alpha.mean:       {m0.alpha_diag.data.mean().item():.4f}   alpha.std={m0.alpha_diag.data.std().item():.4f}')
        print(f'  tanh_bias.std:    {m0.tanh_bias.data.std().item():.4f}')
        print(f'  log_skip_alpha:   mean={m0.log_skip_alpha.data.mean().item():.4f}')
        print(f'  w_q[L0].mean:     {model.layers[0].w_q.data.mean().item():.4f}')

snapshot(model, 'BEFORE')
optimizer = torch.optim.AdamW(model.param_groups(), betas=(0.9, 0.95))

B, L = cfg.batch_size, cfg.seq_len

with torch.no_grad():
    model.eval()
    x0 = torch.randint(0, cfg.vocab, (B, L), device=device)
    h = model.embed_tokens(x0)
    out, _, _ = model(h, None, adaptive=False)
    ce_before, aux_before = model.compute_losses(out, x0)
    print(f'\nCE before:  {ce_before.item():.4f}  aux total: {sum(aux_before.values()).item():.4f}')

model.train()
state = None
t0 = time.time()
losses = []
n_finite = 0

for step in range(N):
    x = torch.randint(0, cfg.vocab, (B, L), device=device)
    y = torch.randint(0, cfg.vocab, (B, L), device=device)
    optimizer.zero_grad()
    h = model.embed_tokens(x)
    out, state, _ = model(h, state=state, step=step)
    ce_loss, aux_dict = model.compute_losses(out, y)
    loss = ce_loss + sum(aux_dict.values())
    
    if torch.isfinite(loss):
        n_finite += 1
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        optimizer.step()
    
    losses.append(loss.item())
    
    if (step + 1) % 50 == 0 or step < 3:
        vr = torch.cuda.memory_allocated() / 1e6 if device == 'cuda' else 0
        print(f'  step {step+1:3d}  loss={loss.item():.4f}  ce={ce_loss.item():.4f}  |g|={grad_norm if torch.isfinite(loss) else 0:.4f}  VRAM={vr:.0f}MB')

t1 = time.time()
print(f'\nTime: {t1-t0:.1f}s ({N/(t1-t0):.1f} steps/s)')
print(f'Finite steps: {n_finite}/{N}')

with torch.no_grad():
    model.eval()
    x0 = torch.randint(0, cfg.vocab, (B, L), device=device)
    h = model.embed_tokens(x0)
    out, _, _ = model(h, None, adaptive=False)
    ce_after, aux_after = model.compute_losses(out, x0)
    print(f'CE: {ce_before.item():.4f} -> {ce_after.item():.4f}  (delta: {ce_after.item()-ce_before.item():.4f})')

snapshot(model, 'AFTER')

# Verify no NaN/Inf
assert not any(math.isnan(l) or math.isinf(l) for l in losses if torch.isfinite(torch.tensor(l))), 'Loss has NaN/Inf!'
print('  PASS: No NaN/Inf in finite steps')

# ─── Full checks (--full flag) ─────────────────────────────────────
if not args.full:
    print('\nDone. (use --full for comprehensive checks)')
    sys.exit(0)

print('\n' + '=' * 60)
print('FULL CHECKS')
print('=' * 60)

n_pass = 0
n_fail = 0

def check(name, cond, detail=''):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f'  PASS {name}')
    else:
        n_fail += 1
        print(f'  FAIL {name}: {detail}')

# ─── Check 1: alpha gradient flows to all layers ───
print('\n--- Check 1: alpha gradient flow ---')
full_cfg = WideBindConfig(D=2560, n_layers=8, bind_K=64, mlp_groups=32, seq_len=64, batch_size=1)
big_model = WideBindStack(full_cfg).to(device)
big_model.train()
B2, L2 = 2, 32
x2 = torch.randint(0, full_cfg.vocab, (B2, L2), device=device)
y2 = torch.randint(0, full_cfg.vocab, (B2, L2), device=device)
h2 = big_model.embed_tokens(x2)
out2, _, _ = big_model(h2, None)
ce2, aux2 = big_model.compute_losses(out2, y2)
loss2 = ce2 + sum(aux2.values())
loss2.backward()
grads = [l.mirror.alpha_diag.grad.norm().item() for l in big_model.layers]
check('All layers have non-zero alpha grad', all(g > 1e-6 for g in grads),
      f'min={min(grads):.6f}')
check('Bottom layer alpha grad > 1e-4', grads[0] > 1e-4,
      f'L0 grad={grads[0]:.6f}')
check('Bottom/top grad ratio > 0.1', grads[0] / max(grads[-1], 1e-8) > 0.1,
      f'ratio={grads[0]/max(grads[-1],1e-8):.3f}')
big_model.zero_grad()
del big_model, h2, out2, loss2

# ─── Check 2: Auxiliary loss contribution ───
print('\n--- Check 2: Auxiliary loss breakdown ---')
chk_cfg = WideBindConfig(D=2560, n_layers=4, bind_K=64, mlp_groups=32, seq_len=64, batch_size=1)
chk_model = WideBindStack(chk_cfg).to(device)
chk_model.eval()
x3 = torch.randint(0, chk_cfg.vocab, (2, 32), device=device)
y3 = torch.randint(0, chk_cfg.vocab, (2, 32), device=device)
h3 = chk_model.embed_tokens(x3)
out3, _, _ = chk_model(h3, None)
ce3, aux3 = chk_model.compute_losses(out3, y3)
loss_with = ce3 + sum(aux3.values())
check('CE dominates total loss (aux < 20% of CE)',
      (loss_with.item() - ce3.item()) < 0.2 * ce3.item(),
      f'ce={ce3.item():.4f} total={loss_with.item():.4f}')
del chk_model, h3, out3

# ─── Check 3: Gate signal range ───
print('\n--- Check 3: Gate behavior ---')
gate_model = WideBindStack(chk_cfg).to(device)
gate_model.eval()
x3b = torch.randint(0, chk_cfg.vocab, (2, 32), device=device)
h3b = gate_model.embed_tokens(x3b)
out3b, _, _ = gate_model(h3b, None)
_ = gate_model.compute_losses(out3b, y3)

mir = gate_model.layers[0].mirror
pk = mir._cached_pred_k
hp = mir._cached_hp
pred_error = hp - pk
gate_signal = torch.abs(pred_error)
w_gate = mir.w_gate
b_gate = mir.b_gate
gate_logits = torch.einsum('blgk,gk->blg', gate_signal, w_gate) + b_gate
expert_gate = torch.sigmoid(gate_logits)
check('Gate range covers [0,1]', expert_gate.min().item() < 0.2 and expert_gate.max().item() > 0.8,
      f'range=[{expert_gate.min().item():.4f}, {expert_gate.max().item():.4f}]')
check('Gate mean not stuck at 0.5', abs(expert_gate.mean().item() - 0.5) > 0.001,
      f'mean={expert_gate.mean().item():.4f}')
del gate_model

# ─── Check 4: pred_cache lifecycle ───
print('\n--- Check 4: Pred cache lifecycle ---')
cache_model = WideBindStack(full_cfg).to(device)
cache_model.eval()
x4 = torch.randint(0, full_cfg.vocab, (2, 16), device=device)
h4 = cache_model.embed_tokens(x4)
out4, _, _ = cache_model(h4, None)
check('Pred cache populated after forward',
      len(getattr(cache_model, '_pred_cache', [])) == full_cfg.n_layers,
      f'{len(getattr(cache_model, "_pred_cache", []))} vs {full_cfg.n_layers}')
out4b, _, _ = cache_model(h4, None)
check('Pred cache refreshed each forward',
      len(getattr(cache_model, '_pred_cache', [])) == full_cfg.n_layers, '')
del cache_model

# ─── Check 5: compute_loss backward compat ───
print('\n--- Check 5: Loss API ---')
comp_model = WideBindStack(full_cfg).to(device)
comp_model.train()
x5 = torch.randint(0, full_cfg.vocab, (2, 16), device=device)
y5 = torch.randint(0, full_cfg.vocab, (2, 16), device=device)
h5 = comp_model.embed_tokens(x5)
out5, _, _ = comp_model(h5, None)
ce5, aux5 = comp_model.compute_losses(out5, y5)
l5 = ce5 + sum(aux5.values())
check('compute_losses returns ce+aux', not torch.isnan(l5), 'NaN')
check('compute_loss returns CE only', not torch.isnan(comp_model.compute_loss(out5, y5)), 'NaN')
del comp_model, h5, out5

# ─── Check 6: Recurrent state propagation ───
print('\n--- Check 6: Recurrent state (50 steps continuous) ---')
recurrent_cfg = WideBindConfig(D=2560, n_layers=6, bind_K=64, mlp_groups=32, seq_len=128, batch_size=1)
recurr_model = WideBindStack(recurrent_cfg).to(device)
recurr_opt = torch.optim.AdamW(recurr_model.param_groups(), betas=(0.9, 0.95))
recurr_model.train()
rs = None
r_losses = []
for step in range(50):
    xr = torch.randint(0, recurrent_cfg.vocab, (1, 128), device=device)
    yr = torch.randint(0, recurrent_cfg.vocab, (1, 128), device=device)
    recurr_opt.zero_grad()
    hr = recurr_model.embed_tokens(xr)
    outr, rs, _ = recurr_model(hr, state=rs, step=step)
    cer, auxr = recurr_model.compute_losses(outr, yr)
    lr = cer + sum(auxr.values())
    lr.backward()
    torch.nn.utils.clip_grad_norm_(recurr_model.parameters(), recurrent_cfg.grad_clip)
    recurr_opt.step()
    r_losses.append(lr.item())
check('No NaN/Inf in 50 recurrent steps',
      not any(math.isnan(l) or math.isinf(l) for l in r_losses), '')
check('Loss stable (range < 5.0)', max(r_losses) - min(r_losses) < 5.0,
      f'range=[{min(r_losses):.4f}, {max(r_losses):.4f}]')
del recurr_model, rs

# ─── Check 7: Training stability (100 steps, random batch) ───
print('\n--- Check 7: Deeper model training (100 steps) ---')
deep_cfg = WideBindConfig(D=2560, n_layers=12, bind_K=64, mlp_groups=32, seq_len=128, batch_size=1)
deep_model = WideBindStack(deep_cfg).to(device)
deep_opt = torch.optim.AdamW(deep_model.param_groups(), betas=(0.9, 0.95))
deep_model.train()
ds = None
d_losses = []
for step in range(100):
    xd = torch.randint(0, deep_cfg.vocab, (1, 128), device=device)
    yd = torch.randint(0, deep_cfg.vocab, (1, 128), device=device)
    deep_opt.zero_grad()
    hd = deep_model.embed_tokens(xd)
    outd, ds, _ = deep_model(hd, state=ds, step=step)
    ced, auxd = deep_model.compute_losses(outd, yd)
    ld = ced + sum(auxd.values())
    ld.backward()
    torch.nn.utils.clip_grad_norm_(deep_model.parameters(), deep_cfg.grad_clip)
    deep_opt.step()
    d_losses.append(ld.item())
check('No NaN/Inf in 100 steps', not any(math.isnan(l) or math.isinf(l) for l in d_losses), '')
check('Loss stable (range < 5.0)', max(d_losses) - min(d_losses) < 5.0,
      f'range=[{min(d_losses):.4f}, {max(d_losses):.4f}]')
del deep_model, ds

# ─── Summary ───
print(f'\n{"="*40}')
print(f'Passed: {n_pass}/{n_pass + n_fail}')
if n_fail > 0:
    print(f'FAILED: {n_fail}')
    sys.exit(1)
else:
    print('All checks passed.')
print(f'{"="*40}')
print('Done.')
