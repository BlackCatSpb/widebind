"""Mini-test: 100 steps + comprehensive checks.
Without args: quick smoke test (100 steps).
With --full: adds alpha grad flow, gate behavior, checkpoint, aux loss checks.
"""
import sys, os, math, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn.functional as F
from core import WideBindConfig, WideBindStack, AdaptiveController, MirrorLRScheduler
from compression import FCF_CPR

parser = argparse.ArgumentParser()
parser.add_argument('--full', action='store_true', help='Run comprehensive checks')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device} ({torch.cuda.get_device_name(0) if device=="cuda" else "N/A"})')

# ─── Smoke test (always) ────────────────────────────────────────────
print('\n=== SMOKE TEST: 100 steps ===')
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
        print(f'  alpha.mean:       {m0.alpha_diag.data.mean().item():.4f}   alpha.std={m0.alpha_diag.data.std().item():.4f}   pred_scale.std={m0.w_pred_scale.data.std().item():.4f}')
        print(f'  log_scale.var (diff):{diff:.6f}')
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
state, gs = None, None
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

# Verify no NaN
assert not math.isnan(loss_after.item()), 'Loss is NaN!'
assert not math.isinf(loss_after.item()), 'Loss is Inf!'
print('  PASS: No NaN/Inf')

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
full_cfg = WideBindConfig(D=896, n_layers=8, mlp_groups=8, seq_len=64, batch_size=1)
big_model = WideBindStack(full_cfg).to(device)
big_model.train()
B2, L2 = 2, 32
x2 = torch.randint(0, full_cfg.vocab, (B2, L2), device=device)
y2 = torch.randint(0, full_cfg.vocab, (B2, L2), device=device)
h2 = big_model.embed_tokens(x2)
out2, _, _ = big_model(h2, None)
loss2 = big_model.compute_loss(out2, y2)
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
chk_cfg = WideBindConfig(D=896, n_layers=4, mlp_groups=8, seq_len=64)
chk_model = WideBindStack(chk_cfg).to(device)
chk_model.eval()
x3 = torch.randint(0, chk_cfg.vocab, (2, 32), device=device)
y3 = torch.randint(0, chk_cfg.vocab, (2, 32), device=device)
h3 = chk_model.embed_tokens(x3)
out3, _, _ = chk_model(h3, None)
# CE only
logits = chk_model.lm_head(out3)
ce_loss = F.cross_entropy(logits.reshape(-1, chk_cfg.vocab), y3.reshape(-1))
# With aux
loss_with = chk_model.compute_loss(out3, y3, pred_weight=0.01)
check('Aux loss weight=0.01 doesn\'t dominate CE',
      (loss_with.item() - ce_loss.item()) < 0.1 * ce_loss.item(),
      f'aux_contrib={(loss_with.item() - ce_loss.item()):.4f} ce={ce_loss.item():.4f}')

# ─── Check 3: Gate signal is |pred_error| ───
print('\n--- Check 3: Gate behavior ---')
mir = chk_model.layers[0].mirror
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
del chk_model, h3, out3

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
print('\n--- Check 5: Backward compatibility ---')
comp_model = WideBindStack(full_cfg).to(device)
comp_model.train()
x5 = torch.randint(0, full_cfg.vocab, (2, 16), device=device)
y5 = torch.randint(0, full_cfg.vocab, (2, 16), device=device)
h5 = comp_model.embed_tokens(x5)
out5, _, _ = comp_model(h5, None)
l2 = comp_model.compute_loss(out5, y5)
l3 = comp_model.compute_loss(out5, y5, pred_weight=0.01)
l0 = comp_model.compute_loss(out5, y5, pred_weight=0.0)
check('compute_loss 2 args works', not torch.isnan(l2), 'NaN')
check('compute_loss 3 args works', not torch.isnan(l3), 'NaN')
check('compute_loss pred_weight=0 works', not torch.isnan(l0), 'NaN')
del comp_model, h5, out5

# ─── Check 6: Checkpoint save/load ───
print('\n--- Check 6: Checkpoint save/load ---')
ckpt_model = WideBindStack(full_cfg).to(device)
ckpt_model.eval()
sd_before = {k: v.clone() for k, v in ckpt_model.state_dict().items()}
cpr = FCF_CPR()
save_path = '_test_ckpt.pt'
cpr.save_compressed({'step': 42, 'model': ckpt_model.state_dict(),
                     'best_val_loss': 7.5, 'cfg': full_cfg}, save_path)
loaded = cpr.load_compressed(save_path)
ckpt_model2 = WideBindStack(full_cfg).to(device)
missing, unexpected = ckpt_model2.load_state_dict(loaded['model'], strict=False)
os.remove(save_path)
check('No missing keys', len(missing) == 0, f'{len(missing)} missing')
check('No unexpected keys', len(unexpected) == 0, f'{len(unexpected)} unexpected')

# ─── Check 7: alpha no weight decay ───
print('\n--- Check 7: alpha param group ---')
wd_model = WideBindStack(full_cfg)
groups = wd_model.param_groups()
wd_ok = True
for g in groups:
    wd = g.get('weight_decay', None)
    for n, p in wd_model.named_parameters():
        if '.alpha' in n and any(id(p) == id(pp) for pp in g['params']):
            if wd != 0:
                print(f'  FAIL: alpha weight_decay={wd}')
                wd_ok = False
check('All alpha have weight_decay=0', wd_ok, '')
del wd_model, ckpt_model, ckpt_model2

# ─── Check 8: Training stability on deeper model ───
print('\n--- Check 8: Deeper model training (50 steps) ---')
deep_cfg = WideBindConfig(D=896, n_layers=12, mlp_groups=8, seq_len=64, batch_size=1)
deep_model = WideBindStack(deep_cfg).to(device)
deep_opt = torch.optim.AdamW(deep_model.param_groups(deep_cfg.lr), betas=(0.9, 0.95))
deep_model.train()
ds, dgs = None, None
losses = []
for step in range(50):
    xd = torch.randint(0, deep_cfg.vocab, (1, 64), device=device)
    yd = torch.randint(0, deep_cfg.vocab, (1, 64), device=device)
    deep_opt.zero_grad()
    hd = deep_model.embed_tokens(xd)
    outd, ds, dgs = deep_model(hd, state=ds, global_state=dgs)
    ld = deep_model.compute_loss(outd, yd)
    ld.backward()
    torch.nn.utils.clip_grad_norm_(deep_model.parameters(), deep_cfg.grad_clip)
    deep_opt.step()
    ds = [(s[0].detach(), s[1].detach(), s[2].detach()) if s is not None else None for s in ds]
    dgs = dgs.detach() if dgs is not None else None
    losses.append(ld.item())
check('No NaN/Inf in 50 steps', not any(math.isnan(l) or math.isinf(l) for l in losses), '')
check('Loss stable (no divergence)', max(losses) - min(losses) < 5.0,
      f'range=[{min(losses):.4f}, {max(losses):.4f}]')
del deep_model, ds, dgs

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
