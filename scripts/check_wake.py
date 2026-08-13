"""Wake-up detector for WideBind training checkpoints.
Checks all watchlist markers from docs/TRAINING_JOURNAL.md §5:
  - MLP W_std vs empirical weight-decay curve (wake marker #1)
  - modulation gate sigmoid(mod_scale_mlp) (wake marker #2)
  - mod_scale_mlp vs mod_scale_mem divergence
  - collective memory slots (births in empty layers, _temp)
  - maturity count (read-only mode marker)
  - core internals: log_temp, token_bias, tau ladder, b_gate, ls_std, conv_std
  - NaN/Inf scan

Usage:
  python scripts/check_wake.py <path/to/checkpoint.pt>
Baselines set at step 10485 (val 8.6706). Verdict:
  PASS | WATCH | WAKE-CANDIDATE
"""
import sys, os, io
if sys.stdout and sys.stdout.buffer:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from torch.serialization import add_safe_globals
from core import WideBindConfig, WideBindStack
add_safe_globals([WideBindConfig])

REF_STEP = 10485
W_STD_REF = 0.0695          # MLP mean W_std at REF_STEP
DECAY_RATE = 1.6e-6         # empirical per-step decay (wd=0.01, lr_eff~8.9e-5)
EMPTY_SLOT_LAYERS = {13, 14, 16, 17, 18, 20, 21}

SIG = {'PASS': 'PASS ', 'WATCH': 'WATCH', 'WAKE': 'WAKE '}


def verdict(report, flag, hits, text):
    report.append(f"  [{SIG[flag]}] {text}")
    for h in hits:
        report.append(f"        {h}")


def main(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    cfg = ckpt['cfg']
    model = WideBindStack(cfg)
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    step = int(ckpt.get('step', 0))
    n_layers = len(model.layers)

    report = [f"Wake-up scan: {os.path.basename(path)} step={step} layers={n_layers}",
              f"  unexpected keys: {len(unexpected)}  missing: {len(missing)}"]

    mlp_wstd = []
    gate_mlp = []
    gate_mem = []
    slot_occ = {}
    temp_vals = []
    mat_counts = []
    for i, layer in enumerate(model.layers):
        mlp = layer.mlp
        if hasattr(mlp, 'W_up'):
            mlp_wstd.append(mlp.W_up.data.std().item())
        elif hasattr(mlp, 'W_gate'):
            mlp_wstd.append(mlp.W_gate.data.std().item())
        mir = layer.mirror
        if hasattr(mir, 'mod_scale_mlp'):
            gate_mlp.append(torch.sigmoid(mir.mod_scale_mlp.data).mean().item())
        if hasattr(mir, 'mod_scale_mem'):
            gate_mem.append(torch.sigmoid(mir.mod_scale_mem.data).mean().item())
        cl = getattr(layer, 'collective', None)
        if cl is not None:
            occ = (cl.N_s.data > 0).sum().item()
            slot_occ[i] = occ
            if hasattr(cl, '_temp') and cl._temp.ndim == 0:
                temp_vals.append(cl._temp.data.item())
        mat = getattr(layer.collective, '_mature_count', None) if getattr(layer, 'collective', None) is not None else None
        if mat is not None:
            mat_counts.append(mat)

    all_vals = torch.cat([p.data.flatten() for p in model.parameters()])
    n_nan = int(torch.isnan(all_vals).sum().item())
    n_inf = int(torch.isinf(all_vals).sum().item())
    report.append(f"  NaN: {n_nan}  Inf: {n_inf}")
    if n_nan or n_inf:
        verdict(report, 'WAKE', [], 'NaN/Inf in parameters -- ABORT analysis')
        return report

    wstd = sum(mlp_wstd) / len(mlp_wstd)
    expected = W_STD_REF * torch.exp(torch.tensor(-DECAY_RATE * (step - REF_STEP))).item()
    dev = wstd - expected
    report.append(f"  MLP W_std mean={wstd:.4f} decay-expected={expected:.4f} dev={dev:+.4f}")
    flag = 'WAKE' if dev > 0.001 else ('WATCH' if dev > 0.0005 else 'PASS')
    verdict(report, flag, [], f"MLP W_std vs decay curve (marker #1, dev {dev:+.4f})")

    per_layer = [(i, mlp_wstd[i] - expected) for i in range(n_layers)]
    per_layer.sort(key=lambda t: -t[1])
    top = per_layer[:3]
    assert_flag = 'WAKE' if top[0][1] > 0.002 else ('WATCH' if top[0][1] > 0.001 else 'PASS')
    hits = [f"L{li}: W_std dev {d:+.5f} (gate {torch.sigmoid(model.layers[li].mirror.mod_scale_mlp.data).mean().item():.3f})" for li, d in top]
    verdict(report, assert_flag, hits, f"per-layer W_std deviation (marker #1b, worst {top[0][1]:+.5f})")

    gmlp_all = [torch.sigmoid(l.mirror.mod_scale_mlp.data) for l in model.layers]
    g_max = [(i, g.max().item()) for i, g in enumerate(gmlp_all)]
    g_max.sort(key=lambda t: -t[1])
    gtop = g_max[:3]
    gflag = 'WAKE' if gtop[0][1] > 0.75 else ('WATCH' if gtop[0][1] > 0.72 else 'PASS')
    ghit = [f"L{li}: gate max {v:.3f} mean {torch.sigmoid(model.layers[li].mirror.mod_scale_mlp.data).mean().item():.3f}" for li, v in gtop]
    verdict(report, gflag, ghit, f"per-layer gate max (marker #2b, worst {gtop[0][1]:.3f}, wake >0.75)")

    g_mlp = sum(gate_mlp) / len(gate_mlp)
    g_mem = sum(gate_mem) / len(gate_mem) if gate_mem else float('nan')
    report.append(f"  sigmoid(mod_scale_mlp) mean={g_mlp:.3f}  sigmoid(mod_scale_mem) mean={g_mem:.3f}")
    flag = 'WAKE' if g_mlp > 0.75 else ('WATCH' if g_mlp > 0.72 else 'PASS')
    verdict(report, flag, [], f"modulation gate (marker #2, >0.75 = WAKE, baseline 0.668)")
    if abs(g_mlp - g_mem) > 0.2:
        verdict(report, 'WAKE', [], f"mlp/mem gate divergence {abs(g_mlp - g_mem):.2f}")

    births = [i for i in EMPTY_SLOT_LAYERS if i in slot_occ and slot_occ[i] > 0]
    full = [i for i, o in slot_occ.items() if o >= 8]
    report.append(f"  slots occupied: {sum(slot_occ.values())}/192, full layers: {len(full)}, births in empty: {births}")
    flag = 'WAKE' if births else 'PASS'
    verdict(report, flag, [], f"slot births in empty layers L13-14/16-18/20-21: {births or 'none'}")
    if temp_vals:
        t_mean = sum(temp_vals) / len(temp_vals)
        report.append(f"  _temp mean={t_mean:.3f}")
        if t_mean < 1.95:
            verdict(report, 'WATCH', [], f"_temp dropped from 2.0 baseline ({t_mean:.3f})")

    if mat_counts:
        n_zero = sum(1 for c in mat_counts if c == 0)
        report.append(f"  _mature_count: {n_zero}/{len(mat_counts)} layers locked (0)")
        if n_zero < len(mat_counts) * 0.5:
            verdict(report, 'WATCH', [], 'maturity lock released in >half layers')

    vsa_log = model._vsa_log_param
    vsa_tau = torch.exp(torch.cumsum(torch.nn.functional.softplus(vsa_log), dim=0)) + 1.0
    t0, t1 = vsa_tau[0].item(), vsa_tau[-1].item()
    td = model._tau_l_dev.data
    b_d = model.layers[0].b_d.data
    b_i = model.layers[0].b_i.data
    report.append(f"  tau ladder: [{t0:.2f}, {t1:.2f}] {t1/t0:.1f}x  tau_l_dev mean={td.mean().item():.4f}  b_d={b_d.mean().item():.4f}({b_d.min().item():.4f},{b_d.max().item():.4f})  b_i={b_i.mean().item():.4f}")
    report.append(f"  vsa_log softplus mean={torch.nn.functional.softplus(vsa_log).mean().item():.4f}")

    lm = model.lm_head
    log_temp = None
    for name, p in lm.named_parameters():
        if 'log_temp' in name:
            log_temp = p.data
    if log_temp is not None:
        report.append(f"  log_temp mean={log_temp.mean().item():+.4f} (ref +0.1351 at 10485)")
        if log_temp.mean().item() > 0.25:
            verdict(report, 'WATCH', [], 'log_temp rising (softmax flattening)')

    for i in (0, 1):
        mir = model.layers[i].mirror
        ls = mir.log_scale.data
        cw = model.layers[i].conv.weight.data
        tb = mir.tanh_bias.data
        report.append(f"  L{i}: ls_std={ls.std().item():.3f} conv_std={cw.std().item():.4f} tanh_bias=[{tb.min().item():.2f},{tb.max().item():.2f}]")

    hdr = 'OK' if not any(r.startswith('  [WAKE') for r in report) else 'WAKE-CANDIDATE'
    report.insert(1, f"  => VERDICT: {hdr}")
    return report


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\black\OneDrive\Desktop\WideBind\checkpoints\best.pt'
    for line in main(path):
        print(line)