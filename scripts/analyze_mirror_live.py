"""Live mirror dissection: load checkpoint, run a real forward, measure K-space.
Answers: why pred=35 (hp inflated?), why |mirror|=51 (log_scale/skip_alpha/gates?),
why signal_ent~0 (which signal dominates), why diversity collapsed.

Usage:
  python scripts/analyze_mirror_live.py <path/to/checkpoint.pt> [--seq 256] [--batch 2] [--device cpu|cuda]
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import argparse
import torch
import torch.nn.functional as F
from torch.serialization import add_safe_globals
from core import WideBindConfig, WideBindStack
add_safe_globals([WideBindConfig])


def dissect(ckpt_path, seq=256, batch=2, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    cfg = ckpt['cfg']
    model = WideBindStack(cfg)
    model.load_state_dict(ckpt['model'], strict=False)
    model.train()
    model.to(device)

    print('=' * 78)
    print(f'CHECKPOINT {os.path.basename(ckpt_path)}  step={ckpt.get("step", "?")}  '
          f'best_val={ckpt.get("best_val_loss", float("inf"))}')
    print('=' * 78)

    x = torch.randint(0, cfg.vocab, (batch, seq), device=device)
    h = model.embed_tokens(x)
    h = h.to(device)
    h_out, _, _, _ = model(h)

    print(f'\nINPUT/OUTPUT:  in_norm={h.norm(dim=-1).mean().item():.3f}  '
          f'out_norm={h_out.norm(dim=-1).mean().item():.3f}  '
          f'out_std={h_out.std().item():.4f}')

    # LM head sanity on random targets
    tgt = torch.randint(1, cfg.vocab, (batch, seq), device=device)
    ls, aux = model.compute_losses(h_out[:, :-1], tgt[:, 1:])
    print(f'CE(random)={ls.item():.3f}  pred={aux["pred"]:.3f}  gate_l1={aux["gate_l1"]:.5f}  '
          f'balance={aux["balance"]:.5f}')

    print()
    print(f'{"L":>3s} {"||hp||":>8s} {"predMSE":>8s} {"|mirror|":>9s} {"gate":>7s} '
          f'{"gate_var":>9s} {"|1-a|":>7s} {"ls_expM":>8s} {"ls_std":>7s} {"skip":>6s} '
          f'{"d_norm":>7s} {"w_delta":>7s}')
    print('-' * 106)
    for i, layer in enumerate(model.layers):
        m = layer.mirror
        hp = m._cached_hp
        hp_norm = hp.norm(dim=-1).mean().item()
        pk = m._cached_pred_k
        pred_mse = F.mse_loss(pk, hp.detach()).item()
        mag = m._last_magnitude.item()
        gate = m._cached_gate_l1.item()
        gvar = m._last_gates.var().item()
        alpha = m.alpha_diag.data
        ls = m.log_scale.data
        ls_exp_max = ls.exp().max().item()
        skip = math.exp(m.log_skip_alpha.data.mean().item())
        delta_n = m._cached_delta_norm.item() if hasattr(m, '_cached_delta_norm') else float('nan')
        wdg = m.w_delta_gate.data.abs().mean().item()
        print(f'{i:3d} {hp_norm:8.3f} {pred_mse:8.3f} {mag:9.3f} {gate:7.5f} {gvar:9.5f} '
              f'{(1-alpha).abs().mean().item():7.4f} {ls_exp_max:8.2f} {ls.std().item():7.4f} '
              f'{skip:6.3f} {delta_n:7.3f} {wdg:7.5f}')

    print()
    print('SIGNALS (sigmoid weights + softmax share) per layer 0, mid, last:')
    for i in [0, len(model.layers) // 2, len(model.layers) - 1]:
        m = model.layers[i].mirror
        w = torch.sigmoid(m._signal_log_weights)
        p = w / (w.sum() + 1e-10)
        names = ['temp', 'pred', 'smooth', 'sym'] + (['help'] if m._has_private_mem else [])
        print(f'  L{i}: sigmoid={["%.3f" % v for v in w.tolist()]}  '
              f'share={["%.3f" % v for v in p.tolist()]}  names={names}')

    print()
    print('MIRROR PARAM DIAGNOSIS per layer 0, mid, last:')
    for i in [0, len(model.layers) // 2, len(model.layers) - 1]:
        m = model.layers[i].mirror
        alpha = m.alpha_diag.data
        ls = m.log_scale.data
        print(f'  L{i}:')
        print(f'    alpha_diag: mean={alpha.mean().item():.4f} min={alpha.min().item():.4f} '
              f'max={alpha.max().item():.4f}  (init range 0.61..0.995)')
        print(f'    log_scale:  mean={ls.mean().item():.4f} max={ls.max().item():.4f} '
              f'exp(max)={ls.exp().max().item():.2f} std={ls.std().item():.4f}')
        print(f'    log_skip_alpha: mean={m.log_skip_alpha.data.mean().item():.4f} '
              f'exp={math.exp(m.log_skip_alpha.data.mean().item()):.4f}')
        print(f'    tanh_bias: mean={m.tanh_bias.data.mean().item():.4f} '
              f'max={m.tanh_bias.data.max().item():.4f}')
        print(f'    W_out: std={m.W_out.data.std().item():.4f}  '
              f'W_proj: std={m.W_proj.data.std().item():.4f}')
        print(f'    w_gate: mean={m.w_gate.data.mean().item():.4f} std={m.w_gate.data.std().item():.4f}  '
              f'b_gate={m.b_gate.data.mean().item():.4f}  gate_bias={m.gate_bias.data.mean().item():.4f}')
        print(f'    mod_scale_mlp(sig)={torch.sigmoid(m.mod_scale_mlp.data).mean().item():.3f}  '
              f'mod_scale_mem(sig)={torch.sigmoid(m.mod_scale_mem.data).mean().item():.3f}')
        if m._has_private_mem:
            print(f'    w_help(sig)={torch.sigmoid(m.w_help.data).mean().item():.3f}  '
                  f'w_contra={m.w_contra.data.mean().item():.4f}  '
                  f'pm_norm={m._private_mem.norm(dim=-1).mean().item():.3f}')
        print(f'    gate_ema={m._gate_ema.data.mean().item():.4f}')

    print()
    print('GLOBAL:')
    vsa_log = model._vsa_log_param
    vsa_tau = torch.exp(torch.cumsum(F.softplus(vsa_log), dim=0)) + 1.0
    print(f'  vsa tau: min={vsa_tau[0].item():.2f} max={vsa_tau[-1].item():.2f}')
    td = model._tau_l_dev.data
    print(f'  tau_l_dev: mean={td.mean().item():.4f} std={td.std().item():.4f}')
    pm = model._private_mem if hasattr(model, '_private_mem') else None
    if pm is not None:
        print(f'  model private_mem norm: {pm.norm(dim=-1).mean().item():.3f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint', type=str)
    p.add_argument('--seq', type=int, default=256)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    dissect(a.checkpoint, a.seq, a.batch, a.device)