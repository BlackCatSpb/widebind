"""Единый анализатор чекпоинтов WideBind.

Объединяет все методы анализа в один прогон:
  - static    — конфиг, per-layer параметры, VSA-лестница, зеркало, MLP, голова, NaN/Inf
  - inspector — сигналы (temp/pred/smooth/sym/help), trust/concept/dominance (private memory)
  - wake      — вердикт PASS/WATCH/WAKE (пробуждение MLP по маркерам watchlist)
  - live      — forward на случайном входе: hp/predMSE/|mirror|/gate/ls/skip, сигналы, параметры
  - head      — декомпозиция bias vs контекст + позиционная карта головы + A/B reasoning
  - cmp       — сравнение нескольких чекпоинтов на одинаковом входе

Usage:
  python scripts/analyze.py <ckpt...> [--no-live] [--no-head] [--quick]
                                       [--windows N] [--prompt TEXT] [--temp T]

  Один чекпоинт — полный разбор; несколько — разбор каждого + сравнение.
"""
import sys, os, math, re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass

import argparse
import torch
import torch.nn.functional as F
from torch.serialization import add_safe_globals

sys.path.insert(0, BASE)
from core import WideBindConfig, WideBindStack
add_safe_globals([WideBindConfig])

PUNCT = re.compile(r'^[\s\.,:;!?\-—–…"«»()\[\]{}]+$')
WORD = re.compile(r'[а-яёА-ЯЁ]')


# ─────────────────────────── загрузка ───────────────────────────

def load_ckpt(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    cfg = ckpt['cfg']
    model = WideBindStack(cfg)
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    model.train()
    if model.explicit_reasoning:
        model.reasoning_enabled_step = int(ckpt.get('reasoning_enabled_step', 0))
    return ckpt, cfg, model, missing, unexpected


def sec(title):
    print()
    print('=' * 78)
    print(title)
    print('=' * 78)


# ─────────────────────────── STATIC ───────────────────────────

def run_static(ckpt, cfg, model, missing, unexpected, tok=None):
    sd = ckpt['model']
    sec('STATIC')
    print(f'File:       {os.path.basename(ckpt.get("_path", "?"))}')
    print(f'Step:       {ckpt.get("step", "?")}')
    print(f'Best val:   {ckpt.get("best_val_loss", float("inf"))}')
    print(f'Params:     {model.param_count() / 1e6:.2f}M')
    print(f'Tensors:    {len(sd)}')
    print(f'Missing:    {len(missing)}   Unexpected: {len(unexpected)}')
    if missing:
        for k in missing[:5]:
            print(f'  MISS: {k}')
    if unexpected:
        for k in unexpected[:5]:
            print(f'  UNEXP: {k}')

    imp = ['D', 'n_layers', 'G', 'k', 'vocab', 'seq_len', 'mlp_groups', 'mlp_expand',
           'bind_K', 'mirror_k', 'lr', 'weight_decay', 'explicit_reasoning',
           'reasoning_max_steps', 'reasoning_adaptive', 'variable_precision',
           'collective_read_out', 'private_mem', 'head_mode', 'bind_twist_mode']
    print('\nCONFIG:')
    for a in imp:
        print(f'  {a:24s} = {getattr(cfg, a, "N/A")}')

    print('\nPARAM GROUPS (LR multipliers):')
    for g in model.param_groups():
        print(f'  lr={g.get("lr", cfg.lr):.6f} wd={g.get("weight_decay", 0):.4f} '
              f'n_params={len(g["params"]):4d}')

    print('\nPER-LAYER (params, без forward):')
    hdr = f'{"L":>3s} {"alpha":>7s} {"|1-a|":>7s} {"ls_std":>7s} {"w_help":>7s} ' \
          f'{"skip":>7s} {"scale_w":>7s} {"conv_std":>8s} {"pm_norm":>8s}'
    print(hdr)
    print('-' * len(hdr))
    for i, layer in enumerate(model.layers):
        m = layer.mirror
        a = m.alpha_diag.data
        ls = m.log_scale.data
        wh = torch.sigmoid(m.w_help.data)
        lsa = torch.sigmoid(m.log_skip_alpha.data)
        sw = torch.sigmoid(layer.scale_w.data)
        cw = layer.conv.weight.data
        pm = m._private_mem.norm(dim=-1).mean().item() if m._has_private_mem else float('nan')
        print(f'{i:3d} {a.mean().item():7.4f} {(1 - a).abs().mean().item():7.4f} '
              f'{ls.std().item():7.4f} {wh.mean().item():7.4f} {lsa.mean().item():7.4f} '
              f'{sw.mean().item():7.4f} {cw.std().item():8.4f} {pm:8.3f}')

    vsa_log = model._vsa_log_param
    vsa_tau = torch.exp(torch.cumsum(F.softplus(vsa_log), dim=0)) + 1.0
    td = model._tau_l_dev.data
    b_d = model.b_d.data if hasattr(model, 'b_d') else None
    b_i = model.b_i.data if hasattr(model, 'b_i') else None
    print(f'\nVSA TAU: tau[0]={vsa_tau[0].item():.2f}  tau[-1]={vsa_tau[-1].item():.2f}  '
          f'ratio={vsa_tau[-1].item() / vsa_tau[0].item():.1f}x  '
          f'tau_l_dev={td.mean().item():.4f} (std {td.std().item():.4f})')
    if b_d is not None:
        print(f'  b_d: mean={b_d.mean().item():.4f} range=[{b_d.min().item():.4f}, {b_d.max().item():.4f}]')
    if b_i is not None:
        print(f'  b_i: mean={b_i.mean().item():.4f} range=[{b_i.min().item():.4f}, {b_i.max().item():.4f}]')

    print('\nMIRROR INTERNALS (L0, mid, last):')
    for i in (0, len(model.layers) // 2, len(model.layers) - 1):
        m = model.layers[i].mirror
        print(f'  L{i}:')
        for attr in ['W_proj', 'W_out', 'w_gate', 'b_gate', 'log_scale', 'log_skip_alpha',
                     'w_help', 'alpha_diag', 'tanh_bias', 'mod_scale_mlp', 'mod_scale_mem']:
            if hasattr(m, attr):
                t = getattr(m, attr).data
                info = f'shape={list(t.shape)} mean={t.mean().item():.4f} std={t.std().item():.4f}'
                if t.numel() <= 10:
                    info += f' vals={[round(v, 3) for v in t.flatten().tolist()]}'
                elif attr == 'tanh_bias':
                    info += f' min={t.min().item():.4f} max={t.max().item():.4f}'
                print(f'    {attr:16s}: {info}')

    lm = getattr(model, 'lm_head', None)
    if lm is not None:
        print('\nLM HEAD:')
        for name, p in lm.named_parameters():
            print(f'  {name:24s}: shape={list(p.data.shape)} mean={p.data.mean().item():.6f} '
                  f'std={p.data.std().item():.6f}')

    all_vals = torch.cat([p.data.flatten() for p in model.parameters()])
    print(f'\nTENSORS: {all_vals.numel()} scalars | mean={all_vals.mean().item():.6f} '
          f'std={all_vals.std().item():.6f} | min={all_vals.min().item():.6f} '
          f'max={all_vals.max().item():.6f}')
    q = torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99])
    try:
        quants = torch.quantile(all_vals, q)
        print('  quantiles: ' + '  '.join(f'Q{qi * 100:3.0f}={qv:.6f}'
                                          for qi, qv in zip(q.tolist(), quants.tolist())))
    except Exception:
        pass
    print(f'  NaN: {int(torch.isnan(all_vals).sum())}  Inf: {int(torch.isinf(all_vals).sum())}')

    if model.explicit_reasoning:
        g = getattr(model, '_reasoning_gates', None)
        print(f'\nREASONING: enabled_step={model.reasoning_enabled_step} '
              f'scale={model.reasoning_scale:.4f}  last gates={g}')

    if model.layers[0].mirror._has_private_mem:
        run_inspector(model)


# ─────────────────────────── INSPECTOR (private memory) ───────────────────────────

def run_inspector(model):
    m0 = model.layers[0].mirror
    w = torch.softmax(m0._signal_log_weights, dim=0)
    print('\nINSPECTOR (private memory):')
    print(f'  signals: ' + '  '.join(f'{n}={w[i].item():.3f}'
                                     for i, n in enumerate(['temp', 'pred', 'smooth', 'sym', 'help'])))
    print(f'  w_help(sigmoid)={torch.sigmoid(m0.w_help).mean().item():.3f}  '
          f'w_contra={m0.w_contra.mean().item():.4f}')
    if m0._concept_sim_ema is not None:
        cs = m0._concept_sim_ema
        print(f'  concept_sim: mean={cs.mean().item():.4f} std={cs.std().item():.4f} '
              f'diag={cs.diag().mean().item():.4f}')
    if m0._behavior_div_ema is not None:
        print(f'  behavior_div: {m0._behavior_div_ema.mean().item():.4f}')
    if m0._trust_matrix is not None:
        tr = m0._trust_matrix
        print(f'  trust: mean={tr.mean().item():.4f} diag={tr.diag().mean().item():.4f}')
    if m0._cached_dominance is not None:
        print(f'  dominance: {[round(x, 3) for x in m0._cached_dominance.tolist()]}')
    if m0._cached_isolation is not None:
        print(f'  isolation: {[round(x, 3) for x in m0._cached_isolation.tolist()]}')
    if m0._cached_contra_expert is not None:
        print(f'  contra_expert: {[round(x, 3) for x in m0._cached_contra_expert.tolist()]}')
    if hasattr(m0, '_pm_step'):
        print(f'  pm_step: {int(m0._pm_step.item())}/{m0._pm_write_delay}')


# ─────────────────────────── WAKE ───────────────────────────

REF_STEP = 1398
W_STD_REF = 0.0705
DECAY_RATE = 1.6e-6
EMPTY_SLOT_LAYERS = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
SIG = {'PASS': 'PASS ', 'WATCH': 'WATCH', 'WAKE': 'WAKE '}


def _verdict(report, flag, hits, text):
    report.append(f'  [{SIG[flag]}] {text}')
    for h in hits:
        report.append(f'        {h}')


def run_wake(model, ckpt):
    step = int(ckpt.get('step', 0))
    n_layers = len(model.layers)
    report = [f'Wake-up scan: step={step} layers={n_layers}']

    mlp_wstd, gate_mlp, gate_mem, slot_occ, temp_vals, mat_counts = [], [], [], {}, [], []
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
            slot_occ[i] = int((cl.N_s.data > 0).sum().item())
            if hasattr(cl, '_temp') and cl._temp.ndim == 0:
                temp_vals.append(cl._temp.data.item())
            mat = getattr(cl, '_mature_count', None)
            if mat is not None:
                mat_counts.append(mat)

    wstd = sum(mlp_wstd) / len(mlp_wstd)
    expected = W_STD_REF * torch.exp(torch.tensor(-DECAY_RATE * (step - REF_STEP))).item()
    dev = wstd - expected
    report.append(f'  MLP W_std mean={wstd:.4f} decay-expected={expected:.4f} dev={dev:+.4f}')
    _verdict(report, 'WAKE' if dev > 0.001 else ('WATCH' if dev > 0.0005 else 'PASS'), [],
             f'MLP W_std vs decay curve (marker #1, dev {dev:+.4f})')

    per_layer = sorted(((i, mlp_wstd[i] - expected) for i in range(n_layers)), key=lambda t: -t[1])
    top = per_layer[:3]
    _verdict(report, 'WAKE' if top[0][1] > 0.002 else ('WATCH' if top[0][1] > 0.001 else 'PASS'),
             [f'L{li}: W_std dev {d:+.5f}' for li, d in top],
             f'per-layer W_std deviation (marker #1b, worst {top[0][1]:+.5f})')

    g_max = sorted(((i, torch.sigmoid(l.mirror.mod_scale_mlp.data).max().item())
                    for i, l in enumerate(model.layers)), key=lambda t: -t[1])
    gtop = g_max[:3]
    _verdict(report, 'WAKE' if gtop[0][1] > 0.75 else ('WATCH' if gtop[0][1] > 0.72 else 'PASS'),
             [f'L{li}: gate max {v:.3f}' for li, v in gtop],
             f'per-layer gate max (marker #2b, worst {gtop[0][1]:.3f}, wake >0.75)')

    g_mlp = sum(gate_mlp) / len(gate_mlp)
    g_mem = sum(gate_mem) / len(gate_mem) if gate_mem else float('nan')
    report.append(f'  sigmoid(mod_scale_mlp) mean={g_mlp:.3f}  '
                  f'sigmoid(mod_scale_mem) mean={g_mem:.3f}')
    _verdict(report, 'WAKE' if g_mlp > 0.75 else ('WATCH' if g_mlp > 0.72 else 'PASS'), [],
             f'modulation gate (marker #2, >0.75 = WAKE, baseline 0.668)')
    if abs(g_mlp - g_mem) > 0.2:
        _verdict(report, 'WAKE', [], f'mlp/mem gate divergence {abs(g_mlp - g_mem):.2f}')

    births = [i for i in EMPTY_SLOT_LAYERS if i in slot_occ and slot_occ[i] > 0]
    full = [i for i, o in slot_occ.items() if o >= 8]
    report.append(f'  slots occupied: {sum(slot_occ.values())}/192, full layers: {len(full)}, '
                  f'births in empty: {births}')
    _verdict(report, 'WAKE' if births else 'PASS', [],
             f'slot births in empty layers L10-23: {births or "none"}')
    if temp_vals:
        t_mean = sum(temp_vals) / len(temp_vals)
        report.append(f'  _temp mean={t_mean:.3f}')
        if t_mean < 1.95:
            _verdict(report, 'WATCH', [], f'_temp dropped from 2.0 baseline ({t_mean:.3f})')

    if mat_counts:
        n_zero = sum(1 for c in mat_counts if c == 0)
        report.append(f'  _mature_count: {n_zero}/{len(mat_counts)} layers locked (0)')
        if n_zero < len(mat_counts) * 0.5:
            _verdict(report, 'WATCH', [], 'maturity lock released in >half layers')

    lm = getattr(model, 'lm_head', None)
    if lm is not None:
        log_temp = None
        for name, p in lm.named_parameters():
            if 'log_temp' in name:
                log_temp = p.data
        if log_temp is not None:
            report.append(f'  log_temp mean={log_temp.mean().item():+.4f} (ref +0.0188 at 1398)')
            if log_temp.mean().item() > 0.25:
                _verdict(report, 'WATCH', [], 'log_temp rising (softmax flattening)')

    hdr = 'OK' if not any(r.startswith('  [WAKE') for r in report) else 'WAKE-CANDIDATE'
    print(f'\nWAKE DETECTOR (verdict: {hdr})')
    for line in report:
        print(line)


# ─────────────────────────── LIVE ───────────────────────────

def run_live(model, cfg, batch=1, seq=128):
    device = 'cpu'
    model.to(device)
    model.train()
    x = torch.randint(0, cfg.vocab, (batch, seq), device=device)
    h = model.embed_tokens(x)
    h_out, _, _ = model(h)

    sec('LIVE (forward на случайном входе)')
    print(f'INPUT/OUTPUT:  in_norm={h.norm(dim=-1).mean().item():.3f}  '
          f'out_norm={h_out.norm(dim=-1).mean().item():.3f}  '
          f'out_std={h_out.std().item():.4f}')
    tgt = torch.randint(1, cfg.vocab, (batch, seq), device=device)
    ls, aux = model.compute_losses(h_out[:, :-1], tgt[:, 1:])
    print(f'CE(random)={ls.item():.3f}  pred={aux["pred"]:.3f}  '
          f'gate_l1={aux["gate_l1"]:.5f}  balance={aux["balance"]:.5f}')

    print()
    hdr = f'{"L":>3s} {"||hp||":>8s} {"predMSE":>8s} {"|mirror|":>9s} {"gate":>7s} ' \
          f'{"gate_var":>9s} {"|1-a|":>7s} {"ls_expM":>8s} {"ls_std":>7s} {"skip":>6s} ' \
          f'{"w_delta":>7s}'
    print(hdr)
    print('-' * len(hdr))
    for i, layer in enumerate(model.layers):
        m = layer.mirror
        hp = m._cached_hp
        pk = m._cached_pred_k
        print(f'{i:3d} {hp.norm(dim=-1).mean().item():8.3f} '
              f'{F.mse_loss(pk, hp.detach()).item():8.3f} '
              f'{m._last_magnitude.item():9.3f} {m._cached_gate_l1.item():7.5f} '
              f'{m._last_gates.var().item():9.5f} '
              f'{(1 - m.alpha_diag.data).abs().mean().item():7.4f} '
              f'{m.log_scale.data.exp().max().item():8.2f} '
              f'{m.log_scale.data.std().item():7.4f} '
              f'{math.exp(m.log_skip_alpha.data.mean().item()):6.3f} '
              f'{m.w_delta_gate.data.abs().mean().item():7.5f}')

    print('\nSIGNALS (sigmoid + softmax share) L0, mid, last:')
    names = ['temp', 'pred', 'smooth', 'sym'] + (
        ['help'] if model.layers[0].mirror._has_private_mem else [])
    for i in (0, len(model.layers) // 2, len(model.layers) - 1):
        m = model.layers[i].mirror
        w = torch.sigmoid(m._signal_log_weights)
        p = torch.softmax(m._signal_log_weights, dim=0)
        print(f'  L{i}: sigmoid={["%.3f" % v for v in w.tolist()]}  '
              f'share={["%.3f" % v for v in p.tolist()]}  names={names}')

    print('\nMIRROR PARAMS L0, mid, last:')
    for i in (0, len(model.layers) // 2, len(model.layers) - 1):
        m = model.layers[i].mirror
        print(f'  L{i}: alpha_diag mean={m.alpha_diag.data.mean().item():.4f} '
              f'min={m.alpha_diag.data.min().item():.4f} max={m.alpha_diag.data.max().item():.4f} | '
              f'log_scale mean={m.log_scale.data.mean().item():.4f} '
              f'max={m.log_scale.data.max().item():.4f} | '
              f'tanh_bias mean={m.tanh_bias.data.mean().item():.4f} '
              f'max={m.tanh_bias.data.max().item():.4f} | '
              f'b_gate={m.b_gate.data.mean().item():.4f} gate_bias={m.gate_bias.data.mean().item():.4f} | '
              f'mod_mlp(σ)={torch.sigmoid(m.mod_scale_mlp.data).mean().item():.3f} '
              f'mod_mem(σ)={torch.sigmoid(m.mod_scale_mem.data).mean().item():.3f} | '
              f'gate_ema={m._gate_ema.data.mean().item():.4f}')
        if m._has_private_mem:
            print(f'      w_help(σ)={torch.sigmoid(m.w_help.data).mean().item():.3f} '
                  f'w_contra={m.w_contra.data.mean().item():.4f} '
                  f'pm_norm={m._private_mem.norm(dim=-1).mean().item():.3f}')

    vsa_log = model._vsa_log_param
    vsa_tau = torch.exp(torch.cumsum(F.softplus(vsa_log), dim=0)) + 1.0
    td = model._tau_l_dev.data
    print(f'\nGLOBAL: vsa tau=[{vsa_tau[0].item():.2f}, {vsa_tau[-1].item():.2f}]  '
          f'tau_l_dev={td.mean().item():.4f} (std {td.std().item():.4f})')
    pm = getattr(model, '_private_mem', None)
    if pm is not None:
        print(f'  model private_mem norm: {pm.norm(dim=-1).mean().item():.3f}')


# ─────────────────────────── HEAD ───────────────────────────

def _head_forward(model, window, tok):
    with torch.no_grad():
        h = model.embed_tokens(window.unsqueeze(0))
        out, _, _ = model(h, None, adaptive=False)
        return model.lm_head(out[0])


def _cat(tid, tok):
    tstr = tok.decode([int(tid)]).strip()
    return 'punct' if PUNCT.match(tstr) else ('word' if WORD.search(tstr) else 'other')


def run_head(model, ckpt, args, tok):
    sec('HEAD: декомпозиция bias vs контекст + позиционная карта + A/B reasoning')
    model.eval()
    text = 'Москва — столица России, и в ней живут миллионы людей. ' * 60
    ids = tok.encode(text).ids
    L = model.cfg.seq_len
    n_w = max(args.windows, 1)
    log_temp = model.lm_head.log_temp.data.mean().item()
    t_eff = args.temp / math.exp(log_temp)
    tb = model.lm_head.token_bias.data
    tb_argmax = int(tb.argmax())
    print(f'  log_temp={log_temp:.4f} -> t_eff={t_eff:.4f} | '
          f'token_bias top: {[repr(tok.decode([int(i)])) for i in torch.topk(tb, 5).indices.tolist()]}')

    stats = {'match': 0, 'total': 0, 'ctx_word': 0, 'ctx_punct': 0,
             'full_word': 0, 'full_punct': 0}
    posmap = {}
    for start in range(0, min(len(ids) - L, n_w * L), L):
        window = torch.tensor(ids[start:start + L], dtype=torch.long)
        logits = _head_forward(model, window, tok)
        ctx = logits - tb.unsqueeze(0)
        pr = F.softmax((logits.double() / t_eff), dim=-1)
        top1 = pr.max(dim=-1)
        H = -(pr * torch.log2(pr.clamp_min(1e-12))).sum(dim=-1)
        for pos in range(L):
            d = posmap.setdefault(pos, {'top1': 0.0, 'H': 0.0, 'cnt': 0, 'punct': 0, 'word': 0})
            d['top1'] += top1.values[pos].item()
            d['H'] += H[pos].item()
            d['cnt'] += 1
            if _cat(top1.indices[pos], tok) == 'punct':
                d['punct'] += 1
            elif _cat(top1.indices[pos], tok) == 'word':
                d['word'] += 1
        for pos in range(0, L, 8):
            lt = logits[pos]
            stats['total'] += 1
            if int(lt.argmax()) == tb_argmax:
                stats['match'] += 1
            for tid, cat, k in ((int(ctx[pos].argmax()), 'ctx', 'ctx'),
                                (int(lt.argmax()), 'full', 'full')):
                c = _cat(tid, tok)
                stats[f'{k}_word'] += c == 'word'
                stats[f'{k}_punct'] += c == 'punct'

    print(f'\n  BIAS-DECOMP: argmax==bias в {stats["match"]}/{stats["total"]} '
          f'({100.0 * stats["match"] / max(stats["total"], 1):.1f}%) | '
          f'ctx: word={stats["ctx_word"]} punct={stats["ctx_punct"]} | '
          f'full: word={stats["full_word"]} punct={stats["full_punct"]}')

    print('\n  POSMAP: pos | top-1 avg | H avg | punct% | word%')
    bins = [(0, 0, 15), (16, 15, 33), (32, 31, 65), (64, 63, 129), (128, 127, 193),
            (192, 191, 225), (224, 223, 241), (240, 239, 249), (250, 249, L - 1)]
    for label, lo, hi in bins:
        span = hi - lo + 1
        top1 = sum(posmap[p]['top1'] / posmap[p]['cnt'] for p in range(lo, hi + 1)) / span
        H = sum(posmap[p]['H'] / posmap[p]['cnt'] for p in range(lo, hi + 1)) / span
        pc = sum(posmap[p]['punct'] for p in range(lo, hi + 1)) / span * 100
        wc = sum(posmap[p]['word'] for p in range(lo, hi + 1)) / span * 100
        print(f'  {label:3d} | {top1:.4f} | {H:.3f} | {pc:5.1f}% | {wc:5.1f}%')

    print('\n  A/B REASONING (одно окно, top-1/H + KL):')
    win = torch.tensor(ids[:L], dtype=torch.long)
    res = {}
    for mode, override in (('OFF', 0.0), ('natural', None), ('FULL', 1.0)):
        model.reasoning_scale_override = override
        model.reset_reasoning()
        logits = _head_forward(model, win, tok)[-1]
        p = F.softmax(logits.double(), dim=-1)
        pe = F.softmax((logits.double() / t_eff), dim=-1)
        Hr = -(p * torch.log2(p.clamp_min(1e-12))).sum().item()
        He = -(pe * torch.log2(pe.clamp_min(1e-12))).sum().item()
        res[mode] = (p.max().item(), Hr, pe.max().item(), He, p)
        print(f'    {mode:8s} top-1={p.max().item():.4f} H={Hr:.3f} bit | '
              f'sampler top-1={pe.max().item():.4f} H={He:.3f} bit')
    model.reasoning_scale_override = None
    pn, po = res['natural'][4].clamp_min(1e-12), res['OFF'][4].clamp_min(1e-12)
    print(f'    KL natural||OFF = {(pn * torch.log2(pn / po)).sum().item():.4f} bit | '
          f'KL OFF||natural = {(po * torch.log2(po / pn)).sum().item():.4f} bit')


# ─────────────────────────── CMP ───────────────────────────

def run_cmp(models, args, tok):
    sec('CMP: сравнение на одинаковом входе')
    prompt = args.prompt or 'В один холодный зимний вечер'
    ids = tok.encode(prompt).ids
    for ckpt_path, (model, _state) in models.items():
        model.eval()
        L = model.cfg.seq_len
        ctx = torch.tensor(ids[-L:], dtype=torch.long).unsqueeze(0)
        log_temp = model.lm_head.log_temp.data.mean().item()
        tb = model.lm_head.token_bias.data
        ro = model.lm_head.readout.data
        print(f'\n  {os.path.basename(ckpt_path)} | step={_state.get("step", "?")} | '
              f'reasoning t={model.reasoning_enabled_step} scale={model.reasoning_scale:.4f}')
        print(f'    head: log_temp={log_temp:.4f} | tb_max={tb.abs().max().item():.3f} '
              f'tb_std={tb.std().item():.3f} | readout_norm={ro.norm().item():.3f}')
        for ov, tag in ((0.0, 'reasoning OFF'), (None, 'natural'), (1.0, 'FULL')):
            model.reasoning_scale_override = ov
            model.reset_reasoning()
            with torch.no_grad():
                h = model.embed_tokens(ctx)
                out, _, _ = model(h, None, adaptive=False)
                logits = model.lm_head(out[:, -1:, :])[0, 0]
            eff = args.temp / math.exp(log_temp)
            for tname, z in (('raw', logits.double()), ('sampler', logits.double() / eff)):
                pr = F.softmax(z, dim=-1)
                topk = torch.topk(pr, 8)
                toks = [tok.decode([int(i)]) for i in topk.indices.tolist()]
                print(f'    [{tag}] {tname:7s} top-1={pr.max().item():.4f} '
                      f'H={-(pr * torch.log2(pr.clamp_min(1e-12))).sum().item():.3f} bit | '
                      f'{" | ".join(f"{t!r}({v:.3f})" for t, v in zip(toks, topk.values.tolist()))}')
        model.reasoning_scale_override = None


# ─────────────────────────── MAIN ───────────────────────────

def main():
    ap = argparse.ArgumentParser(description='WideBind checkpoint analyzer (все методы)')
    ap.add_argument('checkpoints', nargs='+', help='path(s) to .pt')
    ap.add_argument('--no-live', action='store_true', help='skip live forward dissection')
    ap.add_argument('--no-head', action='store_true', help='skip head analysis (bias/posmap/A-B)')
    ap.add_argument('--quick', action='store_true', help='static + wake only')
    ap.add_argument('--windows', type=int, default=2)
    ap.add_argument('--prompt', type=str, default='')
    ap.add_argument('--temp', type=float, default=0.8)
    ap.add_argument('--seq', type=int, default=128, help='live forward seq_len')
    args = ap.parse_args()

    tok = None
    if not args.no_head and not args.quick:
        try:
            from generate import load_russian_tokenizer
            tok = load_russian_tokenizer()
        except Exception as e:
            print(f'[warn] токенизатор недоступен ({e}), head-анализ пропущен')
            tok = None

    models = {}
    for path in args.checkpoints:
        print(f'\n# {path}')
        try:
            ckpt, cfg, model, missing, unexpected = load_ckpt(path)
        except Exception as e:
            print(f'[error] не удалось загрузить: {e}')
            continue
        ckpt['_path'] = path
        models[path] = (model, ckpt)
        try:
            run_static(ckpt, cfg, model, missing, unexpected, tok)
        except Exception as e:
            print(f'[error] static: {e}')
        try:
            run_wake(model, ckpt)
        except Exception as e:
            print(f'[error] wake: {e}')
        if not args.quick:
            if not args.no_live:
                try:
                    run_live(model, cfg, seq=args.seq)
                except Exception as e:
                    print(f'[error] live: {e}')
            if tok is not None:
                try:
                    run_head(model, ckpt, args, tok)
                except Exception as e:
                    print(f'[error] head: {e}')

    if len(models) > 1 and tok is not None:
        try:
            run_cmp(models, args, tok)
        except Exception as e:
            print(f'[error] cmp: {e}')


if __name__ == '__main__':
    main()